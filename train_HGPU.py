import argparse

from libs.dataset.data import get_dataset

from libs.utils.obj import Loss
from libs.utils.eval import db_eval_iou_multi
from libs.utils.utils import make_dir
from libs.utils.utils import get_optimizer
from libs.utils.utils import check_parallel, save_checkpoint_epoch, load_checkpoint_epoch
from libs.model.HGPU import EncoderNet, DecoderNet

from tqdm import tqdm
import torch
from torch.utils import data
from torchvision import transforms

import os
import sys
import time
import random
import numpy as np

import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
import apex
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn


parser = argparse.ArgumentParser(description='HGPU')

parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument('-year', dest='year', default='2017')
parser.add_argument('-imsize', dest='imsize', default=480, type=int)
parser.add_argument('-batch_size', dest='batch_size', default=4, type=int)
parser.add_argument('-num_workers', dest='num_workers', default=8, type=int)

## TRAINING parameters ##
parser.add_argument('--pretrain', dest='pretrain', default=True)
parser.add_argument('--resume', dest='resume', action='store_true',
                    help=('whether to resume training an existing model'
                          '(the one with name model_name will be used)'))
parser.set_defaults(resume=False)
parser.add_argument('-seed', dest='seed', default=123, type=int)
# parser.add_argument('-gpu_id', dest='gpu_id', default=0, type=int)
parser.add_argument('-lr', dest='lr', default=1e-2, type=float)
parser.add_argument('-lr_cnn', dest='lr_cnn', default=1e-3, type=float)
parser.add_argument('-optim_cnn', dest='optim_cnn', default='sgd',
                    choices=['adam', 'sgd', 'rmsprop'])
parser.add_argument('-momentum', dest='momentum', default=0.9, type=float)
parser.add_argument('-weight_decay', dest='weight_decay', default=5e-4,
                    type=float)
parser.add_argument('-weight_decay_cnn', dest='weight_decay_cnn',
                    default=5e-4, type=float)
parser.add_argument('-optim', dest='optim', default='sgd',
                    choices=['adam', 'sgd', 'rmsprop'])
parser.add_argument('--crop', dest='crop', action='store_true')
parser.set_defaults(crop=False)

parser.add_argument('--update_encoder', dest='update_encoder',
                    action='store_true',
                    help='used in sync with finetune_after.'
                         ' no need to activate.')
parser.set_defaults(update_encoder=True)

parser.add_argument('-max_epoch', dest='max_epoch', default=25, type=int)

# visualization and logging
parser.add_argument('-print_every', dest='print_every', default=10,
                    type=int)

# loss weights
parser.add_argument('-iou_weight', dest='iou_weight', default=1.0,
                    type=float)
# augmentation
parser.add_argument('--augment', dest='augment', action='store_true')
parser.set_defaults(augment=True)
parser.add_argument('-rotation', dest='rotation', default=10, type=int)
parser.add_argument('-translation', dest='translation', default=0.1,
                    type=float)
parser.add_argument('-shear', dest='shear', default=0.1, type=float)
parser.add_argument('-zoom', dest='zoom', default=0.7, type=float)

# GPU
parser.add_argument('--cpu', dest='use_gpu', action='store_false')
parser.set_defaults(use_gpu=True)
parser.add_argument('-ngpus', dest='ngpus', default=2, type=int)

parser.add_argument('-model_name', dest='model_name', default='model')
parser.add_argument('-log_file', dest='log_file', default='train.log')

# dataset parameters
parser.add_argument('--resize', dest='resize', action='store_true')
parser.set_defaults(resize=False)
parser.add_argument('-dataset', dest='dataset', default='youtube',
                    choices=['davis2017', 'youtube'])

# testing
parser.add_argument('-eval_split', dest='eval_split', default='test')
parser.add_argument('-mask_th', dest='mask_th', default=0.5, type=float)
parser.add_argument('-max_dets', dest='max_dets', default=100, type=int)
parser.add_argument('-min_size', dest='min_size', default=0.001,
                    type=float)
parser.add_argument('--display', dest='display', action='store_true')
parser.add_argument('--no_display_text', dest='no_display_text',
                    action='store_true')
parser.set_defaults(display=False)
parser.set_defaults(display_route=False)
parser.set_defaults(no_display_text=False)
parser.set_defaults(use_gt_masks=False)


def train_dataloader(args):
    train_loader = {}
    batch_size = args.batch_size
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    image_transforms = transforms.Compose([to_tensor, normalize])
    target_transforms = transforms.Compose([to_tensor])
    split = 'train'
    train_set = get_dataset(
        args, split=split, image_transforms=image_transforms,
        target_transforms=target_transforms,
        augment=args.augment and split == 'train',
        input_size=(512, 512), pre_train=args.pretrain)
    train_sample = data.distributed.DistributedSampler(train_set)
    train_loader = data.DataLoader(train_set,
                                   batch_size=batch_size,
                                   num_workers=args.num_workers,
                                   drop_last=True,
                                   pin_memory=True,
                                   sampler=train_sample,
                                   )
    return train_loader


def test_dataloader(args):
    test_loader = {}
    batch_size = args.batch_size
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    image_transforms = transforms.Compose([to_tensor, normalize])
    target_transforms = transforms.Compose([to_tensor])
    split = 'val'
    test_set = get_dataset(
        args, split=split, image_transforms=image_transforms,
        target_transforms=target_transforms,
        augment=args.augment and split == 'train',
        input_size=(512, 512), pre_train=args.pretrain)

    test_loader = data.DataLoader(test_set,
                                  batch_size=batch_size,
                                  num_workers=args.num_workers,
                                  drop_last=True,
                                  pin_memory=True,
                                  )
    return test_loader


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


args = parser.parse_args()
local_rank = args.local_rank
args.nprocs = torch.cuda.device_count()

torch.manual_seed(args.seed)
random.seed(args.seed)


################## SETUP #################
args.model_name = 'HGPU'
args.batch_size = 5
args.num_works = 8
args.max_epoch = 30
args.year = '2016'
args.lr = 1e-2
args.lr_cnn = 1e-3

# first pre-training on YouTube-VOS (pretrain = True).
# max_epoch all sets to 25.
args.pretrain = True
EncoderNet.flag = 'pre'

# then main-training on DAVIS-16 dataset (pretrain = False)
# epoch_resume set to miou and uncomment the following three lines.

# args.resume = False
# EncoderNet.flag = 'main'
# args.epoch_resume = 0.7467677482618633  # miou

##########################################

torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')
cudnn.benchmark = True

train_loader = train_dataloader(args)
test_loader = test_dataloader(args)

model_dir = os.path.join('model/', args.model_name)
make_dir(model_dir)

epoch_resume = 0
if args.resume and dist.get_rank() == 0:

    encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, load_args = \
        load_checkpoint_epoch(args.model_name, args.epoch_resume,
                              args.use_gpu)
    epoch_resume = args.epoch_resume
    encoder = EncoderNet()
    decoder = DecoderNet()
    encoder_dict, decoder_dict = check_parallel(encoder_dict, decoder_dict)
    encoder.load_state_dict(encoder_dict)
    decoder.load_state_dict(decoder_dict)
    print(epoch_resume)
else:
    encoder = EncoderNet()
    decoder = DecoderNet()

encoder = encoder.to(local_rank)
decoder = decoder.to(local_rank)

criterion = Loss()
criterion = criterion.to(local_rank)

encoder_params = list(encoder.parameters())
decoder_params = list(decoder.parameters())
dec_opt = get_optimizer(args.optim, args.lr, decoder_params,
                        args.weight_decay)
enc_opt = get_optimizer(args.optim_cnn, args.lr_cnn, encoder_params,
                        args.weight_decay_cnn)

[encoder, decoder], [enc_opt, dec_opt] = amp.initialize([encoder, decoder], [enc_opt, dec_opt], opt_level="O0")

encoder = DDP(encoder, delay_allreduce=True)
decoder = DDP(decoder, delay_allreduce=True)

best_iou = 0
start = time.time()

iterator = tqdm(range(args.max_epoch))

for e in iterator:
    print("Epoch", e + 1)
    train_loader.sampler.set_epoch(e)

    epoch_losses = {'train': {'total': [], 'iou': []},
                    'val': {'total': [], 'iou': []}
                    }
    for split in ['train', 'val']:
        if split == 'train':
            encoder.train(True)
            decoder.train(True)
        else:
            encoder.train(False)
            decoder.train(False)

        if split == 'train':
            for batch_idx, (im1, im2, flow, ms1, ms2, negative_pixels1, negative_pixels2) in \
                    enumerate(train_loader):
                im1, im2, flow, mask1, mask2, negative_pixels1, negative_pixels2 = \
                    im1.to(local_rank), im2.to(local_rank), flow.to(local_rank), \
                    ms1.to(local_rank), ms2.to(local_rank), \
                    negative_pixels1.to(local_rank), negative_pixels2.to(local_rank)

                h5_1, h4_1, h3_1, h2_1, \
                h5_2, h4_2, h3_2, h2_2, \
                h5_3, h4_3, h3_3, h2_3 = encoder(im1, im2, flow)
                mask_1, mask_2 = decoder(h5_1, h4_1, h3_1, h2_1,
                                         h5_2, h4_2, h3_2, h2_2,
                                         h5_3, h4_3, h3_3, h2_3)

                mask_loss1 = criterion(mask_1, mask1, negative_pixels1)
                mask_loss2 = criterion(mask_2, mask2, negative_pixels2)

                loss = 0.5 * (mask_loss1 + mask_loss2)

                iou = db_eval_iou_multi(mask1.cpu().detach().numpy(), mask_1.cpu().detach().numpy())

                torch.distributed.barrier()
                reduce_loss = reduce_mean(loss, args.nprocs)

                dec_opt.zero_grad()
                enc_opt.zero_grad()
                with amp.scale_loss(loss, [enc_opt, dec_opt]) as scaled_loss:
                    scaled_loss.backward()
                enc_opt.step()
                dec_opt.step()

                epoch_losses[split]['total'].append(reduce_loss.data.item())
                epoch_losses[split]['iou'].append(iou)

                if local_rank == 0 and ((batch_idx + 1) % args.print_every == 0):
                    mt = np.mean(epoch_losses[split]['total'])
                    miou = np.mean(epoch_losses[split]['iou'])

                    te = time.time() - start
                    print('Epoch: [{}/{}][{}/{}]\tTime {:.3f}s\tLoss: {:.4f}''\tIOU: {:.4f}'.
                          format(e + 1, args.max_epoch, batch_idx,
                                 round(len(train_loader.sampler)/args.batch_size), te, mt, miou))

                    start = time.time()
        else:
            for batch_idx, (im1, im2, flow, ms1, ms2, negative_pixels1, negative_pixels2) in \
                    enumerate(test_loader):
                im1, im2, flow, mask1, mask2, negative_pixels1, negative_pixels2 = \
                    im1.to(local_rank), im2.to(local_rank), flow.to(local_rank), \
                    ms1.to(local_rank), ms2.to(local_rank), \
                    negative_pixels1.to(local_rank), negative_pixels2.to(local_rank)

                with torch.no_grad():
                    h5_1, h4_1, h3_1, h2_1, \
                    h5_2, h4_2, h3_2, h2_2, \
                    h5_3, h4_3, h3_3, h2_3 = encoder(im1, im2, flow)
                    mask_1, mask_2 = decoder(h5_1, h4_1, h3_1, h2_1,
                                             h5_2, h4_2, h3_2, h2_2,
                                             h5_3, h4_3, h3_3, h2_3)

                    mask_loss1 = criterion(mask_1, mask1, negative_pixels1)
                    mask_loss2 = criterion(mask_2, mask2, negative_pixels2)
                    loss = 0.5 * (mask_loss1 + mask_loss2)

                    iou = db_eval_iou_multi(mask1.cpu().detach().numpy(),
                                            mask_1.cpu().detach().numpy())

                torch.distributed.barrier()
                reduce_loss = reduce_mean(loss, args.nprocs)

                epoch_losses[split]['total'].append(reduce_loss.data.item())
                epoch_losses[split]['iou'].append(iou)

                if local_rank == 0 and ((batch_idx + 1) % args.print_every == 0):
                    mt = np.mean(epoch_losses[split]['total'])
                    miou = np.mean(epoch_losses[split]['iou'])

                    te = time.time() - start
                    print('Epoch: [{}/{}][{}/{}]\tTime {:.3f}s\tLoss: {:.4f}''\tIOU: {:.4f}'.
                          format(e + 1, args.max_epoch, batch_idx, len(test_loader), te, mt, miou))

                    start = time.time()

    miou = np.mean(epoch_losses['val']['iou'])
    if args.pretrain and dist.get_rank() == 0:
        save_checkpoint_epoch(args, encoder, decoder,
                              enc_opt, dec_opt, miou, False)
    else:
        if miou > best_iou and dist.get_rank() == 0:
            best_iou = miou
            print(best_iou)
            save_checkpoint_epoch(args, encoder, decoder,
                                  enc_opt, dec_opt, miou, False)
