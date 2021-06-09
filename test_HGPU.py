
import os
import glob

import torch
from PIL import Image
from tqdm import tqdm
from scipy.misc import imresize
from torchvision import transforms
from libs.utils.utils import check_parallel
from libs.utils.utils import load_checkpoint_epoch

from libs.model.HGPU import EncoderNet, DecoderNet


def func(listTemp, n):
    for i in range(0, len(listTemp), n):
        yield listTemp[i:i + n]


def print_list_davis(imagefile):
    temp = []
    imagefiles = []
    temp.extend(imagefile[::])
    temp.extend(imagefile[1: -1:])
    temp.sort()
    li = func(temp, 2)
    for i in li:
        imagefiles.append(i)
    return imagefiles


def flip(x, dim):
    if x.is_cuda:
        return torch.index_select(x, dim, torch.arange(x.size(dim) - 1, -1, -1).\
                                  long().cuda())
    else:
        return torch.index_select(x, dim, torch.arange(x.size(dim) - 1, -1, -1).\
                                  long())


def test():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tr = transforms.ToTensor()
    image_transforms = transforms.Compose([tr, normalize])

    encoder_dict, decoder_dict = load_checkpoint_epoch(model_name, epoch, use_gpu=True, load_opt=False)

    encoder = EncoderNet()
    decoder = DecoderNet()
    encoder_dict, decoder_dict = check_parallel(encoder_dict, decoder_dict)
    encoder.load_state_dict(encoder_dict)
    decoder.load_state_dict(decoder_dict)

    encoder.cuda()
    decoder.cuda()

    encoder.train(False)
    decoder.train(False)

    for video in tqdm(seqs):
            im_dir = os.path.join(davis_image_dir, video)
            flow_dir = os.path.join(davis_flow_dir, video)

            imagefile = sorted(glob.glob(os.path.join(im_dir, '*.jpg')))
            imagefiles = []
            imagefiles.extend(print_list_davis(imagefile))

            flowfiles = sorted(glob.glob(os.path.join(flow_dir, '*.png')))

            with torch.no_grad():
                for imagefile, flowfile in zip(imagefiles, flowfiles):
                    im1 = Image.open(imagefile[0]).convert('RGB')
                    im2 = Image.open(imagefile[1]).convert('RGB')
                    flow = Image.open(flowfile).convert('RGB')
                    width, height = im1.size

                    im1 = imresize(im1, img_size)
                    im2 = imresize(im2, img_size)
                    flow = imresize(flow, img_size)

                    im1 = image_transforms(im1)
                    im2 = image_transforms(im2)
                    flow = image_transforms(flow)

                    im1 = im1.unsqueeze(0)
                    im2 = im2.unsqueeze(0)
                    flow = flow.unsqueeze(0)

                    im1, im2, flow = im1.cuda(), im2.cuda(), flow.cuda()

                    h5_1, h4_1, h3_1, h2_1, \
                    h5_2, h4_2, h3_2, h2_2, \
                    h5_3, h4_3, h3_3, h2_3 = encoder(im1, im2, flow)

                    mask_1, mask_2 = decoder(h5_1, h4_1, h3_1, h2_1,
                                             h5_2, h4_2, h3_2, h2_2,
                                             h5_3, h4_3, h3_3, h2_3)

                    if use_flip:
                        im1_flip = flip(im1, 3)
                        im2_flip = flip(im2, 3)
                        flow_flip = flip(flow, 3)
                        h5_1, h4_1, h3_1, h2_1, \
                        h5_2, h4_2, h3_2, h2_2, \
                        h5_3, h4_3, h3_3, h2_3 = encoder(im1_flip, im2_flip, flow_flip)
                        mask_flip_1, mask_flip_2 = decoder(h5_1, h4_1, h3_1, h2_1,
                                                           h5_2, h4_2, h3_2, h2_2,
                                                           h5_3, h4_3, h3_3, h2_3)
                        mask_flip_1 = flip(mask_flip_1, 3)
                        mask_flip_2 = flip(mask_flip_2, 3)
                        mask_1 = (mask_1 + mask_flip_1) / 2.0
                        mask_2 = (mask_2 + mask_flip_2) / 2.0

                    mask_1 = mask_1[0, 0, :, :]
                    mask_2 = mask_2[0, 0, :, :]
                    mask_1 = Image.fromarray(mask_1.cpu().detach().numpy() * 255).convert('L')
                    mask_2 = Image.fromarray(mask_2.cpu().detach().numpy() * 255).convert('L')

                    save_mask_folder = '{}/{}_epoch{}/{}'.format(davis_mask_dir, model_name, epoch, video)
                    if not os.path.exists(save_mask_folder):
                        os.makedirs(save_mask_folder)

                    save_file1 = os.path.join(save_mask_folder,
                                              os.path.basename(imagefile[0])[:-4] + '.png')
                    save_file2 = os.path.join(save_mask_folder,
                                              os.path.basename(imagefile[1])[:-4] + '.png')

                    mask_1 = mask_1.resize((width, height))
                    mask_2 = mask_2.resize((width, height))
                    mask_1.save(save_file1)
                    mask_2.save(save_file2)


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    img_size = (512, 512)
    use_flip = True

    model_name = 'HGPU'
    epoch = 0.8394088008745166

    val_set_v1 = './libs/dataset/val_v1.txt'
    val_set_v2 = './libs/dataset/val_v2.txt'
    davis_image_dir = '/YourPath/DAVIS/JPEGImages/480p'
    davis_flow_dir = '/YourPath/DAVIS/davis-flow'
    davis_mask_dir = './outputs/DAVIS-16'

    EncoderNet.flag = 'pre'
    with open(val_set_v1) as f:
        seqs = f.readlines()
        seqs = [seq.strip() for seq in seqs]
    test()

    EncoderNet.flag = 'main'
    with open(val_set_v2) as f:
        seqs = f.readlines()
        seqs = [seq.strip() for seq in seqs]
    test()