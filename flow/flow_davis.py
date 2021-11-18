import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
import torch.backends.cudnn as cudnn


DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def run(flo, save_file):
    flo = flo[0].permute(1,2,0).cpu().numpy()
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    cv2.imwrite(save_file, flo)

def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model = torch.nn.DataParallel(RAFT(args), device_ids=[0])
    model.load_state_dict(torch.load(args.model))
    cudnn.benchmark = True

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():

        davis_folder = 'Your_Path/YouTube-VOS/train/JPEGImages'
        save_dir = 'Your_Path/YouTube-VOS/train/YouTube-flow'


        videos = os.listdir(davis_folder)

        for idx, video in enumerate(videos):
            print('process {}[{}/{}]'.format(video, idx, len(videos)))
            save_dir_video = os.path.join(save_dir, video)
            if not os.path.exists(save_dir_video):
                os.makedirs(save_dir_video)

            imagefiles = sorted(glob.glob(os.path.join(davis_folder, video, '*.jpg')))
            for i in range(len(imagefiles)-1):
                f1 = imagefiles[i]
                f2 = imagefiles[i+1]

                save_name = os.path.basename(f1)[:-4] + '_' + os.path.basename(f2)[:-4] + '.png'
                save_file = os.path.join(save_dir_video, save_name)
                if not os.path.exists(save_file):

                   image1 = load_image(f1)
                   image2 = load_image(f2)
               
                   padder = InputPadder(image1.shape)
                   image1, image2 = padder.pad(image1, image2)
                
                   flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
                   run(flow_up, save_file)
                else:
                    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    args = parser.parse_args()

    demo(args)
