
import os
import cv2
import glob
import lmdb
import numpy as np
from PIL import Image
import os.path as osp
from scipy.misc import imresize

from torch.utils import data

from torchvision import transforms
from .base import Sequence, Annotation
from libs.dataset import transform as tr

from libs.utils.config_davis import cfg as cfg_davis
from libs.utils.config_davis import db_read_sequences as db_read_sequences_davis
from libs.utils.config_youtubevos import cfg as cfg_youtubevos
from libs.utils.config_youtubevos import db_read_sequences_train as db_read_sequences_train_youtubevos


def func(listTemp, n):
    for i in range(0, len(listTemp), n):
        yield listTemp[i:i + n]


def print_list_davis(imagefile):
    temp = []
    imagefiles = []
    temp.extend(imagefile[: :])
    temp.extend(imagefile[1: -1:])
    temp.sort()
    li = func(temp, 2)
    for i in li:
        imagefiles.append(i)
    return imagefiles


class DataLoader(data.Dataset):

    def __init__(self, args, split, input_size, augment=False,
                 transform=None, target_transform=None, pre_train=False):
        self._year = args.year
        self._phase = split
        self.transform = transform
        self.target_transform = target_transform
        self.input_size = input_size
        self.augment = augment
        self.augment_transform = None
        self.pre_train = pre_train
        self._single_object = False

        assert args.year == "2017" or args.year == "2016"

        if augment:
            self.augment_transform = transforms.Compose([
                tr.RandomHorizontalFlip(),
                tr.ScaleNRotate(rots=(-args.rotation, args.rotation),
                                scales=(.75, 1.25))])

        self.image_files = []
        self.mask_files = []
        self.flow_files = []
        self.hed_files = []

        if split == 'train':
            if pre_train:
                self.load_youtubevos(args)
            else:
                self.load_davis(args)
        else:
            self.load_davis(args)

    def __len__(self):
        return len(self.flow_files)

    def __getitem__(self, index):

        image_file1 = self.image_files[index][0]
        image_file2 = self.image_files[index][1]
        mask_file1 = self.mask_files[index][0]
        mask_file2 = self.mask_files[index][1]
        flow_file = self.flow_files[index]
        hed_file1 = self.hed_files[index][0]
        hed_file2 = self.hed_files[index][1]

        image1 = Image.open(image_file1).convert('RGB')
        image2 = Image.open(image_file2).convert('RGB')
        flow = Image.open(flow_file).convert('RGB')

        mask1 = cv2.imread(mask_file1, 0)
        mask1[mask1 > 0] = 255
        mask2 = cv2.imread(mask_file2, 0)
        mask2[mask2 > 0] = 255

        hed1 = cv2.imread(hed_file1, 0)
        hed2 = cv2.imread(hed_file2, 0)

        # enlarge the object mask
        kernel = np.ones((11, 11), np.uint8)  # use a large kernel
        dilated_mask = cv2.dilate(mask1, kernel, iterations=1)
        inverse_dilated_mask = (255.0 - dilated_mask) / 255.0
        inverse_hed = (255.0 - hed1) / 255.0
        negative_pixels = inverse_hed * inverse_dilated_mask
        kernel = np.ones((5, 5), np.uint8)  # use a small kernel
        negative_pixels = cv2.dilate(negative_pixels, kernel, iterations=1)

        mask1 = Image.fromarray(mask1)
        negative_pixels1 = Image.fromarray(negative_pixels)

        kernel = np.ones((11, 11), np.uint8)  # use a large kernel
        dilated_mask = cv2.dilate(mask2, kernel, iterations=1)
        inverse_dilated_mask = (255.0 - dilated_mask) / 255.0
        inverse_hed = (255.0 - hed2) / 255.0
        negative_pixels = inverse_hed * inverse_dilated_mask
        kernel = np.ones((5, 5), np.uint8)  # use a small kernel
        negative_pixels = cv2.dilate(negative_pixels, kernel, iterations=1)

        mask2 = Image.fromarray(mask2)
        negative_pixels2 = Image.fromarray(negative_pixels)

        if self.input_size is not None:
            image1 = imresize(image1, self.input_size)
            flow = imresize(flow, self.input_size)
            mask1 = imresize(mask1, self.input_size, interp='nearest')
            negative_pixels1 = imresize(negative_pixels1, self.input_size, interp='nearest')

            image2 = imresize(image2, self.input_size)
            mask2 = imresize(mask2, self.input_size, interp='nearest')
            negative_pixels2 = imresize(negative_pixels2, self.input_size, interp='nearest')

        sample = {'image1': image1, 'image2': image2, 'flow': flow,
                  'mask1': mask1, 'mask2': mask2,
                  'negative_pixels1': negative_pixels1,'negative_pixels2': negative_pixels2}

        if self.augment_transform is not None:
            sample = self.augment_transform(sample)

        image1, image2, flow, mask1, mask2, negative_pixels1, negative_pixels2 =\
            sample['image1'], sample['image2'], sample['flow'],\
            sample['mask1'], sample['mask2'],\
            sample['negative_pixels1'], sample['negative_pixels2']

        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
            flow = self.transform(flow)

        if self.target_transform is not None:
            mask1 = mask1[:, :, np.newaxis]
            mask2 = mask2[:, :, np.newaxis]
            negative_pixels1 = negative_pixels1[:, :, np.newaxis]
            negative_pixels2 = negative_pixels2[:, :, np.newaxis]
            mask1 = self.target_transform(mask1)
            mask2 = self.target_transform(mask2)
            negative_pixels1 = self.target_transform(negative_pixels1)
            negative_pixels2 = self.target_transform(negative_pixels2)

        return image1, image2, flow, mask1, mask2, negative_pixels1, negative_pixels2

    def load_youtubevos(self, args):

        self._db_sequences = db_read_sequences_train_youtubevos()

        # Check lmdb existance. If not proceed with standard dataloader.
        lmdb_env_seq_dir = osp.join(cfg_youtubevos.PATH.DATA, 'lmdb_seq')
        lmdb_env_annot_dir = osp.join(cfg_youtubevos.PATH.DATA, 'lmdb_annot')

        if osp.isdir(lmdb_env_seq_dir) and osp.isdir(lmdb_env_annot_dir):
            lmdb_env_seq = lmdb.open(lmdb_env_seq_dir)
            lmdb_env_annot = lmdb.open(lmdb_env_annot_dir)
        else:
            lmdb_env_seq = None
            lmdb_env_annot = None
            print('LMDB not found. This could affect the data loading time.'
                  ' It is recommended to use LMDB.')

        # Load sequences
        self.sequences = [Sequence(self._phase, s, lmdb_env=lmdb_env_seq)
                          for s in self._db_sequences]

        # Load sequences
        videos = []
        for seq, s in zip(self.sequences, self._db_sequences):
            videos.append(s)

        for _video in videos:
            image_file = sorted(glob.glob(os.path.join(
                cfg_youtubevos.PATH.SEQUENCES_TRAIN, _video, '*.jpg')))
            mask_file = sorted(glob.glob(os.path.join(
                cfg_youtubevos.PATH.ANNOTATIONS_TRAIN, _video, '*.png')))
            flow_file = sorted(glob.glob(os.path.join(
                cfg_youtubevos.PATH.FLOW, _video, '*.png')))
            hed_file = sorted(glob.glob(os.path.join(
                cfg_youtubevos.PATH.HED, _video, '*.jpg')))

            self.image_files.extend(print_list_davis(image_file))
            self.mask_files.extend(print_list_davis(mask_file))
            self.flow_files.extend(flow_file)
            self.hed_files.extend(print_list_davis(hed_file))

        assert(len(self.image_files) == len(self.mask_files) ==
               len(self.flow_files) == len(self.hed_files))

    def load_davis(self, args):

        self._db_sequences = list(db_read_sequences_davis(args.year, self._phase))

        # Check lmdb existance. If not proceed with standard dataloader.
        lmdb_env_seq_dir = osp.join(cfg_davis.PATH.DATA, 'lmdb_seq')
        lmdb_env_annot_dir = osp.join(cfg_davis.PATH.DATA, 'lmdb_annot')

        if osp.isdir(lmdb_env_seq_dir) and osp.isdir(lmdb_env_annot_dir):
            lmdb_env_seq = lmdb.open(lmdb_env_seq_dir)
            lmdb_env_annot = lmdb.open(lmdb_env_annot_dir)
        else:
            lmdb_env_seq = None
            lmdb_env_annot = None
            print('LMDB not found. This could affect the data loading time.'
                  ' It is recommended to use LMDB.')

        self.sequences = [Sequence(self._phase, s.name, lmdb_env=lmdb_env_seq)
                          for s in self._db_sequences]
        self._db_sequences = db_read_sequences_davis(args.year, self._phase)

        # Load annotations
        self.annotations = [Annotation(
            self._phase, s.name, self._single_object, lmdb_env=lmdb_env_annot)
            for s in self._db_sequences]
        self._db_sequences = db_read_sequences_davis(args.year, self._phase)

        # Load Videos
        videos = []
        for seq, s in zip(self.sequences, self._db_sequences):
            if s['set'] == self._phase:
                videos.append(s['name'])

        for _video in videos:
            image_file = sorted(glob.glob(os.path.join(
                cfg_davis.PATH.SEQUENCES, _video, '*.jpg')))
            mask_file = sorted(glob.glob(os.path.join(
                cfg_davis.PATH.ANNOTATIONS, _video, '*.png')))
            flow_file = sorted(glob.glob(os.path.join(
                cfg_davis.PATH.FLOW, _video, '*.png')))
            hed_file = sorted(glob.glob(os.path.join(
                cfg_davis.PATH.HED, _video, '*.jpg')))

            self.image_files.extend(print_list_davis(image_file))
            self.mask_files.extend(print_list_davis(mask_file))
            self.flow_files.extend(flow_file)
            self.hed_files.extend(print_list_davis(hed_file))

        assert(len(self.image_files) == len(self.mask_files) ==
               len(self.flow_files) == len(self.hed_files))
