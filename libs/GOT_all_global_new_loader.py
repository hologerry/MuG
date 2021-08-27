import copy
import os
import random
import time
from os.path import exists, join, split

import cv2
import numpy as np
import scipy.io
import torch
from numpy.random import randint
from PIL import Image
from torchvision import datasets
from torch.utils.data import Dataset

import libs.transforms_multi as transforms

# def video_loader(video_path, frame_end, step, frame_start=0):
#     #cap = cv2.VideoCapture(video_path)
#     #cap.set(1, frame_start - 1)
#     video = []
# 	for i in range(frame_start - 1, frame_end, step):
# 		cap.set(1, i)
# 		success, image = cap.read()
# 		if not success:
# 			raise Exception('Error while reading video {}'.format(video_path))
# 		pil_im = image
# 		video.append(pil_im)
# 	return video


class VideoRecord(object):
    pass


def _sample_indices(record):
    """
    :param record: VideoRecord
    :return: list
    """

    average_duration = (record.num_frames - 1 + 1) // 8
    if average_duration > 0:
        offsets = np.multiply(list(range(8)), average_duration) + randint(average_duration,
                                                                          size=8)
    elif record.num_frames > 8:
        offsets = np.sort(randint(record.num_frames - 1 + 1, size=8))
    else:
        offsets = np.zeros((8,))
    return offsets + 1


def framepair_loader(data_dir, video_path, frame_start, frame_end, record, indices):
    # print('frame read:')
    pair = []
    org_pair = []
    segments = []
    id_ = np.zeros(2)

    frame_num = frame_end - frame_start

    if frame_end > 50:
        id_[0] = random.randint(frame_start, frame_end-50)
        id_[1] = id_[0] + random.randint(1, 50)
    else:
        id_[0] = random.randint(frame_start, frame_end)
        id_[1] = random.randint(frame_start, frame_end)

    images = list()
    for seg_ind in indices:
        image = cv2.imread(os.path.join(data_dir, video_path, '{:08d}.jpg'.format(int(seg_ind))), cv2.IMREAD_COLOR)
        if image is None:
            print('im path (seg):', type(image), seg_ind, os.path.join(data_dir, video_path, '{:08d}.jpg'.format(int(seg_ind))))
        h, w, _ = image.shape
        meanval = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        h = (h // 64) * 64
        w = (w // 64) * 64
        image = cv2.resize(image, (w, h))

        image = image.astype(np.uint8)
        pil_im = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        segments.append(pil_im)


    for ii in range(2):
        image = cv2.imread(os.path.join(data_dir, video_path,  '{:08d}.jpg'.format(int(id_[ii]))), cv2.IMREAD_COLOR)
        if image is None:
            print('im path (ii):',type(image), id_[ii], os.path.join(data_dir, video_path, '{:08d}.jpg'.format(int(id_[ii]))))
        h, w, _ = image.shape
        h = (h // 64) * 64
        w = (w // 64) * 64
        img = copy.deepcopy(image)
        img = cv2.resize(img, (256, 256))
        img = np.array(img, dtype=np.float32)
        img = img / 255
        img = np.subtract(img, np.array(meanval, dtype=np.float32))
        img = img / std
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        image = cv2.resize(image, (w, h))
        image = image.astype(np.uint8)
        pil_im = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        org_pair.append(img)
        pair.append(pil_im)

    return pair, segments, org_pair


def video_frame_counter(video_path):
    cap = cv2.VideoCapture(video_path)
    return cap.get(7)


class VidListV1(Dataset):
    # for warm up, random crop both
    def __init__(self, video_path, patch_size, rotate=10, scale=1.2, is_train=True, moreaug=True):
        super(VidListV1, self).__init__()
        csv_path = "data/GOT-new.csv"
        filenames = open(csv_path).readlines()
        frame_all = [filename.split(',')[0].strip() for filename in filenames]
        n_frames = [int(filename.split(',')[1].strip()) for filename in filenames]
        self.data_dir = video_path
        self.list = frame_all
        self.n_frames = n_frames
        normalize = transforms.Normalize(mean=(128, 128, 128), std=(128, 128, 128))

        t = []
        if rotate > 0:
            t.append(transforms.RandomRotate(rotate))
        if scale > 0:
            t.append(transforms.RandomScale(scale))
        t.extend([transforms.RandomCrop(patch_size, separate=moreaug), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                  normalize])

        self.transforms = transforms.Compose(t)

        self.is_train = is_train
        self.video_list = []
        for filename in filenames:
            record = VideoRecord()
            record.path = os.path.join(video_path, filename.split(',')[0].strip())
            record.num_frames = int(filename.split(',')[1].strip())  # len(glob.glob(os.path.join(video_path, '*.jpg')))
            record.label = filename.split(',')[0].strip()
            self.video_list.append(record)

    def __getitem__(self, idx):
        video_ = self.list[idx]
        record = self.video_list[idx]
        segment_indices = _sample_indices(record)
        frame_end = self.n_frames[idx]-1  # video_frame_counter(video_)-1
        pair_, _, _ = framepair_loader(self.data_dir, video_, 1, frame_end, record, segment_indices)
        data = list(self.transforms(*pair_))
        return tuple(data)

    def __len__(self):
        return len(self.list)


class VidListV2(Dataset):
    # for localization, random crop frame1
    def __init__(self, video_path, patch_size, window_len, rotate=10, scale=1.2, full_size=640, is_train=True):
        super(VidListV2, self).__init__()
        csv_path = "data/GOT-train.csv"
        filenames = open(csv_path).readlines()

        frame_all = [filename.split(',')[0].strip() for filename in filenames]

        n_frames = [int(filename.split(',')[1].strip()) for filename in filenames]
        self.data_dir = video_path
        self.list = frame_all
        self.window_len = window_len
        self.n_frames = n_frames
        normalize = transforms.Normalize(mean=(128, 128, 128), std=(128, 128, 128))
        self.transforms1 = transforms.Compose([
            transforms.RandomRotate(rotate),
            # transforms.RandomScale(scale),
            transforms.ResizeandPad_1(full_size),
            transforms.RandomCrop(patch_size),
            transforms.ToTensor(),
            normalize])
        self.transforms2 = transforms.Compose([
            transforms.ResizeandPad_1(full_size),
            transforms.ToTensor(),
            normalize])

        self.is_train = is_train
        self.video_list = []
        for filename in filenames:
            record = VideoRecord()
            record.path = os.path.join(video_path, filename.split(',')[0].strip())
            record.num_frames = int(filename.split(',')[1].strip())  # len(glob.glob(os.path.join(video_path, '*.jpg')))
            record.label = filename.split(',')[0].strip()
            self.video_list.append(record)

    def __getitem__(self, idx):
        video_ = self.list[idx]
        frame_end = self.n_frames[idx] - 1  # video_frame_counter(video_)-1
        record = self.video_list[idx]
        segment_indices = _sample_indices(record)
        pair_, segments_, org_pair = framepair_loader(self.data_dir, video_, 1, frame_end, record, segment_indices)
        # print('input length:', len(pair_), len(segments_))
        data1 = list(self.transforms1(*pair_))
        data2 = list(self.transforms2(*pair_))
        data_all = []
        for ii in range(0, 8):
            temp = self.transforms2(*[pair_[1], segments_[ii]])
            data_all.append(temp[1])
        # data_all = list(self.transforms2(*segments_))
        # print('data size:', len(data_all), data_all[0].size(),data2[0].size(),org_pair[0].size())
        if self.window_len == 2:
            data = [data1[0], data2[1]]
        else:
            data = [data1[0], data2[1], data2[2]]
        return tuple(data), data2, data_all, tuple(org_pair)

    def __len__(self):
        # print('video length:', len(self.list))
        return len(self.list)
