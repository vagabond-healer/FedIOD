import torch
import os
import cv2

import pandas as pd
import numpy as np

from numbers import Number
from typing import Container
from collections import defaultdict
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.utilities.file_and_folder_operations import *
from .txt_utils import txt2list
from skimage import io, color
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
from collections import OrderedDict


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    # print(images)
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)
    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


def norm_zscore(tx):
    tx = np.array(tx)
    tx = tx.astype(np.float32)
    tx_flat = tx.flatten()
    if np.sum(tx_flat) > 0:
        tx_flat_no = tx_flat[tx_flat > 0]
        tx_normal = (tx - np.mean(tx_flat_no)) / (np.std(tx_flat_no) + 1e-5)
        tx_normal[tx == 0] = 0
    else:
        tx_normal = tx
    return tx_normal


def chk_mkdir(*paths: Container) -> None:
    """
    Creates folders if they do not exist.

    Args:
        paths: Container of paths to be created.
    """
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


class Logger:
    def __init__(self, verbose=False):
        self.logs = defaultdict(list)
        self.verbose = verbose

    def log(self, logs):
        for key, value in logs.items():
            self.logs[key].append(value)

        if self.verbose:
            print(logs)

    def get_logs(self):
        return self.logs

    def to_csv(self, path):
        pd.DataFrame(self.logs).to_csv(path, index=None)


class MetricList:
    def __init__(self, metrics):
        assert isinstance(metrics, dict), '\'metrics\' must be a dictionary of callables'
        self.metrics = metrics
        self.results = {key: 0.0 for key in self.metrics.keys()}

    def __call__(self, y_out, y_batch):
        for key, value in self.metrics.items():
            self.results[key] += value(y_out, y_batch)

    def reset(self):
        self.results = {key: 0.0 for key in self.metrics.keys()}

    def get_results(self, normalize=False):
        assert isinstance(normalize, bool) or isinstance(normalize, Number), '\'normalize\' must be boolean or a number'
        if not normalize:
            return self.results
        else:
            return {key: value / normalize for key, value in self.results.items()}


class JointTransform2D:
    """
    Performs augmentation on image and mask when called. Due to the randomness of augmentation transforms,
    it is not enough to simply apply the same Transform from torchvision on the image and mask separetely.
    Doing this will result in messing up the ground truth mask. To circumvent this problem, this class can
    be used, which will take care of the problems above.

    Args:
        crop: tuple describing the size of the random crop. If bool(crop) evaluates to False, no crop will
            be taken.
        p_flip: float, the probability of performing a random horizontal flip.
        color_jitter_params: tuple describing the parameters of torchvision.transforms.ColorJitter.
            If bool(color_jitter_params) evaluates to false, no color jitter transformation will be used.
        p_random_affine: float, the probability of performing a random affine transform using
            torchvision.transforms.RandomAffine.
        long_mask: bool, if True, returns the mask as LongTensor in label-encoded format.
    """

    def __init__(self, img_size=256, crop=(32, 32), p_flip=0.0, p_rota=0.0, p_scale=0.0, p_gaussn=0.0, p_contr=0.0,
                 p_gama=0.0, p_distor=0.0, z_score=False, color_jitter_params=(0.1, 0.1, 0.1, 0.1), p_random_affine=0,
                 long_mask=False):
        self.crop = crop
        self.p_flip = p_flip
        self.p_rota = p_rota
        self.p_scale = p_scale
        self.p_gaussn = p_gaussn
        self.p_gama = p_gama
        self.p_contr = p_contr
        self.p_distortion = p_distor
        self.zscore = z_score
        self.img_size = img_size
        self.color_jitter_params = color_jitter_params
        if color_jitter_params:
            self.color_tf = T.ColorJitter(*color_jitter_params)
        self.p_random_affine = p_random_affine
        self.long_mask = long_mask

    def __call__(self, image, mask):
        #  gamma enhancement
        if np.random.rand() < self.p_gama:
            c = 1
            g = np.random.randint(10, 25) / 10.0
            # g = 2
            image = (np.power(image / 255, 1.0 / g) / c) * 255
            image = image.astype(np.uint8)
        # transforming to PIL image
        image, mask = F.to_pil_image(image), F.to_pil_image(mask)
        # random crop
        if self.crop:
            i, j, h, w = T.RandomCrop.get_params(image, self.crop)
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
        # random horizontal flip
        if np.random.rand() < self.p_flip:
            image, mask = F.hflip(image), F.hflip(mask)
        # random rotation
        if np.random.rand() < self.p_rota:
            angle = T.RandomRotation.get_params((-30, 30))
            image, mask = F.rotate(image, angle), F.rotate(mask, angle)
        # random scale and center resize to the original size
        if np.random.rand() < self.p_scale:
            scale = np.random.uniform(1, 1.3)
            new_h, new_w = int(self.img_size * scale), int(self.img_size * scale)
            image, mask = F.resize(image, (new_h, new_w), 2), F.resize(mask, (new_h, new_w), 0)
            # image = F.center_crop(image, (self.img_size, self.img_size))
            # mask = F.center_crop(mask, (self.img_size, self.img_size))
            i, j, h, w = T.RandomCrop.get_params(image, (self.img_size, self.img_size))
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
        # random add gaussian noise
        if np.random.rand() < self.p_gaussn:
            ns = np.random.randint(3, 15)
            noise = np.random.normal(loc=0, scale=1, size=(self.img_size, self.img_size)) * ns
            noise = noise.astype(int)
            image = np.array(image) + noise
            image[image > 255] = 255
            image[image < 0] = 0
            image = F.to_pil_image(image.astype('uint8'))
        # random change the contrast
        if np.random.rand() < self.p_contr:
            contr_tf = T.ColorJitter(contrast=(0.8, 2.0))
            image = contr_tf(image)
        # random distortion
        if np.random.rand() < self.p_distortion:
            distortion = T.RandomAffine(0, None, None, (5, 30))
            image = distortion(image)
        # color transforms || ONLY ON IMAGE
        if self.color_jitter_params:
            image = self.color_tf(image)
        # random affine transform
        if np.random.rand() < self.p_random_affine:
            affine_params = T.RandomAffine(180).get_params((-90, 90), (1, 1), (2, 2), (-45, 45), self.crop)
            image, mask = F.affine(image, *affine_params), F.affine(mask, *affine_params)

        if self.zscore:
            image = norm_zscore(image)
            image = torch.from_numpy(image[None, :, :])
        else:
            image = F.to_tensor(image)

        if not self.long_mask:
            mask = F.to_tensor(mask)
        else:
            mask = to_long_tensor(mask)
        return image, mask


class Dataset_Fed(Dataset):
    def __init__(self, dataset_path: str, label_value_scale=255, split='pancreas_train.txt', joint_transform: Callable = None, FI_condition_list = None) -> None:

        self.dataset_path = dataset_path  # '../data/Fed/'
        self.img_path     = os.path.join(dataset_path, 'img')    # '../data/Fed/img'
        self.label_path   = os.path.join(dataset_path, 'label')  # '../data/Fed/label'
        self.txt_path     = os.path.join(dataset_path, split)    # # '../data/Fed/pancreas_train.txt'
        self.label_value_scale = label_value_scale

        if FI_condition_list != None:
            ids = [id.split('.')[0] for id in txt2list(self.txt_path) if id.split('_')[0] + '_' + id.split('_')[1] in FI_condition_list]
        else:
            ids = [id.split('.')[0] for id in txt2list(self.txt_path)]  # id = pancreas_0001_1
        self.ids = sorted(ids, key=lambda x: (int(x.split('_')[1]), int(x.split('_')[2])))  # x = pancreas_0001_1

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

        self.patient_dict = self.create_patient_dict()
        self.depth_median = self.cal_depth_median()
        self.depth_list   = self.collect_depth_for_GD()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        image = cv2.imread(os.path.join(self.img_path, id_ + '.png'), 0)
        mask  = cv2.imread(os.path.join(self.label_path, id_ + '.png'), 0)

        image, mask = correct_dims(image, mask)
        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)
        mask = (mask / self.label_value_scale).float()  # 因为在保存训练集的时候，label的值都乘了255

        if mask.unique().tolist() != [0] and mask.unique().tolist() != [0, 1]:  # 因为保存的时候mask本身就是二值图
            print('data loader error', id_, mask.unique().tolist())

        return image, mask, id_ + '.png'

    def create_patient_dict(self):
        '''

        :return: {'liver_0001': [liver_0001_1.png, liver_0001_2.png, liver_0001_2.png, ...],
                  'liver_0002': [liver_0002_1.png, liver_0002_2.png, liver_0002_2.png, ...],
                  ......}
        '''
        patient_dict = OrderedDict()
        for id in self.ids:
            key = id.split('_')[0] + '_' + id.split('_')[1]
            slice = id + '.png'
            # print('id: ', id, 'slice: ', slice)
            if key not in patient_dict:
                patient_dict[key] = []
                patient_dict[key].append(slice)
            else:
                patient_dict[key].append(slice)
        for key, value in patient_dict.items():
            patient_dict[key] = sorted(value, key=lambda x: (int(x.split('.')[0].split('_')[1]), int(x.split('.')[0].split('_')[2])))
        return patient_dict

    def cal_depth_median(self):
        depth_list = []
        for key, value in self.patient_dict.items():
            depth_list.append(len(value))
        # print('self.patient_dict: ', list(self.patient_dict.keys())[0], len(self.patient_dict[list(self.patient_dict.keys())[0]]))
        # print('depth_list: ', depth_list)
        return round(np.median(np.array(depth_list)).item())

    def collect_depth_for_GD(self):
        depth_list = []
        for key, value in self.patient_dict.items():
            depth_list.append(len(value))
        return depth_list


class Dataset_BCV(Dataset):
    def __init__(self, dataset_path: str, full_label_value_scale=80, joint_transform: Callable = None) -> None:
        self.dataset_path = dataset_path  # ''../data/Fed/''
        self.img_path     = os.path.join(dataset_path, 'BCV', 'img')    # '../data/Fed/BCV/img'
        self.label_path   = os.path.join(dataset_path, 'BCV', 'label')  # '../data/Fed/BCV/label'

        self.label_value_scale = full_label_value_scale

        ids = [id.split('.')[0] for id in os.listdir(self.img_path)]  # id = BCV_0001_1
        self.ids = sorted(ids, key=lambda x: (int(x.split('_')[1]), int(x.split('_')[2])))  # x = BCV_0001_1

        # print('self.ids: ', self.ids)
        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        image = cv2.imread(os.path.join(self.img_path, id_ + '.png'), 0)
        mask  = cv2.imread(os.path.join(self.label_path, id_ + '.png'), 0)

        image, mask = correct_dims(image, mask)
        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)

        mask = (mask / self.label_value_scale).float()  # 因为在保存测试集的时候，label的值都乘了80

        list = [0, 1, 2, 3]
        if not set(mask.unique().tolist()) <= set(list):
            print('mask error', id_, mask.unique().tolist())

        return image, mask, id_ + '.png'


def create_datasets(data_path, datasets_name, partial_label_value_scale, full_label_value_scale, tf_train, tf_val, args):
    for dataset_name in datasets_name:
        if dataset_name == "liver":
            liver_train = Dataset_Fed(dataset_path=data_path, label_value_scale=partial_label_value_scale, split=args.liver_train, joint_transform=tf_train)
            liver_val   = Dataset_Fed(dataset_path=data_path, label_value_scale=partial_label_value_scale, split=args.liver_val,   joint_transform=tf_val)
            liver_test  = Dataset_Fed(dataset_path=data_path, label_value_scale=partial_label_value_scale, split=args.liver_test,  joint_transform=tf_val)
            liver_A     = Dataset_Fed(dataset_path=data_path, label_value_scale=partial_label_value_scale, split=args.liver_train, joint_transform=tf_val)
            liver_weight = torch.tensor([1.0, args.liver_solo])
        elif dataset_name == "kidney":
            kidney_train = Dataset_Fed(dataset_path=data_path, label_value_scale=partial_label_value_scale, split=args.kidney_train, joint_transform=tf_train)
            kidney_val   = Dataset_Fed(dataset_path=data_path, label_value_scale=partial_label_value_scale, split=args.kidney_val,   joint_transform=tf_val)
            kidney_test  = Dataset_Fed(dataset_path=data_path, label_value_scale=partial_label_value_scale, split=args.kidney_test,  joint_transform=tf_val)
            kidney_A     = Dataset_Fed(dataset_path=data_path, label_value_scale=partial_label_value_scale, split=args.kidney_train, joint_transform=tf_val)
            kidney_weight = torch.tensor([1.0, args.kidney_solo])
        elif dataset_name == "pancreas":
            pancreas_train = Dataset_Fed(dataset_path=data_path, label_value_scale=partial_label_value_scale, split=args.pancreas_train, joint_transform=tf_train)
            pancreas_val   = Dataset_Fed(dataset_path=data_path, label_value_scale=partial_label_value_scale, split=args.pancreas_val,   joint_transform=tf_val)
            pancreas_test  = Dataset_Fed(dataset_path=data_path, label_value_scale=partial_label_value_scale, split=args.pancreas_test,  joint_transform=tf_val)
            pancreas_A     = Dataset_Fed(dataset_path=data_path, label_value_scale=partial_label_value_scale, split=args.pancreas_train, joint_transform=tf_val)
            pancreas_weight = torch.tensor([1.0, args.pancreas_solo])
        elif dataset_name == "BCV":
            bcv_test = Dataset_BCV(dataset_path=data_path, full_label_value_scale=full_label_value_scale, joint_transform=tf_val)
        else:
            print('create dataset error')

    liver_dict    = {'train': liver_train,    'val': liver_val,    'test':[liver_test,    bcv_test], 'weight':liver_weight,    'A':liver_A}
    kidney_dict   = {'train': kidney_train,   'val': kidney_val,   'test':[kidney_test,   bcv_test], 'weight':kidney_weight,   'A':kidney_A}
    pancreas_dict = {'train': pancreas_train, 'val': pancreas_val, 'test':[pancreas_test, bcv_test], 'weight':pancreas_weight, 'A':pancreas_A}

    return [liver_dict, kidney_dict, pancreas_dict]




