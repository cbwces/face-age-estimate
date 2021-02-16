import json
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from PIL import Image, ImageEnhance

class ImageCroper(object):

    def __init__(self, img_info, margin_rate):
        f = open(img_info, 'r')
        self.img_info_dict = json.load(f)
        f.close()
        self.margin_rate = margin_rate

    def crop_image(self, src_img, src_img_path):

        bbox = self.img_info_dict[src_img_path][0]['bbox']
        height, weight = src_img.shape[0], src_img.shape[1]
        x_margin = (bbox[2] - bbox[0]) * self.margin_rate
        y_margin = (bbox[3] - bbox[1]) * self.margin_rate
        height_pad_value_before = max(int(y_margin-bbox[1]), 0)
        height_pad_value_after = max(int(bbox[3]+y_margin-height), 0)
        weight_pad_value_before = max(int(x_margin-bbox[0]), 0)
        weight_pad_value_after = max(int(bbox[2]+x_margin-weight), 0)
        bbox_with_margin = [max(bbox[0]-x_margin, 0), max(bbox[1]-y_margin, 0), min(bbox[2]+x_margin, weight), min(bbox[3]+y_margin, height)]
        bbox_with_margin = [int(i) for i in bbox_with_margin]
        dst_img = src_img[bbox_with_margin[1]:bbox_with_margin[3], bbox_with_margin[0]:bbox_with_margin[2], :]
        dst_img = np.pad(dst_img, ((height_pad_value_before, height_pad_value_before), (weight_pad_value_before, weight_pad_value_after), (0, 0)))

        return  dst_img

def Resize(img, size):
    return cv2.resize(img, (size, size))

def CenterCrop(img, size):
    w, h = img.shape[1], img.shape[0]
    w_edge = w - size
    h_edge = h - size
    return img[h_edge:h-h_edge, w_edge:w-w_edge, :]

def RandomFlip(img, p):
    if p > random.random():
        img = img[:, ::-1, :]
    return img

def RandomCrop(img, crop_size): # [w, h]
    origin_w, origin_h = img.shape[1], img.shape[0]
    rand_crop_ptr = [random.random()*i for i in range(2)]
    return img[origin_h*rand_crop_ptr[1]:crop_size[1], origin_w*rand_crop_ptr[0]:crop_size[0], :]

def RandomJit(img, factor_list):
    rand_factor_list = [i[0] + (i[1]-i[0])*random.random() for i in factor_list]
    table_brightness = np.array([i * rand_factor_list[0] for i in range(0, 256)]).clip(0, 255).astype(np.uint8)
    img = cv2.LUT(img.astype(np.uint8), table_brightness)
    table_contrast = np.array([(i - 74) * rand_factor_list[1] + 74 for i in range(0, 256)]).clip(0, 255).astype(np.uint8)
    img = cv2.LUT(img.astype(np.uint8), table_contrast)
    img = Image.fromarray(img)
    img = ImageEnhance.Color(img).enhance(rand_factor_list[2])
    return np.array(img)

def RandomRotate(img, rotate_angle):
    angle = rotate_angle[0] + (rotate_angle[1] - rotate_angle[0])*random.random()
    w, h = img.shape[1], img.shape[0]
    x_center = int(w / 2)
    y_center = int(h / 2)
    r_mat = cv2.getRotationMatrix2D((x_center, y_center), angle, 1.0)
    return cv2.warpAffine(img, r_mat, (w, h))

augmentation_dict = {'resize': Resize,
                     'center_crop': CenterCrop,
                     'random_flip': RandomFlip,
                     'random_crop': RandomCrop, 
                     'random_jit': RandomJit, 
                     'random_rotate': RandomRotate}

class SplitDataset(object):
    def __init__(self, df_file_list):
        self.fit(df_file_list)

    def fit(self, df_file_list):
        self.img_paths = []
        self.img_labels = []
        f = open(df_file_list, 'r')
        for line in f.read().strip().split('\n'):
            self.img_paths.append(line.split(' ')[0])
            self.img_labels.append(int(line.split(' ')[1]))
        f.close()

    def transform(self, shuffle=False, test_size=None, train_size=None, stratify=True):
        if test_size != None:
            assert train_size == None
            if stratify == True:
                train_X, test_X, train_y, test_y = train_test_split(self.img_paths, self.img_labels, shuffle=shuffle, test_size=test_size, stratify=self.img_labels, random_state=42)
            else:
                train_X, test_X, train_y, test_y = train_test_split(self.img_paths, self.img_labels, shuffle=shuffle, test_size=test_size, random_state=42)
        if train_size != None:
            assert test_size == None
            if stratify == True:
                train_X, test_X, train_y, test_y = train_test_split(self.img_paths, self.img_labels, shuffle=shuffle, train_size=train_size, stratify=self.img_labels, random_state=42)
            else:
                train_X, test_X, train_y, test_y = train_test_split(self.img_paths, self.img_labels, shuffle=shuffle, train_size=train_size, random_state=42)

        return (train_X, train_y), (test_X, test_y)

class AgeData(Dataset):

    def __init__(self, data_pair, is_train, img_size, num_classes, normal_aug=None, test_time_aug=None, mode='normal', crop_info="", crop_margin=0):
        self.img_paths = data_pair[0]
        self.labels = data_pair[1]
        self.max_age = num_classes
        self.is_train = is_train
        if self.is_train == False:
            self.test_time_aug = test_time_aug
        self.normal_aug = normal_aug
        self.img_size = img_size
        self.mode = mode
        self.crop_info = crop_info
        if crop_info != "":
            self.img_croper = ImageCroper(crop_info, crop_margin)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        if self.crop_info != "":
            img = self.img_croper.crop_image(img, img_path)
        for func in self.normal_aug.keys():
            img = augmentation_dict[func](img, self.normal_aug[func])
        if self.is_train ==  False:
            ori_img = cv2.resize(img, (self.img_size, self.img_size))
            if self.mode == 'normal':
                return torch.from_numpy(np.transpose(ori_img, (2, 0, 1)) / 255.0), self.labels[idx]
            elif self.mode == 'tta':
                tta_list = []
                tta_list.append(torch.from_numpy(np.transpose(ori_img, (2, 0, 1)) / 255.0))
                for func in self.test_time_aug.keys():
                    aug_img = augmentation_dict[func](img, self.test_time_aug[func])
                    aug_img = cv2.resize(aug_img, (self.img_size, self.img_size))
                    tta_list.append(torch.from_numpy(np.transpose(aug_img, (2, 0, 1)) / 255.0))
                return tta_list, self.labels[idx]
            else:
                raise KeyError
        else:
            img = cv2.resize(img, (self.img_size, self.img_size))
            label =  [1] * (self.labels[idx] - 1) + [0] * (self.max_age - self.labels[idx])
            return torch.from_numpy(np.transpose(img, (2, 0, 1)) / 255.0), torch.Tensor(label)

    def __len__(self):
        return len(self.labels)

