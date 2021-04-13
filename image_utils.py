import os
import copy
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from PIL import Image, ImageEnhance
from image_croper import ImageCropper

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
            if len(line.split(' ')) > 2:
                self.img_paths.append(" ".join(line.split(' ')[:-1]))
                self.img_labels.append(int(line.split(' ')[-1]))
            else:
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

def save_split_to_file(train_tuple, valid_tuple, file_path='.'):
    with open(os.path.join(file_path, 'train_split_file.txt'), 'w') as f:
        for (img_path, label) in zip(train_tuple[0], train_tuple[1]):
            f.write(img_path + " " + str(label) + "\n")
    with open(os.path.join(file_path, 'valid_split_file.txt'), 'w') as f:
        for (img_path, label) in zip(valid_tuple[0], valid_tuple[1]):
            f.write(img_path + " " + str(label) + "\n")

def load_split_from_file(file_path='.'):
    train_path_list = []
    train_label_list = []
    valid_path_list = []
    valid_label_list = []
    with open(os.path.join(file_path, 'train_split_file.txt'), 'r') as f:
        for line in f.read().strip().split("\n"):
            train_path_list.append(line.split(" ")[0])
            train_label_list.append(int(line.split(" ")[1]))
    with open(os.path.join(file_path, 'valid_split_file.txt'), 'r') as f:
        for line in f.read().strip().split("\n"):
            valid_path_list.append(line.split(" ")[0])
            valid_label_list.append(int(line.split(" ")[1]))
    return (train_path_list, train_label_list), (valid_path_list, valid_label_list)

class AgeData(Dataset):

    def __init__(self, data_pair, is_train, img_size, num_classes, normal_aug=None, test_time_aug=None, mode=-1, crop_info="", crop_margin=0, is_affine=False):
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
        self.affine = is_affine
        if crop_info != "":
            self.img_croper = ImageCropper(crop_info, crop_margin, self.img_paths, if_affine=is_affine, img_size=img_size)

    def __getitem__(self, idx):

        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        if self.crop_info != "":
            if self.affine == False:
                img = self.img_croper.crop_image(img, idx)
            else:
                img = self.img_croper.affine_image(img, idx)
        if self.normal_aug != None:
            for func in self.normal_aug.keys():
                img = augmentation_dict[func](img, self.normal_aug[func])
        if self.is_train ==  False:
            ori_img = cv2.resize(img, (self.img_size, self.img_size))
            if self.mode < 2:
                return torch.from_numpy(np.transpose(ori_img, (2, 0, 1)) / 255.0), self.labels[idx]
            else:
                tta_list = []
                for turn in range(self.mode):
                    aug_img = copy.deepcopy(ori_img)
                    for func in self.test_time_aug.keys():
                        aug_img = augmentation_dict[func](aug_img, self.test_time_aug[func])
                    aug_img = np.transpose(aug_img, (2, 0, 1)) / 255.0
                    tta_list.append(aug_img)
                return tta_list, self.labels[idx]
        else:
            img = cv2.resize(img, (self.img_size, self.img_size))
            label =  [1] * (self.labels[idx] - 1) + [0] * (self.max_age - self.labels[idx])
            return torch.from_numpy(np.transpose(img, (2, 0, 1)) / 255.0), torch.Tensor(label)

    def __len__(self):
        return len(self.labels)

