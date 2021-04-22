'''
@author: cbwces
@github: https://github.com/cbwces
@contact: sknyqbcbw@gmail.com
'''
import json
import numpy as np
import cv2
from skimage import transform as tform

class ImageCropper(object):

    def __init__(self, img_info, margin_rate, total_img_list, if_affine=False, img_size=None, dst=None):
        f = open(img_info, 'r')
        img_info_dict = json.load(f)
        f.close()
        self.margin_rate = margin_rate
        self.img_bboxes = np.zeros((len(total_img_list), 4))
        if if_affine == True:
            self.img_size = img_size
            self.img_pts = np.zeros((len(total_img_list), 5, 2))
            for pos, k in enumerate(total_img_list):
                self.img_bboxes[pos] = img_info_dict[k][0]['bbox']
                self.img_pts[pos] = np.array(img_info_dict[k][0]['pts']).reshape(2, 5).T
            self.dst = dst
            if dst == None:
                self.dst = np.array([
                  [38.2946, 51.6963],
                  [73.5318, 51.5014],
                  [56.0252, 71.7366],
                  [41.5493, 92.3655],
                  [70.7299, 92.2041]], dtype=np.float32)
                self.dst  = self.dst * self.img_size / 112 
        else:
            for pos, k in enumerate(total_img_list):
                self.img_bboxes[pos] = img_info_dict[k][0]['bbox']
            

    def crop_image(self, src_img, src_img_idx):

        bbox = self.img_bboxes[src_img_idx]
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

        return dst_img

    def affine_image(self, src_img, src_img_idx):

        src_img = self.crop_image(src_img, src_img_idx)
        bbox = self.img_bboxes[src_img_idx]
        pts = self.img_pts[src_img_idx]
        height, weight = src_img.shape[0], src_img.shape[1]
        x_margin = (bbox[2] - bbox[0]) * self.margin_rate
        y_margin = (bbox[3] - bbox[1]) * self.margin_rate
        pts[:, 0] = pts[:, 0] - bbox[0] + x_margin
        pts[:, 1] = pts[:, 1] - bbox[1] + y_margin
        estimator = tform.SimilarityTransform()
        estimator.estimate(pts.astype(np.float32), self.dst)
        warp_mat = estimator.params[:2, :]
        warp_dst = cv2.warpAffine(src_img, warp_mat, (self.img_size, self.img_size))

        return warp_dst

