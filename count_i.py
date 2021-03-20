import json
import cv2
import numpy as np
from pycocotools import mask
import pycocotools.mask as mask_util
import os
from pycocotools.coco import COCO

# coco_folder = '/xuhanzhu/mscoco2014'
# json_path = coco_folder + '/annotations/densepose_coco_2014_train.json'
# cihp_coco = COCO(json_path)
# chip_json = json.load(open(json_path, 'r'))
# images = chip_json['images']
# categories = chip_json['categories']
# annotations = []
# im_ids = cihp_coco.getImgIds()
# I = np.zeros(25)
# for i, im_id in enumerate(im_ids):
#     ann_ids = cihp_coco.getAnnIds(imgIds=im_id)
#     anns = cihp_coco.loadAnns(ann_ids)
#     im = cihp_coco.loadImgs(im_id)[0]
#     height = im['height']
#     width = im['width']
#
#     img = np.zeros((height, width))
#     for ii, ann in enumerate(anns):
#         if 'dp_masks' in ann:
#             for II in range(1, 25):
#                 for i_l in ann['dp_I']:
#                     if int(i_l)==II:
#                         I[II]+=1
#
# print(I)
# np.save("I.npy", I)
I = np.load("I.npy")
I = I/10000
print(I.astype(np.int))