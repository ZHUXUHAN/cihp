import json
import cv2
import numpy as np
from pycocotools import mask
import pycocotools.mask as mask_util
import os
from pycocotools.coco import COCO

coco_folder = '/xuhanzhu/mscoco2014'
json_path = coco_folder + '/annotations/densepose_parsing_coco_2014_train.json'
cihp_coco = COCO(json_path)
chip_json = json.load(open(json_path, 'r'))
images = chip_json['images']
categories = chip_json['categories']
annotations = []
im_ids = cihp_coco.getImgIds()
num = 0
for i, im_id in enumerate(im_ids):
    ann_ids = cihp_coco.getAnnIds(imgIds=im_id)
    anns = cihp_coco.loadAnns(ann_ids)
    # print(anns[0])
    for ann in anns:
        if 'parsing' in ann:
            num+=1
print(num)