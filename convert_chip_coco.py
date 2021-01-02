import cv2
import os
from pycocotools.coco import COCO
import colormap as colormap_utils
import numpy as np


def convert_parsing():
    coco_folder = '/xuhanzhu/CIHP/'
    cihp_coco = COCO(coco_folder + '/annotations/CIHP_train.json')
    im_ids = cihp_coco.getImgIds()
    ann_ids = cihp_coco.getAnnIds(imgIds=im_ids)
    anns = cihp_coco.loadAnns(ann_ids)
    for i, obj in enumerate(anns):
        if i % 50 == 0:
            print(i)
        parsing_name = os.path.join(coco_folder + 'train_parsing', obj['parsing'])
        parsing = cv2.imread(parsing_name)
        for p in range(19):
            if p == 1 or p == 2 or p == 4 or p == 13:
                parsing[parsing == p] = 14
            elif p == 5 or p == 10 or p == 7:
                parsing[parsing == p] = 1
            elif p == 18:
                parsing[parsing == p] = 4
            elif p == 19:
                parsing[parsing == p] = 5
            else:
                parsing[parsing == p] = 0
        save_name = os.path.join(coco_folder + 'train_parsing_uv', obj['parsing'])
        cv2.imwrite(save_name, parsing)


def vis_parsing(path):
    parsing = cv2.imread(path, 0)
    parsing_color_list = eval('colormap_utils.{}'.format('CIHP20'))  # CIHP20
    parsing_color_list = colormap_utils.dict_bgr2rgb(parsing_color_list)
    colormap = colormap_utils.dict2array(parsing_color_list)
    parsing_color = colormap[parsing.astype(np.int)]
    cv2.imwrite('vis_train_{}_{}.png'.format(str(0), str(0)), parsing_color)

convert_parsing()
vis_parsing("/xuhanzhu/CIHP/train_parsing_uv/0000006-1.png")
