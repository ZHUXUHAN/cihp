import json
import cv2
import numpy as np
from pycocotools import mask
import pycocotools.mask as mask_util
import os
from pycocotools.coco import COCO
import colormap as colormap_utils


def GetDensePoseMask(Polys):
    MaskGen = np.zeros([256, 256])
    for i in range(1, 15):
        if (Polys[i - 1]):
            current_mask = mask_util.decode(Polys[i - 1])
            MaskGen[current_mask > 0] = i
    return MaskGen


def seg2mask():
    coco_folder = '/xuhanzhu/mscoco2014'
    cihp_coco = COCO(coco_folder + '/annotations/densepose_coco_2014_train.json')
    im_ids = cihp_coco.getImgIds()[0]
    ann_ids = cihp_coco.getAnnIds(imgIds=im_ids)
    anns = cihp_coco.loadAnns(ann_ids)
    segment = anns[0]["dp_masks"]
    print(len(segment))
    Mask = GetDensePoseMask(segment)
    cv2.imwrite("Mask.png", Mask)


def vis_parsing(path):
    parsing = cv2.imread(path, 0)
    parsing_color_list = eval('colormap_utils.{}'.format('CIHP20'))  # CIHP20
    parsing_color_list = colormap_utils.dict_bgr2rgb(parsing_color_list)
    colormap = colormap_utils.dict2array(parsing_color_list)
    parsing_color = colormap[parsing.astype(np.int)]
    cv2.imwrite('vis_train_{}_{}.png'.format(str(0), str(0)), parsing_color)


def convert_json():
    coco_folder = '/xuhanzhu/mscoco2014'
    json_path = coco_folder + '/annotations/densepose_coco_2014_train.json'
    cihp_coco = COCO(json_path)
    chip_json = json.load(open(json_path, 'r'))
    print(coco_folder + '/annotations/densepose_mask_coco_2014_train.json')
    images = chip_json['images']
    categories = chip_json['categories']
    annotations = []

    im_ids = cihp_coco.getImgIds()
    for i, im_id in enumerate(im_ids):
        if i % 100 == 0:
            print(i)
        ann_ids = cihp_coco.getAnnIds(imgIds=im_id)
        anns = cihp_coco.loadAnns(ann_ids)
        im = cihp_coco.loadImgs(im_id)[0]
        for ii, ann in enumerate(anns):
            if 'dp_masks' in ann:
                segment = ann["dp_masks"]
                Mask = GetDensePoseMask(segment)
                new_name = os.path.splitext(im['file_name'])[0] + '_%d'%ii + '.png'
                new_path = os.path.join(coco_folder, 'mask_ann', new_name)
                # cv2.imwrite(new_path, Mask)
                ann["parsing"] = new_name
                annotations.append(ann)
            else:
                continue

    save_json_path = coco_folder + '/annotations/densepose_mask_coco_2014_train.json'
    data_coco = {'images': images, 'categories': categories, 'annotations': annotations}
    json.dump(data_coco, open(save_json_path, 'w'), indent=4)


if __name__ == "__main__":
    path = "/home/zhuxuhan/cihp/Mask.png"

    # m = seg2mask()
    # vis_parsing(path)
    convert_json()
