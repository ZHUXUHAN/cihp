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


def vis_parsing(path):
    parsing = cv2.imread(path, 0)
    parsing_color_list = eval('colormap_utils.{}'.format('CIHP20'))  # CIHP20
    parsing_color_list = colormap_utils.dict_bgr2rgb(parsing_color_list)
    colormap = colormap_utils.dict2array(parsing_color_list)
    parsing_color = colormap[parsing.astype(np.int)]
    cv2.imwrite('vis_train_{}_{}.png'.format(str(1), str(1)), parsing_color)


coco_folder = '/xuhanzhu/mscoco2014'
cihp_coco = COCO(coco_folder + '/annotations/densepose_coco_2014_train.json')
im_ids = cihp_coco.getImgIds()[0]
im = cihp_coco.loadImgs(ids=im_ids)[0]
ann_ids = cihp_coco.getAnnIds(imgIds=im_ids)
anns = cihp_coco.loadAnns(ann_ids)
height = im['height']
width = im['width']
for i, ann in enumerate(anns):
    bbr = np.array(ann['bbox']).astype(int)  # the box.
    if 'dp_masks' in ann.keys():  # If we have densepose annotation for this ann
        Mask = GetDensePoseMask(ann['dp_masks'])
        # I_vis = cv2.imread(os.path.join(coco_folder, 'train2014', im["file_name"]))
        I_vis = np.zeros((im['height'], im['width']))
        ################
        x1, y1, x2, y2 = bbr[0], bbr[1], bbr[0] + bbr[2], bbr[1] + bbr[3]
        x2 = min([x2, I_vis.shape[1]]);
        y2 = min([y2, I_vis.shape[0]])
        ################
        MaskIm = cv2.resize(Mask, (int(x2 - x1), int(y2 - y1)), interpolation=cv2.INTER_NEAREST)
        I_vis[y1:y2, x1:x2] = MaskIm
        # MaskBool = np.tile((MaskIm == 0)[:, :, np.newaxis], [1, 1, 3])
        #  Replace the visualized mask image with I_vis.
        # Mask_vis = cv2.applyColorMap((MaskIm * 15).astype(np.uint8), cv2.COLORMAP_PARULA)[:, :, :]
        # Mask_vis[MaskBool] = I_vis[y1:y2, x1:x2, :][MaskBool]
        # I_vis[y1:y2, x1:x2, :] = I_vis[y1:y2, x1:x2, :] * 0.3 + Mask_vis * 0.7
        cv2.imwrite("vis_densepose_ann_mask_%d.png" % i, I_vis)

        mask_im = cv2.imread("vis_densepose_ann_mask_0.png", 0)
        mask_im = mask_im[y1:y2, x1:x2]
        mask_im = cv2.resize(mask_im, (256, 256), interpolation=cv2.INTER_NEAREST)
        parsing_color_list = eval('colormap_utils.{}'.format('CIHP20'))  # CIHP20
        parsing_color_list = colormap_utils.dict_bgr2rgb(parsing_color_list)
        colormap = colormap_utils.dict2array(parsing_color_list)
        parsing_color = colormap[mask_im.astype(np.int)]
        cv2.imwrite("temp.png", parsing_color)
