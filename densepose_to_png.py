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
    im = cihp_coco.loadImgs(ids=im_ids)[0]
    ann_ids = cihp_coco.getAnnIds(imgIds=im_ids)
    anns = cihp_coco.loadAnns(ann_ids)
    print(anns[0].keys())
    print(im['height'], im['width'])
    height = im['height']
    width = im['width']
    box = anns[0]["bbox"]
    box_x = box[0]
    box_y = box[1]
    box_w = box[2]
    box_h = box[3]
    x_targets = (np.arange(x1, x2, (x2 - x1) / w) - x1_source) * (256. / (x2_source - x1_source))
    y_targets = (np.arange(y1, y2, (y2 - y1) / h) - y1_source) * (256. / (y2_source - y1_source))
    x_targets = x_targets[0:w]  ## Strangely sometimes it can be M+1, so make sure size is OK!
    y_targets = y_targets[0:h]
    [X_targets, Y_targets] = np.meshgrid(x_targets, y_targets)

    # segmentation = anns[0]["segmentation"][0]#im['height']im['width']
    # box = anns[0]["bbox"]
    # box_x = box[0]
    # box_y = box[1]
    # box_w = box[2]
    # box_h = box[3]
    # length = len(segmentation)
    # print(length//2)
    # for i in range(length//2):
    #     segmentation[i*2] = (segmentation[i*2] - box_x) / box_w * 256
    #     segmentation[i*2+1] = (segmentation[i*2+1] - box_y) / box_h * 256
    #
    # compactedRLE = mask_util.frPyObjects([np.array(segmentation)], 256, 256)
    # compactedRLE[0]['counts'] = compactedRLE[0]['counts'].decode('UTF-8')
    # print(compactedRLE)
    # Mask = mask_util.decode(compactedRLE)
    # Mask = cv2.resize(Mask, (256, 256))
    # print(anns[0]["dp_masks"])
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
    cv2.imwrite('vis_train_{}_{}.png'.format(str(1), str(1)), parsing_color)


def convert_json():
    coco_folder = '/xuhanzhu/mscoco2014'
    json_path = coco_folder + '/annotations/densepose_coco_2014_valminusminival.json'
    cihp_coco = COCO(json_path)
    chip_json = json.load(open(json_path, 'r'))
    images = chip_json['images']
    categories = chip_json['categories']
    annotations = []
    im_ids = cihp_coco.getImgIds()
    for i, im_id in enumerate(im_ids):
        if i % 10 == 0:
            print(i)

        ann_ids = cihp_coco.getAnnIds(imgIds=im_id)
        anns = cihp_coco.loadAnns(ann_ids)
        im = cihp_coco.loadImgs(im_id)[0]
        height = im['height']
        width = im['width']
        for ii, ann in enumerate(anns):
            if 'dp_masks' in ann:
                box = anns[0]["bbox"]
                box_x = box[0]
                box_y = box[1]
                box_w = box[2]
                box_h = box[3]

                segment = ann["dp_masks"]
                Mask = GetDensePoseMask(segment)
                new_name = os.path.splitext(im['file_name'])[0] + '_%d'%ii + '.png'
                new_path = os.path.join(coco_folder, 'mask_ann', new_name)
                cv2.imwrite(new_path, Mask)
                ann["parsing"] = new_name
                annotations.append(ann)
            else:
                continue
            # if 'segmentation' in ann and 'dp_masks' in ann:
            #     segmentation = ann["segmentation"][0]
            #     box = anns[0]["bbox"]
            #     box_x = box[0]
            #     box_y = box[1]
            #     box_w = box[2]
            #     box_h = box[3]
            #     length = len(segmentation)
            #     for i in range(length // 2):
            #         segmentation[i * 2] = (segmentation[i * 2] - box_x) / box_w * 256
            #         segmentation[i * 2 + 1] = (segmentation[i * 2 + 1] - box_y) / box_h * 256
            #     compactedRLE = mask_util.frPyObjects([np.array(segmentation)], 256, 256)
            #     compactedRLE[0]['counts'] = compactedRLE[0]['counts'].decode('UTF-8')
            #     ann["segmentationRLE"] = compactedRLE
            #     annotations.append(ann)
            # else:
            #     continue

    save_json_path = coco_folder + '/annotations/densepose_segmentation_coco_2014_valminusminival.json'
    data_coco = {'images': images, 'categories': categories, 'annotations': annotations}
    json.dump(data_coco, open(save_json_path, 'w'), indent=4)


if __name__ == "__main__":
    path = "/xuhanzhu/CIHP/val_parsing/0011578-5.png"
    im = cv2.imread('/xuhanzhu/CIHP/val_img/0011578.jpg')
    cv2.imwrite("0011578.png", im)
    # m = seg2mask()
    vis_parsing(path)
    # convert_json()
