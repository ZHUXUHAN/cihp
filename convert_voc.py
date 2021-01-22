import json
import cv2
import os
import shutil
import numpy as np
import colormap as colormap_utils
from pycocotools.coco import COCO


def vis_parsing(path, dir):
    parsing = cv2.imread(path, 0)
    parsing_color_list = eval('colormap_utils.{}'.format('CIHP20'))  # CIHP20
    parsing_color_list = colormap_utils.dict_bgr2rgb(parsing_color_list)
    colormap = colormap_utils.dict2array(parsing_color_list)
    parsing_color = colormap[parsing.astype(np.int)]
    cv2.imwrite(os.path.join(dir, os.path.basename(path)), parsing_color)


def convert_voc():
    json_path = '/xuhanzhu/pascal_person_part/voc_2012_val.json'
    label_path = '/xuhanzhu/pascal_person_part/pascal_person_part_label'
    label_train = '/xuhanzhu/pascal_person_part'
    gt_path = '/xuhanzhu/pascal_person_part/pascal_person_part_gt'
    coco_folder = '/xuhanzhu/pascal_person_part'
    img_dir = '/xuhanzhu/pascal_person_part/JPEGImages'
    img_path = '/xuhanzhu/pascal_person_part/val_img'

    cihp_coco = COCO(json_path)
    im_ids = cihp_coco.getImgIds()

    chip_json = json.load(open(json_path, 'r'))
    images = chip_json['images']
    categories = [{'name': 'person', 'id': 1, 'supercategory': 'person', 'parsing': ['background', \
                    'Head', 'Torso', 'Upper-arms', 'Lower-arms', 'Upper-legs', 'Lower-legs']}]
    # print()
    annotations = []
    for i, im_id in enumerate(im_ids):
        if i % 100 == 0:
            print(i)
        ann_ids = cihp_coco.getAnnIds(imgIds=im_id)
        anns = cihp_coco.loadAnns(ann_ids)
        im = cihp_coco.loadImgs(im_id)[0]
        height = im['height']
        width = im['width']
        id = os.path.splitext(im['file_name'])[0]
        label_name = id + '.png'
        if os.path.exists(os.path.join(label_path, label_name)):
            MaskIm = cv2.imread(os.path.join(label_path, label_name), 0)
        else:
            continue
        shutil.copy(os.path.join(img_dir, im['file_name']), os.path.join(img_path, im['file_name']))
        for ii, ann in enumerate(anns):
            if ann['category_id'] != 15:
                continue
            # img = np.zeros((height, width))
            # # img = cv2.imread(os.path.join(gt_path, im['file_name'].replace('jpg', 'png')))
            # bbr = np.array(ann['bbox']).astype(int)  # the box.
            # x1, y1, x2, y2 = bbr[0], bbr[1], bbr[0] + bbr[2] + 1, bbr[1] + bbr[3] + 1
            # x2 = min([x2, width]);
            # y2 = min([y2, height])
            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
            # img[y1:y2, x1:x2] = MaskIm[y1:y2, x1:x2]
            new_name = os.path.splitext(im['file_name'])[0] + '_%d' % ii + '.png'
            new_path = os.path.join(label_train, 'val_parsing', new_name)
            # cv2.imwrite(new_path, img)
            ann["parsing"] = new_name
            ann['category_id'] = 1
            annotations.append(ann)
    save_json_path = coco_folder + '/annotations/pascal_person_part_val.json'
    data_coco = {'images': images, 'categories': categories, 'annotations': annotations}
    json.dump(data_coco, open(save_json_path, 'w'), indent=4)


convert_voc()
# vis_parsing('/xuhanzhu/pascal_person_part/train_parsing/2008_000041_2.png', './')

