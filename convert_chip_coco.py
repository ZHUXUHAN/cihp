import cv2
import os
from pycocotools.coco import COCO
import colormap as colormap_utils
import numpy as np


def convert_parsing():
    coco_folder = '/xuhanzhu/CIHP/'
    cihp_coco = COCO(coco_folder + '/annotations/CIHP_train.json')
    im_ids = cihp_coco.getImgIds()
    for i, im_id in enumerate(im_ids):
        # if i==1:
        #     break
        if i % 50 == 0:
            print(i)
        ann_ids = cihp_coco.getAnnIds(imgIds=im_id)
        anns = cihp_coco.loadAnns(ann_ids)
        im = cihp_coco.loadImgs(im_id)[0]
        height = im['height']
        width = im['width']
        for ii, ann in enumerate(anns):
            bbr = np.array(ann['bbox']).astype(int)  # the box.
            parsing_name = os.path.join(coco_folder + 'train_parsing', ann['parsing'])
            parsing = cv2.imread(parsing_name)
            parsing_copy = parsing.copy()
            # x1, y1, x2, y2 = bbr[0], bbr[1], bbr[0] + bbr[2], bbr[1] + bbr[3]
            # x2 = min([x2, width]);
            # y2 = min([y2, height])
            # parsing = parsing[y1:y2, x1:x2]
            # parsing = cv2.resize(parsing, (256, 256), interpolation=cv2.INTER_NEAREST)
            for p in range(0, 20):
                if p == 1 or p == 2 or p == 4 or p == 13:
                    parsing_copy[parsing == p] = 14
                elif p == 5 or p == 10 or p == 7:
                    parsing_copy[parsing == p] = 1
                elif p == 18:
                    parsing_copy[parsing == p] = 4
                elif p == 19:
                    parsing_copy[parsing == p] = 5
                else:
                    parsing_copy[parsing == p] = 0
            save_name = os.path.join(coco_folder + 'train_parsing_uv', ann['parsing'])
            cv2.imwrite(save_name, parsing_copy)


def convert_parsing_2():
    coco_folder = '/xuhanzhu/CIHP/'
    cihp_coco = COCO(coco_folder + '/annotations/CIHP_train.json')
    im_ids = cihp_coco.getImgIds()
    for i, im_id in enumerate(im_ids):
        # if i == 1:
        #     break
        if i % 50 == 0:
            print(i)
        ann_ids = cihp_coco.getAnnIds(imgIds=im_id)
        anns = cihp_coco.loadAnns(ann_ids)
        im = cihp_coco.loadImgs(im_id)[0]
        height = im['height']
        width = im['width']
        for ii, ann in enumerate(anns):
            parsing_save = np.zeros((height, width))
            bbr = np.array(ann['bbox']).astype(int)  # the box.
            parsing_name = os.path.join(coco_folder + 'train_parsing_uv_1', ann['parsing'].split('-')[0] + '.png')
            if os.path.exists(parsing_name):
                parsing = cv2.imread(parsing_name, 0)
                x1, y1, x2, y2 = bbr[0], bbr[1], bbr[0] + bbr[2], bbr[1] + bbr[3]
                x2 = min([x2, width]);
                y2 = min([y2, height])
                parsing_save[y1:y2, x1:x2] = parsing[y1:y2, x1:x2]
            else:
                print(parsing_name)
                parsing_name = os.path.join(coco_folder + 'train_parsing', ann['parsing'])
                parsing = cv2.imread(parsing_name)
                parsing_save = parsing.copy()
                for p in range(0, 20):
                    if p == 1 or p == 2 or p == 4 or p == 13:
                        parsing_save[parsing == p] = 14
                    elif p == 5 or p == 10 or p == 7:
                        parsing_save[parsing == p] = 1
                    elif p == 18:
                        parsing_save[parsing == p] = 4
                    elif p == 19:
                        parsing_save[parsing == p] = 5
                    else:
                        parsing_save[parsing == p] = 0
            save_name = os.path.join(coco_folder + 'train_parsing_uv', ann['parsing'])
            cv2.imwrite(save_name, parsing_save)


def convert_seg():
    file_set = {}
    path = '/xuhanzhu/pascal_person_part/train_parsing'
    files = os.listdir(path)
    save_path = '/xuhanzhu/pascal_person_part/train_seg'
    for file in files:
        # filename = file.split("-")[0]
        # filename = '_'.join(file.split("_")[0:3])
        filename = file[:11] + '.png'
        if filename not in file_set:
            file_set[filename] = [file]
        else:
            file_set[filename].append(file)
    i = 0
    for filename in file_set:
        if i % 100 == 0:
            print(i, filename)
        mask = cv2.imread(os.path.join(path, file_set[filename][0]), 0)
        for f in file_set[filename][1:]:
            mask_c = cv2.imread(os.path.join(path, f), 0)
            mask[mask==0] = mask_c[mask==0]
        cv2.imwrite(os.path.join(save_path, filename[:11]+'.png'), mask)
        i += 1


def vis_parsing(path):
    parsing = cv2.imread(path, 0)
    parsing_color_list = eval('colormap_utils.{}'.format('MHP59'))  # CIHP20
    parsing_color_list = colormap_utils.dict_bgr2rgb(parsing_color_list)
    colormap = colormap_utils.dict2array(parsing_color_list)
    parsing_color = colormap[parsing.astype(np.int)]
    cv2.imwrite('vis_train_{}_{}.png'.format(str(5), str(5)), parsing_color)


def move_person_data():
    import shutil
    coco_folder = '/xuhanzhu/CIHP'
    img_dir = '/xuhanzhu/CIHP/train_img'
    cihp_coco = COCO(coco_folder + '/annotations/CIHP_train.json')
    im_ids = cihp_coco.getImgIds()
    new_dir = '/xuhanzhu/train_input_cihp'
    for im_id in im_ids:
        im = cihp_coco.loadImgs(im_id)[0]
        filename = im['file_name']
        print(filename)
        ori_path = os.path.join(img_dir, filename)
        new_path = os.path.join(new_dir, filename)
        shutil.copy(ori_path, new_path)


def convert_parsing_2():
    coco_folder = '/xuhanzhu/CIHP/'
    cihp_coco = COCO(coco_folder + '/annotations/CIHP_train.json')
    parsing_dir = '/xuhanzhu/CDCL-human-part-segmentation/output_coco'
    target_dir = '/xuhanzhu/mscoco2014'
    im_ids = cihp_coco.getImgIds()
    for i, im_id in enumerate(im_ids):
        if i % 50 == 0:
            print(i)
        ann_ids = cihp_coco.getAnnIds(imgIds=im_id)
        anns = cihp_coco.loadAnns(ann_ids)
        im = cihp_coco.loadImgs(im_id)[0]
        height = im['height']
        width = im['width']
        filename = im['file_name']
        for ii, ann in enumerate(anns):
            c = ann['category_id']
            if c == 1:
                parsing_save = np.zeros((height, width))
                bbr = np.array(ann['bbox']).astype(int)  # the box.
                parsing_name = os.path.join(parsing_dir, filename.replace('.jpg', '.png'))
                if os.path.exists(parsing_name):
                    parsing = cv2.imread(parsing_name, 0)
                    x1, y1, x2, y2 = bbr[0], bbr[1], bbr[0] + bbr[2], bbr[1] + bbr[3]
                    x2 = min([x2, width]);
                    y2 = min([y2, height])
                    parsing_save[y1:y2, x1:x2] = parsing[y1:y2, x1:x2]
                save_name = os.path.join(target_dir + '/train_parsing_cdcl', ann['parsing'])
                cv2.imwrite(save_name, parsing_save)


# convert_parsing()
# convert_seg()
# vis_parsing("/xuhanzhu/pascal_person_part/train_seg/2008_000266.png")
# convert_parsing_2()
# vis_parsing("/xuhanzhu/my_pet/iccv_inference/COCO_val2014_000000564058_vissegs.jpg")
# move_person_data()
convert_parsing_2()
