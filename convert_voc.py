import json
import cv2
import os
import shutil
import glob
import numpy as np
import colormap as colormap_utils
from pycocotools.coco import COCO


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def cal_one_mean_iou(image_array, label_array, num_parsing):
    hist = fast_hist(label_array, image_array, num_parsing).astype(np.float)
    num_cor_pix = np.diag(hist)
    num_gt_pix = hist.sum(1)
    union = num_gt_pix + hist.sum(0) - num_cor_pix
    iu = num_cor_pix / (num_gt_pix + hist.sum(0) - num_cor_pix)
    return iu


def vis_parsing(path, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    parsing = cv2.imread(path, 0)
    parsing_color_list = eval('colormap_utils.{}'.format('CIHP20'))  # CIHP20
    parsing_color_list = colormap_utils.dict_bgr2rgb(parsing_color_list)
    colormap = colormap_utils.dict2array(parsing_color_list)
    parsing_color = colormap[parsing.astype(np.int)]
    cv2.imwrite(os.path.join(dir, os.path.basename(path)), parsing_color)


def convert_voc():
    json_path = '/xuhanzhu/pascal_person_part/voc_2012_train.json'
    label_path = '/xuhanzhu/pascal_person_part/pascal_person_part_label'
    label_train = '/xuhanzhu/pascal_person_part'
    gt_path = '/xuhanzhu/pascal_person_part/pascal_person_part_gt'
    coco_folder = '/xuhanzhu/pascal_person_part'
    img_dir = '/xuhanzhu/pascal_person_part/JPEGImages'
    img_path = '/xuhanzhu/pascal_person_part/train_img'

    cihp_coco = COCO(json_path)
    im_ids = cihp_coco.getImgIds()

    chip_json = json.load(open(json_path, 'r'))
    # images = chip_json['images']

    categories = [{'name': 'person', 'id': 1, 'supercategory': 'person', 'parsing': ['background', \
                                                                                     'Head', 'Torso', 'Upper-arms',
                                                                                     'Lower-arms', 'Upper-legs',
                                                                                     'Lower-legs']}]
    annotations = []
    images = []
    for i, im_id in enumerate(im_ids):
        if (i + 1) % 100 == 0:
            print(i + 1)
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
        ann_num = 0
        for ii, ann in enumerate(anns):
            if ann['category_id'] != 15:
                continue
            img = np.zeros((height, width))
            # img = cv2.imread(os.path.join(gt_path, im['file_name'].replace('jpg', 'png')))
            bbr = np.array(ann['bbox']).astype(int)  # the box.
            x1, y1, x2, y2 = bbr[0], bbr[1], bbr[0] + bbr[2] + 1, bbr[1] + bbr[3] + 1
            x2 = min([x2, width]);
            y2 = min([y2, height])
            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
            img[y1:y2, x1:x2] = MaskIm[y1:y2, x1:x2]
            new_name = os.path.splitext(im['file_name'])[0] + '_%d' % ii + '.png'
            new_path = os.path.join(label_train, 'val_parsing', new_name)
            cv2.imwrite(new_path, img)
            ann["parsing"] = new_name
            ann['category_id'] = 1
            annotations.append(ann)
            ann_num += 1
        if ann_num != 0:
            shutil.copy(os.path.join(img_dir, im['file_name']), os.path.join(img_path, im['file_name']))
            images.append(im)
        else:
            print(im['file_name'])
    save_json_path = coco_folder + '/annotations/pascal_person_part_train.json'
    data_coco = {'images': images, 'categories': categories, 'annotations': annotations}
    json.dump(data_coco, open(save_json_path, 'w'), indent=4)


def convert_seg():
    file_set = {}
    path = '/xuhanzhu/pascal_person_part/train_parsing'
    files = os.listdir(path)
    save_path = '/xuhanzhu/pascal_person_part/train_seg'
    for file in files:
        # filename = file.split("-")[0]
        filename = '_'.join(file.split("_")[0:2])
        if filename not in file_set:
            print(filename)
            file_set[filename] = [file]
        else:
            file_set[filename].append(file)
    i = 0
    for filename in file_set:
        if i % 100 == 0:
            print(i, filename)
        mask = cv2.imread(os.path.join(path, file_set[filename][0]))
        for f in file_set[filename][1:]:
            mask += cv2.imread(os.path.join(path, f))
        cv2.imwrite(os.path.join(save_path, filename + '.png'), mask)
        i += 1


def compute_iou():
    pred_dir = '/xuhanzhu/my_pet/ckpts/rcnn/CIHP/uvann/baseline-R50-FPN-COCO_s1x_ms/test/parsing_instances'
    pred_all_dir = '/xuhanzhu/my_pet/ckpts/rcnn/CIHP/uvann/baseline-R50-FPN-COCO_s1x_ms/test/parsing_predict'
    predict_files = os.listdir(pred_all_dir)
    predict_fs = []
    save_dir = '/home/xuhanzhu/inference_par/vis_ori_par'
    save_dir_par = '/home/xuhanzhu/inference_par/par'
    if not os.path.exists(save_dir_par):
        os.makedirs(save_dir_par)
    for p_f in predict_files:
        filename = os.path.basename(p_f).split('.')[0]
        predict_fs.append(filename)
    ann_root = '/xuhanzhu/pascal_person_part/train_parsing/'
    d = 0
    predict_fs = sorted(predict_fs)

    for p_f in predict_fs:
        if d % 100 == 0:
            print(d)
        # p_f_s = p_f.split('_')[0]
        p_f_s = p_f
        par_pred_list = glob.glob(os.path.join(pred_dir, p_f_s + '*.png'))
        par_f_list = glob.glob(os.path.join(ann_root, p_f_s + '*.png'))
        ann_all = cv2.imread('/xuhanzhu/pascal_person_part/train_seg/{}.png'.format(p_f_s), 0)
        pred_all = cv2.imread(os.path.join(pred_all_dir, p_f_s + '.png'), 0)
        # vis_parsing(os.path.join(pred_all_dir, p_f_s + '.png'), '/home/xuhanzhu/inference_par/vis_par_ins')
        save_name = p_f_s + '.png'
        result_numpy = np.zeros_like(pred_all)
        for p in par_pred_list:
            pre = cv2.imread(p, 0)
            for f in par_f_list:
                ann = cv2.imread(f, 0)
                ann_p_copy = np.zeros_like(ann)
                pred_p_copy = np.zeros_like(pre)
                ann_p_copy[ann > 0] = 1
                pred_p_copy[pre > 0] = 1
                iu = cal_one_mean_iou(ann_p_copy, pred_p_copy, 2)

                if iu[1] > 0.5:
                    #### 每个人的
                    for i in range(15):
                        if (pre == i).any():
                            bin_number = np.bincount(ann[pre == i])
                            maxid = np.argsort(bin_number)
                            if len(maxid) > 1:
                                for mid in reversed(range(len(maxid))):
                                    # if mid < len(maxid)-1:
                                    #     continue
                                    if maxid[mid] and bin_number[maxid[mid]] > 0:
                                        result_bool_1 = (pre == i) & (ann == maxid[mid]) & (result_numpy == 0)
                                        result_numpy[result_bool_1] = i
                            else:
                                if maxid[0] > 0:
                                    result_bool_1 = (pre == i) & (ann == maxid[0])
                                    result_numpy[result_bool_1] = i

        # for i in range(15):
        #     if (pred_all == i).any():
        #         bin_number = np.bincount(ann_all[pred_all == i])
        #         maxid = np.argsort(bin_number)
        #         if len(maxid) > 1:
        #             for mid in reversed(range(len(maxid))):
        #                 if mid < len(maxid) - 2:
        #                     continue
        #                 if maxid[mid] and bin_number[maxid[mid]] > 0:
        #                     if (maxid[mid] == 1 and i in [12, 13]) or (maxid[mid] == 3 and i in [6, 7])\
        #                             or (maxid[mid] == 5 and i in [8, 9]) or (maxid[mid] == 6 and i in [10, 11]):
        #                         print(maxid[mid], i)
        #                         result_bool_1 = (ann_all == maxid[mid]) & (result_numpy == 0)
        #                         result_numpy[result_bool_1] = i
        #         else:
        #             if maxid[0] > 0:
        #                 if (maxid[0] == 1 and i in [12, 13]) or (maxid[0] == 3 and i in [6, 7]) \
        #                         or (maxid[0] == 5 and i in [8, 9]) or (maxid[0] == 6 and i in [10, 11]):
        #                     result_bool_1 = (ann_all == maxid[0]) & (result_numpy == 0)
        #                     result_numpy[result_bool_1] = i
        # for i in range(7):
        #     if (pred_all == i).any():
        #         bin_number = np.bincount(pred_all[ann_all == i])
        #         maxid = np.argsort(bin_number)
        #         if len(maxid) > 1:
        #             for mid in reversed(range(len(maxid))):
        #                 if mid < len(maxid) - 1:
        #                     continue
        #                 if maxid[mid] and bin_number[maxid[mid]] > 0:
        #                     # if (maxid[mid] == 1 and i in [12, 13]) or (maxid[mid] == 3 and i in [6, 7])\
        #                     #         or (maxid[mid] == 5 and i in [8, 9]) or (maxid[mid] == 6 and i in [10, 11]):
        #                     #     print(maxid[mid], i)
        #                     if (i == 1 and maxid[mid] in [12, 13]) or (i == 3 and maxid[mid] in [6, 7]) \
        #                             or (i == 5 and maxid[mid] in [8, 9]) or (i == 6 and maxid[mid] in [10, 11]):
        #                         result_bool_1 = (ann_all == i) & (result_numpy == 0)
        #                         result_numpy[result_bool_1] = maxid[mid]
        #         else:
        #             if maxid[0] > 0:
        #                 # if (maxid[0] == 1 and i in [12, 13]) or (maxid[0] == 3 and i in [6, 7]) \
        #                 #         or (maxid[0] == 5 and i in [8, 9]) or (maxid[0] == 6 and i in [10, 11]):
        #                 if (i == 1 and maxid[0] in [12, 13]) or (i == 3 and maxid[0] in [6, 7]) \
        #                         or (i == 5 and maxid[0] in [8, 9]) or (i == 6 and maxid[0] in [10, 11]):
        #                     result_bool_1 = (ann_all == i) & (result_numpy == 0)
        #                     result_numpy[result_bool_1] = maxid[mid]

        result_bool_1 = (ann_all == 2) & (result_numpy == 0)
        result_numpy[result_bool_1] = 14
        result_bool_1 = (ann_all == 4) & (result_numpy == 0)
        result_numpy[result_bool_1] = 1
        cv2.imwrite(os.path.join(save_dir_par, save_name), result_numpy)
        vis_parsing(os.path.join(save_dir_par, save_name), '/home/xuhanzhu/inference_par/vis_par')
        # vis_parsing('/xuhanzhu/pascal_person_part/train_seg/{}.png'.format(p_f_s), save_dir)
        d += 1


# convert_voc()
# convert_seg()
compute_iou()
# vis_parsing('/xuhanzhu/pascal_person_part/train_parsing/2008_000041_2.png', './')
