import os
import cv2
import numpy as np
import json
import pickle
import glob
from pycocotools.coco import COCO
import colormap as colormap_utils


def vis_parsing(path, dir):
    parsing = cv2.imread(path, 0)
    parsing_color_list = eval('colormap_utils.{}'.format('CIHP20'))  # CIHP20
    parsing_color_list = colormap_utils.dict_bgr2rgb(parsing_color_list)
    colormap = colormap_utils.dict2array(parsing_color_list)
    parsing_color = colormap[parsing.astype(np.int)]
    cv2.imwrite(os.path.join(dir, os.path.basename(path)), parsing_color)


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


def comput_iou():
    a = '/home/zhuxuhan/par_pretrain/ckpts/rcnn/CIHP/my_experiment/baseline_R-50-FPN-COCO_s1x_ms/test/parsing_instances/0000001_1.png'
    b = '/xuhanzhu/CIHP/val_seg/0000001.png'
    ann = cv2.imread(b, 0)
    pre = cv2.imread(a, 0)
    ann_copy = np.zeros_like(ann)
    pre_copy = np.zeros_like(pre)
    iu_numpy = np.zeros((20, 15, 2))
    result_numpy = np.zeros_like(ann)
    for i in range(20):
        for n in range(15):
            if ann[ann == i].shape[0] > 0:
                ann_copy[pre == n] = 1
                pre_copy[pre == i] = 1
                max_id = np.argmax(np.bincount(ann_copy[ann == max_id2]))
                iu = cal_one_mean_iou(ann_copy, pre_copy, 2)
                iu_numpy[i, n, :] = iu
                # if iu[1] > 0.3:
                #     print(iu[1], i, n)
                #     result_numpy[pre == n] = n
                #     result_numpy[ann == i] = n
                ann_copy = np.zeros_like(ann)
                pre_copy = np.zeros_like(pre)

            # print(np.bincount(pre[ann == i]))
            # bin = np.bincount(pre[ann == i])
            # ann_i, ann_ii = np.argsort(bin)[-2:]

    for i in range(15):
        max_id = np.argmax(iu_numpy[:, i, 1])
        if iu_numpy[max_id, i, 1] > 0.0:
            print(max_id, i, iu_numpy[max_id, i, 1])
            ann_pre_bool = (ann == max_id) & (pre == i)
            result_numpy[ann_pre_bool] = i
    for i in range(20):
        # if len(np.argsort(iu_numpy[:, i, 1])[-2:])>1:
        #     max_id1, max_id2 = np.argsort(iu_numpy[:, i, 1])[-2:]
        #     if max_id2 != 0:
        #         if iu_numpy[max_id1, i, 1] > 0.0:
        #             ann_pre_bool_a = (ann == max_id1) & (result_numpy == 0) & (pre == i)
        #             result_numpy[ann_pre_bool_a] = i
        #         # if iu_numpy[max_id2, i, 1] > 0.0 and np.bincount(ann[ann == max_id2])[-1] > 100:
        #         #     ann_pre_bool_a = (ann == max_id2) & (result_numpy == 0) & (pre == i)
        #         #     result_numpy[ann_pre_bool_a] = i
        #     else:
        #         max_id1 = np.argmax(iu_numpy[:, i, 1])
        #         if iu_numpy[max_id1, i, 1] > 0.0:
        #             ann_pre_bool_a = (ann == max_id1) & (result_numpy == 0)
        #             result_numpy[ann_pre_bool_a] = i
        # else:
        #     print("DEdedede")
        max_id = np.argmax(iu_numpy[i, :, 1])
        if iu_numpy[i, max_id, 1] > 0.0:
            ann_pre_bool_a = (ann == i) & (result_numpy == 0)
            result_numpy[ann_pre_bool_a] = max_id

    ##
    cv2.imwrite("test_ann.png", result_numpy)
    vis_parsing("test_ann.png", './')


def compute_ious():
    inference_dir = '/home/zhuxuhan/par_pretrain/ckpts/rcnn/CIHP/my_experiment/baseline_R-50-FPN-COCO_s1x_ms/test/parsing_predict'
    ann_dir = '/xuhanzhu/CIHP/val_seg_uv'
    save_pre_dir = '/home/zhuxuhan/par_dir/predict_cihp'
    save_ann_dir = '/home/zhuxuhan/par_dir/ann_cihp_uv'
    predict_files = os.listdir(inference_dir)
    ann_files = os.listdir(ann_dir)
    for predict_file in predict_files:
        if predict_file in ann_files:
            vis_parsing(os.path.join(ann_dir, predict_file), save_ann_dir)
            # vis_parsing(os.path.join(inference_dir, predict_file), save_pre_dir)
        else:
            print(predict_file)
            pass


def vis_keypoints(kp_preds, img):
    kp_x = kp_preds[::3]
    kp_y = kp_preds[1::3]
    vs = kp_preds[2::3]
    # img = np.zeros_like(img)
    for n in range(len(kp_x)):
        if vs[n] == 0:
            continue
        cor_x, cor_y = int(kp_x[n]), int(kp_y[n])
        bg = img.copy()
        cv2.circle(bg, (int(cor_x), int(cor_y)), 3, (0, 255, 255))
        transparency = 0.7
        img = cv2.addWeighted(bg, transparency, img, 1 - transparency, 2)
        cv2.imwrite("test_kp.png", img)


def keypoints():
    json_path = '/xuhanzhu/CIHP/annotations/CIHP_val_with_kp.json'
    cihp_coco = COCO(json_path)
    im_ids = cihp_coco.getImgIds()
    for i, im_id in enumerate(im_ids):
        if i == 1:
            break
        ann_ids = cihp_coco.getAnnIds(imgIds=im_id)
        anns = cihp_coco.loadAnns(ann_ids)
        im = cihp_coco.loadImgs(im_id)[0]
        for kp_ann in anns:
            # img = cv2.imread(os.path.join('/xuhanzhu/CIHP/val_img', im['file_name']))
            img = cv2.imread('/home/zhuxuhan/par_dir/ann_cihp/0000001.png')
            vis_keypoints(kp_ann['keypoints'], img)

    # kp_json = json.load(open(json_path, 'r'))
    # kp_ann = kp_json['annotations'][0]

    # Draw limbs
    # for i, (start_p, end_p) in enumerate(l_pair):
    #     if start_p in part_line and end_p in part_line:
    #         start_xy = part_line[start_p]
    #         end_xy = part_line[end_p]
    #         # bg = img.copy()
    #         X = (start_xy[0], end_xy[0])
    #         Y = (start_xy[1], end_xy[1])
    #         mX = np.mean(X)
    #         mY = np.mean(Y)
    #         length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
    #         angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
    #         stickwidth = (kp_scores[start_p] + kp_scores[end_p]) + 1
    #         polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
    #         cv2.fillConvexPoly(bg, polygon, line_color[i])
    # cv2.line(bg, start_xy, end_xy, line_color[i], (2 * (kp_scores[start_p] + kp_scores[end_p])) + 1)
    # transparency = max(0, min(1, 0.5*(kp_scores[start_p] + kp_scores[end_p]))).item()
    # img = cv2.addWeighted(bg, transparency, img, 1-transparency, 0)

    # img = cv2.addWeighted(bg, 0.7, img, 0.3, 0)
    # img1 = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)


def comput_iou_pp():
    # a = '/home/zhuxuhan/par_pretrain/ckpts/rcnn/CIHP/my_experiment/baseline_R-50-FPN-COCO_s1x_ms/test/parsing_instances/0000001_2.png'
    # b = '/xuhanzhu/CIHP/val_parsing/0000001-1.png'
    pred_dir = '/home/zhuxuhan/par_pretrain/ckpts/rcnn/CIHP/my_experiment/baseline_R-50-FPN-COCO_s1x_ms/test/parsing_instances/'
    pred_all_dir = '/home/zhuxuhan/par_pretrain/ckpts/rcnn/CIHP/my_experiment/baseline_R-50-FPN-COCO_s1x_ms/test/parsing_predict/'
    predict_files = os.listdir(pred_all_dir)
    predict_fs = []
    save_dir = '/home/zhuxuhan/inference_par/vis_ori_par'
    save_dir_par = '/home/zhuxuhan/inference_par/par'
    for p_f in predict_files:
        filename = os.path.basename(p_f).split('.')[0]
        predict_fs.append(filename)
    ann_root = '/xuhanzhu/CIHP/val_parsing/'
    d = 0
    predict_fs = sorted(predict_fs)
    for p_f in predict_fs:
        if d==10:
            break
        # p_f_s = p_f.split('_')[0]
        p_f_s = p_f
        par_pred_list = glob.glob(pred_dir + p_f_s + '*.png')
        par_f_list = glob.glob(ann_root + p_f_s + '*.png')
        ann_all = cv2.imread('/xuhanzhu/CIHP/val_seg/{}.png'.format(p_f_s), 0)
        pred_all = cv2.imread(os.path.join(pred_all_dir, p_f_s + '.png'), 0)
        save_name = p_f_s + '.png'
        result_numpy = np.zeros_like(pred_all)
        for p in par_pred_list:
            pre = cv2.imread(p, 0)
            print(p)
            for f in par_f_list:
                ann = cv2.imread(f, 0)
                ann_p_copy = np.zeros_like(ann)
                pred_p_copy = np.zeros_like(pre)
                ann_p_copy[ann > 0] = 1
                pred_p_copy[pre > 0] = 1
                iu = cal_one_mean_iou(ann_p_copy, pred_p_copy, 2)
                if iu[1] > 0.5:
                    max_numpy = np.zeros((15, 20))
                    #### 每个人的
                    for i in range(15):
                        if (pre == i).any():
                            bin_number = np.bincount(ann[pre == i])
                            maxid = np.argsort(bin_number)
                            if len(maxid) > 1:
                                for mid in reversed(range(len(maxid))):
                                    # if mid < len(maxid)-2:
                                    #     continue
                                    if maxid[mid] and bin_number[maxid[mid]] > 0:
                                        if mid == len(maxid):
                                            result_bool_1 = (pre == i) & (ann == maxid[mid])
                                            result_numpy[result_bool_1] = i
                                        else:
                                            result_bool_2 = (pre == i) & (ann == maxid[mid]) & (result_numpy == 0)
                                            result_numpy[result_bool_2] = i
                            else:
                                if maxid[0] > 0:
                                    result_bool_1 = (pre == i) & (ann == maxid[0])
                                    result_numpy[result_bool_1] = i

        for i in range(15):
            if (pred_all == i).any():
                bin_number = np.bincount(ann_all[pred_all == i])
                maxid = np.argsort(bin_number)
                if len(maxid) > 1:
                    for mid in reversed(range(len(maxid))):
                        if mid < len(maxid) - 2:
                            continue
                        if maxid[mid] and bin_number[maxid[mid]] > 0:
                            if mid == len(maxid):
                                result_bool_1 = (ann_all == maxid[mid]) & (result_numpy == 0)
                                result_numpy[result_bool_1] = i
                            else:
                                result_bool_2 = (ann_all == maxid[mid]) & (result_numpy == 0)
                                result_numpy[result_bool_2] = i
                else:
                    if maxid[0] > 0:
                        result_bool_1 = (ann_all == maxid[0]) & (result_numpy == 0)
                        result_numpy[result_bool_1] = i
        cv2.imwrite(os.path.join(save_dir_par, save_name), result_numpy)
        vis_parsing(os.path.join(save_dir_par, save_name), '/home/zhuxuhan/inference_par/vis_par')
        vis_parsing('/xuhanzhu/CIHP/val_seg/{}.png'.format(p_f_s), save_dir)
        d += 1

# compute_ious()
# keypoints()
comput_iou_pp()
# vis_parsing("/home/zhuxuhan/par_dir/0000001.png", './')
