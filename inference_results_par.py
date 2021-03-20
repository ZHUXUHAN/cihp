import os
import cv2
import numpy as np
import shutil
import json
import pickle
import glob
from pycocotools.coco import COCO
import colormap as colormap_utils


def vis_parsing(path, dir, colormap, im_ori, draw_contours):
    parsing = cv2.imread(path, 0)
    parsing_color_list = eval('colormap_utils.{}'.format(colormap))  # CIHP20
    parsing_color_list = colormap_utils.dict_bgr2rgb(parsing_color_list)
    colormap = colormap_utils.dict2array(parsing_color_list)
    parsing_color = colormap[parsing.astype(np.int)]

    parsing_alpha = 0.9
    idx = np.nonzero(parsing)
    im_ori = im_ori.astype(np.float32)
    im_ori[idx[0], idx[1], :] *= 1.0 - parsing_alpha
    im_ori += parsing_alpha * parsing_color
    ######
    if draw_contours:
        contours, _ = cv2.findContours(parsing.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        parsing_color = parsing_color.astype(np.uint8)
        cv2.drawContours(im_ori, contours, -1, (0, 0, 255), 1)
    # M = cv2.moments(contours[1])  # 计算第一条轮廓的各阶矩,字典形式
    # center_x = int(M["m10"] / M["m00"])
    # center_y = int(M["m01"] / M["m00"])
    # cv2.circle(parsing_color, (center_x, center_y), 30, 128, -1)  # 绘制中心点
    # print(center_x, center_y)

    cv2.imwrite(os.path.join(dir, os.path.basename(path)), im_ori)


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


def compute_iou_pp():
    datatset = 'CIHP-COCO' # source data
    model = 'CIHP'
    class_num = 15
    pred_dir = '/xuhanzhu/iccv_panet/ckpts/ICCV/Parsing/parsing_R-101-FPN-COCO-PAR-USEANN_s1x_ms/test/parsing_instances/'
    pred_all_dir = '/xuhanzhu/iccv_panet/ckpts/ICCV/Parsing/parsing_R-101-FPN-COCO-PAR-USEANN_s1x_ms/test/parsing_predict/'
    # pred_dir = '/xuhanzhu/mscoco2014/train_parsing_cdcl/'
    # pred_all_dir = '/xuhanzhu/CDCL-human-part-segmentation/output_coco/'
    predict_files = os.listdir(pred_all_dir)
    predict_fs = []
    save_dir = '/xuhanzhu/inference_par/%s/vis_ori_par' % datatset  # train_seg 原标注的可视化图
    save_dir_par = '/xuhanzhu/inference_par/%s/par' % datatset  # 变换后的mask未上色图
    save_dir_par_ins = '/xuhanzhu/inference_par/%s/vis_par_ins' % datatset  # 预测结果
    save_dir_par_ori = '/xuhanzhu/inference_par/%s/vis_par' % datatset  # 变换后的结果
    # ori_img_dir = '/xuhanzhu/%s/train2014' % datatset
    ori_img_dir = '/xuhanzhu/CIHP/train_img'
    img_dir = '/xuhanzhu/inference_par/%s/img' % datatset
    con_dir = '/xuhanzhu/inference_par/%s/vis_par_ins_con' % datatset
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir_par):
        os.makedirs(save_dir_par)
    if not os.path.exists(save_dir_par_ins):
        os.makedirs(save_dir_par_ins)
    if not os.path.exists(save_dir_par_ori):
        os.makedirs(save_dir_par_ori)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    if not os.path.exists(con_dir):
        os.makedirs(con_dir)

    for p_f in predict_files:
        filename = os.path.basename(p_f).split('.')[0]
        predict_fs.append(filename)
    ann_root = '/xuhanzhu/CIHP/train_parsing/'
    d = 0
    predict_fs = sorted(predict_fs)

    for p_f in predict_fs:
        if d % 100 == 0:
            print(d)
        # p_f_s = p_f.split('_')[0]
        p_f_s = p_f
        # shutil.copy(os.path.join(ori_img_dir, p_f_s + '.jpg'), os.path.join(img_dir, p_f_s + '.jpg'))
        ori_img = cv2.imread(os.path.join(ori_img_dir, p_f_s + '.jpg'))
        par_pred_list = glob.glob(pred_dir + p_f_s + '*.png')
        par_f_list = glob.glob(ann_root + p_f_s + '*.png')
        ann_all = cv2.imread('/xuhanzhu/CIHP/train_seg/{}.png'.format(p_f_s), 0)
        pred_all = cv2.imread(os.path.join(pred_all_dir, p_f_s + '.png'), 0)
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
                    ### 每个人的
                    for i in range(class_num):
                        if (pre == i).any():
                            bin_number = np.bincount(ann[pre == i])
                            maxid = np.argsort(bin_number)
                            if len(maxid) > 1:
                                for mid in reversed(range(len(maxid))):
                                    # if mid < len(maxid)-1:
                                    #     continue
                                    if maxid[mid] and bin_number[maxid[mid]] > 0:
                                        # if i not in [14, 15, 16, 17, 18, 19]:
                                        #         result_bool_1 = (pre == i)
                                        #         result_numpy[result_bool_1] = i
                                        # else:
                                        result_bool_1 = (pre == i) & (ann == maxid[mid]) & (result_numpy == 0) #
                                        result_numpy[result_bool_1] = i
                                        result_bool_1 = (pre == i) & (ann == maxid[mid]) & (result_numpy == 0)  #
                                        result_numpy[result_bool_1] = i #
                            else:
                                if maxid[0] > 0:
                                    result_bool_1 = (pre == i)
                                    result_numpy[result_bool_1] = i
                else:
                    result_bool_1 = pre > 0
                    result_numpy[result_bool_1] = pre[pre > 0] #

        for i in range(1, class_num):
            if i in [1, 14]:
                if (pred_all == i).any():
                    bin_number = np.bincount(ann_all[pred_all == i])
                    maxid = np.argsort(bin_number)
                    print(bin_number[maxid[-1]] / np.sum(bin_number))
                    if len(maxid) > 1 and bin_number[maxid[-1]] / np.sum(bin_number) > 0.2:
                        for mid in reversed(range(len(maxid))):
                            if mid < len(maxid) - 1:
                                continue
                            if not maxid[mid] in [2]:
                                if maxid[mid] and bin_number[maxid[mid]] > 0:
                                    result_bool_1 = (ann_all == maxid[mid]) & (result_numpy == 0)
                                    result_numpy[result_bool_1] = i
                            else:
                                continue
                    else:
                        if maxid[0] > 0:
                            result_bool_1 = (ann_all == maxid[0]) & (result_numpy == 0)
                            result_numpy[result_bool_1] = i
            # elif i == 2 or i == 3: # or i == 2 or i == 3
            #     if (pred_all == i).any():
            #         bin_number = np.bincount(ann_all[pred_all == i])
            #         maxid = np.argsort(bin_number)
            #         if len(maxid) > 1 and bin_number[maxid[-1]] / np.sum(bin_number) > 0.5:
            #             for mid in reversed(range(len(maxid))):
            #                 if mid < len(maxid) - 1:
            #                     continue
            #                 if maxid[mid] and bin_number[maxid[mid]] > 0:
            #                         result_bool_1 = (ann_all == maxid[mid]) & (result_numpy == 0)
            #                         result_numpy[result_bool_1] = i
            #         else:
            #             if maxid[0] > 0:
            #                 result_bool_1 = (ann_all == maxid[0]) & (result_numpy == 0)
            #                 result_numpy[result_bool_1] = i
            # elif i == 1:  # or i == 2 or i == 3
            #     if (pred_all == i).any():
            #         bin_number = np.bincount(ann_all[pred_all == i])
            #         maxid = np.argsort(bin_number)
            #         print(bin_number[maxid[-1]] / np.sum(bin_number))
            #         if len(maxid) > 1 and bin_number[maxid[-1]] / np.sum(bin_number) > 0.5:
            #             for mid in reversed(range(len(maxid))):
            #                 if mid < len(maxid) - 1:
            #                     continue
            #                 if maxid[mid] not in [10, 14, 15]:
            #                     if maxid[mid] and bin_number[maxid[mid]] > 0:
            #                         result_bool_1 = (ann_all == maxid[mid]) & (result_numpy == 0)
            #                         result_numpy[result_bool_1] = i
            #                 else:
            #                     continue
            #         else:
            #             if maxid[0] > 0:
            #                 result_bool_1 = (ann_all == maxid[0]) & (result_numpy == 0)
            #                 result_numpy[result_bool_1] = i
            # for i in range(1, class_num):
            #     if i == 2:
            #         result_bool_1 = (ann_all == 14) & (result_numpy == 0)
            #         result_numpy[result_bool_1] = i
            #     elif i == 4:
            #         result_bool_1 = (ann_all == 1) & (result_numpy == 0)
            #         result_numpy[result_bool_1] = i
            #     elif i == 6:
            #         result_bool_1 = ((ann_all == 10) | (ann_all == 11)) & (result_numpy == 0)
            #         result_numpy[result_bool_1] = i
            #     elif i == 1:
            #         result_bool_1 = ((ann_all == 12) | (ann_all == 13)) & (result_numpy == 0)
            #         result_numpy[result_bool_1] = i
            #     elif i == 3:
            #         result_bool_1 = ((ann_all == 6) | (ann_all == 7)) & (result_numpy == 0)
            #         result_numpy[result_bool_1] = i
            #     elif i == 5:
            #         result_bool_1 = ((ann_all == 8) | (ann_all == 9)) & (result_numpy == 0)
            #         result_numpy[result_bool_1] = i
        # for i in range(1, class_num):
        #     if i == 14:
        #         result_bool_1 = (ann_all == 2) & (result_numpy == 0)
        #         result_numpy[result_bool_1] = i
        #     if i == 1:
        #         result_bool_1 = (ann_all == 4) & (result_numpy == 0)
        #         result_numpy[result_bool_1] = i
        #     elif i == 12 or i == 13:
        #         result_bool_1 = (ann_all == 1) & (result_numpy == 0)
        #         result_numpy[result_bool_1] = i
        #     elif i == 10 or i == 11:
        #         result_bool_1 = (ann_all == 6) & (result_numpy == 0)
        #         result_numpy[result_bool_1] = i
        #
        cv2.imwrite(os.path.join(save_dir_par, save_name), result_numpy)
        # vis_parsing(os.path.join(save_dir_par, save_name), save_dir_par_ori, 'MHP59', ori_img, True)
        # vis_parsing('/xuhanzhu/%s/train_seg_uv/{}.png'.format(p_f_s) % datatset, save_dir, 'CIHP20', ori_img, True)
        # vis_parsing(os.path.join(pred_all_dir, p_f_s + '.png'), save_dir_par_ins, 'MHP59', ori_img, False)
        # draw_contous(p_f_s, con_dir, datatset)
        d += 1


def draw_contous(id, dir, dataset):
    mask = cv2.imread('/xuhanzhu/inference_par/%s/par/%s.png' % (dataset, id), 0)
    img = cv2.imread('/xuhanzhu/inference_par/%s/vis_par_ins/%s.png' % (dataset, id))
    #####
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
    cv2.imwrite(os.path.join(dir, '%s_ins.png' % id), img)


def convert_parsing_2():
    coco_folder = '/xuhanzhu/CIHP/'
    cihp_coco = COCO(coco_folder + '/annotations/CIHP_train.json')
    parsing_dir = '/xuhanzhu/inference_par/CIHP-COCO/par'
    target_dir = '/xuhanzhu/CIHP'
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
                # print(parsing_name)
                if os.path.exists(parsing_name):
                    parsing = cv2.imread(parsing_name, 0)
                    x1, y1, x2, y2 = bbr[0], bbr[1], bbr[0] + bbr[2], bbr[1] + bbr[3]
                    x2 = min([x2, width]);
                    y2 = min([y2, height])
                    parsing_save[y1:y2, x1:x2] = parsing[y1:y2, x1:x2]
                save_name = os.path.join(target_dir + '/train_parsing_cdcl_coco', ann['parsing'])
                cv2.imwrite(save_name, parsing_save)


def vis_parsing_dir(new_dir, colormap, img_dir, parsing_dir, draw_contours):
    coco_folder = '/xuhanzhu/mscoco2014'
    cihp_coco = COCO(coco_folder + '/annotations/densepose_coco_2014_train.json')
    im_ids = cihp_coco.getImgIds()

    for im_id in sorted(im_ids):

        im = cihp_coco.loadImgs(im_id)[0]
        filename = im['file_name']

        ori_path = os.path.join(img_dir, filename)
        new_path = os.path.join(new_dir, filename)
        parsing_path = os.path.join(parsing_dir, filename.replace('.jpg', '.png'))
        parsing = cv2.imread(parsing_path, 0)
        if os.path.exists(ori_path):
            im_ori = cv2.imread(ori_path)
            parsing_color_list = eval('colormap_utils.{}'.format('MHP59'))  # CIHP20
            parsing_color_list = colormap_utils.dict_bgr2rgb(parsing_color_list)
            colormap = colormap_utils.dict2array(parsing_color_list)
            parsing_color = colormap[parsing.astype(np.int)]
            # parsing_color = parsing_color[..., ::-1]

            parsing_alpha = 0.9
            idx = np.nonzero(parsing)
            im_ori = im_ori.astype(np.float32)
            im_ori[idx[0], idx[1], :] *= 1.0 - parsing_alpha
            im_ori += parsing_alpha * parsing_color
            #####
            if draw_contours:
                contours, _ = cv2.findContours(parsing.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(im_ori, contours, -1, (0, 0, 255), 1)

            cv2.imwrite(new_path, im_ori)
        else:
            continue

#
def vis_parsing333(path, dir):
    parsing = cv2.imread(path, 0)
    parsing_color_list = eval('colormap_utils.{}'.format('MHP59'))  # CIHP20
    parsing_color_list = colormap_utils.dict_bgr2rgb(parsing_color_list)
    colormap = colormap_utils.dict2array(parsing_color_list)
    parsing_color = colormap[parsing.astype(np.int)]
    cv2.imwrite(os.path.join(dir, os.path.basename(path)), parsing_color)


def vis_parsing_dir2():
    dir = '/xuhanzhu/CIHP/train_parsing_cdcl_coco'
    files = os.listdir(dir)
    for file in files:
        path = os.path.join(dir, file)
        vis_parsing333(path, './vis')


# compute_ious()
# keypoints()
# compute_iou_pp()
# draw_contous('0000015')
# vis_parsing("/home/zhuxuhan/par_dir/0000001.png", './')
# convert_parsing_2()
# img_dir = '/xuhanzhu/mscoco2014/val2014'
# new_dir = '/xuhanzhu/anno'
# img_dir = '/xuhanzhu/output'
# new_dir = '/xuhanzhu/output_anno'
# parsing_dir = '/xuhanzhu/mscoco2014/train_parsing_cihp'
# vis_parsing_dir(new_dir, 'MHP59', img_dir, parsing_dir, True)
# vis_parsing_dir2()
convert_parsing_2()

#
