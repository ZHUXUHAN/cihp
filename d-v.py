## coco-ppp
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
                            if maxid[mid] and bin_number[maxid[mid]] > 0:
                                result_bool_1 = (pre == i) & (ann == maxid[mid]) & (result_numpy == 0)  #
                                result_numpy[result_bool_1] = i  #
                    else:
                        if maxid[0] > 0:
                            result_bool_1 = (pre == i)
                            result_numpy[result_bool_1] = i
        else:
            result_bool_1 = pre > 0
            result_numpy[result_bool_1] = pre[pre > 0]
            
        for i in range(1, class_num):
            if i == 2:
                result_bool_1 = (ann_all == 14) & (result_numpy == 0)
                result_numpy[result_bool_1] = i
            elif i == 4:
                result_bool_1 = (ann_all == 1) & (result_numpy == 0)
                result_numpy[result_bool_1] = i
            elif i == 6:
                result_bool_1 = ((ann_all == 10) | (ann_all == 11)) & (result_numpy == 0)
                result_numpy[result_bool_1] = i
            elif i == 1:
                result_bool_1 = ((ann_all == 12) | (ann_all == 13)) & (result_numpy == 0)
                result_numpy[result_bool_1] = i
            elif i == 3:
                result_bool_1 = ((ann_all == 6) | (ann_all == 7)) & (result_numpy == 0)
                result_numpy[result_bool_1] = i
            elif i == 5:
                result_bool_1 = ((ann_all == 8) | (ann_all == 9)) & (result_numpy == 0)
                result_numpy[result_bool_1] = i

## coco-cihp