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
                                if i not in [14, 15, 16, 17, 18, 19]:
                                    result_bool_1 = (pre == i)
                                    result_numpy[result_bool_1] = i
                                else:
                                    result_bool_1 = (pre == i) & (ann == maxid[mid]) & (result_numpy == 0)  #
                                    result_numpy[result_bool_1] = i
                                result_bool_1 = (pre == i) & (ann == maxid[mid]) & (result_numpy == 0)  #
                                result_numpy[result_bool_1] = i  #
                    else:
                        if maxid[0] > 0:
                            result_bool_1 = (pre == i)
                            result_numpy[result_bool_1] = i
        else:
            result_bool_1 = pre > 0
            result_numpy[result_bool_1] = pre[pre > 0]  #

for i in range(1, class_num):
    if i in [1, 2, 14, 15, 16, 17, 18, 19]:
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