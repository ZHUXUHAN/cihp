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
    cihp_coco = COCO(coco_folder + '/annotations/densepose_coco_2014_minival.json')
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
    cv2.imwrite('vis_train_{}_{}.png'.format(str(10), str(10)), parsing_color)


def convert_seg():
    coco_folder = '/xuhanzhu/mscoco2014'
    json_path = coco_folder + '/annotations/densepose_coco_2014_train.json'
    cihp_coco = COCO(json_path)
    im_ids = cihp_coco.getImgIds()
    for i, im_id in enumerate(im_ids):
        if i % 50 == 0:
            print(i)

        ann_ids = cihp_coco.getAnnIds(imgIds=im_id)
        anns = cihp_coco.loadAnns(ann_ids)
        im = cihp_coco.loadImgs(im_id)[0]
        height = im['height']
        width = im['width']

        img = np.zeros((height, width))
        new_name = os.path.splitext(im['file_name'])[0] + '.png'
        new_path = os.path.join(coco_folder, 'train_seg_uv', new_name)
        print(new_path)
        d = 0
        for ii, ann in enumerate(anns):
            if 'dp_masks' in ann:
                d += 1
                bbr = np.array(ann['bbox']).astype(int)  # the box.
                x1, y1, x2, y2 = bbr[0], bbr[1], bbr[0] + bbr[2], bbr[1] + bbr[3]
                x2 = min([x2, width]);
                y2 = min([y2, height])

                segment = ann["dp_masks"]
                Mask = GetDensePoseMask(segment)
                MaskIm = cv2.resize(Mask, (int(x2 - x1), int(y2 - y1)), interpolation=cv2.INTER_NEAREST)
                img[y1:y2, x1:x2] = MaskIm
        if d > 0:
            d = 0
            cv2.imwrite(new_path, img)
        else:
            continue


def convert_json():
    coco_folder = '/xuhanzhu/mscoco2014'
    json_path = coco_folder + '/annotations/densepose_coco_2014_minival.json'
    cihp_coco = COCO(json_path)
    chip_json = json.load(open(json_path, 'r'))
    images = chip_json['images']
    categories = chip_json['categories']
    annotations = []
    im_ids = cihp_coco.getImgIds()
    for i, im_id in enumerate(im_ids):
        if i % 50 == 0:
            print(i)

        ann_ids = cihp_coco.getAnnIds(imgIds=im_id)
        anns = cihp_coco.loadAnns(ann_ids)
        im = cihp_coco.loadImgs(im_id)[0]
        height = im['height']
        width = im['width']

        img = np.zeros((height, width))

        for ii, ann in enumerate(anns):
            if 'dp_masks' in ann:
                img = np.zeros((height, width))
                bbr = np.array(ann['bbox']).astype(int)  # the box.
                x1, y1, x2, y2 = bbr[0], bbr[1], bbr[0] + bbr[2], bbr[1] + bbr[3]
                x2 = min([x2, width]);
                y2 = min([y2, height])

                segment = ann["dp_masks"]
                Mask = GetDensePoseMask(segment)
                MaskIm = cv2.resize(Mask, (int(x2 - x1), int(y2 - y1)), interpolation=cv2.INTER_NEAREST)
                img[y1:y2, x1:x2] = MaskIm
                new_name = os.path.splitext(im['file_name'])[0] + '_%d' % ii + '.png'
                new_path = os.path.join(coco_folder, 'val_parsing_uv', new_name)
                cv2.imwrite(new_path, img)
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

    save_json_path = coco_folder + '/annotations/densepose_parsing_coco_2014_minival.json'
    data_coco = {'images': images, 'categories': categories, 'annotations': annotations}
    json.dump(data_coco, open(save_json_path, 'w'), indent=4)


def get_ann(id=4219):
    coco_folder = '/xuhanzhu/mscoco2014'
    cihp_coco = COCO(coco_folder + '/annotations/densepose_parsing_coco_2014_minival.json')
    im_ids = cihp_coco.getImgIds()
    # ann_ids = cihp_coco.getAnnIds(imgIds=[sorted(im_ids)[id-1]])
    ann_ids = cihp_coco.getAnnIds(imgIds=[id])
    for ann_id in ann_ids:
        ann = cihp_coco.loadAnns(ann_id)[0]
        # print(ann['bbox'])  # the box.)
        print(ann)


def move_person_data():
    import shutil
    coco_folder = '/xuhanzhu/mscoco2014'
    img_dir = '/xuhanzhu/mscoco2014/train2014'
    cihp_coco = COCO(coco_folder + '/annotations/densepose_coco_2014_train.json')
    im_ids = cihp_coco.getImgIds()
    new_dir = '/xuhanzhu/train_input'
    for im_id in im_ids:
        im = cihp_coco.loadImgs(im_id)[0]
        filename = im['file_name']
        print(filename)
        ori_path = os.path.join(img_dir, filename)
        new_path = os.path.join(new_dir, filename)
        shutil.copy(ori_path, new_path)


if __name__ == "__main__":
    # path = "/xuhanzhu/mscoco2014/train_parsing_uv/COCO_train2014_000000329592_0.png"
    # path1 = "/xuhanzhu/CIHP/val_parsing_uv/0015242-6.png"
    # m = seg2mask()
    # vis_parsing(path)
    # convert_json()
    # convert_seg()

    # get_ann(id=5)
    # img_uv = cv2.imread(path)
    # img_parsing = cv2.imread(path1)
    # move_person_data()
    get_ann(id=2324)
