import json
from pycocotools.coco import COCO

coco_folder = '/xuhanzhu/CIHP'
cihp_coco = COCO(coco_folder + '/annotations/CIHP_train.json')
im_ids = cihp_coco.getImgIds()
numbers = 0
for i, im_id in enumerate(im_ids):
    ann_ids = cihp_coco.getAnnIds(imgIds=im_id)
    anns = cihp_coco.loadAnns(ann_ids)
    im = cihp_coco.loadImgs(im_id)[0]
    height = im['height']
    width = im['width']
    ann_have_label = 0
    for ii, ann in enumerate(anns):
        if 'parsing' in ann:
            numbers += 1
    # if ann_have_label != 0:
    #     numbers += 1

print(numbers)
