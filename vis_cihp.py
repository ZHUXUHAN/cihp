from pycocotools.coco import COCO
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_util
from random import randint
import matplotlib as mpl
import colormap as colormap_utils
from torchvision.transforms import functional as F
mpl.use('Agg')


def vis_parsing(img, parsing, colormap, show_segms=True):
    """Visualizes a single binary parsing."""

    img = img.astype(np.float32)
    idx = np.nonzero(parsing)
    # for i in range(20):
    #     print(parsing[parsing == i])
    #     if i == 15:
    #         parsing[parsing == i] = 1
    #         print(i)

    parsing_alpha = 0.4
    colormap = colormap_utils.dict2array(colormap)
    parsing_color = colormap[parsing.astype(np.int)]

    border_color = (255, 255, 255)
    border_thick = 1

    img[idx[0], idx[1], :] *= (1.0 - parsing_alpha)
    # img[idx[0], idx[1], :] += alpha * parsing_color
    print(img.shape, parsing_color.shape)
    img += parsing_alpha * parsing_color

    # if cfg.VIS.SHOW_PARSS.SHOW_BORDER and not show_segms:
    #     _, contours, _ = cv2.findContours(parsing.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    #     cv2.drawContours(img, contours, -1, border_color, border_thick, cv2.LINE_AA)
    return parsing_color.astype(np.uint8)
    # if return img+mask using flowling code.
    # return img.astype(np.uint8)


coco_folder = '/xuhanzhu/CIHP/'
cihp_coco = COCO(coco_folder + '/annotations/CIHP_train.json')


im_ids = cihp_coco.getImgIds()
Selected_im = im_ids[randint(0, len(im_ids))]

Selected_im = 17144
im = cihp_coco.loadImgs(Selected_im)[0]
ann_ids = cihp_coco.getAnnIds(imgIds=im['id'])
anns = cihp_coco.loadAnns(ann_ids)
im_name = os.path.join(coco_folder + 'train_img', im['file_name'])
mask_name = os.path.join(coco_folder + 'train_seg', im['file_name'].replace('jpg', 'png'))
for i, obj in enumerate(anns):
    parsing_name = os.path.join(coco_folder + 'train_parsing', obj['parsing'])
    parsing = cv2.imread(parsing_name, 0)
    parsing_color_list = eval('colormap_utils.{}'.format('CIHP20'))  # CIHP20
    parsing_color_list = colormap_utils.dict_bgr2rgb(parsing_color_list)
    colormap = colormap_utils.dict2array(parsing_color_list)
    parsing_color = colormap[parsing.astype(np.int)]
    cv2.imwrite('vis_train_{}_{}.png'.format(str(Selected_im), str(i)), parsing_color)
# if os.path.exists(im_name) and os.path.exists(mask_name):
#     print('Exits', im_name, mask_name)
#     I = cv2.imread(im_name)
#     mask = cv2.imread(mask_name, 0)
#     # ins_pars = mask.numpy()  # [instance].parsing
#     # ins_pars = np.array(F.to_pil_image(mask))
#     mask = np.reshape(mask, I.shape[:2])
#     # h, w = mask.shape[:2]
#     parsing_color_list = eval('colormap_utils.{}'.format('CIHP20'))  # CIHP20
#     parsing_color_list = colormap_utils.dict_bgr2rgb(parsing_color_list)
#     # I[0:h, 0:w, :] = vis_parsing(I[0:h, 0:w, :], mask, parsing_color_list)
#     I = vis_parsing(I, mask, parsing_color_list)
#     cv2.imwrite('vis_train_{}_tmp.png'.format(str(Selected_im)), I)
# fig, ax = plt.subplots()
#
# print
# plt.imshow(I[:, :, ::-1])
# plt.axis('off')
# height, width, channels = I.shape
# # 如果dpi=300，那么图像大小=height*width
# fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
# plt.margins(0, 0)
#
# plt.savefig('vis_train_{}_tmp.png'.format(str(Selected_im)))
# plt.close()
