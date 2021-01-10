import cv2
import os


def vis_parsing(path):
    parsing = cv2.imread(path, 0)
    parsing_color_list = eval('colormap_utils.{}'.format('CIHP20'))  # CIHP20
    parsing_color_list = colormap_utils.dict_bgr2rgb(parsing_color_list)
    colormap = colormap_utils.dict2array(parsing_color_list)
    parsing_color = colormap[parsing.astype(np.int)]
    cv2.imwrite('vis_train_{}_{}.png'.format(str(10), str(10)), parsing_color)


path = '/xuhanzhu/mscoco2014/train_parsing_uv'
files = os.listdir(path)

for file in files:
    img = cv2.imread(os.path.join(path, file), 0)
for h in range(img.shape[0]):
    for w in range(img.shape[1]):
        if img[h][w] not in list(range(15)):
            print("WRONG")
