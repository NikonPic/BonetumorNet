# %%
#
#  extend_training.py
#  BonetumorNet
#
#  Created by Nikolas Wilhelm on 2021-06-29.
#  Copyright © 2021 Nikolas Wilhelm. All rights reserved.
#

# %%

# Idea extend the training dataset via "Copy-Paste"

from PIL import Image, ImageDraw
import json
from ipywidgets import widgets
from shapely.geometry import Polygon

IMGDIR = '../PNG'
IMGKEY = 'images'
IMGFKEY = 'file_name'

EXT_DIR = '../EXT'

ANNKEY = 'annotations'
ANNSEGKEY = 'segmentation'

with open('../train.json') as fp:
    data = json.load(fp)


def get_anno(imgdata, data):
    """get the annotation from the imgdata"""
    img_id = imgdata['id']

    for anno in data[ANNKEY]:
        anno_id = anno['id']

        if anno_id == img_id:
            return anno

    return False


def visualize_anno(imgdata, annodata, draw=True):
    """visualize the annotation"""
    img = Image.open(f'{IMGDIR}/{imgdata[IMGFKEY]}')
    if draw:
        draw = ImageDraw.Draw(img)
        draw.polygon(annodata[ANNSEGKEY][0])
    return img


def visualize_mask(imgdata, imgdata2, annodata, annodata2):
    original_copy = Image.open(f'{IMGDIR}/{imgdata[IMGFKEY]}')
    original_paste = Image.open(f'{IMGDIR}/{imgdata2[IMGFKEY]}')
    poly = annodata[ANNSEGKEY][0]
    poly2 = annodata2[ANNSEGKEY][0]
    mask = Image.new("L", original_copy.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(poly, fill=255, outline=0)
    result = Image.composite(original_copy, original_paste, mask)
    draw2 = ImageDraw.Draw(result)
    draw2.polygon(poly2, fill=None, outline=0)

    return result


def calc_iou(annodata, annodata2):
    """get the intersection over union between the two polygons"""
    poly = anno_2_poly(annodata)
    area = poly.area

    poly2 = anno_2_poly(annodata2)
    area2 = poly2.area

    poly_inter = poly.intersection(poly2)
    inter_area = poly_inter.area

    iou = inter_area / (area + area2)

    return iou


def check_iou_lim(annodata, annodata2, lim=0.05):
    """only include annotations, if th iou is lower than {lim}"""
    iou = calc_iou(annodata, annodata2)
    if (lim >= iou):
        return True
    return False


def check_poly_lim(poly, img):
    """check if the polygon is within the image range"""
    x_max = max(poly[0::2])
    y_max = max(poly[1::2])
    img_x_max, img_y_max = img.size
    if (img_x_max >= x_max) and (img_y_max >= y_max):
        return True
    return False


def anno_2_poly(annodata):
    """transform the annotation to polygon format"""
    poly = annodata[ANNSEGKEY][0]
    x_arr = poly[0::2]
    y_arr = poly[1::2]
    poly = [(x, y) for (x, y) in zip(x_arr, y_arr)]
    poly = Polygon(poly)
    return poly


def update(idx):
    imgdata = data[IMGKEY][idx]
    imgdata2 = data[IMGKEY][idx + 1]

    annodata = get_anno(imgdata, data)
    annodata2 = get_anno(imgdata2, data)

    img = visualize_mask(imgdata, imgdata2, annodata, annodata2)

    inter = calc_iou(annodata, annodata2)

    img.show()


# %%
if __name__ == '__main__':
    idx = widgets.IntSlider(0, 0, 50)
    widgets.interactive(update, idx=idx)
# %%
idx = widgets.IntSlider(0, 0, 50)
widgets.interactive(update, idx=idx)
# %%

# %%
