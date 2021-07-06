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
import matplotlib.pyplot as plt
import numpy as np

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


def visualize_mask(imgdata, imgdata2, annodata, annodata2, highlight=False):
    outl = 0 if highlight else None

    # get both images
    original_copy = Image.open(f'{IMGDIR}/{imgdata[IMGFKEY]}')
    original_paste = Image.open(f'{IMGDIR}/{imgdata2[IMGFKEY]}')
    # get both polygons
    poly = annodata[ANNSEGKEY][0]
    poly2 = annodata2[ANNSEGKEY][0]
    # create new blank image
    mask = Image.new("L", original_copy.size, 0)
    draw = ImageDraw.Draw(mask)
    # draw new segmentation on blank image
    draw.polygon(poly, fill=255, outline=outl)
    # merge the images
    result = Image.composite(original_copy, original_paste, mask)
    draw2 = ImageDraw.Draw(result)

    print(check_poly_lim(poly, original_paste))

    if highlight:
        draw2.polygon(poly2, fill=None, outline=outl)

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


def check_iou_lim(annodata, annodata2, lim=0.01):
    """only include annotations, if th iou is lower than {lim}"""
    try:
        iou = calc_iou(annodata, annodata2)
    except:
        return False
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


def update(idx, highlight=False):
    imgdata = data[IMGKEY][idx]
    imgdata2 = data[IMGKEY][idx + 1]

    annodata = get_anno(imgdata, data)
    annodata2 = get_anno(imgdata2, data)

    img = visualize_mask(imgdata, imgdata2, annodata,
                         annodata2, highlight=highlight)

    inter = calc_iou(annodata, annodata2)
    print(inter)

    imgarr = np.asarray(img)
    plt.imshow(imgarr)

# ok we need a logic, that imports multiple images and adds them to the paste img by randomness


def include_multiple_imgs(idx, maxlen=650, max_append=20, highlight=False):
    """load img and include multiple annotations for this image randomly"""
    # get the original img data
    imgdata_org = data[IMGKEY][idx]
    annodata_org = get_anno(imgdata_org, data)
    img_paste = Image.open(f'{IMGDIR}/{imgdata_org[IMGFKEY]}')

    img_arr = np.asarray(img_paste)
    print(np.mean(img_arr))
    mean_img = np.mean(img_arr)

    # get a random number of annotations to add to the img
    append_num = np.random.randint(1, high=max_append)
    append_idx_list = []
    anno_list = []
    img_list = []

    # get all idx to use and fill img and annos
    for _ in range(append_num):
        # take a random integer an check, that it is not already included
        randint = np.random.randint(0, maxlen)

        if randint != idx and randint not in append_idx_list:
            append_idx_list.append(randint)
            imgdata = data[IMGKEY][randint]
            img_list.append(imgdata)
            annodata = get_anno(imgdata, data)
            anno_list.append(annodata)

    # now start adding the segmnented images to the original image
    # only if the new annotation doesnt collide with all previous annotations!
    annos_on_img = [annodata_org]
    for loc_imgdata, loc_annodata in zip(img_list, anno_list):
        include = True
        # get the local polygon
        loc_poly = loc_annodata[ANNSEGKEY][0]
        # only continue if the dimensions match
        if check_poly_lim(loc_poly, img_paste):
            # compare polygon to all previous polygons
            for anno_on_img in annos_on_img:
                x, y = loc_annodata['bbox'][:2]
                print(np.mean(img_arr[x, y, :]))
                mean_seg = np.mean(img_arr[x, y])

                include = include if 2 * mean_img < mean_seg else False
                include = include if check_iou_lim(
                    loc_annodata, anno_on_img) else False
        else:
            include = False

        # finally add the polygon if include remains true:
        if include:
            img_copy = Image.open(f'{IMGDIR}/{loc_imgdata[IMGFKEY]}')
            mask = Image.new("L", img_copy.size, 0)
            draw = ImageDraw.Draw(mask)
            # draw new segmentation on blank image
            draw.polygon(loc_poly, fill=255, outline=None)
            img_paste = Image.composite(img_copy, img_paste, mask)
            annos_on_img.append(loc_annodata)

    if highlight:
        draw2 = ImageDraw.Draw(img_paste)
        for anno in annos_on_img:
            poly_org = anno[ANNSEGKEY][0]
            draw2.polygon(poly_org, fill=None, outline=0)

        imgarr = np.asarray(img_paste)
        plt.figure(figsize=(20, 20))
        plt.imshow(imgarr)
    else:
        return img_paste, annos_on_img


def extend_training_data(original_path='PNG', max_append=30):
    """extend the original dataset by using copy-paste"""
    data_extended = data.copy()
    # update the image names:
    for index, imgdata in enumerate(data_extended[IMGKEY]):
        data_extended[IMGKEY][index][IMGFKEY] = f'{original_path}/{imgdata[IMGFKEY]}'

    # now add new images
    for index, _ in enumerate(data[IMGKEY]):
        img, annos = include_multiple_imgs(
            index, maxlen=650, max_append=max_append, highlight=False)


        # %%
if __name__ == '__main__':
    idx = widgets.IntSlider(0, 0, 50)
    widgets.interactive(update, idx=idx)
    # %%
    idx = widgets.IntSlider(0, 0, 50)
    widgets.interactive(include_multiple_imgs, idx=idx)
    # %%
