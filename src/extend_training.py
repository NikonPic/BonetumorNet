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

from PIL import Image, ImageDraw, ImageFilter
import json
from ipywidgets import widgets
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import shutil

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMGDIR = '../PNG'
IMGKEY = 'images'
IMGFKEY = 'file_name'

EXT_DIR = 'EXT'

ANNKEY = 'annotations'
ANNSEGKEY = 'segmentation'
ANNIMGIDKEY = 'image_id'
ANNIDKEY = 'id'

TRAIN_JSON = '../train.json'

with open(TRAIN_JSON) as fp:
    data = json.load(fp)


def get_anno(imgdata, data):
    """get the annotation from the imgdata"""
    img_id = imgdata['id']

    for anno in data[ANNKEY]:
        anno_id = anno['id']

        if anno_id == img_id:
            return anno.copy()

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

    if poly.is_valid and poly2.is_valid:
        poly_inter = poly.intersection(poly2)
        inter_area = poly_inter.area

        iou = inter_area / (area + area2)

        return iou

    return 1


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


def include_multiple_imgs(idx, gen_round=1, maxlen=650, max_append=60, highlight=False, outline=False):
    """load img and include multiple annotations for this image randomly"""
    # get the original img data
    imgdata_org = data[IMGKEY][idx]
    annodata_org = get_anno(imgdata_org, data)
    img_paste = Image.open(f'{IMGDIR}/{imgdata_org[IMGFKEY]}')
    id_org = annodata_org[ANNIDKEY]
    new_id = id_org + gen_round * maxlen + 1
    img_arr = np.asarray(img_paste)

    # get a random number of annotations to add to the img
    append_num = np.random.randint(int(max_append / 2), high=max_append)
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
    annos_on_img[0][ANNIDKEY] = new_id
    annos_on_img[0][ANNIMGIDKEY] = new_id

    include_count = 0

    if len(img_arr.shape) < 3:
        print(f'Name: {imgdata_org[IMGFKEY]}')
        print(f'Shape invalid: {idx}')
        return new_id, img_paste, annos_on_img

    for loc_imgdata, loc_annodata in zip(img_list, anno_list):
        img_copy = Image.open(f'{IMGDIR}/{loc_imgdata[IMGFKEY]}')
        img_copy_arr = np.asarray(img_copy)

        if len(img_copy_arr.shape) < 3:
            print('Shape invalid')
            continue

        include = True
        # get the local polygon
        loc_poly = loc_annodata[ANNSEGKEY][0]
        # only continue if the dimensions match
        if check_poly_lim(loc_poly, img_paste):
            # compare polygon to all previous polygons
            for anno_on_img in annos_on_img:
                x, y = loc_annodata['bbox'][:2]
                width, height = loc_annodata['bbox'][2:4]
                mean_seg = np.mean(img_arr[y:y+height, x: x+width, :])
                mean_img = np.mean(img_copy_arr[y:y+height, x: x+width, :])

                include = include if (
                    0.8 * mean_img < mean_seg and 1.2*mean_img > mean_seg) else False
                include = include if check_iou_lim(
                    loc_annodata, anno_on_img) else False
        else:
            include = False

        # finally add the polygon if include remains true:
        if include:
            include_count += 1
            mask = Image.new("L", img_copy.size, 0)
            draw = ImageDraw.Draw(mask)

            # draw new segmentation on blank image
            draw.polygon(loc_poly, fill=255, outline=None)

            # blur the mask
            mask_blur = mask.filter(ImageFilter.GaussianBlur(10))

            # composite of images
            img_paste = Image.composite(img_copy, img_paste, mask_blur)

            # change the annotation ids
            loc_annodata[ANNIDKEY] = new_id
            loc_annodata[ANNIMGIDKEY] = new_id * 100000 + include_count

            annos_on_img.append(loc_annodata)

    if highlight:
        draw2 = ImageDraw.Draw(img_paste)
        if outline:
            for anno in annos_on_img:
                poly_org = anno[ANNSEGKEY][0]
                draw2.polygon(poly_org, fill=None, outline=None)

        imgarr = np.asarray(img_paste)
        plt.figure(figsize=(20, 20))
        plt.imshow(imgarr)
    else:
        return new_id, img_paste, annos_on_img


def extend_training_data(original_path='PNG', max_append=60, gen_round=1):
    """extend the original dataset by using copy-paste"""

    # manage paths
    if os.path.isdir(f'../{EXT_DIR}'):
        shutil.rmtree(f'../{EXT_DIR}')
    os.mkdir(f'../{EXT_DIR}')

    with open(TRAIN_JSON) as fp:
        data = json.load(fp)

    # copy the original training data
    data_extended = data.copy()
    max_len = len(data['images'])

    # update the image names:
    for index, imgdata in enumerate(data_extended[IMGKEY]):
        data_extended[IMGKEY][index][IMGFKEY] = f'{original_path}/{imgdata[IMGFKEY]}'

    with open(TRAIN_JSON) as fp:
        data = json.load(fp)

    # perform the annotation process multiple times per image?
    for _ in range(1, gen_round+1):
        # now add new images
        for index, imgdata in tqdm(enumerate(data[IMGKEY])):

            # genertae new image and annotations
            new_id, img, annos = include_multiple_imgs(
                index, maxlen=max_len, max_append=max_append, highlight=False)

            # save the image
            loc_img_name = imgdata[IMGFKEY].split('.')[0]
            filename = f'{EXT_DIR}/{loc_img_name}_{new_id}.png'
            img.save(f'../{filename}')

            # create new image_data
            img_data_new = {
                "id": new_id,
                "file_name": filename,
                "height": imgdata['height'],
                "width": imgdata['width']
            },

            # append the dataset by the new annotations
            data_extended[IMGKEY].append(img_data_new)
            data_extended[ANNKEY].extend(annos)

    # finally save the new dataset
    save_file = '../training_extended.json'
    print(f'Saving to: {save_file}')
    with open(save_file, 'w') as file_p:
        json.dump(data_extended, file_p, indent=2)


# %%
if __name__ == '__main__':
    idx = widgets.IntSlider(626, 0, 654)

    # %%
    widgets.interactive(include_multiple_imgs, idx=idx)

    # %%
    extend_training_data(max_append=120, gen_round=1)

# %%
