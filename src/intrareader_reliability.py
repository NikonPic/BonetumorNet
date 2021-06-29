# %%
#
#  intrareader_reliability.py
#  BonetumorNet
#
#  Created by Nikolas Wilhelm on 2020-11-01.
#  Copyright © 2020 Nikolas Wilhelm. All rights reserved.
#
import os
import nrrd
import numpy as np
from PIL import Image
from tqdm import tqdm
from detec_helper import compare_masks, get_bb_from_mask


p1 = 'SEG'
p2 = 'Intrareader_seg'
im_p = 'PNG2'

all_segs = os.listdir(p1)
comp_segs = os.listdir(p2)


def get_nrrd_mask(filename, img_path, nrrd_path, fac=15, nrrd_key='Segmentation_ReferenceImageExtentOffset'):
    # load nrrd image
    readdata, header = nrrd.read(f'{nrrd_path}/{filename}')
    nrrd_img = np.transpose(readdata[:, :, 0] * fac)

    # get the offsets
    offset = header[nrrd_key].split()
    offset = [int(off) for off in offset]
    offset = offset[0:2]

    # load true image
    background = Image.open(f'{img_path}/{filename[:-9]}.png')
    foreground = Image.fromarray(nrrd_img)

    # generate masked image
    mask = Image.fromarray(np.array(background) * 0)
    mask.paste(foreground, offset, foreground)

    return np.array(mask)[:, :, 0]


def make_bool(x_var):
    if x_var > 0:
        return True
    return False


def make_1(x_var):
    if x_var:
        return 1
    return 0


vfunc = np.vectorize(make_bool)
vfunc1 = np.vectorize(make_1)
comp2_seg = []

# ensure all files exist
for seg in all_segs:
    if seg in comp_segs:
        comp2_seg.append(seg)


ious_bb = []
dice_scores_bb = []

ious_mask = []
dice_scores_mask = []

# go trough all segmentations
for seg in tqdm(comp2_seg):
    mask1 = get_nrrd_mask(seg, im_p, p1)
    mask2 = get_nrrd_mask(seg, im_p, p2)

    mask1_bb = vfunc1(get_bb_from_mask(mask1))
    mask2_bb = vfunc1(get_bb_from_mask(mask2))

    mask1 = vfunc1(mask1)
    mask2 = vfunc1(mask2)

    iou_loc, dice_loc = compare_masks(mask1, mask2)
    iou_loc_bb, dice_loc_bb = compare_masks(mask1_bb, mask2_bb)

    ious_bb.append(iou_loc_bb)
    dice_scores_bb.append(dice_loc_bb)

    ious_mask.append(iou_loc)
    dice_scores_mask.append(dice_loc)


# %%
iou_mean, iou_std = np.array(ious_mask).mean().round(
    decimals=2), np.array(ious_mask).std().round(decimals=3)
dice_mean, dice_std = np.array(dice_scores_mask).mean().round(
    decimals=2), np.array(dice_scores_mask).std().round(decimals=2)
print(f'IOU MASK: {iou_mean} +/- {iou_std}')
print(f'DICE MASK: {dice_mean} +/- {dice_std}')


iou_mean, iou_std = np.array(ious_mask).mean().round(
    decimals=2), np.array(ious_mask).std().round(decimals=2)
dice_mean, dice_std = np.array(dice_scores_mask).mean().round(
    decimals=2), np.array(dice_scores_mask).std().round(decimals=2)
print(f'IOU BB: {iou_mean} +/- {iou_std}')
print(f'DICE BB: {dice_mean} +/- {dice_std}')
# %%

seg = comp2_seg[0]
mask1 = get_nrrd_mask(seg, im_p, p1)
mask2 = get_nrrd_mask(seg, im_p, p2)
