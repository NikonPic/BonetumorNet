# %%
#
#  utils_detectron.py
#  BonetumorNet
#
#  Created by Nikolas Wilhelm on 2020-08-01.
#  Copyright © 2020 Nikolas Wilhelm. All rights reserved.
#

# define some useful functionalities for detectron2
import os
import json
import itertools
import copy
import math
import random
import logging


from tqdm.notebook import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from PIL import Image
from sklearn.metrics import confusion_matrix, roc_curve, auc
from torchvision.transforms import functional as F
from pycocotools.coco import COCO

# detectron core specific
from fvcore.common.file_io import PathManager
from fvcore.transforms.transform import Transform

# detectron specific
from detectron2.data import build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.engine import DefaultTrainer
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.evaluation import COCOEvaluator
from detectron2.data.transforms.augmentation import TransformGen


# personal functionality
if __name__ == '__main__':
    from categories import cat_mapping_new, malign_int, benign_int, make_cat_advanced
    from utils_tumor import get_advanced_dis_data_fr, CLASS_KEY, ENTITY_KEY, F_KEY
else:
    from src.categories import cat_mapping_new, malign_int, benign_int, make_cat_advanced
    from src.utils_tumor import get_advanced_dis_data_fr, CLASS_KEY, ENTITY_KEY, F_KEY


class MyEvaluator(DatasetEvaluator):
    """Evaluator for Any Dataset, here redefined from source to avieche more funcionality"""

    def __init__(self, cfg, distributed, output_dir=None, *, use_fast_impl=True):
        self._tasks = self._tasks_from_config(cfg)
        self._distributed = distributed
        self._output_dir = output_dir
        self._use_fast_impl = use_fast_impl

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self._predicitions = []

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        self._predicitions = []


class CocoTrainer(DefaultTrainer):
    """
    customized training class, overwriteing some default functionalities
    """

    @classmethod
    def build_train_loader(cls, cfg):
        """add the idividual train_loader:"""
        return get_dataloader(cfg, is_train=True)


class CocoTrainer2(DefaultTrainer):
    """
    customized training class, overwriteing some default functionalities
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        """add the idividual train_loader:"""
        return get_dataloader(cfg, is_train=True)

# %% Apply the Rotation:


class RotTransform(Transform):
    """
    Perform rotation on image
    """

    def __init__(self, degree, height, width):
        super().__init__()
        self.degree, self.height, self.width = degree, height, width
        self.center_x = width // 2
        self.center_y = height // 2
        self.sind = math.sin(degree * (math.pi / 180))
        self.cosd = math.cos(degree * (math.pi / 180))

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Rotate the image(s).
        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: the flipped image(s).
        """
        # to PIL image
        img = Image.fromarray(img)

        # rotate the whole Image
        img = F.rotate(img, self.degree)
        # back to numpy:
        img = np.asarray(img)
        return img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Rotate the coordinates.
        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: the flipped coordinates.
        Note:
            The inputs are floating point coordinates, not pixel indices.
            Therefore they are flipped by `(W - x, H - y)`, not
            `(W - 1 - x, H - 1 - y)`.
        """
        # x' = cos(alp) * x_c - sin(alp) * y_c
        x_new_c = self.cosd * (coords[:, 0] - self.center_x) + \
            self.sind * (coords[:, 1] - self.center_y)
        # y' = sin(alp) * x_c + cos(alp) * y_c
        y_new_c = - self.sind * (coords[:, 0] - self.center_x) + \
            self.cosd * (coords[:, 1] - self.center_y)

        # reapply to edge
        coords[:, 0] = x_new_c + self.center_x
        coords[:, 1] = y_new_c + self.center_y

        return coords


class RandomRot(TransformGen):
    """
    Randomly rotate the image and annotations
    """

    def __init__(self, deg_range=30):
        """
        Args:
            crop_type (str): one of "relative_range", "relative", "absolute".
                See `config/defaults.py` for explanation.
            crop_size (tuple[float]): the relative ratio or absolute pixels of
                height and width
        """
        super().__init__()
        self.deg_range = deg_range

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        return random.uniform(degrees[0], degrees[1])

    def get_transform(self, img):
        # get the height / width
        height, width = img.shape[:2]
        # get random angle in range
        ang = self.get_params([-self.deg_range, self.deg_range])
        # return rotation operation
        return RotTransform(ang, height=height, width=width)

# %%


def softmax(x_var):
    """softmax norm vector to 1"""
    return np.exp(x_var)/sum(np.exp(x_var))


def get_active_idx(data_fr, mode, external=False):
    """Get the currently active indexes depending on the dataset mode(=test)"""
    dis = get_advanced_dis_data_fr(data_fr, mode=external)
    # get the indices of the dataset
    if mode == "test":
        active_idx = dis["test"]["idx"]
    elif mode == "valid":
        active_idx = dis["valid"]["idx"]
    elif mode == 'train':
        active_idx = dis["train"]["idx"]
    else:
        active_idx = dis["test_external"]["idx"]

    return active_idx


def auroc_helper(out, pred, entity, targets2, preds2, count2, pred_score):
    """
    determine the softmax score between first guess and second unequal guess
    """
    cla_malig = (
        1 if entity in [
            "Chondrosarkom",
            "Osteosarkom",
            "Ewing-Sarkom",
            "Plasmozytom / Multiples Myelom",
            "NHL vom B-Zell-Typ",
        ] else 0
    )
    targets2.append(cla_malig)

    pred_malig = 1 if pred in [0, 1, 2, 3, 4] else 0
    preds2.append(pred_malig)
    correct = 1 if cla_malig == pred_malig else 0
    count2 += correct

    # further get the score
    score_0 = out[:1].scores[0]
    score_1 = out[:1].scores[0]

    copy_correct = correct
    loc_count = 1

    # go trough the other suggestions and pic one uneqaul to the first
    while correct == copy_correct and loc_count < 100:
        pred_loc = out[loc_count].pred_classes[0]
        score_0 = out[loc_count].scores[0]
        pred_malig = 1 if pred_loc in [0, 1, 2] else 0
        copy_correct = 1 if cla_malig == pred_malig else 0
        loc_count += 1

    pred_score.append(softmax([score_0, score_1])[correct])

    return count2


def check_if_correct(pred_int, true_int):
    """outsourced checking func"""
    loc = 0
    if (pred_int in malign_int) and (true_int in malign_int):
        loc += 1
    if (pred_int in benign_int) and (true_int in benign_int):
        loc += 1
    return loc


def get_cla_map(simple, entity, true_int, cat_mapping, malignant):
    """define the class mapping"""
    # get the mapped integer
    cla = cat_mapping[malignant] if simple else cat_mapping[entity]
    if simple:
        cla = 1 if true_int in malign_int else 0

    return cla


def personal_score(predictor, data_fr, mode="test", simple=True, imgpath="./PNG2", external=False):
    """define the accuracy"""
    # get the dataset distribution
    active_idx = get_active_idx(data_fr, mode, external=external)

    # get the active files
    if external:
        files = [os.path.join(imgpath, f"{f}.png") for f in data_fr['id']]
    else:
        files = [os.path.join(imgpath, f"{f}.png") for f in data_fr[F_KEY]]

    # apply the category mapping dep. on simple-mode
    _, cat_mapping = make_cat_advanced(simple)

    # counters during evaluation
    count, count2 = 0, 0

    # to be filled arrays
    preds, preds2, targets, targets2 = [], [], [], []

    # auroc score (how likely compared to other solution)
    pred_score = []

    # Go over the whole dataset
    for idx in tqdm(active_idx):
        with torch.no_grad():
            # load image
            img = cv2.imread(files[idx])
            outputs = predictor(img)

            # get predicitions
            out = outputs["instances"].to("cpu")
            pred = out[:1].pred_classes[0]
            pred = pred if simple else pred + 1

        preds.append(pred)

        # select the relevant name from the data_fr
        malignant = data_fr[CLASS_KEY][idx]
        entity = data_fr[ENTITY_KEY][idx]

        _ = auroc_helper(out, pred, entity, targets2,
                         preds2, count2, pred_score)

        pred_int = pred
        true_int = cat_mapping_new[entity][0]
        count2 += check_if_correct(pred_int, true_int)

        # get the mapped integer
        cla = get_cla_map(simple, entity, true_int, cat_mapping, malignant)
        targets.append(cla)
        # increase count if true
        count += 1 if cla == pred else 0

    if simple:
        targets2 = targets
        preds2 = preds

    fpr, tpr, _ = roc_curve(targets2, pred_score)
    auc_score = auc(fpr, tpr)

    conf = confusion_matrix(targets, preds)
    conf2 = confusion_matrix(targets2, preds2)

    res = {
        "preds": preds,
        "targets": targets,
        "acc": count / len(active_idx),
        "acc2": count2 / len(active_idx),
        "conf": conf,
        "conf2": conf2,
        "rocauc": (fpr, tpr, auc_score),
    }

    return res

# %%


def plot_confusion_matrix(
    conf_mat, target_names, title="Confusion matrix", cmap=None, normalize=False, font=16
):
    """
    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    accuracy = np.trace(conf_mat) / float(np.sum(conf_mat))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap("Blues")

    fig = plt.figure(figsize=(16, 16))
    plt.imshow(conf_mat, interpolation="nearest", cmap=cmap)
    plt.title(title, fontsize=font)
    # plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90, fontsize=font)
        plt.yticks(tick_marks, target_names, fontsize=font)

    if normalize:
        conf_mat = conf_mat.astype(
            "float") / conf_mat.sum(axis=1)[:, np.newaxis]

    thresh = conf_mat.max() / 1.5 if normalize else conf_mat.max() / 2
    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        if normalize:
            plt.text(
                j,
                i,
                "{:0.4f}".format(conf_mat[i, j]),
                horizontalalignment="center",
                color="white" if conf_mat[i, j] > thresh else "black", fontsize=font
            )
        else:
            plt.text(
                j,
                i,
                "{:,}".format(conf_mat[i, j]),
                horizontalalignment="center",
                color="white" if conf_mat[i, j] > thresh else "black", fontsize=font
            )

    # plt.tight_layout()
    plt.ylabel("True label", fontsize=font)
    plt.xlabel(
        "Predicted label\naccuracy={:0.4f}; misclass={:0.4f}".format(
            accuracy, misclass), fontsize=font
    )
    plt.show()
    return fig

# %%


def get_dicts_from_coco(imgdir="../PNG2", mode="train"):
    """Custom Dataset Functionality for Detectron"""

    # load json
    json_file = f"../{mode}.json"
    with open(json_file) as file:
        coco_data = json.load(file)

    # create the list of dictionaries
    dataset_dict = []

    # itearte over all images
    for img, anns in zip(coco_data["images"], coco_data["annotations"]):
        # collect the data
        filename = os.path.join(imgdir, img["filename"])
        # transcript anns ot obj
        obj = {
            "bbox": anns["bbox"],
            "iscrowd": anns["iscrowd"],
            "segmentation": anns["segmentation"],
            "area": anns["area"]
        }
        # write the data to the dictionary
        record = {
            "file_name": filename,
            "image_id": img["id"],
            "height": img["height"],
            "width": img["width"],
            "annotations": obj,
        }
        # append the dictionary to the running list
        dataset_dict.append(record)

    # finally return the list
    return dataset_dict


def do_train(dataset_dict):
    """provide dict in train mode"""
    # USER: Modify this if you want to keep them for some reason.
    dataset_dict.pop("annotations", None)
    dataset_dict.pop("sem_seg_file_name", None)
    return dataset_dict


def do_sem_seg(dataset_dict, loc_transforms):
    if "sem_seg_file_name" in dataset_dict:
        with PathManager.open(dataset_dict.pop("sem_seg_file_name"), "rb") as loc_file:
            sem_seg_gt = Image.open(loc_file)
            sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
        sem_seg_gt = loc_transforms.apply_segmentation(sem_seg_gt)
        sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
        dataset_dict["sem_seg"] = sem_seg_gt

    return dataset_dict


class MyDatasetMapper():
    """
    Customized Datasetmapper, strongly based on the default one:
    https://detectron2.readthedocs.io/_modules/detectron2/data/dataset_mapper.html#DatasetMapper
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(
                cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)

        else:
            self.crop_gen = None

        self.tfm_gens = build_transform_gen(cfg, is_train)

        # fmt: off
        self.img_format = cfg.INPUT.FORMAT
        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.keypoint_on = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        # fmt: on
        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(
                cfg.DATASETS.TRAIN)
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train

    def do_annotations(self, dataset_dict, loc_transforms, loc_image_shape):
        """add annotations if required"""
        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, loc_transforms, loc_image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, loc_image_shape, mask_format=self.mask_format
            )
            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(
            dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(
            dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1)))

        # USER: Remove if you don't use pre-computed proposals.
        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, self.min_box_side_len, self.proposal_topk
            )

        if not self.is_train:
            return do_train(dataset_dict)

        dataset_dict = self.do_annotations(
            dataset_dict, transforms, image_shape)
        dataset_dict = do_sem_seg(dataset_dict, transforms)

        return dataset_dict


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Now it includes resizing and flipping.
    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(
            len(min_size)
        )

    # Print the transformations
    logger = logging.getLogger(__name__)
    tfm_gens = []

    # always set to uniform scale
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))

    # add all the personalized transformations for training here:
    if is_train:
        # Crop
        tfm_gens.insert(0, (T.RandomCrop(
            crop_type="relative_range", crop_size=[0.7, 1])))
        # Horizontal
        tfm_gens.append(T.RandomFlip(horizontal=True))
        # Vertical
        tfm_gens.append(T.RandomFlip(horizontal=False, vertical=True))
        # Lightning
        tfm_gens.append(T.RandomLighting(scale=3))
        # Brightness
        tfm_gens.append(T.RandomBrightness(0.9, 1.1))
        # Contrast
        tfm_gens.append(T.RandomContrast(0.9, 1.1))
        # Intensity
        tfm_gens.append(T.RandomSaturation(
            intensity_min=0.7, intensity_max=1.3))
        # NEW: Rotation
        tfm_gens.append(RandomRot(deg_range=60))

        logger.info("TransformGens used in training: %s", str(tfm_gens))

        print(tfm_gens)

    return tfm_gens


# use this dataloader instead of the default
def get_dataloader(cfg, is_train):
    """
    summarize all functionality in the dataloader
    """
    mapper = MyDatasetMapper(cfg, is_train)
    data_loader = build_detection_train_loader(cfg, mapper=mapper)
    return data_loader

# %% Evaluation:


def eval_iou_dice(predictor, data_fr, proposed=1, mode="test"):
    """
    Calculate the IoU and Dice Score for the predictor on the <proposed number>
    """
    file_list = [os.path.join("./PNG2", f"{f}.png") for f in data_fr[F_KEY]]

    active_idx = get_active_idx(data_fr, mode)

    with open(f'{mode}.json', 'r') as file:
        true_data = json.load(file)

    coco = COCO(f'{mode}.json')

    ious_box = []
    dices_box = []

    ious_mask = []
    dices_mask = []

    # go over all segmentations
    for i, idx in tqdm(enumerate(active_idx)):
        img = cv2.imread(file_list[idx])
        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")[:proposed]

        # PREDICTION
        # bbox
        pred_box = instances.pred_boxes.tensor[0].numpy()

        # mask
        pred_mask = instances.pred_masks[0].numpy()

        # GROUND TRUTH
        truth = true_data['annotations'][i]
        # box
        truth_bbox = truth['bbox']
        truth_bbox = np.array([truth_bbox[0], truth_bbox[1],
                               truth_bbox[2], truth_bbox[3]])
        # mask
        try:
            true_mask = coco.annToMask(truth)
        except FileNotFoundError:
            true_mask = pred_mask * 0

        # RESULT
        iou_box, dice_box = bb_iou_dice(pred_box, truth_bbox)
        iou_box, dice_box = max(0, iou_box), max(0, dice_box)

        iou_mask, dice_mask = mask_iou_dice(
            pred_mask, true_mask)
        iou_mask, dice_mask = max(0, iou_mask), max(0, dice_mask)

        # append
        ious_box.append(iou_box)
        dices_box.append(dice_box)

        ious_mask.append(iou_mask)
        dices_mask.append(dice_mask)

    return ious_box, dices_box, ious_mask, dices_mask


def bb_iou_dice(boxa, boxb):
    """IoU and Dice for bbox"""
    x_min = max(boxa[0], boxb[0])
    y_min = max(boxa[1], boxb[1])
    x_max = min(boxa[2], boxb[0] + boxb[2])
    y_max = min(boxa[3], boxb[1] + boxb[3])

    inter_area = (x_max - x_min) * (y_max - y_min)

    boxa_area = (boxa[2]-boxa[0]) * (boxa[3] - boxa[1])
    boxb_area = boxb[2] * boxb[3]

    iou = inter_area / float(boxa_area + boxb_area - inter_area)
    dice = (2 * inter_area) / float(boxa_area + boxb_area)

    return iou, dice


def mask_iou_dice(maska, maskb):
    """IoU and Dice for mask"""
    maska = maska > 0
    maskb = maskb > 0
    inter_area = sum(sum(np.logical_and(maska, maskb)))

    maska_area = sum(sum(maska))
    maskb_area = sum(sum(maskb))

    iou = inter_area / float(maska_area + maskb_area - inter_area)
    dice = (2 * inter_area) / float(maska_area + maskb_area)

    return iou, dice


def personal_score_simple(predictor, data_fr, active_idx, imgpath="./PNG"):
    """define the accuracy"""

    # get the actibe files
    files = [os.path.join(imgpath, f"{f}.png") for f in data_fr[F_KEY]]

    # counters during evaluation
    count = 0
    preds = []

    # Go over the whole dataset
    for idx in tqdm(active_idx):
        # load image
        img = cv2.imread(files[idx])

        with torch.no_grad():
            outputs = predictor(img)

            # get predicitions
            out = outputs["instances"].to("cpu")
            pred = out[:1].pred_classes[0]

        preds.append(pred)

        # select the relevant name from the data_fr
        malignant = data_fr[CLASS_KEY][idx]

        if malignant and pred <= 4:
            count += 1

        if not malignant and (pred >= 4 or None):
            count += 1

    res = count / len(active_idx)

    return res


# %%
