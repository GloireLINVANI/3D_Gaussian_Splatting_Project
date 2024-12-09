import numpy as np
from transformers import pipeline
from PIL import Image
import requests

"""
This file is about to evaluate the succes of the segmentation model. It compares a mask of a segmented object
with the labeled data by using the IoU (intersection over union) metric
The labeled data can be generated via labelme
"""

img1 = np.array([[1,2,3,0],
               [4,5,6,0],
               [7,8,9,0],
               [0,0,0,0]])

img2 = np.array([[0,0,0,0],
               [0,1,2,3],
               [0,4,5,6],
               [0,7,8,9]])


def IoU(img1, img2):
    """
    img1: np array with num != 0 for labeled pixels and 0 for the others
    img2: np array with num != 0 for labeled pixels and 0 for the others
    """
    img1[np.where(img1 != 0)] = 1
    img2[np.where(img2 != 0)] = 1

    intersection = np.logical_and(img1, img2).astype(int)
    union = np.logical_or(img1, img2).astype(int)

    return np.sum(intersection)/np.sum(union)


def get_ious_from_masks(masks, ground_truths):
    """
    masks: a list of numpy arrays for each instances
    ground truths: a list of label mask representing the ground truth
    returns a list of tuple (max_iou, gt_idx)
    """
    max_ious = []
    #for each mask find the maximum iou
    for mask in masks:
        max_iou = 0
        gt_idx = 0
        for i, gt in enumerate(ground_truths):
            iou = IoU(mask, gt)
            if iou > max_iou:
                max_iou = iou
                gt_idx = i
        max_ious.append((max_iou, gt_idx))
    
    return max_ious


if __name__ == "__main__":
    #IoU test
    iou = IoU(img1, img2)
    print(f"iou is {iou}")

    #segmentation model
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/segmentation_input.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    semantic_segmentation = pipeline("image-segmentation", "nvidia/segformer-b1-finetuned-cityscapes-1024-1024", device='cuda')
    results = semantic_segmentation(image)
    print(results)


