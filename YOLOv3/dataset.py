import torch
import numpy as np
import os
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import iou_width_height as iou, non_max_suppression as nms

# Make sure there are no errors when loading the images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLOv3Dataset(Dataset):
    def __init__(self, csv_file, img_dir, label_dir, anchors, image_size=416, S=[13,26,52], C=20, transforms=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2]) # for all three scales (3 anchors each)
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3 
        self.C = C
        # only 1 anchor box should be responsible for each of three possible object midpoints that might appear in the grid cell
        # However, other anchor boxes might have high IOU --> ignore them if their IOU is greater than 0.5
        self.ignore_iou_threshold = 0.5

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        # load data from text file using np.loadtxt (each label path is a text file) as a numpy array
        # ndmin = number of dimensions (2 dimensions -- row for each bounding box, each bounding box has 5 values)
        # np.roll() by 4 "rolls" along axis = 1 (columns) so that row [p, x, y, w, h] becomes [x, y, w, h, p] (for data augmentation later)
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        # ensure it is numpy array for augmentation later
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # dim 0 = number of anchor boxes
        # dim 1 & 2 = grid size
        # dim 3 = 6 --> [x, y, w, h, c] (c is NOT one-hot-encoded yet, should be)
        # REMEMBER: self.S = [13, 26, 52] --> 3 targets for the 3 predictions
        targets = [torch.zeroes((self.num_anchors // 3, S, S, 6)) for S in self.S]

        for box in bboxes:
            # broadcasting done here to make iou be taken with box and ALL the anchors
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            # no p_o in box (not in label)
            x, y, width, height, class_label = box
            has_anchor = [False, False, False]

            # start w/ anchor w/ highest IOU with ground truth box
            # anchor that has highest IOU with ground-truth label will be responsible for that object
            for anchor_index in anchor_indices:
                scale_index = anchor_index // self.num_anchors_per_scale # 0, 1, or 2
                anchor_on_scale = anchor_index % self.num_anchors_per_scale # 0,1,or 2 (3 anchors per scale)
                S = self.S[scale_index]
                i, j = int(S*y), int(S*x) # coordinates of the cell 
                anchor_taken = targets[scale_index][anchor_on_scale, i, j, 0]

                if not anchor_taken and not has_anchor[scale_index]:
                    # set probability of object equal to 1
                    # note: p_o is first index because np.roll() was done for bboxes, NOT targets
                    targets[scale_index][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S*x - j, S*y - i
                    width_cell, height_cell = S*width, S*height

                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    targets[scale_index][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_index][anchor_on_scale, i, j, 5] = int(class_label)

                    has_anchor[scale_index] = True

                elif not anchor_taken and iou_anchors[anchor_index] > self.ignore_iou_threshold:
                    # ignore this anchor box prediction (don't punish it in the loss function), as specified in the paper
                    targets[scale_index][anchor_on_scale, i, j, 0] = -1 

        return image, tuple(targets)

