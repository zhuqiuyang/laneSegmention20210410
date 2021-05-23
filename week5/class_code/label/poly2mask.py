#!/usr/bin/env python
# coding: utf-8

import cv2
import json
import numpy as np

ann_path = '171206_054430023_Camera_5.json'
with open(ann_path, 'r') as f:
    ann = json.load(f)

h, w = ann['imageHeight'], ann['imageWidth']
ann_img = np.zeros((h, w), dtype=np.uint8)
polys = []
for poly_ann in ann['shapes']:
    if poly_ann['shape_type'] != 'polygon':
        continue
    poly = np.array(poly_ann['points']).astype(np.int64)
    polys.append(poly)

#多边形填充
cv2.fillPoly(ann_img, polys, 255)
cv2.imwrite('171206_054430023_Camera_5.png', ann_img)

