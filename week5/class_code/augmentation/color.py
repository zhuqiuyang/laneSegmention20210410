import cv2
import numpy as np

img = cv2.imread('imgs/cat.png')

print('contrast / brightness')
alpha = 1.0
beta = 0.0
while True:
    new_img = (img * alpha + beta)
    new_img = np.clip(new_img, 0, 255).astype(np.uint8)
    cv2.imshow('show', new_img)
    key = cv2.waitKey(1)
    if key != -1:
        print(key)
    if key == 97: # left/a
        beta -= 2
    if key == 100: # right/d
        beta += 2
    if key == 115: # down/s
        alpha -= 0.05
    if key == 119: # up/w
        alpha += 0.05
    if key == 27: # esc
        break

print('hue / saturation / value(brightness)')
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h = 0.0
s = 0.0
v = 0.0
while True:
    new_img_hsv = img_hsv + np.array([h, s, v]).reshape(1, 1, 3)
    new_img_hsv = np.clip(new_img_hsv, 0, 255).astype(np.uint8)
    new_img = cv2.cvtColor(new_img_hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('show', new_img)
    if key != -1:
        print(key)
    key = cv2.waitKey(1)
    if key == 97: # left
        h -= 5
    if key == 100: # right
        h += 5
    if key == 115: # down
        s -= 5
    if key == 119: # up
        s += 5
    if key == 113: # q
        v -= 5
    if key == 101: # e
        v += 5
    if key == 27: # esc
        break

print('equalization')
b = cv2.equalizeHist(img[..., 0])
g = cv2.equalizeHist(img[..., 1])
r = cv2.equalizeHist(img[..., 2])
new_img = np.stack([b, g, r], axis=-1)
cv2.imshow('show', new_img)
key = cv2.waitKey(0)
