import cv2
import numpy as np

img = cv2.imread('imgs/cat.png')

print('blur')
new_img = cv2.blur(img, (5, 5))
show_img = np.concatenate([img, new_img], axis=1)
cv2.imshow('show', show_img)
cv2.waitKey(0)

print('sharpen')
new_img = cv2.blur(img, (5, 5))
new_img = cv2.addWeighted(img, 1.5, new_img, -0.5, 0)
show_img = np.concatenate([img, new_img], axis=1)
cv2.imshow('show', show_img)
cv2.waitKey(0)

print('gaussian noise')
mean = 0
std = 50
noise = np.random.randn(*img.shape) * std
new_img = np.clip((img.astype(np.float32) + noise), 0, 255).astype(np.uint8)
show_img = np.concatenate([img, new_img], axis=1)
cv2.imshow('show', show_img)
cv2.waitKey(0)

print('poisson noise')
crop_img = img[100:200, 100:200]
lam = 100.0
noise = np.random.poisson(lam=lam, size=crop_img.shape) - lam
new_img = np.clip(crop_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
show_img = np.concatenate([crop_img, new_img], axis=1)
show_img = cv2.resize(show_img, (1000, 500))
cv2.imshow('show', show_img)
cv2.waitKey(0)

print('salt pepper noise')
rate = 0.02
noise = np.random.choice((0, 256, -256), (img.shape[0], img.shape[1]), p=(1-rate, rate/2, rate/2))
noise = np.expand_dims(noise, axis=-1)
new_img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
show_img = np.concatenate([img, new_img], axis=1)
cv2.imshow('show', show_img)
cv2.waitKey(0)
