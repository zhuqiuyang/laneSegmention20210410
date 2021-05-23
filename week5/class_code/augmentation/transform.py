import cv2
import numpy as np

img = cv2.imread('imgs/cat.png')
h, w = img.shape[:2]

# rigid
theta = 30.0 / 180.0 * np.pi
tx = 10
ty = 20
rigid_mat = np.array([[np.cos(theta), -np.sin(theta), tx],
                      [np.sin(theta),  np.cos(theta), ty]], dtype=np.float32)
rigid_img = cv2.warpAffine(img, rigid_mat, (w, h))
show_img = np.concatenate([img, rigid_img], axis=1)
cv2.imwrite('imgs/rigid.png', show_img)

# similar
theta = 30.0 / 180. * np.pi
tx = 50
ty = -150
s = 0.5
similar_mat = np.array([[s * np.cos(theta), -s * np.sin(theta), tx],
                        [s * np.sin(theta),  s * np.cos(theta), ty]], dtype=np.float32)
similar_img = cv2.warpAffine(img, similar_mat, (w, h))
show_img = np.concatenate([img, similar_img], axis=1)
cv2.imwrite('imgs/similar.png', show_img)

# affine
a, b, c, d = 1.1, 0.9, 0.1, 1.0
tx = -100
ty = 30
affine_mat = np.array([[a, b, tx],
                       [c, d, ty]], dtype=np.float32)
affine_img = cv2.warpAffine(img, affine_mat, (w, h))
show_img = np.concatenate([img, affine_img], axis=1)
cv2.imwrite('imgs/affine.png', show_img)

# perspective 
a, b, c, d = 1.0, 0.0, 0.0, 1.0
tx = 50
ty = 0
v1 = 0
v2 = 1e-3
persp_mat = np.array([[ a,  b, tx],
                      [ c,  d, ty],
                      [v1, v2,  1]], dtype=np.float32)
persp_img = cv2.warpPerspective(img, persp_mat, (w, h))
show_img = np.concatenate([img, persp_img], axis=1)
cv2.imwrite('imgs/persp.png', show_img)
