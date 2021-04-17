import cv2
import numpy as np
import time


# 循环版本
def bilinear_interpolation(src, dst_size):
    src_h, src_w, channel = src.shape
    dst_w, dst_h = dst_size
    x_scale = dst_w / src_w
    y_scale = dst_h / src_h
    dst = np.zeros((dst_h, dst_w, channel), dtype=src.dtype)
    for c in range(channel):
        for dst_x in range(dst_w):
            for dst_y in range(dst_h):
                src_x = (dst_x + 0.5) / x_scale - 0.5
                src_y = (dst_y + 0.5) / y_scale - 0.5

                src_x1 = int(src_x)
                src_y1 = int(src_y)
                src_x2 = src_x1 + 1
                src_y2 = src_y1 + 1

                def clip(v, vmin, vmax):
                    v = v if v >= vmin else vmin
                    v = v if v <= vmax else vmax
                    return v

                src_x1 = clip(src_x1, 0, src_w - 1)
                src_x2 = clip(src_x2, 0, src_w - 1)
                src_y1 = clip(src_y1, 0, src_h - 1)
                src_y2 = clip(src_y2, 0, src_h - 1)

                y1_value = (src_x - src_x1) * src[src_y1, src_x2, c] + (src_x2 - src_x) * src[src_y1, src_x1, c]
                y2_value = (src_x - src_x1) * src[src_y2, src_x2, c] + (src_x2 - src_x) * src[src_y2, src_x1, c]
                dst[dst_y, dst_x, c] = (src_y - src_y1) * y2_value + (src_y2 - src_y) * y1_value
    return dst


# 矩阵版本
def bilinear_interpolation_fast(src, dst_size):
    src_h, src_w, channel = src.shape
    dst_w, dst_h = dst_size
    x_scale = dst_w / src_w
    y_scale = dst_h / src_h

    dst_y, dst_x = np.mgrid[:dst_h, :dst_w]
    src_x = (dst_x + 0.5) / x_scale - 0.5  # 小数
    src_y = (dst_y + 0.5) / y_scale - 0.5

    src_x1 = src_x.astype(np.int64)  # 整数
    src_y1 = src_y.astype(np.int64)
    src_x2 = src_x1 + 1
    src_y2 = src_y1 + 1

    # 解决超出边界的情况
    src_x1 = np.clip(src_x1, 0, src_w-1)
    src_x2 = np.clip(src_x2, 0, src_w-1)
    src_y1 = np.clip(src_y1, 0, src_h-1)
    src_y2 = np.clip(src_y2, 0, src_h-1)

    '''
    np.expand_dims(src_x - src_x1, -1)
    -1 :最后一个纬度
    '''
    y1_value = np.expand_dims(src_x - src_x1, -1) * src[src_y1, src_x2] + \
               np.expand_dims(src_x2 - src_x, -1) * src[src_y1, src_x1]

    y2_value = np.expand_dims(src_x - src_x1, -1) * src[src_y2, src_x2] + \
               np.expand_dims(src_x2 - src_x, -1) * src[src_y2, src_x1]

    dst = np.expand_dims(src_y - src_y1, -1) * y2_value + \
          np.expand_dims(src_y2 - src_y, -1) * y1_value

    dst = dst.astype(src.dtype)
    return dst


if __name__ == '__main__':
    img = cv2.imread('./week1/cat.png')
    t1 = time.time()
    img_b = bilinear_interpolation(img, (512, 512))
    t2 = time.time()
    img_bf = bilinear_interpolation_fast(img, (512, 512))
    t3 = time.time()
    print('slow: ', t2 - t1)
    print('fast: ', t3 - t2)
    print('Difference slow-fast: ', ((img_bf - img_b) ** 2).sum())
    cv2.imwrite('show.png',img_bf)
    #cv2.imshow('show', img_bf)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    '''
    slow:  8.563268184661865
    fast:  0.07288289070129395
    Difference slow-fast:  0
    '''
