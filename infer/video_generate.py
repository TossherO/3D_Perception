import os
import cv2
import numpy as np

# img_path = './infer/results/'
img_path = './data/indoor2/imgs/'
img_save_path = './infer/video_results/indoor2_cam.mp4'
# idx = [[250, 550], [1000, 1250], [1725, 2025]]
idx = [[1, 1387]]
img = cv2.imread(img_path + '%04d.png' % idx[0][0])
height, width, _ = img.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(img_save_path, fourcc, 10, (width, height), True)
for i in idx:
    for j in range(i[0], i[1]):
        img = cv2.imread(img_path + '%04d.png' % j)
        video.write(img)
video.release()