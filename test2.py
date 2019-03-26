import myconfig as cfg
import cv2
import random
import numpy as np
import pickle

from manager import DetectMng

detect_manager = DetectMng(cfg.detect_cfg, cfg.detect_weights)

for vfn in cfg.video_fn_list:
    vid = cv2.VideoCapture(vfn)
    success = True
    T = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    while success:
        success, image = vid.read()
        if success:
            count+=1
            if count>10 and count<T-10: continue
            output = detect_manager.detect_img(image)
            image = detect_manager.visualize_img(image, output)
            cv2.imwrite(str(count)+'.jpg',image)
    break
