import myconfig as cfg
import cv2
import random
import numpy as np
import pickle

from manager import DetectMng, TrackMng

detect_manager = DetectMng(cfg.detect_cfg, cfg.detect_weights)
tracker_manager = TrackMng()


vfn = 'data/video_example.avi'
vid = cv2.VideoCapture(vfn)
success = True
T = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
count = 0

out_fn = './detect_track.avi'
out_vid = None

while success:
    success, image = vid.read()

    if out_vid is None:
        h,w,_ = image.shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_vid = cv2.VideoWriter(out_fn,fourcc, 20.0, (w,h))

    if success:
        count+=1
        print('processing frame: '+str(count))
        output = detect_manager.detect_img(image)

        ### keep person detection results only
        output = output[output[:,-1]==0,:]
        #image = detect_manager.visualize_img(image, output)
        bboxes = output[:,1:5]
        bboxes[:,2]-=bboxes[:,0]
        bboxes[:,3]-=bboxes[:,1]
        bboxes = bboxes[bboxes[:,2]>0,:]  ## removing dummy detection results
        bboxes = bboxes[bboxes[:,3]>0,:]

        tracker_manager.track_img(bboxes, None, None)
        tracker_manager.visualize_img(image)
        #print(count, output.shape, len(tracker_manager.tracker.tracks))
        #cv2.imwrite('output/'+str(count)+'.jpg',image)
        out_vid.write(image)
        
print('\nDone')
if out_vid is not None:
    out_vid.release()
