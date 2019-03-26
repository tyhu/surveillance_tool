import glob
import os.path as osp

video_data_root = '/home/ubuntu/Backup/datasets/airport/september/camera3'
video_fn_list = glob.glob(osp.join(video_data_root,'*.mp4'))


### for object detection
detect_cfg = 'yolov3/cfg/yolov3.cfg'
detect_weights = 'yolov3/weights/yolov3.weights'
