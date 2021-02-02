import glob
import os.path as osp


### for object detection
## wget -c https://pjreddie.com/media/files/yolov3.weights
detect_cfg = 'yolov3/cfg/yolov3.cfg'
detect_weights = 'yolov3/weights/yolov3.weights'

vggface_model_fn = 'vggface2/resnet50_ft_weight.pkl'
