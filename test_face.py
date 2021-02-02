import sys
import myconfig as cfg
import cv2
import random
import numpy as np
from manager import MTCNNMng, VggFace2Mng

mtcnn_manager = MTCNNMng()
vggface2_manager = VggFace2Mng(modelfn=cfg.vggface_model_fn)

imgfn = 'data/face_example.jpg'

img = cv2.imread(imgfn)
### face detection
bboxes,_ = mtcnn_manager.detect_img(img)

### visualize
for i in range(len(bboxes)):
    x1,y1,x2,y2 = int(bboxes[i,0]), int(bboxes[i,1]), int(bboxes[i,2]), int(bboxes[i,3])
    cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
cv2.imwrite("face_out.jpg", img)

### extract vggface2 embedding feature, face recognition or reid
vggface_features = vggface2_manager.extract_feature(img, bboxes)
