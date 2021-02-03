import myconfig as cfg
import cv2
from PIL import Image
import random
import math
import numpy as np
import pickle

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

### Object Detection
from yolov3.darknet import Darknet
from yolov3.util import write_results, load_classes, prep_image, post_rescale

### Face Detection
from mtcnn.model import PNet, RNet, ONet
import mtcnn.box_utils as mtcnn_util
#from mtcnn.box_utils import nms, calibrate_box, get_image_boxes, convert_to_square, _preprocess

### Person ReID
#from deep_reid.reid_util import load_person_reid_model, extract_reid_feats

### Object Tracking (deep_sort)
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

### vggface2
import vggface2.resnet as vgg_resnet

### reid
#from torchreid.utils import FeatureExtractor

"""
Manager for Object Detection
"""
class DetectMng(object):
    def __init__(self, cfg_fn, weight_fn, class_fn='yolov3/data/coco.names', color_fn='yolov3/pallete', conf=0.5, nms_thres=0.4):
        self.model = Darknet(cfg_fn)
        self.model.load_weights(weight_fn)
        self.model.cuda()
        self.model.eval()
        self.model.net_info['height'] = 416

        self.classes = load_classes(class_fn)
        self.colors = pickle.load(open(color_fn, "rb"))
        self.inp_dim = self.model.net_info['height']
        self.confidence = conf
        self.num_classes = len(self.classes)
        self.nms_thesh = nms_thres
       
    def detect_img(self, image):
        image, ori_im, dim = prep_image(image, self.inp_dim)
        image = image.cuda()
        with torch.no_grad():
            output = self.model(Variable(image), True)
            output = write_results(output, self.confidence, self.num_classes, nms = True, nms_conf = self.nms_thesh)
            output = output.cpu()
            output = post_rescale(output, self.inp_dim, dim)
        return output.numpy()

        
    def visualize_img(self, image, output):
        list(map(lambda x: self._write(x, image), output))
        return image

    def _write(self, x, img):
        #c1 = tuple(x[1:3].int())
        #c2 = tuple(x[3:5].int())
        c1 = tuple(x[1:3].astype('int'))
        c2 = tuple(x[3:5].astype('int'))
        cls = int(x[-1])
        label = "{0}".format(self.classes[cls])
        color = random.choice(self.colors)
        cv2.rectangle(img, c1, c2,color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2,color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
        return img


"""
Manager for Face Detection
"""
class MTCNNMng(object):
    def __init__(self, min_face_size=20.0, thresholds=[0.6, 0.7, 0.8], nms_thresholds=[0.7, 0.7, 0.7]):
        self.pnet, self.rnet, self.onet = PNet(), RNet(), ONet()
        self.onet.eval()

        self.min_face_size = min_face_size
        self.min_detection_size = 12
        self.factor = 0.707 # sqrt(0.5)
        self.thresholds = thresholds
        self.nms_thresholds = nms_thresholds
        self.m = self.min_detection_size/min_face_size

    def detect_img(self, image):
        image = Image.fromarray(image)
        width, height = image.size
        min_length = min(height, width)
        min_length *= self.m
        scales = []

        factor_count = 0
        while min_length > self.min_detection_size:
            scales.append(self.m*self.factor**factor_count)
            min_length *= self.factor
            factor_count += 1

        ### P-Net
        bboxes = self._run_pnet(image, scales)
        if len(bboxes)==0: return [],[]

        ### R-Net
        bboxes = self._run_rnet(image, bboxes)
        
        ### O-Net
        bboxes, landmarks, offsets = self._run_onet(image, bboxes)
        
        ### calibration
        if len(bboxes)==0: return [],[]

        width = bboxes[:, 2] - bboxes[:, 0] + 1.0
        height = bboxes[:, 3] - bboxes[:, 1] + 1.0
        xmin, ymin = bboxes[:, 0], bboxes[:, 1]
        landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1)*landmarks[:, 0:5]
        landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1)*landmarks[:, 5:10]

        bboxes = mtcnn_util.calibrate_box(bboxes, offsets)
        keep = mtcnn_util.nms(bboxes, self.nms_thresholds[2], mode='min')
        bboxes = bboxes[keep]
        landmarks = landmarks[keep]

        return bboxes, landmarks

    def _run_pnet(self, image, scales):
        all_bboxes = []
        for s in scales:
            bboxes = self._run_pnet_once(image, s)
            all_bboxes.append(bboxes)
        all_bboxes = [i for i in all_bboxes if i is not None]
        if len(all_bboxes)==0: return []
        all_bboxes = np.vstack(all_bboxes)
        return self._bbox_post_process(all_bboxes, self.nms_thresholds[0])


    def _run_pnet_once(self, image, scale):
        width, height = image.size
        sw, sh = math.ceil(width*scale), math.ceil(height*scale)
        img = image.resize((sw, sh), Image.BILINEAR)
        img = np.asarray(img, 'float32')
        img = torch.FloatTensor(mtcnn_util._preprocess(img))
    
        output = self.pnet(img)
        probs = output[1].data.numpy()[0, 1, :, :]
        offsets = output[0].data.numpy()

        boxes = self._generate_bboxes(probs, offsets, scale, self.thresholds[0])
        if len(boxes) == 0:
            return None
        keep = mtcnn_util.nms(boxes[:, 0:5], overlap_threshold=0.5)
        return boxes[keep]

    def _run_rnet(self, image, bboxes):
        img_boxes = mtcnn_util.get_image_boxes(bboxes, image, size=24)
        img_boxes = torch.FloatTensor(img_boxes)
        output = self.rnet(img_boxes)
        offsets = output[0].data.numpy()  # shape [n_boxes, 4]
        probs = output[1].data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > self.thresholds[1])[0]
        bboxes = bboxes[keep]
        bboxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]

        keep = mtcnn_util.nms(bboxes, self.nms_thresholds[1])
        bboxes = bboxes[keep]
        bboxes = mtcnn_util.calibrate_box(bboxes, offsets[keep])
        bboxes = mtcnn_util.convert_to_square(bboxes)
        bboxes[:, 0:4] = np.round(bboxes[:, 0:4])
        return bboxes

    def _run_onet(self, image, bboxes):
        img_boxes = mtcnn_util.get_image_boxes(bboxes, image, size=48)
        if len(img_boxes) == 0: 
            return [], [], []
        img_boxes = torch.FloatTensor(img_boxes)
        output = self.onet(img_boxes)
        landmarks = output[0].data.numpy()  # shape [n_boxes, 10]
        offsets = output[1].data.numpy()  # shape [n_boxes, 4]
        probs = output[2].data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > self.thresholds[2])[0]
        bboxes = bboxes[keep]
        bboxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]
        landmarks = landmarks[keep]
        return bboxes, landmarks, offsets

    def _bbox_post_process(self, bboxes, nms_thres):
        keep = mtcnn_util.nms(bboxes[:,0:5], nms_thres)
        bboxes = bboxes[keep]
        bboxes = mtcnn_util.calibrate_box(bboxes[:,0:5], bboxes[:,5:])
        bboxes = mtcnn_util.convert_to_square(bboxes)
        bboxes[:,0:4] = np.round(bboxes[:,0:4])
        return bboxes

    def _generate_bboxes(self, probs, offsets, scale, threshold):
        """
        Generate bounding boxes at places where there is probably a face.
        """
        stride = 2
        cell_size = 12

        inds = np.where(probs > threshold)

        if inds[0].size == 0:
            return np.array([])

        tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]

        offsets = np.array([tx1, ty1, tx2, ty2])
        score = probs[inds[0], inds[1]]

        # P-Net is applied to scaled images, so we need to rescale bounding boxes back
        bounding_boxes = np.vstack([
            np.round((stride*inds[1] + 1.0)/scale),
            np.round((stride*inds[0] + 1.0)/scale),
            np.round((stride*inds[1] + 1.0 + cell_size)/scale),
            np.round((stride*inds[0] + 1.0 + cell_size)/scale),
            score, offsets
        ])

        return bounding_boxes.T


"""
Manager for Object Tracking
"""
class TrackMng(object):
    def __init__(self, max_cosine_distance=-0.1, nn_budget=5, color_fn='deep_sort/pallete'):
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)
        self.colors = pickle.load(open(color_fn,'rb'))
        
    def track_img(self, bboxes, scores, features):
        if scores is None:
            scores = np.array([1]*len(bboxes))
        if features is None:
            features = np.array([[1,1]]*len(bboxes))
        detections = self._create_detections(bboxes, scores, features)
        self.tracker.predict()
        self.tracker.update(detections)

    def _create_detections(self, bboxes, scores, features):
        detection_list = []
        for i in range(len(bboxes)):
            bbox, score, feature = bboxes[i], scores[i], features[i]
            detection_list.append(Detection(bbox, score, feature))
        return detection_list

    def visualize_img(self, img):
        for track in self.tracker.tracks:
            if track.state!=2: continue
            if track.time_since_update>1: continue
            bbox = track.to_tlbr()
            c1 = tuple(bbox[:2].astype('int'))
            c2 = tuple(bbox[2:4].astype('int'))
            color = self.colors[track.track_id]
            label = 'id_'+str(track.track_id)

            cv2.rectangle(img, c1, c2, color, 1)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
            c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
            cv2.rectangle(img, c1, c2,color, -1)
            cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);


class VggFace2Mng(object):
    def __init__(self, modelfn=''):
        N_IDENTITY = 8631
        with open(modelfn, 'rb') as f:
            obj = f.read()
            weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
        model = vgg_resnet.resnet50(num_classes=N_IDENTITY, include_top=False)
        model.load_state_dict(weights)
        self.model = model

        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),  # Pre-trained model uses 224x224 input images
            transforms.ToTensor(),
        ])

        self.mean_bgr = torch.Tensor([91.4953, 103.8827, 131.0912])

    """
    img: numpy in (w,h,c)
    bboxes: [[x1,y1,x2,y2]]
    """
    def extract_feature(self, img, bboxes):
        img = img[:, :, ::-1]
        feat_list = []
        for i in range(len(bboxes)):
            x1,y1,x2,y2 = int(bboxes[i,0]), int(bboxes[i,1]), int(bboxes[i,2]), int(bboxes[i,3])
            patch = img[y1:y2, x1:x2, :]
            patch = self.preprocess(patch)
            patch = patch - self.mean_bgr.unsqueeze(1).unsqueeze(1)
            feat = self.model(patch.unsqueeze(0))
            feat = feat.squeeze(0).squeeze(-1).squeeze(-1)
            feat_list.append(feat.detach().numpy())
        return np.array(feat_list)
        

"""
Person reID
"""

class ReIDMng(object):
    def __init__(self, modelfn=''):
        self.extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path=modelfn,
            device='cuda'
        )


    """
    img: np.array (from cv2)
    bboxes: list of (x1,y1,x2,y2)
    """
    def torchreid_feature(self, img, bboxes):
        img = img[:, :, ::-1]
        feat_list = []
        for i in range(len(bboxes)):
            x1,y1,x2,y2 = int(bboxes[i,0]), int(bboxes[i,1]), int(bboxes[i,2]), int(bboxes[i,3])
            patch = img[y1:y2, x1:x2, :]
            feat = self.extractor(patch)
            feat_list.append(feat.detach().numpy())
        return np.array(feat_list)
 
