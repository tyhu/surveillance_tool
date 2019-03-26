import myconfig as cfg
import cv2
import random
import numpy as np
import pickle

import torch
from torch.autograd import Variable

from yolov3.darknet import Darknet
from yolov3.util import write_results, load_classes

def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas



def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


def post_rescale(output, input_dim, dim):
    im_dim = torch.FloatTensor(dim).repeat(1,2)
    im_dim = im_dim.repeat(output.size(0), 1)
    scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)
    output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
    output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
    output[:,1:5] /= scaling_factor
    for i in range(output.shape[0]): 
        output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
        output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])
    return output

def write(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img


model = Darknet(cfg.detect_cfg)
model.load_weights(cfg.detect_weights)
model.cuda()
model.eval()
model.net_info['height'] = 416
inp_dim = model.net_info['height']
confidence = 0.5
num_classes = 80
nms_thesh = 0.4

classes = load_classes('yolov3/data/coco.names')
colors = pickle.load(open("yolov3/pallete", "rb"))

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
            image, ori_im, dim = prep_image(image, inp_dim)
            image = image.cuda()
            with torch.no_grad():
                output = model(Variable(image), True)
                output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)
                output = output.cpu()
                output = post_rescale(output, inp_dim, dim)
                print(output)
                
                list(map(lambda x: write(x, ori_im), output))
                cv2.imwrite(str(count)+'.jpg',ori_im)
    break
