import sys

import torch
from .DenseNet import DenseNet121
from .transforms import Resize, Compose, ToTensor, Normalize

transform = Compose([
    Resize((256, 128)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_person_reid_model(modelfn='/home/tingyaoh/github/deep-person-reid/log/dense_market_600.pth.tar', num_classes=751):
    model = DenseNet121(num_classes=num_classes)
    checkpoint = torch.load(modelfn)
    model.load_state_dict(checkpoint['state_dict'])
    return model.cuda()

## bboxes: x1,y1,x2,y2
def extract_reid_feats(model, img, bboxes):
    inputs = []
    feats = []
    with torch.no_grad():
        for i in range(len(bboxes)):
            bbox = bboxes[i]
            x1,y1,x2,y2 = bbox
            patch = img.crop((x1,y1,x2,y2))
            input = transform(patch)
            inputs.append(input)
            if len(inputs)==16 or i==len(bboxes)-1:
                inputs_th = torch.stack(inputs,dim=0).cuda()
                feat = model.forward(inputs_th).cpu()
                feats.append(feat)
                inputs = []
    if len(bboxes)>0:
        feats = torch.cat(feats,dim=0)
        return feats
    else: return None
    

