A collection of efficient surveillance tools in PyTorch

This toolbox includes:\
	1. person detection (YOLOv3 [1])\
	2. object tracking (Deep SORT [2])\
	3. person reid (TorchReID [3])\
 	4. face detection (MTCNN [4])\
	5. face recognition (VGGFace2 [5])\


==============\
Usage:\
	1. Download pretrained models:\
		YOLOv3: wget -c https://pjreddie.com/media/files/yolov3.weights  => yolov3/weights/yolov3.weights\
		VGGFace2: https://drive.google.com/open?id=1A94PAAnwk6L7hXdBXLFosB_s0SzEhAFU => vggface2/resnet50_ft_weight.pkl\

	2. Run demos:\
		python test_person_track.py\
		python test_face.py\


==============\
Reference:\
    git repos:\
	https://github.com/eriklindernoren/PyTorch-YOLOv3.git\
	https://github.com/nwojke/deep_sort.git\
	https://github.com/KaiyangZhou/deep-person-reid.git\
	https://github.com/TropComplique/mtcnn-pytorch.git\
	https://github.com/cydonia999/VGGFace2-pytorch.git\

    papers:\
	[1] Farhadi, Ali, and Joseph Redmon. "Yolov3: An incremental improvement." CVPR 2018.\
	[2] Wojke, Nicolai, Alex Bewley, and Dietrich Paulus. "Simple online and realtime tracking with a deep association metric." ICIP 2017.\
	[3] Zhou, Kaiyang, and Tao Xiang. "Torchreid: A library for deep learning person re-identification in pytorch." arXiv preprint arXiv:1910.10093 (2019).\
	[4] Zhang, Kaipeng, et al. "Joint face detection and alignment using multitask cascaded convolutional networks." IEEE Signal Processing Letters.\
	[5] Cao, Qiong, et al. "Vggface2: A dataset for recognising faces across pose and age." FG 2018.\
