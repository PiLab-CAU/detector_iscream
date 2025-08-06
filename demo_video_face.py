# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F

import cv2
import time
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

from data.config import cfg
from models.eresfd import build_model
from models.recog import MobileFacenet
from utils.augmentations import to_chw_bgr

from mtcnn.core.detect import create_mtcnn_net, MtcnnDetector


parser = argparse.ArgumentParser(description="pyramidbox demo")
parser.add_argument(
    "--save_dir", type=str, default="tmp/", help="Directory for detect result"
)
parser.add_argument(
    "--model", type=str, default="weights/eresfd_16.pth", help="trained model"
)
parser.add_argument(
    "--video_path", type=str, default="video/sing1/test1_IMG_1260.MOV", help="video_path"
)
parser.add_argument(
    "--reference_path", type=str, default="video/sing1/test1_IMG_3326.JPG", help="reference_face"
)
parser.add_argument(
    "--thresh", default=0.4, type=float, help="Final confidence threshold"
)
parser.add_argument("--width_mult", default=0.0625, type=float, help="width-multiplier")
parser.add_argument(
    "--anchor_steps",
    type=int,
    nargs="+",
    default=cfg.STEPS,
    help="anchor stride settings. default is [4, 8, 16, 32, 64, 128]",
)
parser.add_argument(
    "--anchor_sizes",
    type=int,
    nargs="+",
    default=[16, 32, 64, 128, 256, 512],
    help="anchor size settings. default is [16, 32, 64, 128, 256, 512]",
)
parser.add_argument(
    "--anchor_scales",
    type=float,
    nargs="+",
    default=[1.0],
    action="append",
    help="anchor size scales per location. default is 1",
)
parser.add_argument(
    "--anchor_size_ratio",
    type=float,
    nargs="+",
    default=[1.25],
    action="append",
    help="anchor size ratio. default is 1.",
)
args = parser.parse_args()

# Standard MobileFaceNet reference points for 112x112 faces
REFERENCE_FIVE_POINTS = np.array([
    [38.2946, 51.6963],  # left eye
    [73.5318, 51.5014],  # right eye
    [56.0252, 71.7366],  # nose
    [41.5493, 92.3655],  # left mouth corner
    [70.7299, 92.2041]   # right mouth corner
], dtype=np.float32)

# anchor settings
# if anchor scales are same along the feature maps
if len(args.anchor_scales) == 1:
    args.anchor_scales = args.anchor_scales[0]

    anchors = []
    for anchor_size in args.anchor_sizes:
        anchors_per_size = []
        if isinstance(args.anchor_scales, list):
            for anchor_scale in args.anchor_scales:
                anchors_per_size.append(anchor_size * anchor_scale)
        else:
            anchors_per_size.append(anchor_size)
        anchors.append(anchors_per_size)
else:
    anchors = []
    for layer_idx, anchor_size in enumerate(args.anchor_sizes):
        try:
            anchor_scales = args.anchor_scales[layer_idx + 1]
        except IndexError as e:
            pass

        anchors_per_size = []
        for anchor_scale in anchor_scales:
            anchors_per_size.append(anchor_size * anchor_scale)
        anchors.append(anchors_per_size)

cfg.ANCHOR_SIZES = anchors
# # if anchor size ratio is given as default value (i.e. 1)
if len(args.anchor_size_ratio) == 1:
    args.anchor_size_ratio = [[el] for el in args.anchor_size_ratio]
    args.anchor_size_ratio = args.anchor_size_ratio * 6
else:
    args.anchor_size_ratio = args.anchor_size_ratio[1:]

    # if anchor size ratio is same along the detection layers
    if len(args.anchor_size_ratio) == 1:
        args.anchor_size_ratio = [el for el in args.anchor_size_ratio]
        args.anchor_size_ratio = args.anchor_size_ratio * 6

    # if anchor size ratio is different along the detection layers
    else:
        assert len(args.anchor_size_ratio) == 6


cfg.ANCHOR_SIZE_RATIO = args.anchor_size_ratio
print("ANCHOR ratio settings: ", cfg.ANCHOR_SIZE_RATIO)

cfg.STEPS = args.anchor_steps
print("ANCHOR stride settings: ", cfg.STEPS)


if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    torch.set_default_tensor_type("torch.FloatTensor")
    

def align_face(img, five_points, output_size=(112, 112), reference_points=REFERENCE_FIVE_POINTS):
    """
    Align face using 5 facial landmarks and output a cropped, aligned face image.

    Args:
        img (np.ndarray): Original image (HWC, BGR or RGB).
        five_points (np.ndarray): 5x2 array of landmarks from MTCNN.
        output_size (tuple): Output image size. Default is (112,112).
        reference_points (np.ndarray): Standard facial landmarks for alignment.

    Returns:
        aligned_img (np.ndarray): Aligned face image.
    """
    src = np.array(five_points).astype(np.float32)
    dst = np.array(reference_points).astype(np.float32)

    # Estimate affine transform matrix
    transform_matrix = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)[0]

    # Apply affine transformation
    aligned_img = cv2.warpAffine(img, transform_matrix, output_size, borderValue=0.0)

    return aligned_img

def blur_region(image, left_top, right_bottom, ksize=(99, 99), sigma=30):
    """
    Apply Gaussian blur to a rectangular region in the image.

    Args:
        image (np.ndarray): Input BGR image.
        left_top (tuple): (y1, x1) left top corner of the region.
        right_bottom (tuple): (y2, x2) right bottom corner of the region.
        ksize (tuple): Kernel size for Gaussian blur (must be odd).
        sigma (int): Sigma value for Gaussian blur.

    Returns:
        np.ndarray: Image with the specified region blurred.
    """
    x1, y1 = left_top
    x2, y2 = right_bottom
    roi = image[y1:y2, x1:x2]

    if roi.size == 0:
        return image  # Skip if invalid region

    blurred_roi = cv2.GaussianBlur(roi, ksize, sigma)
    image[y1:y2, x1:x2] = blurred_roi
    return image

def get_features(faces, net, mtcnn, gpu=True, load_weight='weights/recog.ckpt'):
    """
    Apply mobilefacenet features of the faces in the image
    """      
    
    aligned = []
    for face in faces:
        _, landmarks = mtcnn_detector.detect_face(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        if len(landmarks)>1:
            landmarks = landmarks[0]
        if len(landmarks) > 0 :
            aligned.append(to_chw_bgr(align_face(face, landmarks.reshape((5, 2)))))
    
    if len(aligned)>0:        
        aligned_tensor = torch.from_numpy(np.stack(aligned))

        if gpu:
            aligned_tensor=aligned_tensor.to('cuda')
            
        aligned_tensor = (aligned_tensor - 127.5) / 128.0 #following sphereface format
        
        feat = net(aligned_tensor)
    
        return feat
    else:
        return None
    

def get_reference_feat(img, net, net_rec, mtcnn, thresh):
    
    #convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = np.array(img)
    height, width, _ = img.shape
    max_im_shrink = np.sqrt(320 * 480 / (img.shape[0] * img.shape[1]))
    image = cv2.resize(
        img,
        None,
        None,
        fx=max_im_shrink,
        fy=max_im_shrink,
        interpolation=cv2.INTER_LINEAR,
    )

    x = to_chw_bgr(image)
    x = x.astype("float32")
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]
    # x = x * cfg.scale

    x = Variable(torch.from_numpy(x).unsqueeze(0))
    if use_cuda:
        x = x.cuda()
    t1 = time.time()
    y = net(x)
    detections = y.data
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    
    for i in range(detections.size(1)):
        
        j = 0
        face_list = []
        while detections[0, i, j, 0] >= thresh:
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy().astype(int)
            face_list.append(img[pt[1]:pt[3], pt[0]:pt[2]])
            j += 1
        
        feats = get_features(face_list, net_rec, mtcnn)   
        
    return feats
    


def detect(net, net_rec, mtcnn, img, thresh, max_h = 640, max_w=640, target_feat='None', thres = 0.35):
    
    #convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = np.array(img)
    height, width, _ = img.shape
    max_im_shrink = np.sqrt(320 * 480 / (img.shape[0] * img.shape[1]))
    image = cv2.resize(
        img,
        None,
        None,
        fx=max_im_shrink,
        fy=max_im_shrink,
        interpolation=cv2.INTER_LINEAR,
    )

    x = to_chw_bgr(image)
    x = x.astype("float32")
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]
    # x = x * cfg.scale

    x = Variable(torch.from_numpy(x).unsqueeze(0))
    if use_cuda:
        x = x.cuda()
    t1 = time.time()
    y = net(x)
    detections = y.data
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])

    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= thresh:
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy().astype(int)
            left_top, right_bottom = (pt[0], pt[1]), (pt[2], pt[3])
            j += 1
            
            feat = get_features([img[pt[1]:pt[3], pt[0]:pt[2]]], net_rec, mtcnn)
            if feat is not None:
                sim = F.cosine_similarity(target_feat, feat)
                #print(sim)
                if sim>thres:
                    img = blur_region(img, left_top=left_top, right_bottom=right_bottom) 
            else:        
                img = blur_region(img, left_top=left_top, right_bottom=right_bottom)
                
                   
                    

    t2 = time.time()
    print("timer:{}".format(t2 - t1))

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    


if __name__ == "__main__":
    net = build_model("test", cfg.NUM_CLASSES, args.width_mult)
    net.load_weights(args.model)
    net.eval()   
          
    net_rec = MobileFacenet()   
    mtcnn = MTCNN(image_size=112)
    pnet, rnet, onet = create_mtcnn_net(p_model_path="./weights/pnet_epoch.pt", r_model_path="./weights/rnet_epoch.pt", o_model_path="./weights/onet_epoch.pt", use_cuda=use_cuda)
    
    if use_cuda:
        net.cuda()       
        net_rec.cuda()        
        ckpt = torch.load('weights/recog.ckpt')
        cudnn.benckmark = True
    net_rec.load_state_dict(ckpt['net_state_dict'])
    net_rec.eval()
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)
    
    #get reference feature
    reference_image = cv2.imread(args.reference_path)
    reference_feat = get_reference_feat(reference_image, net, net_rec, mtcnn_detector, args.thresh)[0]
    #if img.mode == "L":
    #    img = img.convert("RGB")

    #img = np.array(img)
       
    
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print("Error opening video.")
        exit()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    save_file = os.path.join(args.video_path.split()[-1].split('/')[-1])
    out = cv2.VideoWriter(args.save_dir, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out_frame=detect(net, net_rec, mtcnn_detector, frame, args.thresh, target_feat=reference_feat)

        out.write(out_frame)
        cv2.imshow('Blurred Faces', out_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


    #img_path = "./img"
    #img_list = [
    #    os.path.join(img_path, x) for x in os.listdir(img_path) if x.endswith("jpg")
    #]
    #for path in img_list:
    #    with torch.no_grad():
    #        detect(net, path, args.thresh)
