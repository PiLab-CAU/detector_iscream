# -*- coding:utf-8 -*-

# Command 예시
# python demo_video.py --video_path ../video/test1_IMG_1260.MOV    (default model: Resnet50)
# python demo_video.py --model weights/mobilenet0.25_Final.pth --network mobile0.25 --video_path ../video/test1_IMG_1260.MOV

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import cv2
import time
import numpy as np
from PIL import Image

# RetinaFace imports
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms

parser = argparse.ArgumentParser(description="RetinaFace video blur demo")
parser.add_argument(
    "--save_dir", type=str, default="tmp/", help="Directory for detect result"
)
parser.add_argument(
    "--model", type=str, default="weights/Resnet50_Final.pth", help="trained model"
)
parser.add_argument(
    "--video_path", type=str, default="video/test1_IMG_1260.MOV", help="video_path"
)
parser.add_argument(
    "--thresh", default=0.4, type=float, help="Final confidence threshold"
)
parser.add_argument(
    "--nms_threshold", default=0.4, type=float, help="NMS threshold"
)
parser.add_argument(
    "--network", default="resnet50", type=str, help="Backbone network: resnet50 or mobile0.25"
)
parser.add_argument(
    "--max_size", default=1080, type=int, help="Max size for input image"
)
parser.add_argument(
    "--debug", action='store_true', help="Enable debug mode to visualize detections"
)
parser.add_argument(
    "--save_detection", action='store_true', help="Save detection results as images"
)
parser.add_argument(
    "--output_height", default=480, type=int, help="Output video height (width will be scaled proportionally)"
)
args = parser.parse_args()

# RetinaFace configuration
cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

cfg_re50 = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    torch.set_default_tensor_type("torch.FloatTensor")

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path):
    print('Loading pretrained model from {}'.format(pretrained_path))
    pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def blur_region(image, left_top, right_bottom, ksize=(99, 99), sigma=30):
    """
    Apply Gaussian blur to a rectangular region in the image.

    Args:
        image (np.ndarray): Input BGR image.
        left_top (tuple): (x1, y1) left top corner of the region.
        right_bottom (tuple): (x2, y2) right bottom corner of the region.
        ksize (tuple): Kernel size for Gaussian blur (must be odd).
        sigma (int): Sigma value for Gaussian blur.

    Returns:
        np.ndarray: Image with the specified region blurred.
    """
    x1, y1 = left_top
    x2, y2 = right_bottom
    
    # Ensure coordinates are within image bounds
    height, width = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    
    roi = image[y1:y2, x1:x2]

    if roi.size == 0:
        return image  # Skip if invalid region

    blurred_roi = cv2.GaussianBlur(roi, ksize, sigma)
    image[y1:y2, x1:x2] = blurred_roi
    return image

def draw_detections(image, dets, thresh):
    """Draw bounding boxes on image for debugging"""
    img_debug = image.copy()
    for det in dets:
        if det[4] < thresh:
            continue
        x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
        conf = det[4]
        
        # Draw rectangle
        cv2.rectangle(img_debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw confidence score
        cv2.putText(img_debug, f'{conf:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img_debug

def detect(net, img, cfg, thresh, nms_threshold, frame_num=0, target_height=None):
    """
    Detect faces using RetinaFace and apply blur
    Returns: (processed_image, detection_time, total_time)
    """
    img_raw = img.copy()
    original_height, original_width = img.shape[:2]
    
    # Resize to target height if specified
    if target_height and target_height != original_height:
        resize_factor = target_height / original_height
        target_width = int(original_width * resize_factor)
        img_for_detection = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
        print(f"  Input resized from {original_width}x{original_height} to {target_width}x{target_height}")
    else:
        resize_factor = 1.0
        img_for_detection = img.copy()
        target_height = original_height
        target_width = original_width
    
    img_for_detection = np.float32(img_for_detection)
    
    # Get image dimensions for detection
    im_height, im_width, _ = img_for_detection.shape
    
    print(f"  Detection image size: {im_width}x{im_height}")
    
    # Further resize for detection if needed (max_size)
    if im_height > args.max_size or im_width > args.max_size:
        detect_resize = float(args.max_size) / float(im_height) if im_height > im_width else float(args.max_size) / float(im_width)
    else:
        detect_resize = 1
    
    if detect_resize != 1:
        img_for_detection = cv2.resize(img_for_detection, None, None, fx=detect_resize, fy=detect_resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img_for_detection.shape
        print(f"  Further resized to: {im_width}x{im_height} for detection")
    
    scale = torch.Tensor([img_for_detection.shape[1], img_for_detection.shape[0], img_for_detection.shape[1], img_for_detection.shape[0]])
    img_for_detection -= (104, 117, 123)
    img_for_detection = img_for_detection.transpose(2, 0, 1)
    img_for_detection = torch.from_numpy(img_for_detection).unsqueeze(0)
    
    if use_cuda:
        img_for_detection = img_for_detection.cuda()
        scale = scale.cuda()

    t1 = time.time()
    
    # Forward pass
    with torch.no_grad():
        loc, conf, landms = net(img_for_detection)
    
    # Get prior boxes
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    if use_cuda:
        priors = priors.cuda()
    prior_data = priors.data
    
    # Decode predictions
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    
    # Filter by confidence threshold
    inds = np.where(scores > thresh)[0]
    boxes = boxes[inds]
    scores = scores[inds]
    
    print(f"  Detections before NMS: {len(boxes)}")
    
    # Apply NMS
    if len(boxes) > 0:
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        scores = scores[order]
        
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        dets = dets[keep, :]
    else:
        dets = np.array([])
    
    t2 = time.time()
    detection_time = t2 - t1
    print(f"  Detection time: {detection_time:.3f}s")
    print(f"  Faces detected: {len(dets)}")
    
    # Apply blur to detected faces on original resolution image
    blur_count = 0
    for i, det in enumerate(dets):
        if det[4] < thresh:
            continue
        
        # Get coordinates on detection image
        x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
        
        # Scale back to target resolution (480p)
        if detect_resize != 1:
            x1 = int(x1 / detect_resize)
            y1 = int(y1 / detect_resize)
            x2 = int(x2 / detect_resize)
            y2 = int(y2 / detect_resize)
        
        # Scale back to original resolution
        if resize_factor != 1.0:
            x1 = int(x1 / resize_factor)
            y1 = int(y1 / resize_factor)
            x2 = int(x2 / resize_factor)
            y2 = int(y2 / resize_factor)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, original_width - 1))
        y1 = max(0, min(y1, original_height - 1))
        x2 = max(0, min(x2, original_width - 1))
        y2 = max(0, min(y2, original_height - 1))
        
        # print(f"    Face {i+1}: Detection coords: ({int(det[0])}, {int(det[1])}) - ({int(det[2])}, {int(det[3])})")
        # print(f"    Face {i+1}: Original coords: ({x1}, {y1}) - ({x2}, {y2}), confidence: {det[4]:.2f}")
        
        if x2 > x1 and y2 > y1:  # Valid box
            img_raw = blur_region(img_raw, left_top=(x1, y1), right_bottom=(x2, y2))
            blur_count += 1
        else:
            print(f"    Face {i+1}: Invalid box, skipping")
    
    t3 = time.time()
    blur_time = t3 - t1
    print(f"  Blur time: {blur_time:.3f}s")
    print(f"  Blurred {blur_count} faces")
    
    
    # Debug mode: save detection visualization
    if args.debug and len(dets) > 0 and args.save_detection and frame_num % args.save_every == 0:
        # Adjust detections for visualization on original image
        debug_dets = dets.copy()
        for j in range(len(debug_dets)):
            if detect_resize != 1:
                debug_dets[j][0:4] = debug_dets[j][0:4] / detect_resize
            if resize_factor != 1.0:
                debug_dets[j][0:4] = debug_dets[j][0:4] / resize_factor
        
        debug_img = draw_detections(img_raw, debug_dets, thresh)
        debug_path = os.path.join(args.save_dir, f"debug_frame_{frame_num}.jpg")
        cv2.imwrite(debug_path, debug_img)
        print(f"  Debug image saved")
    
    return img_raw, detection_time, blur_time


if __name__ == "__main__":
    # Select config based on network
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    else:
        cfg = cfg_re50
    
    print(f"Using network: {args.network}")
    print(f"Confidence threshold: {args.thresh}")
    print(f"NMS threshold: {args.nms_threshold}")
    print(f"Debug mode: {args.debug}")
    
    # Build and load model
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, args.model)
    net.eval()
    
    print('Finished loading model!')
    
    if use_cuda:
        net = net.cuda()
        cudnn.benchmark = True
        print("Using CUDA")
    else:
        print("Using CPU")
    
    # Open video
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print("Error opening video.")
        exit()

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate output dimensions
    if args.output_height:
        output_height = args.output_height
        output_width = int(original_width * (output_height / original_height))
        # Ensure width is even for video encoding
        output_width = output_width if output_width % 2 == 0 else output_width + 1
    else:
        output_width = original_width
        output_height = original_height

    print(f"Original video: {original_width}x{original_height}, {fps} fps, {total_frames} frames")
    print(f"Output video: {output_width}x{output_height}")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_name = os.path.basename(args.video_path).replace(".MOV", "_blurred.mp4").replace(".mp4", "_blurred.mp4")
    save_file = os.path.join(args.save_dir, video_name)
    out = cv2.VideoWriter(save_file, fourcc, fps, (output_width, output_height))

    frame_count = 0
    total_faces_detected = 0
    detection_times = []
    total_times = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        print(f"\nProcessing frame {frame_count}/{total_frames}")
        
        # Detect and blur faces
        # Pass target height for detection
        detection_height = args.output_height if args.output_height else original_height
        out_frame, det_time, total_time = detect(net, frame, cfg, args.thresh, args.nms_threshold, frame_count, detection_height)
        
        # Skip first frame for timing statistics (initialization overhead)
        if frame_count > 1:
            detection_times.append(det_time)
            total_times.append(total_time)
        
        # Resize output frame to target resolution
        if args.output_height and (output_height != original_height):
            out_frame = cv2.resize(out_frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
        
        out.write(out_frame)
        
        # Optional: Display frame
        if args.debug:
            display_frame = cv2.resize(out_frame, (output_width//2, output_height//2)) if output_width > 960 else out_frame
            cv2.imshow('Blurred Faces', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\nVideo saved to: {save_file}")
    print(f"Total frames processed: {frame_count}")
    
    # Print timing statistics
    if len(detection_times) > 0:
        avg_detection_time = sum(detection_times) / len(detection_times)
        avg_total_time = sum(total_times) / len(total_times)
        
        print(f"\n=== Timing Statistics (excluding first frame) ===")
        print(f"Average detection time: {avg_detection_time:.3f}s")
        print(f"Average detection+blur time: {avg_total_time:.3f}s")
        print(f"Average FPS (detection only): {1/avg_detection_time:.1f}")
        print(f"Average FPS (total): {1/avg_total_time:.1f}")
        print(f"Min detection time: {min(detection_times):.3f}s")
        print(f"Max detection time: {max(detection_times):.3f}s")
        print(f"Min total time: {min(total_times):.3f}s")
        print(f"Max total time: {max(total_times):.3f}s")
        print(f"Total time: {sum(total_times):.3f}s")