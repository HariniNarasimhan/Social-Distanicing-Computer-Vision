from __future__ import absolute_import, division, print_function
%matplotlib inline

import os
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
import cv2
import torch
from torchvision import transforms

import networks
from utils import download_model_if_doesnt_exist

from gluoncv import model_zoo, data, utils
import time
import mxnet as mx

from scipy.spatial.distance import euclidean


parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

parser.add_argument('--video_path', type=str,
                        help='path to a test image or folder of images', required=True)


model_name = "monocular_depth_model"

download_model_if_doesnt_exist(model_name)
encoder_path = os.path.join("models", model_name, "encoder.pth")
depth_decoder_path = os.path.join("models", model_name, "depth.pth")

# LOADING PRETRAINED MODEL
encoder = networks.ResnetEncoder(18, False)
depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)

loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
depth_decoder.load_state_dict(loaded_dict)

encoder.eval()
depth_decoder.eval();


detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
detector.reset_class(["person"], reuse_weights=['person'])

def person_boxes(img, bboxes, scores=None, labels=None, thresh=0.5,
                 class_names=None, colors=None,
                 absolute_coordinates=True, scale=1.0, linewidth=2):
    bounding_boxs=[]
    
    

    if labels is not None and not len(bboxes) == len(labels):
        raise ValueError('The length of labels and bboxes mismatch, {} vs {}'
                         .format(len(labels), len(bboxes)))
    if scores is not None and not len(bboxes) == len(scores):
        raise ValueError('The length of scores and bboxes mismatch, {} vs {}'
                         .format(len(scores), len(bboxes)))

    if isinstance(img, mx.nd.NDArray):
        img = img.asnumpy()
    if isinstance(bboxes, mx.nd.NDArray):
        bboxes = bboxes.asnumpy()
    if isinstance(labels, mx.nd.NDArray):
        labels = labels.asnumpy()
    if isinstance(scores, mx.nd.NDArray):
        scores = scores.asnumpy()
    if len(bboxes) < 1:
        return img

    if not absolute_coordinates:
        # convert to absolute coordinates using image shape
        height = img.shape[0]
        width = img.shape[1]
        bboxes[:, (0, 2)] *= width
        bboxes[:, (1, 3)] *= height
    else:
        bboxes *= scale


    # use random colors if None is provided
    if colors is None:
        colors = dict()
    for i, bbox in enumerate(bboxes):
        if scores is not None and scores.flat[i] < thresh:
            continue
        if labels is not None and labels.flat[i] < 0:
            continue
        cls_id = int(labels.flat[i]) if labels is not None else -1
        if cls_id not in colors:
            if class_names is not None:
                colors[cls_id] = plt.get_cmap('hsv')(cls_id / len(class_names))
            else:
                colors[cls_id] = (random.random(), random.random(), random.random())
        xmin, ymin, xmax, ymax = [int(x) for x in bbox]
        bounding_boxs.append([xmin, ymin, xmax, ymax])
    return bounding_boxs


def rect_distance(pt1,pt2):
    x1, y1, x1b, y1b=pt1
    x2, y2, x2b, y2b=pt2
    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2
    if top and left:
        return euclidean([x1, y1b], [x2b, y2])
    elif left and bottom:
        return euclidean([x1, y1], [x2b, y2b])
    elif bottom and right:
        return euclidean([x1b, y1], [x2, y2b])
    elif right and top:
        return euclidean([x1b, y1b], [x2, y2])
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    else:
        return 15


cap = cv2.VideoCapture(args.video_path)
count=0
while True: 
#     print('reading')
    #Capture frame-by-frame
    __, img = cap.read()
    if img is not None:
        count=count+1
        x,img = data.transforms.presets.ssd.transform_test(mx.nd.array(img),short=512)
        class_IDs, scores, bounding_boxes = detector(x)
        input_image = pil.fromarray(img)
        original_width, original_height = input_image.size

        feed_height = loaded_dict_enc['height']
        feed_width = loaded_dict_enc['width']
        input_image_resized = input_image.resize((feed_width, feed_height), pil.LANCZOS)

        input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)
        with torch.no_grad():
            features = encoder(input_image_pytorch)
            outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(disp,
        (original_height, original_width), mode="bilinear", align_corners=False)

        # Saving colormapped depth image
        disp_resized_np = disp_resized.squeeze().cpu().numpy()
        
        vmax = np.percentile(disp_resized_np, 95)

#         plt.imshow(disp_resized_np, cmap='magma', vmax=vmax)
        bounding_boxs = person_boxes(disp_resized_np, bounding_boxes[0], scores[0],class_IDs[0], class_names=detector.classes)
        pers=[]
        for i in range(len(bounding_boxs)):
            if i not in pers:
                for j in range(len(bounding_boxs)):
                    if i!=j:
#                         print(bounding_boxs[i],bounding_boxs[j])
                        xmin,ymin,xmax,ymax = bounding_boxs[i]
                        depth1= np.average(disp_resized_np[ymin:ymax,xmin:xmax])
                        xmin,ymin,xmax,ymax = bounding_boxs[j]
                        depth2= np.average(disp_resized_np[ymin:ymax,xmin:xmax])
                        depth = abs(depth1-depth2)
                        if depth<0.1:
                            distance = rect_distance(bounding_boxs[i],bounding_boxs[j])
                            if distance<10:
                                xmin,ymin,xmax,ymax = bounding_boxs[i]
                                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
                                y = ymin - 15 if ymin - 15 > 15 else ymin + 15
                                cv2.putText(img,'NO SD',
                                            (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, min(1.0/2, 2),
                                             (0,0,255), min(int(1.0), 5), lineType=cv2.LINE_AA)
                                xmin,ymin,xmax,ymax = bounding_boxs[j]
                                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
                                y = ymin - 15 if ymin - 15 > 15 else ymin + 15
                                cv2.putText(img,'NO SD',
                                            (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, min(1.0/2, 2),
                                             (0,0,255), min(int(1.0), 5), lineType=cv2.LINE_AA)
                                pers.append(i)
                                pers.append(j)
                                break
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 500, 300)
    cv2.imshow('frame',img)
    cv2.namedWindow('Depth_information', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Depth_information', 500, 300)
    disp_resized_np = np.array(disp_resized_np * 255, dtype = np.uint8)
    disp_resized_np = cv2.applyColorMap(disp_resized_np, cv2.COLORMAP_BONE)
    cv2.imshow('Depth_information',disp_resized_np)
    
    if cv2.waitKey(1) &0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()       