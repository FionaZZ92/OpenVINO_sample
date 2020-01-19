#!/usr/bin/env python
"""
 Copyright (C) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from __future__ import print_function
import sys
import os
from math import exp as exp
from argparse import ArgumentParser, SUPPRESS
import cv2
import colorsys
import time
import logging as log
from datetime import datetime
import numpy as np
from openvino.inference_engine import IENetwork, IECore
import threading
from PIL import Image
from yolo3.utils import letterbox_image
from PIL import Image, ImageFont, ImageDraw

def parseValuePerDevice(devices, values_string):
    ## Format: <device1>:<value1>,<device2>:<value2> or just <value>
    result = {}
    if not values_string:
      return result
    device_value_strings = values_string.upper().split(',')
    for device_value_string in device_value_strings:
        device_value_vec = device_value_string.split(':')
        if len(device_value_vec) == 2:
            for device in devices:
                if device == device_value_vec[0]:
                    value = int(device_value_vec[1])
                    result[device_value_vec[0]] = value
                    break
        elif len(device_value_vec) == 1:
            value = int(device_value_vec[0])
            for device in devices:
                result[device] = value
        elif not device_value_vec:
            raise Exception("Unknown string format: " + values_string)
    return result

def parseDevices(device_string):
    devices = device_string
    if ':' in devices:
        devices = devices.partition(':')[2]
    return [ d[:d.index('(')] if '(' in d else d for d in devices.split(',') ]

def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

def entry_index(side, coord, classes, location, entry):
    side_power_2 = side ** 2
    n = location // side_power_2
    loc = location % side_power_2
    return int(side_power_2 * (n * (coord + classes + 1) + entry) + loc)


def scale_bbox(x, y, h, w, class_id, confidence, h_scale, w_scale):
    xmin = int((x - w / 2) * w_scale)
    ymin = int((y - h / 2) * h_scale)
    xmax = int(xmin + w * w_scale)
    ymax = int(ymin + h * h_scale)
    return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id, confidence=confidence)

class YoloParams:
    # ------------------------------------------- Extracting layer parameters ------------------------------------------
    # Magic numbers are copied from yolo samples
    def __init__(self, param, side):
        self.num = 3 if 'num' not in param else int(param['num'])
        self.coords = 4 if 'coords' not in param else int(param['coords'])
        self.classes = 80 if 'classes' not in param else int(param['classes'])
        self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0,
                        198.0,
                        373.0, 326.0] if 'anchors' not in param else [float(a) for a in param['anchors'].split(',')]

        if 'mask' in param:
            mask = [int(idx) for idx in param['mask'].split(',')]
            self.num = len(mask)

            maskedAnchors = []
            for idx in mask:
                maskedAnchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
            self.anchors = maskedAnchors

        self.side = side
        self.isYoloV3 = 'mask' in param  # Weak way to determine but the only one.


    def log_params(self):
        params_to_print = {'classes': self.classes, 'num': self.num, 'coords': self.coords, 'anchors': self.anchors}
        [log.info("         {:8}: {}".format(param_name, param)) for param_name, param in params_to_print.items()]

def parse_yolo_region(blob, resized_image_shape, original_im_shape, params, threshold):
    # ------------------------------------------ Validating output parameters ------------------------------------------
    _, _, out_blob_h, out_blob_w = blob.shape
    assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
                                     "be equal to width. Current height = {}, current width = {}" \
                                     "".format(out_blob_h, out_blob_w)
    # ------------------------------------------ Extracting layer parameters -------------------------------------------
    orig_im_h, orig_im_w = original_im_shape
    resized_image_h, resized_image_w = resized_image_shape
    objects = list()
    predictions = blob.flatten()
    side_square = params.side * params.side

    # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
    for i in range(side_square):
        row = i // params.side
        col = i % params.side
        for n in range(params.num):
            obj_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, params.coords)
            scale = predictions[obj_index]
            if scale < threshold:
                continue
            box_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, 0)
            # Network produces location predictions in absolute coordinates of feature maps.
            # Scale it to relative coordinates.
            x = (col + predictions[box_index + 0 * side_square]) / params.side
            y = (row + predictions[box_index + 1 * side_square]) / params.side
            # Value for exp is very big number in some cases so following construction is using here
            try:
                w_exp = exp(predictions[box_index + 2 * side_square])
                h_exp = exp(predictions[box_index + 3 * side_square])
            except OverflowError:
                continue
            # Depends on topology we need to normalize sizes by feature maps (up to YOLOv3) or by input shape (YOLOv3)
            w = w_exp * params.anchors[2 * n] / (resized_image_w if params.isYoloV3 else params.side)
            h = h_exp * params.anchors[2 * n + 1] / (resized_image_h if params.isYoloV3 else params.side)
            for j in range(params.classes):
                class_index = entry_index(params.side, params.coords, params.classes, n * side_square + i,
                                          params.coords + 1 + j)
                confidence = scale * predictions[class_index]
                if confidence < threshold:
                    continue
                objects.append(scale_bbox(x=x, y=y, h=h, w=w, class_id=j, confidence=confidence,
                                          h_scale=orig_im_h, w_scale=orig_im_w))
    return objects

def intersection_over_union(box_1, box_2):
    width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
    height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
    if width_of_overlap_area < 0 or height_of_overlap_area < 0:
        area_of_overlap = 0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
    box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
    area_of_union = box_1_area + box_2_area - area_of_overlap
    if area_of_union == 0:
        return 0
    return area_of_overlap / area_of_union

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=str)
    args.add_argument("-i", "--input",
                      help="Required. Path to video file or image. 'cam' for capturing video stream from camera",
                      required=True, type=str, nargs="+")
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with the "
                           "kernels implementations.", type=str, default=None)
    args.add_argument("-pp", "--plugin_dir", help="Optional. Path to a plugin folder", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is "
                           "acceptable. The demo will look for a suitable plugin for device specified. "
                           "Default value is CPU", default="CPU", type=str)
    args.add_argument("--labels", help="Optional. Path to labels mapping file", default='coco_classes.txt', type=str)
    #args.add_argument("-pt", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
    #                  default=0.5, type=float)
    args.add_argument('-api', '--api_type', type=str, required=False, default='async', choices=['sync', 'async'],
                      help="Optional. Enable using sync/async API. Default value is async.")
    args.add_argument('-niter', '--number_iterations', type=int, required=False, default=None,
                      help="Optional. Number of iterations. "
                           "If not specified, the number of iterations is calculated depending on a device.")
    args.add_argument('-nstreams', '--number_streams', type=str, required=False, default=None,
                      help="Optional. Number of streams to use for inference on the CPU/GPU in throughput mode "
                           "(for HETERO and MULTI device cases use format <device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>).")
    args.add_argument('-t', '--prob_threshold', help="Optional. Probability threshold for detections filtering", default=0.3, type=float)
    args.add_argument('-iout', '--iou_threshold', help="Optional. Intersection over union threshold for overlapping "
                      "detections filtering", default=0.3, type=float)
    args.add_argument('-nireq', '--number_infer_requests', type=int, required=False, default=None,
                      help="Optional. Number of infer requests. Default value is determined automatically for device.")
    args.add_argument("-pc", "--perf_counts", type=str2bool, required=False, default=False, nargs='?', const=True,
                      help="Optional. Report performance counters.", )

    return parser

def get_class(classes):
    classes_path = os.path.expanduser(classes)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors):
    anchors_path = os.path.expanduser(anchors)
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    # Plugin initialization for specified device and load extensions library if specified
    log.info("Initializing plugin for {} device...".format(args.device))
    device_nstreams = parseValuePerDevice(args.device, args.number_streams)
    ie = IECore()
    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, "CPU")
    
    # Read IR
    log.info("Reading IR...")
    net = IENetwork(model=model_xml, weights=model_bin)

    if "CPU" in args.device:
        ie.set_config({'CPU_THROUGHPUT_STREAMS': str(device_nstreams.get(args.device))
                                                         if args.device in device_nstreams.keys()
                                                         else 'CPU_THROUGHPUT_AUTO' }, args.device)
        device_nstreams[args.device] = int(ie.get_config(args.device, 'CPU_THROUGHPUT_STREAMS'))
        ie.add_extension(args.cpu_extension, "CPU")
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(args.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)
    elif "MYRIAD" in args.device:
        ie.set_config({'LOG_LEVEL': 'LOG_INFO',
                           'VPU_LOG_LEVEL': 'LOG_WARNING'}, MYRIAD_DEVICE_NAME)
    
    input_blob = next(iter(net.inputs))
    netoutput = iter(net.outputs)
    out_blob1 = next(netoutput)
    print("output1:",out_blob1)
    print("shape:",net.outputs[out_blob1].shape)
    out_blob2 = next(netoutput)
    print("output2:",out_blob2)
    print("shape:",net.outputs[out_blob2].shape)
    out_blob3 = next(netoutput)
    print("output3:",out_blob3)
    print("shape:",net.outputs[out_blob3].shape)

    log.info("Loading IR to the plugin...")
    config = { 'PERF_COUNT' : ('YES' if args.perf_counts else 'NO')}
    
   
    exec_net = ie.load_network(network=net, device_name=args.device)
    # Read and pre-process input image
    n, c, h, w = net.inputs[input_blob].shape
    
    image = Image.open(args.input[0])
    ori_width,ori_height=image.size
    print("image ori shape:{},{}".format(ori_width,ori_height))
    boxed_image = letterbox_image(image, tuple(reversed((416,416))))
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0) 
    image_data = image_data.transpose((0,3, 1, 2))

    print("image shape:{}".format(image_data.shape))

    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        labels_map = None
    
    inf_start = time.time()
    
    output = exec_net.infer({input_blob: image_data})


    ## wait the latest inference executions
    inf_end = time.time()
    det_time = inf_end - inf_start
    print("[Performance] inference time:{} ms".format(det_time*1000))
    #post-processing part
    objects = list()
    for layer_name, out_blob in output.items():
        out_blob = out_blob.reshape(net.layers[net.layers[layer_name].parents[0]].shape)
        layer_params = YoloParams(net.layers[layer_name].params, out_blob.shape[2])
        log.info("Layer {} parameters: ".format(layer_name))
        layer_params.log_params()
        objects += parse_yolo_region(out_blob, image_data.shape[2:],
                                    (416,416), layer_params, args.prob_threshold)

    objects = sorted(objects, key=lambda obj : obj['confidence'], reverse=True)
    for i in range(len(objects)):
        if objects[i]['confidence'] == 0:
            continue
        for j in range(i + 1, len(objects)):
            if intersection_over_union(objects[i], objects[j]) > args.iou_threshold:
                objects[j]['confidence'] = 0
        
        # Drawing objects with respect to the --prob_threshold CLI parameter
    objects = [obj for obj in objects if obj['confidence'] >= args.prob_threshold]
    log.info("\nDetected boxes for batch {}:".format(1))
    log.info(" Class ID | Confidence | XMIN | YMIN | XMAX | YMAX | COLOR ")

    origin_im_size = (416,416)
    for obj in objects:
        draw = ImageDraw.Draw(boxed_image)
        # Validation bbox of detected object
        if obj['xmax'] > origin_im_size[1] or obj['ymax'] > origin_im_size[0] or obj['xmin'] < 0 or obj['ymin'] < 0:
            continue
        color = (int(min(obj['class_id'] * 12.5, 255)),
                min(obj['class_id'] * 7, 255), min(obj['class_id'] * 5, 255))
        det_label = labels_map[obj['class_id']] if labels_map and len(labels_map) >= obj['class_id'] else \
            str(obj['class_id'])
        draw.rectangle(
                [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']],
                outline=color)
        
        del draw
        log.info(
            "{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} | {} ".format(det_label, obj['confidence'], obj['xmin'],
                                                                  obj['ymin'], obj['xmax'], obj['ymax'],
                                                                  color))

    boxed_image.show()
    

    if args.perf_counts:
        for ni in range(int(args.number_infer_requests)):
            perf_counts = exe_network.requests[ni].get_perf_counts()
            logger.info("Pefrormance counts for {}-th infer request".format(ni))
            for layer, stats in perf_counts.items():
                max_layer_name = 30
                print("{:<30}{:<15}{:<30}{:<20}{:<20}{:<20}".format(layer[:max_layer_name - 4] + '...' if (len(layer) >= max_layer_name) else layer,
                                                                        stats['status'],
                                                                        'layerType: ' + str(stats['layer_type']),
                                                                        'realTime: ' + str(stats['real_time']),
                                                                        'cpu: ' + str(stats['cpu_time']),
                                                                        'execType: ' + str(stats['exec_type'])))
        
            



if __name__ == '__main__':
    sys.exit(main() or 0)
