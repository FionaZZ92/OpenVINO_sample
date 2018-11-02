# please check your installed packages in current python env
from __future__ import print_function
import sys
import os
from argparse import ArgumentParser
import numpy as np
import cv2
import time

import tensorflow as tf
from tensorflow.python.platform import gfile

from openvino.inference_engine import IENetwork, IEPlugin


class _model_preprocess():
    def __init__(self):
        graph = tf.Graph()
        f_handle = gfile.FastGFile("./model/DeeplabV3plus_mobileNetV2.pb", 'rb')
        graph_def = tf.GraphDef.FromString(f_handle.read())
        with graph.as_default():
            tf.import_graph_def(graph_def, name='')
        self. sess = tf.Session(graph=graph)

    def _pre_process(self, image):
        seg_map = self.sess.run('sub_7:0', feed_dict={'ImageTensor:0': [image]})
        return seg_map


class _model_postprocess():
    def __init__(self):
        graph = tf.Graph()
        f_handle = gfile.FastGFile("./model/DeeplabV3plus_mobileNetV2.pb", 'rb')
        graph_def = tf.GraphDef.FromString(f_handle.read())
        with graph.as_default():
            new_input = tf.placeholder(tf.int64, shape=(1, 513, 513), name='new_input')
            tf.import_graph_def(graph_def, input_map={'ArgMax:0': new_input}, name='')
        self.sess = tf.Session(graph=graph)

    def _post_process(self, image_ir, image):
        seg_map = self.sess.run('SemanticPredictions:0', feed_dict={'ImageTensor:0': [image], 'new_input:0': np.int64(image_ir)})
        return seg_map


_pre = _model_preprocess()
_post = _model_postprocess()


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model.", required=True, type=str)
    parser.add_argument("-i", "--input", help="Path to a folder with images or path to an image files", required=True,
                        type=str)
    parser.add_argument("-l", "--cpu_extension",
                        help="MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels "
                             "impl.", type=str, default=None)
    parser.add_argument("-pp", "--plugin_dir", help="Path to a plugin folder", type=str, default=None)
    parser.add_argument("-d", "--device",
                        help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device specified (CPU by default)", default="CPU",
                        type=str)
    parser.add_argument("-nt", "--number_top", help="Number of top results", default=10, type=int)
    parser.add_argument("-pc", "--performance", help="Enables per-layer performance report", action='store_true')

    return parser


def main_IE_infer():
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # Read & reszie  input image
    m_input_size=513
    image = cv2.imread(args.input)
    ratio = 1.0 * m_input_size / max(image.shape[0], image.shape[1]) #513 is the shape of inputfor model
    shrink_size = (int(ratio * image.shape[1]), int(ratio * image.shape[0]))
    image = cv2.resize(image, shrink_size, interpolation=cv2.INTER_CUBIC)

    # Plugin initialization for specified device and load extensions library if specified
    plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)
    if args.cpu_extension and 'CPU' in args.device:
        plugin.add_cpu_extension(args.cpu_extension)
    if args.performance:
        plugin.set_config({"PERF_COUNT": "YES"})
    # Read IR
    net = IENetwork.from_ir(model=model_xml, weights=model_bin)

    input_blob = next(iter(net.inputs))
    exec_net = plugin.load(network=net)
    for itr in range(5):
        now = time.time()
        image_ = _pre._pre_process(image)
        image_ = image_.transpose((0, 3, 1, 2))  # Change data layout from NHWC to NCHW
        # Do inference
        res = exec_net.infer(inputs={input_blob: image_})
        result = _post._post_process(res['ArgMax/Squeeze'], image)[0]
        print('time cost:', time.time() - now)
        result[result > 0] = 255
        cv2.imwrite('./test_img/result_deeplabv3.jpg', result)
    del net
    del exec_net
    del plugin


if __name__ == '__main__':
    sys.exit(main_IE_infer() or 0)
