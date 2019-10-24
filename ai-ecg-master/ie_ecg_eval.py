import numpy as np
import keras
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import util
import load
import logging as log
import time
from datetime import datetime
import scipy.stats as sst
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import ecg_menu
import threading
from cpuinfo import get_cpu_info
from matplotlib.widgets import Button
from openvino.inference_engine import IENetwork, IECore

ecg_height = 8960

class InferReqWrap:
    def __init__(self, request, id, callbackQueue):
        self.id = id
        self.request = request
        self.request.set_completion_callback(self.callback, self.id)
        self.callbackQueue = callbackQueue

    def callback(self, statusCode, userdata):
        if (userdata != self.id):
            print("Request ID {} does not correspond to user data {}".format(self.id, userdata))
        elif statusCode != 0:
            print("Request {} failed with status code {}".format(self.id, statusCode))
        self.callbackQueue(self.id, self.request.latency)

    def startAsync(self, input_data):
        self.request.async_infer(input_data)

    def infer(self, input_data):
        self.request.infer(input_data)
        self.callbackQueue(self.id, self.request.latency);

class InferRequestsQueue:
    def __init__(self, requests):
      self.idleIds = []
      self.requests = []
      self.times = []
      for id in range(0, len(requests)):
          self.requests.append(InferReqWrap(requests[id], id, self.putIdleRequest))
          self.idleIds.append(id)
      self.startTime = datetime.max
      self.endTime = datetime.min
      self.cv = threading.Condition()

    def resetTimes(self):
      self.times.clear()

    def getDurationInSeconds(self):
      return (self.endTime - self.startTime).total_seconds()

    def putIdleRequest(self, id, latency):
      self.cv.acquire()
      self.times.append(latency)
      self.idleIds.append(id)
      self.endTime = max(self.endTime, datetime.now())
      self.cv.notify()
      self.cv.release()

    def getIdleRequest(self):
        self.cv.acquire()
        while len(self.idleIds) == 0:
            self.cv.wait()
        id = self.idleIds.pop();
        self.startTime = min(datetime.now(), self.startTime);
        self.cv.release()
        return self.requests[id]

    def waitAll(self):
        self.cv.acquire()
        while len(self.idleIds) != len(self.requests):
            self.cv.wait()
        self.cv.release()

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

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with the "
                           "kernels implementations.", type=str, default=None)
    args.add_argument("-pp", "--plugin_dir", help="Optional. Path to a plugin folder", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is "
                           "acceptable. The demo will look for a suitable plugin for device specified. "
                           "Default value is CPU", default="CPU", type=str)
    args.add_argument("--labels", help="Optional. Path to labels mapping file", default=None, type=str)
    args.add_argument("-pt", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
                      default=0.5, type=float)
    args.add_argument("-pc", "--perf_counts", type=str2bool, required=False, default=False, nargs='?', const=True,
                      help="Optional. Report performance counters.", )

    return parser


def on_select(item):
    ax1 = fig.add_subplot(gs[2, :])
    ax2 = fig.add_subplot(gs[1, 3])
    image = plt.imread("openvino-logo.png")
    ax2.axis('off')
    ax2.imshow(image)
    if 'clear' in (item.labelstr):
        ax1.cla()

    else:
        log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
        args = build_argparser().parse_args()
        #read input data
        if 'Async' in (item.labelstr):
            ecg_data = load.load_ecg("A00001.mat")
        else:
            ecg_data = load.load_ecg(item.labelstr)
        preproc = util.load(".")
        input_ecg = preproc.process_x([ecg_data])
        ecg_n, ecg_h, ecg_w = input_ecg.shape
        log.info("Input ecg file shape: {}".format(input_ecg.shape))

        input_ecg_plot = np.squeeze(input_ecg)
        
        # raw signal plot
        Fs = 1000
        N = len(input_ecg_plot)
        T = (N-1)/Fs
        ts = np.linspace(0, T, N, endpoint=False)
        ax1.plot(ts, input_ecg_plot, label=item.labelstr, lw=2)
        ax1.set_ylabel('Amplitude')
        ax1.set_title("ECG Raw signal: length - {}, Freq - 1000 Hz".format(ecg_h))
        ax1.legend(loc='upper right')

        #choose proper IRs
        if (input_ecg.shape[1]==8960):
            model_xml = "tf_model_8960_fp16.xml"
            model_bin = os.path.splitext(model_xml)[0] + ".bin"
        elif (input_ecg.shape[1]==17920):
            model_xml = "tf_model_17920_fp16.xml"
            model_bin = os.path.splitext(model_xml)[0] + ".bin"
        # Plugin initialization for specified device and load extensions library if specified
        log.info("OpenVINO Initializing plugin for {} device...".format(args.device))
        ie = IECore()

        # Read IR
        log.info("OpenVINO Reading IR...")

        net = IENetwork(model=model_xml, weights=model_bin)
        assert len(net.inputs.keys()) == 1, "Demo supports only single input topologies"

        if args.cpu_extension and 'CPU' in args.device:
            ie.add_extension(args.cpu_extension, "CPU")
        config = { 'PERF_COUNT' : ('YES' if args.perf_counts else 'NO')}
        device_nstreams = parseValuePerDevice(args.device, None)
        if ('Async' in (item.labelstr)) and ('CPU' in (args.device)):
            ie.set_config({'CPU_THROUGHPUT_STREAMS': str(device_nstreams.get(args.device))
                                                         if args.device in device_nstreams.keys()
                                                         else 'CPU_THROUGHPUT_AUTO' }, args.device)
            device_nstreams[args.device] = int(ie.get_config(args.device, 'CPU_THROUGHPUT_STREAMS'))
   
        #prepare input blob
        input_blob = next(iter(net.inputs))
        #load IR to plugin
        log.info("Loading network with plugin...")
      
        n, h, w = net.inputs[input_blob].shape
        log.info("Network input shape: {}".format(net.inputs[input_blob].shape))
        if 'Async' in (item.labelstr):
            exec_net = ie.load_network(net,
                                      args.device,
                                      config=config,
                                      num_requests=12)
            infer_requests = exec_net.requests
            request_queue = InferRequestsQueue(infer_requests)
        else:
            exec_net = ie.load_network(net,args.device)
        output_blob = next(iter(net.outputs))
        del net
        
        #Do infer 
        inf_start = time.time()

        if 'Async' in (item.labelstr):
            for i in range(12):
                infer_request = request_queue.getIdleRequest()
                if not infer_request:
                    raise Exception("No idle Infer Requests!")
                infer_request.startAsync({input_blob: input_ecg})
            request_queue.waitAll()
        else:

            res = exec_net.infer({input_blob: input_ecg})
    
        inf_end = time.time()
        
        if 'Async' in (item.labelstr):
            det_time = (inf_end - inf_start)/12
            res = exec_net.requests[0].outputs[output_blob]
        else:
            det_time = inf_end - inf_start
            res = res[output_blob]

        del exec_net
        print("[Performance] each inference time:{} ms".format(det_time*1000))
        prediction = sst.mode(np.argmax(res, axis=2).squeeze())[0][0]
        result = preproc.int_to_class[prediction]

        ax1.set_xlabel('File: {}, Intel OpenVINO Infer_perf for each input: {}ms, classification_result: {}'.format(item.labelstr, det_time*1000, result), fontsize=15, color="c", fontweight='bold')
        ax1.grid()
        

    
    

if __name__ == '__main__':
    

    fig = plt.figure(figsize=(15,12))
    fig.suptitle('Select ECG file of The Physionet 2017 Challenge from below list:', color="#009999", fontsize=18, fontweight='bold')
    widths = [1, 1, 1, 1]
    heights = [1, 1, 8, 7]
    gs = gridspec.GridSpec(ncols=4, nrows=4, width_ratios=widths, height_ratios=heights, figure=fig)
    ax = plt.gca()
    #Menu
    props = ecg_menu.ItemProperties(labelcolor='black', bgcolor='#00cc66', fontsize=15, alpha=0.2)
    hoverprops = ecg_menu.ItemProperties(labelcolor='white', bgcolor='#4c0099',
                            fontsize=15, alpha=0.2)
    menuitems = []

    for label in ('A00001.mat','A00005.mat','A00008.mat','A00022.mat','A00125.mat', 'Async 12 inputs', 'clear'):
        item = ecg_menu.MenuItem(fig, label, props=props, hoverprops=hoverprops, on_select=on_select)
        menuitems.append(item)
    menu = ecg_menu.Menu(fig, menuitems, 50, 1100)

 
    #The Physionet 2017 Challenge

    
    info = get_cpu_info()
    t = "CPU info: " +info['brand'] + ", num of core(s): " +str(info['count'])

    t1 = ("In this Challenge, we treat all non-AF abnormal rhythms as a single "
          "class and require the Challenge entrant to classify the rhythms as:"
         )
    t2 = ("1) N - Normal sinus rhythm")
    t3 = ("2) A - Atrial Fibrillation (AF)")
    t4 = ("3) O - Other rhythm")
    t5 = ("4) ~ - Too noisy to classify")
    t6 = ("*Algo refer to: Stanford Machine Learning Group ECG classification DNN model")
    t7 = ("https://stanfordmlgroup.github.io/projects/ecg2/")
    t8 = ("Demo created by: Zhao, Zhen (Fiona), VMC, IoTG, Intel")
    ax.text(.5, .25, t, fontsize=16, style='oblique', ha='center', va='top', wrap=True, color="#0066cc")
    ax.text(.5, .2, t1, fontsize=16, style='oblique', ha='center', va='top', wrap=True)
    ax.text(.0, .14, t2, fontsize=16, style='oblique', ha='left', va='top', wrap=True)
    ax.text(.0, .11, t3, fontsize=16, style='oblique', ha='left', va='top', wrap=True, color="#cc0066")
    ax.text(.0, .08, t4, fontsize=16, style='oblique', ha='left', va='top', wrap=True, color="#6600cc")
    ax.text(.0, .05, t5, fontsize=16, style='oblique', ha='left', va='top', wrap=True)
    ax.text(1,  .0, t6, fontsize=10, style='oblique', ha='right', va='top', wrap=True)
    ax.text(1,  -.03, t7, fontsize=10, style='oblique', ha='right', va='top', wrap=True, color='b')
    ax.text(1, -.08, t8, fontsize=12, style='oblique', ha='right', va='top', wrap=True, color='c', fontweight='bold')
    plt.axis('off')
    
    plt.show()
    #out = ecg.ecg(signal=input_ecg_plot, sampling_rate=1000., show=True)       





