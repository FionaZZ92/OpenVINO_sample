# OpenVINO Common used Model Conversion & Inference 

This repo contains the model conversion and inference steps/samples with Intel® Distribution of OpenVINO™ Toolkit (or [Intel® OpenVINO™](https://01.org/openvinotoolkit)), and those TF/Caffe/MxNet/ONNX models are widely used for Classification, Object Detection, Semantics Segmentation, Speech Recognition, Optical Character Recognition, etc.

You can download [Intel® Distribution of OpenVINO™ Toolkit](https://software.intel.com/en-us/openvino-toolkit) from Intel offical website, or build source code of [Intel® OpenVINO™](https://github.com/opencv/dldt).

To quick ramp-up OpenVINO (including release note, what's new, HW/SW/OS requirement and demo usage), please follow online document: https://docs.openvinotoolkit.org/
To check all OpenVINO 2018 R5 release supported frameworks and supported layers of each framework by following link:
https://docs.openvinotoolkit.org/R5/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html


# TensorFlow Model - DeepLabV3+

The DeeplabV3+ model can be refer to https://github.com/tensorflow/models/tree/master/research/deeplab which supports encoder-decoder structure contains atrous spatial pyramid pooling(ASPP) module and Xception Convolution structure. To optimize the inference work with Intel OpenVINO, please use the script to convert TF model with Model Optimizer and use attached python program to do inference.

The Intel OpenVINO probably will do support conversion with the whole model, currently, use model cutting feature to cut pre-processing part of this model. The main workload with MobilenetV2 will be kept for inference. Other operations still implemented by TensorFlow.

# ONNX Model - CRNN
This CRNN model can be refer to https://github.com/meijieru/crnn.pytorch, the origin pre-trained model is pytorch (crnn.pth). Current OpenVINO 2018 R5 version still not directly support pyTorch framework, my solution is to convert pyTorch model to ONNX model, then use OpenVINO Model Optimizer to finish IR files generation. 

In this case, I provide sample code for pyTorch2onnx conversion. User can test to download the pytorch model, copy CRNN/pytorch2onnx/convert_pytorch2onnx.py to ./crnn.pytorch. If succefully, you will get crnn.onnx model under the same path. Use the crnn.onnx model as input model for OpenVINO Model Optimizer, set options and flags as the command line in script.sh.

Please pay attension, this demo actually use grayscale image with the size of 100x37 (width x height). Thus, during the model optimizer conversion, please set input shape as [1,1,37,100]. The output layer should be with the name of "234", thus during the inference, please get the result of "234" layer.
