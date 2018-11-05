# OpenVINO Common used Model Conversion & Inference 

This repo contains the model conversion and inference steps/samples with Intel® Distribution of OpenVINO™ Toolkit (or Intel® OpenVINO™), and those TF/Caffe/MxNet models are widely used for classification, object detection and semantics segmentation.

The DeeplabV3+ model can be refer to https://github.com/tensorflow/models/tree/master/research/deeplab which supports encoder-decoder structure contains atrous spatial pyramid pooling(ASPP) module and Xception Convolution structure. To optimize the inference work with Intel OepnVINO, please use the script to convert TF model with Model Optimizer and use attached python program to do inference.

The Intel OpenVINO probably will do support conversion with the whole model, use model cutting feature to cut pre-processing part of this model. The main workload with MobilenetV2 will be kept for inference. Other operations still implemented by TensorFlow.
