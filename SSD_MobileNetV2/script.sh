#!/bin/bash

R='\033[0;31m'
G='\033[0;32m'
LB='\033[1;34m'
NC='\033[0m' # No Color

echo -e "${G}----Enter into virtual python env for OpenVINO----${NC}\n"
echo -e "${LB}[command] source ~/openvino_lab/vt_py3/bin/activate${NC}\n"
source ~/openvino_lab/vt_py3/bin/activate

#setup OPenVINO environment path
echo -e "\n\n${G}----setup OpenVINO environement path${NC}\n"
echo -e "${LB}[command] source /opt/intel/computer_vision_sdk_2018.3.343/bin/setupvars.sh${NC}\n"
source /opt/intel/computer_vision_sdk_2018.3.343/bin/setupvars.sh

#run command to convert model to IRs
echo -e "\n\n${G}----Model Optimizer----${NC}\n"
echo -e "${LB}[ info  ] TensorFlow SSD MobileNetV2 model${NC}\n"
echo -e "${LB}[command] python3 /opt/intel/computer_vision_sdk_2018.3.343/deployment_tools/model_optimizer/mo_tf.py --input_model ~/openvino_lab/openvino_lab_demo/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb --input_shape [1,300,300,3] --tensorflow_use_custom_operations_config /opt/intel/computer_vision_sdk_2018.3.343/deployment_tools/model_optimizer/extensions/front/tf/ssd_support.json --tensorflow_object_detection_api_pipeline_config ~/openvino_lab/openvino_lab_demo/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config --reverse_input_channels --output_dir ~/openvino_lab/openvino_lab_demo/ssd_mobilenet_v2_coco_2018_03_29/${NC}\n"


python3 /opt/intel/computer_vision_sdk_2018.3.343/deployment_tools/model_optimizer/mo_tf.py --input_model ~/openvino_lab/openvino_lab_demo/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb --input_shape [1,300,300,3] --tensorflow_use_custom_operations_config /opt/intel/computer_vision_sdk_2018.3.343/deployment_tools/model_optimizer/extensions/front/tf/ssd_support.json --tensorflow_object_detection_api_pipeline_config ~/openvino_lab/openvino_lab_demo/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config --reverse_input_channels --output_dir ~/openvino_lab/openvino_lab_demo/ssd_mobilenet_v2_coco_2018_03_29/

#run command to infer model:
echo -e "\n\n${G}----Inference Engine----${NC}\n"
echo -e "${LB}[info] Run python_ssd_v2_demo.py${NC}\n"
echo -e "${LB}/opt/intel/computer_vision_sdk_2018.3.343/deployment_tools/inference_engine/samples/build/intel64/Release/./object_detection_demo_ssd_async -m ~/openvino_lab/openvino_lab_demo/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml -i ~/openvino_lab/openvino_lab_demo/video/Pedestrain_Detect_2_1_1.mp4 -d CPU -l /opt/intel/computer_vision_sdk_2018.3.343/deployment_tools/inference_engine/samples/build/intel64/Release/lib/libcpu_extension.so -t 0.6${NC}\n"


/opt/intel/computer_vision_sdk_2018.3.343/deployment_tools/inference_engine/samples/build/intel64/Release/./object_detection_demo_ssd_async -m ~/openvino_lab/openvino_lab_demo/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml -i ~/openvino_lab/openvino_lab_demo/video/Pedestrain_Detect_2_1_1.mp4 -d CPU -l /opt/intel/computer_vision_sdk_2018.3.343/deployment_tools/inference_engine/samples/build/intel64/Release/lib/libcpu_extension.so -t 0.6

#/opt/intel/computer_vision_sdk_2018.3.343/deployment_tools/inference_engine/samples/build/intel64/Release/./object_detection_demo_ssd_async -m ~/openvino_lab/openvino_lab_demo/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml -i ~/openvino_lab/openvino_lab_demo/video/Pedestrain_Detect_2_1_1.mp4 -d GPU -t 0.6
