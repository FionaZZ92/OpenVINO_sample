#!/bin/bash

py_find_tf=$(pip list | grep "tensorflow" | wc -l)
py3_find_tf=$(pip3 list | grep "tensorflow" | wc -l)
py_major=3

echo -e "$(tput setaf 2)----Check If you installed TF----\n"
if [ "$py_find_tf" -eq 0 ] && [ "$py3_find_tf" -eq 0 ];then
	echo "$(tput setaf 1)Please install TensorFlow firstly!$(tput sgr 0)\n"
else
	py_major=$(python -V 2>&1 | grep -Po '(?<=Python )(.)')
	if [ "$py_major" -eq 2 ];then
		py_major=""
	fi
	py_ver=$(python -V 2>&1 | grep -Po '[0-9]+(.).')
	
	#setup OPenVINO environment path
	echo -e "\n\n$(tput setaf 2)----setup OpenVINO environement path$(tput sgr 0)\n"
	echo -e "$(tput setaf 6)[command] source /opt/intel/computer_vision_sdk_2018.3.343/bin/setupvars.sh$(tput sgr 0)\n"
	source /opt/intel/computer_vision_sdk_2018.3.343/bin/setupvars.sh
	echo -e "export PYTHONPATH=${INTEL_CVSDK_DIR}/python/python$py_ver:${PYTHONPATH}"
	export PYTHONPATH="${INTEL_CVSDK_DIR}/python/python$py_ver:${INTEL_CVSDK_DIR}/python/python$py_ver/ubuntu16:${PYTHONPATH}"
	
	#run command to convert model to IRs
	echo -e "\n\n$(tput setaf 2)----Model Optimizer----$(tput sgr 0)\n"
	echo -e "$(tput setaf 6)[ info  ] TensorFlow SSD MobileNetV2 model$(tput sgr 0)\n"
	echo -e "$(tput setaf 6)[command] python$py_major ${INTEL_CVSDK_DIR}/deployment_tools/model_optimizer/mo_tf.py --input_model ./model/DeeplabV3plus_mobileNetV2.pb --input 0:MobilenetV2/Conv/Conv2D --output ArgMax --input_shape [1,513,513,3] --output_dir ./model$(tput sgr 0)\n"

	python$py_major ${INTEL_CVSDK_DIR}/deployment_tools/model_optimizer/mo_tf.py --input_model ./model/DeeplabV3plus_mobileNetV2.pb --input 0:MobilenetV2/Conv/Conv2D --output ArgMax --input_shape [1,513,513,3] --output_dir ./model


	#run command to infer model:
	echo -e "\n\n$(tput setaf 2)----Inference Engine----$(tput sgr 0)\n"
	echo -e "$(tput setaf 6)[info] Run python_ssd_v2_demo.py$(tput sgr 0)\n"
	echo -e "$(tput setaf 6)python$py_major infer_IE_TF.py -m ./model/DeeplabV3plus_mobileNetV2.xml -i ./test_img/test.jpg -d CPU -l ${INTEL_CVSDK_DIR}/deployment_tools/inference_engine/samples/intel64/Release/lib/libcpu_extension.so $(tput sgr 0)\n"

	python$py_major infer_IE_TF.py -m ./model/DeeplabV3plus_mobileNetV2.xml -i ./test_img/test.jpg -d CPU -l ${INTEL_CVSDK_DIR}/deployment_tools/inference_engine/samples/intel64/Release/lib/libcpu_extension.so

fi








