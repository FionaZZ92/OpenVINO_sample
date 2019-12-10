#!/bin/bash

py_find_tf=$(pip list | grep "onnx" | wc -l)
py3_find_tf=$(pip3 list | grep "onnx" | wc -l)
py_major=3

echo -e "$(tput setaf 2)----Check If you installed ONNX----\n"
if [ "$py_find_tf" -eq 0 ] && [ "$py3_find_tf" -eq 0 ];then
	echo "$(tput setaf 1)Please install ONNX firstly!$(tput sgr 0)\n"
else
	py_major=$(python -V 2>&1 | grep -Po '(?<=Python )(.)')
	if [ "$py_major" -eq 2 ];then
		py_major=""
	fi
	py_ver=$(python -V 2>&1 | grep -Po '[0-9]+(.).')
	
	#setup OpenVINO environment path
	echo -e "\n\n$(tput setaf 2)----setup OpenVINO environement path$(tput sgr 0)\n"
	echo -e "$(tput setaf 6)[command] source /opt/intel/openvino/bin/setupvars.sh$(tput sgr 0)\n"
	source /opt/intel/openvino/bin/setupvars.sh
	echo -e "export PYTHONPATH=${INTEL_CVSDK_DIR}/python/python$py_ver:${PYTHONPATH}"
	export PYTHONPATH="${INTEL_CVSDK_DIR}/python/python$py_ver:${INTEL_CVSDK_DIR}/python/python$py_ver/ubuntu16:${PYTHONPATH}"
	
	#run command to convert model to IRs
	echo -e "\n\n$(tput setaf 2)----Model Optimizer----$(tput sgr 0)\n"
	echo -e "$(tput setaf 6)[ info  ] ONNX CRNN model$(tput sgr 0)\n"
	echo -e "$(tput setaf 6)[command] python$py_major ${INTEL_CVSDK_DIR}/deployment_tools/model_optimizer/mo_onnx.py --input_model ./model/crnn.onnx --input_shape [1,1,37,100] --output_dir ./model$(tput sgr 0)\n"

	python$py_major ${INTEL_CVSDK_DIR}/deployment_tools/model_optimizer/mo_onnx.py --input_model ./model/crnn.onnx --input_shape [1,1,37,100] --output_dir ./model


	#run command to infer model:
	echo -e "\n\n$(tput setaf 2)----Inference Engine----$(tput sgr 0)\n"
	echo -e "$(tput setaf 6)[info] Run openvino_crnn_demo.py$(tput sgr 0)\n"
	echo -e "$(tput setaf 6)python$py_major openvino_crnn_demo.py -m ./model/crnn.xml -i ./test_img/image_data2.npy -d CPU -l ${INTEL_CVSDK_DIR}/deployment_tools/inference_engine/samples/build/intel64/Release/lib/libcpu_extension.so $(tput sgr 0)\n"

	python$py_major openvino_crnn_demo.py -m ./model/crnn.xml -i ./test_img/image_data2.npy -d CPU -l ${INTEL_CVSDK_DIR}/deployment_tools/inference_engine/samples/build/intel64/Release/lib/libcpu_extension.so

fi








