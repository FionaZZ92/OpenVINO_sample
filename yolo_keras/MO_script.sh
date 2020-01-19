#!/bin/bash
python /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model ./tf_yolov3_fullx.pb --input input_1,Placeholder_366 --input_shape [1,416,416,3],[1] --freeze_placeholder_with_value "Placeholder_366->0" --tensorflow_use_custom_operations_config ./yolov3_keras.json
