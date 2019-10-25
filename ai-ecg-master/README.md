# AI ECG with Intel OpenVINO for Atrial Fibrillation detection

Case study use [Stanford ML group public ECG model](https://stanfordmlgroup.github.io/projects/ecg2/) with [The Physionet 2017 Challenge dataset](https://www.physionet.org/content/challenge-2017/1.0.0/) for 1D convolutional deep neural network to detect arrhythmias in arbitrary length ECG time-series.

In this case, takes as input the raw ECG data (sampled at 200 Hz), highly optimized NN inference processing with Intel OpenVINO based on x86 platform. To simply demonstrate the low power patient monitor workflow:

![alt text](workloads.png)

+ Intel(R) Core(TM) i7-8700K CPU @ 3.7GHz
+ Ubuntu 16.04.6
+ gcc 5.4.0

## Installation requirements
+ Intel(R) OpenVINO 2019 R3
+ Numpy==1.14.3
+ scikit-learn
+ scipy==1.1.0
+ tensorflow==1.8.0
+ Keras==2.1.6
+ matplotlib (python-tk)
+ tqdm
+ py-cpuinfo

## Setup OpneVINO env
source /opt/intel/openvino/bin/setupvars.sh

## Start AI-ECG demo
python ie_ecg_eval.py -d CPU -l /opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_avx2.so

## Screenshot
![alt text](ecg3.png)

## Description
Click button with file name (e.g A00001.mat), you will get ECG single raw data drawing on the chart, and inference performance and classification result. To refer to a correct result, please check with following table to see acuuracy:

| File name | training reference | TensorFlow result | OpenVINO IE result | OpenVINO IE perf |
| ------ | ------ | ------ | ------ | ------ |
| A00001 | N | N | N | 22.47ms |
| A00005 | A | A | A | 42.26ms |
| A00008 | O | O | O | 42.91ms |
| A00022 | ~ | A | A | 22.82ms |
| A00125 | ~ | O | O | 22.62ms |

Use Async 12 files which inference 12 times parallel on saperate infer requests emulate ECG with 12 leads parallel computing. The Perf of inference A00001.mat with 12 times parallel:

[Performance] each inference time: 18.48 ms which has been optimized 1.22x compare with single inference with A0001
