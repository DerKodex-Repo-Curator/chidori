---
title:  'PyTorch'
sidebar_position: 2
draft: true
---

# PyTorch

Juice can accelerate PyTorch applications with a remote GPU.  For example run the sample image classification applications for [ResNet](https://pytorch.org/hub/pytorch_vision_resnet/) and [YOLOv5](https://docs.ultralytics.com/#what-is-yolov5) with Juice by launching them through the `juice run` command line tool:

## ResNet

There are examples of running ResNet and YOLO with PyTorch in an Anaconda environment included with the Juice client archive.  To run these examples install [Anaconda](https://www.anaconda.com/), re-open your shell after installation, and then run the following from the directory you extracted the Juice client archive to.

Create an Anaconda environment and install the latest PyTorch with CUDA 11.7 support:

~~~
conda create -n juice
conda activate juice
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
~~~

Run the ResNet example with Juice:

~~~
juice run python pytorch/resnet.py
~~~

When you see a result like:

~~~
Juice GPU
Using cache found in C:\Users\charles/.cache\torch\hub\pytorch_vision_v0.9.0
Samoyed 0.7907482981681824
Pomeranian 0.08977659046649933
white wolf 0.0361027754843235
keeshond 0.02681436948478222
Arctic fox 0.022788094356656075
~~~

... you've confirmed that Juice is working for ResNet.

## YOLO

Install extra dependencies required by YOLO in the same Anaconda environment before running the YOLO example:

~~~
pip install -r pytorch/requirements.txt
~~~

Run the YOLO example with Juice from the Anaconda prompt:

~~~
juice run python pytorch/yolo.py
~~~

When you see a result like:

~~~
Juice GPU
Using cache found in C:\Users\charles/.cache\torch\hub\ultralytics_yolov5_master
YOLOv5  2022-7-19 Python-3.10.6 torch-1.11.0 CUDA:0 (Juice GPU, 8192MiB)

Fusing layers...
YOLOv5s summary: 213 layers, 7225885 parameters, 0 gradients
Adding AutoShape...
image 1/1: 720x1280 2 persons, 2 ties
Speed: 1185.4ms pre-process, 54.8ms inference, 184.0ms NMS per image at shape (1, 3, 384, 640)

~~~

... you've confirmed that Juice is working for YOLO.
