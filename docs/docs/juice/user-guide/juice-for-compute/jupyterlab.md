---
title:  'Jupyterlab'
sidebar_position: 2
---

# Jupyterlab

Use these commands to use Juice with JupyterLab on Windows. 

## Install JupyterLab

~~~sh
pip install jupyterlab 
~~~

Visit the [Jupyter Labs documentation](https://docs.jupyter.org/en/latest/) for more information.

## Run JupyterLab with Juice

~~~~sh
juice run jupyter lab 
~~~

## Test Juice JupyterLab Connection

Run this command in a new notebook:

~~~
!nvidia-smi 
~~~

If successful you should see an output similar to the one below:

~~~sh
Wed Jul 17 11:56:15 2024      
+-----------------------------------------------------------------------------------------+ 

| NVIDIA-SMI 555.85                 Driver Version: 556.12         CUDA Version: 12.5     | 

|-----------------------------------------+------------------------+----------------------+ 

| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC | 

| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. | 

|                                         |                        |               MIG M. | 

|=========================================+========================+======================| 

|   0  NVIDIA GeForce RTX 4090      ERR!  |   00000000:01:00.0 Off |                  Off | 

|  0%   45C    P0             17W /  450W |       0MiB /  24564MiB |      0%      Default | 

|                                         |                        |                  N/A | 

+-----------------------------------------+------------------------+----------------------+ 

                                                                                        

+-----------------------------------------------------------------------------------------+ 

| Processes:                                                                              | 

|  GPU   GI   CI        PID   Type   Process name                              GPU Memory | 

|        ID   ID                                                               Usage      | 

|=========================================================================================| 

|  No running processes found                                                             | 

+-----------------------------------------------------------------------------------------+ 
~~~
