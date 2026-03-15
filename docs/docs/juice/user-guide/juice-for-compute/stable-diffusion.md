---
title:  'Stable Diffusion'
sidebar_position: 2
---

# Stable Diffusion

Set up [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui):

1. Install Python 3.10.6 `sudo apt install python3.10 python3.10-venv`
2. Install Git `sudo apt install git`
3. Clone `https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`
4. Download [sd-v1-4.ckpt](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt) to `stable-diffusion-webui/models/Stable-diffusion`
5. Download [GFPGANv1.4.pth](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth) to `stable-diffusion-webui`

Run Stable Diffusion with Juice on Windows from a command prompt in the directory you cloned Stable Diffusion to:

~~~
juice run webui.bat
~~~

Run Stable Diffusion with Juice on Linux from a shell in the directory you cloned Stable Diffusion to:

~~~
juice run webui.sh
~~~

Browse to [http://127.0.0.1:7860](http://127.0.0.1:7860) to view the user interface and generate images.

## Active Paths

Activate the directory that you cloned Stable Diffusion to automatically run Stable Diffusion when a Juice session is current.

~~~
juice session add-path <stable-diffusion-directory>
~~~
