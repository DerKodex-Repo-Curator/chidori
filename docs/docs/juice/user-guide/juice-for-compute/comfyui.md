---
title:  'ComfyUI'
sidebar_position: 2
---

# ComfyUI

Download and install [ComfyUI](https://www.comfy.org/) on Windows.

Run ComfyUI with Juice on Windows:

~~~
juice run "%USERPROFILE%\AppData\Local\Programs\@comfyorgcomfyui-electron\ComfyUI.exe" --disable-cuda-malloc
~~~

The `--disable-cuda-malloc` argument is specified for optimal performance with Juice.  ComfyUI will still run through Juice without this argument but responses will take longer to generate.

Replace `%USERPROFILE%` with the path to your user directory on Windows, e.g. `C:\Windows\joeb`, if you prefer.

## Active Paths

Activate `%USERPROFILE%\AppData\Local\Programs\@comfyorgcomfyui-electron` to run ComfyUI through Juice automatically when a Juice session is created:

~~~
juice session add-path %USERPROFILE%\AppData\Local\Programs\@comfyorgcomfyui-electron
~~~

Replace `%USERPROFILE%` with the path to your user directory on Windows, e.g. `C:\Windows\joeb`, if you prefer.
