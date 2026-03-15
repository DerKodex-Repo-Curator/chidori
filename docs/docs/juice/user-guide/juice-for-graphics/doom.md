---
title:  'Doom'
sidebar_position: 2
---

# Doom

Run the Vulkan version of Doom 2016 with Juice after creating `C:\Program Files (x86)\Steam\steamapps\common\DOOM\steam_appid.txt` with the Steam App ID for your Steam install of Doom 2016:

~~~
juice run "C:\Program Files (x86)\Steam\steamapps\common\DOOM\DOOMx64vk.exe"
~~~

NOTE: Juice specifically supports the Vulkan version of DOOM 2016 so you'll need to make sure that you're running the Vulkan executable `DOOMx64vk.exe` and have configured Doom to use Vulkan.

## Active Paths

Activate Doom through Juice automatically when a Juice session is current:

~~~cmd
juice session add-path "C:\Program Files (x86)\Steam\steamapps\common\DOOM"
~~~
