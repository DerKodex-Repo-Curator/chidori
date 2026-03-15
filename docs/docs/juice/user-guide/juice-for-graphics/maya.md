---
title:  'Maya'
sidebar_position: 2
---

# Maya

Run [Maya](https://www.autodesk.com/products/maya/overview) with Juice:

~~~cmd
juice run "C:\Program Files\Autodesk\Maya2022\bin\maya.exe"
~~~

Replace `Maya2022` with the version of Maya that you have installed.

## Active Paths

Activate Maya through Juice automatically when a Juice session is current:

~~~cmd
juice session add-path "C:\Program Files\Autodesk\Maya2022\bin\maya.exe"
~~~
