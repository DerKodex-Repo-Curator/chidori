---
title:  'Visual Studio Code'
sidebar_position: 2
---

# Visual Studio Code

Run Visual Studio Code with Juice on Windows:

~~~cmd
juice run "C:\Program Files\Microsoft VS Code\Code.exe"
~~~

Run Visual Studio Code with Juice on Linux:

~~~sh
juice run /usr/local/bin/code
~~~

## Active Paths

Activate `C:\Program Files\Microsoft VS Code` to run Visual Studio Code through Juice automatically when a Juice session is created:

~~~
juice session add-path "C:\Program Files\Microsoft VS Code"
~~~

Use Visual Studio Code as you normally would.  The integrated terminal and extensions utilize the remote Juice GPU as needed.

Make sure to close all instances of VSCode before you connect to a remote GPU.
