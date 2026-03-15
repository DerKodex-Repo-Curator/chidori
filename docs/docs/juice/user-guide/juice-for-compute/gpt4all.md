---
title:  'GPT4all'
sidebar_position: 2
---

# gpt4all

Download and install [gpt4all](https://www.nomic.ai/gpt4all) on Windows.

Run gpt4all with Juice on Windows:

~~~cmd
juice run "%USERPROFILE%\gpt4all\bin\chat.exe"
~~~

Replace `%USERPROFILE%` with the path to your user directory, e.g. `C:\Windows\joebloggs`, if you prefer.

## Active Paths

Activate `%USERPROFILE%\gpt4all` to run gpt4all through Juice automatically when a Juice session is created:

~~~cmd
juice session add-path %USERPROFILE%\gpt4all
~~~

Replace `%USERPROFILE%` with the path to your user directory, e.g. `C:\Windows\joebloggs`, if you prefer.
