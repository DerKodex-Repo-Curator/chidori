---
title:   'Running an Application on a Remote GPU'
sidebar_position: 3
---
# Running an Application on a Remote GPU

The most straightforward way to run an application is to use the run command, like this: 
    ```powershell
    ./juice run <application> [<arg>...]
    ```

The run command encapsulates several steps: it requests a session (connecting you to a remote GPU), runs the application using that remote GPU, then releases the session, freeing that remote GPU back into its pool. 

    :::tip
    Try run -h for more advanced run capabilities, including the ability to narrow your remote GPU target(s) by pool(s), session(s), or even specific GPUs.

