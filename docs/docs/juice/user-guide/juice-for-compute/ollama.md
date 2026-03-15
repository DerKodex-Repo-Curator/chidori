---
title:  'Ollama'
sidebar_position: 3
---

# Ollama

Ollama is a client server application where the server uses the GPU and the client request responses to chat messages from the server.  Make sure to run the Ollama server with Juice *not* the Ollama client.  Running the Ollama client through Juice without restarting the Ollama server will use the local GPU or CPU instead of the remote Juice GPU.

Download and install Ollama from the [Download Ollama](https://ollama.com/download) page.

On Windows stop the Ollama server by right clicking on the Ollama icon in the system tray and choosing "Quit Ollama" from the context menu.  On Linux stop the Ollama service, e.g. on Ubuntu using `systemd` run `sudo systemctl stop ollama.server`.  You will need to do stop the Ollama server each time you reboot your machine.

Run Ollama server through Juice:

~~~
juice run ollama serve
~~~

Then run the Ollama client to chat through the server using the Phi-2 model:

~~~
ollama run phi
~~~

See `ollama run --help` for more arguments to running the Ollama client.  Visit [Ollama's model search](https://ollama.com/search) for more available models.

**Windows**: 

- C:\Users\[user]\AppData\Local\Programs\Ollama
- C:\Users\[user]\.ollama (Models)

## Active Paths

Activate `%USERPROFILE%\AppData\Local\Programs\Ollama` to run Ollama through Juice automatically when a Juice session is created:

~~~
juice session add-path %USERPROFILE%\AppData\Local\Programs\Ollama
~~~

Replace `%USERPROFILE%` with the path to your user directory on Windows, e.g. `C:\Windows\joeb`, if you prefer.
