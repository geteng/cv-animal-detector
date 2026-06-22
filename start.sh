#!/bin/bash
export PATH=$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
cd /home/ubuntu
exec python3 main.py
