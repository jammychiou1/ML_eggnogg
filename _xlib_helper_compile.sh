#!/bin/sh
g++ _xlib_helper.cpp -shared -fPIC -lX11 -o _xlib_helper.so
