import xlib_helper
import time
from ctypes import *

def press(keyname):
    xlib_helper.so_press(xlib_helper.dpy, xlib_helper.window, c_char_p(keyname.encode('utf-8')))
    
def release(keyname):
    xlib_helper.so_release(xlib_helper.dpy, xlib_helper.window, c_char_p(keyname.encode('utf-8')))
    
def tap(keyname):
    print('tapping ' + keyname)
    press(keyname)
    time.sleep(0.03)
    release(keyname)
    time.sleep(0.03)
