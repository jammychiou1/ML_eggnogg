import xlib_helper
import time
import numpy as np
from ctypes import *

def act_to_key(act):
    tmp = np.unravel_index(act, (3, 3, 2, 2))
    return np.array([tmp[0] & 1, (tmp[0] & 2) >> 1, tmp[1] & 1, (tmp[1] & 2) >> 1, tmp[2], tmp[3]])

def press(keyname):
    xlib_helper.so_press(xlib_helper.dpy, xlib_helper.window, c_char_p(keyname.encode('utf-8')))
    
def release(keyname):
    xlib_helper.so_release(xlib_helper.dpy, xlib_helper.window, c_char_p(keyname.encode('utf-8')))
    
def tap(keyname):
    print('tapping ' + keyname)
    press(keyname)
    time.sleep(0.06)
    release(keyname)
    time.sleep(0.06)
