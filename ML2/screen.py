import xlib_helper
from ctypes import *
from PIL import Image
from matplotlib import pyplot as plt
import time

def get_screen():
    w = 960
    h = 640
    size = w * h * 3
    data = (c_ubyte * size)()
    xlib_helper.so_get_screen(xlib_helper.dpy, xlib_helper.window, w, h, data)
    return Image.frombuffer('RGB', (w, h), data, 'raw', 'BGR', 0, 1)
    
def get_box():
    left = c_int()
    up = c_int()
    right = c_int()
    down = c_int()
    xlib_helper.so_get_box(xlib_helper.dpy, xlib_helper.window, byref(left), byref(up), byref(right), byref(down))
    return left.value, up.value, right.value, down.value

def capture():
    screen = get_screen()
    screen.thumbnail((240, 160), Image.NEAREST)
    return screen

if __name__ == '__main__':
    xlib_helper.init()
    #time.sleep(3)
    print('capturing')
    screen = capture()
    plt.imshow(screen)
    plt.show()
    
