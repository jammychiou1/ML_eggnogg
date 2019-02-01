from ctypes import *
import time

Display_p = c_void_p
Window = c_long

so = CDLL('./_xlib_helper.so')

so_init = so.init
so_init.argtypes = [POINTER(Display_p), POINTER(Window)]

so_press = so.press
so_press.argtypes = [Display_p, Window, c_char_p]

so_release = so.release
so_release.argtypes = [Display_p, Window, c_char_p]

so_get_screen = so.get_screen
so_get_screen.argtypes = [Display_p, Window, c_int, c_int, POINTER(c_ubyte)]

so_get_box = so.get_box
so_get_box.argtypes = [Display_p, Window, POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int)]

dpy = Display_p()
window = Window()

def init():
    #print('initing')
    so_init(byref(dpy), byref(window))
    #print('python:', hex(dpy.value), window.value)
    if window == 0:
        print('error: cannot find window')
    #print('finish init')

W = 0x11
A = 0x1E
S = 0x1F
D = 0x20
V = 0x2F
B = 0x30
