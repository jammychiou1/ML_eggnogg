#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <cstring>
#include <cstdio>
using namespace std;

Window get_handle(Display *dpy) {
    Atom a = XInternAtom(dpy, "_NET_CLIENT_LIST", true);
    Atom actualType;
    int format;
    unsigned long numItems, bytesAfter;
    unsigned char *data = NULL;
    int status = XGetWindowProperty(dpy, XDefaultRootWindow(dpy), a, 0, 1024, false,
                                    AnyPropertyType, &actualType, &format, &numItems,
                                    &bytesAfter, &data);

    if (status == Success) {
        long *array = (long*) data;
        for (unsigned long k = 0; k < numItems; k++) {
            Window w = (Window) array[k];
            char *name = NULL;
            status = XFetchName(dpy, w, &name);
            if (status) {
                if (strcmp(name, "eggnoggplus") == 0) {
                    return w;
                }
            }
            XFree(name);
        }
        XFree(data);
    }
    //puts("not found");
    return 0;
}

extern "C" int init(Display** dpy, Window* window) {
    puts("_xlib_helper initializing");
    *dpy = XOpenDisplay(NULL);
    if (*dpy == NULL) {
        return -1;
    }
    *window = get_handle(*dpy);
    if (*window == 0) {
        return -1;
    }
    //printf("c++: %p %d\n", (void*)*dpy, (int)*window);
    XSetInputFocus(*dpy, *window, RevertToNone, CurrentTime);
    XRaiseWindow(*dpy, *window);
    XSync(*dpy, 1);
    return 0;
}

extern "C" void press(Display* dpy, Window window, char* key) {
    XKeyEvent e;
    e.type = KeyPress;
    e.keycode = XKeysymToKeycode(dpy, XStringToKeysym(key));
    e.state = 0;
    e.window = window;
    XSendEvent(dpy, window, true, KeyPressMask, (XEvent*)&e);
    XFlush(dpy);
}

extern "C" void release(Display* dpy, Window window, char* key) {
    XKeyEvent e;
    e.type = KeyRelease;
    e.keycode = XKeysymToKeycode(dpy, XStringToKeysym(key));
    e.state = 0;
    e.window = window;
    XSendEvent(dpy, window, true, KeyReleaseMask, (XEvent*)&e);
    XFlush(dpy);
}

extern "C" void get_screen(Display* dpy, Window window, int w, int h, unsigned char* data) {
    XImage *image = XGetImage(dpy, window, 0, 0, w, h, AllPlanes, ZPixmap);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            unsigned long pixel = XGetPixel(image,x,y);
            //memcpy(data + (x + y*w) * 3 + 0, &pixel + 2, 1); //R
            //memcpy(data + (x + y*w) * 3 + 1, &pixel + 1, 1); //G
            //memcpy(data + (x + y*w) * 3 + 2, &pixel + 0, 1); //B
            memcpy(data + (x + y*w) * 3, &pixel, 3); 
        }
    }
    XDestroyImage(image);
}

extern "C" void get_box(Display* dpy, Window window, int* left, int* up, int* right, int* down) {
    XTranslateCoordinates(dpy, window, XDefaultRootWindow(dpy), 0, 0, left, up, NULL);
    XTranslateCoordinates(dpy, window, XDefaultRootWindow(dpy), 960, 640, left, up, NULL);
}
