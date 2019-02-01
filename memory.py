import ctypes

class iovec(ctypes.Structure):
    _fields_ = [("iov_base", ctypes.c_void_p), ("iov_len", ctypes.c_size_t)]

libc = ctypes.CDLL('libc.so.6')
readmem = libc.process_vm_readv
readmem.argtypes = [ctypes.c_int, ctypes.POINTER(iovec), ctypes.c_ulong, ctypes.POINTER(iovec), ctypes.c_ulong, ctypes.c_ulong]

room = ctypes.c_long()
state = ctypes.c_long()

read_room = iovec(0x75B268, 4)
read_state = iovec(0x720fC0, 4)
read_arr = (iovec * 2)(read_room, read_state)

write_room = iovec(ctypes.cast(ctypes.byref(room), ctypes.c_void_p), 4)
write_state = iovec(ctypes.cast(ctypes.byref(state), ctypes.c_void_p), 4)
write_arr = (iovec * 2)(write_room, write_state)

pid = 0

def init(_pid):
    global pid
    pid = _pid

def read():
    readmem(pid, write_arr, 2, read_arr, 2, 0)
    return room.value, state.value
