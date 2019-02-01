import keyboard
import numpy as np
# N, L, R
# N, D, U

last1 = (0, 0, 0, 0)
last2 = (0, 0, 0, 0)

btn1 = [['A', 'D'], ['S', 'W'], ['V'], ['B']]
btn2 = [['Left', 'Right'], ['Down', 'Up'], ['comma'], ['period']]
nums = [2, 2, 1, 1]

def change(num, btns, a, b):
    for i in range(num):
        bit_a = (a & (1 << i) != 0)
        bit_b = (b & (1 << i) != 0)
        if bit_a and not bit_b:
            #print('release ' + btns[i])
            keyboard.release(btns[i])
        if not bit_a and bit_b:
            #print('press ' + btns[i])
            keyboard.press(btns[i])
        
def update(act1, act2):
    global last1, last2
    new1 = np.unravel_index(act1, (3, 3, 2, 2))
    new2 = np.unravel_index(act2, (3, 3, 2, 2))
    for i in range(4):
        change(nums[i], btn1[i], last1[i], new1[i])
        change(nums[i], btn2[i], last2[i], new2[i])
    last1 = new1
    last2 = new2
