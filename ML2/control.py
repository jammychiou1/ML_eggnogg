import keyboard
import numpy as np
# N, L, R
# N, D, U

last1 = [0, 0, 0, 0, 0, 0]
last2 = [0, 0, 0, 0, 0, 0]

btn1 = ['A', 'D', 'S', 'W', 'V', 'B']
btn2 = ['Left', 'Right', 'Down', 'Up', 'comma', 'period']

def act_to_key(act):
    tmp = np.unravel_index(act, (3, 3, 2, 2))
    return [tmp[0] & 1, (tmp[0] & 2) >> 1, tmp[1] & 1, (tmp[1] & 2) >> 1, tmp[2], tmp[3]]
    
def update(act1, act2):
    global last1, last2
    new1 = act_to_key(act1)
    new2 = act_to_key(act2)
    for i in range(6):
        if last1[i] and not new1[i]:
            keyboard.release(btn1[i])
        if last2[i] and not new2[i]:
            keyboard.release(btn2[i])
        if not last1[i] and new1[i]:
            keyboard.press(btn1[i])
        if not last2[i] and new2[i]:
            keyboard.press(btn2[i])
    last1 = new1
    last2 = new2
