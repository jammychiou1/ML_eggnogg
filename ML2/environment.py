import numpy as np
import subprocess
import time
import torch
import os

import keyboard
import screen
import memory
import control
import xlib_helper

step = 0
screens = np.zeros([300, 3, 240, 160], dtype=np.uint8)
controls1 = np.zeros(300, dtype=np.uint8)
controls2 = np.zeros(300, dtype=np.uint8)
rooms = np.zeros(300, dtype=np.uint8)
td_errors = np.zeros(2700, dtype=np.float)

def init():
    game = subprocess.Popen('./eggnoggplus', cwd='..')
    time.sleep(1)
    memory.init(game.pid)
    xlib_helper.init()
    time.sleep(1)

def reset():
    global step, screens, controls1, controls2, rooms
    step = 0
    screens = np.zeros([300, 3, 240, 160], dtype=np.uint8)
    controls1 = np.zeros(300, dtype=np.uint8)
    controls2 = np.zeros(300, dtype=np.uint8)
    rooms = np.zeros(300, dtype=np.uint8)
    control.update(0, 0)
    keyboard.tap('V')
    keyboard.tap('V')
    print('starting')
    time.sleep(3)

def observe():
    global step, screens, controls1, controls2, rooms
    screens[step % 300] = np.transpose(screen.capture(), (2, 1, 0))
    room, mode = memory.read()
    rooms[step % 300] = room
    rew1 = 0
    rew2 = 0
    winner = 0
    if step != 0:
        rm1 = rooms[(step + 299) % 300]
        rm2 = rooms[step % 300]
        if rm1 < rm2:
            rew1 = 1
            rew2 = -1
        if rm1 > rm2:
            rew1 = -1
            rew2 = 1
        if room == 0:
            rew1 = -5
            rew2 = 5
            winner = 2
        if room == 10:
            rew1 = 5
            rew2 = -5
            winner = 1
    return screens[step % 300], winner, rew1, rew2

def save_tdE(tdE1, tdE2):
    td_errors[step-1] = tdE1 + tdE2

def act(control1, control2):
    global step, preroom, screens, controls1, controls2, rooms
    controls1[step % 300] = control1
    controls2[step % 300] = control2
    control.update(control1, control2)
    step += 1

def checkpoint(episode):
    global step, screens, controls1, controls2, rooms
    directory = './training_data/{}/'.format(episode % 8)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save({'winner': 0, 'length': step, 'episode': episode, 'td_errors': td_errors[:step-1]}, directory + 'info.pt')
    np.savez_compressed(directory + str(step // 300 - 1), screens=screens, controls1=controls1, controls2=controls2, rooms=rooms)

def finish(episode, winner):
    global step, screens, controls1, controls2, rooms
    directory = './training_data/{}/'.format(episode % 8)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save({'winner': winner, 'length': step+1, 'episode': episode, 'td_errors': td_errors[:step]}, directory + 'info.pt')
    np.savez_compressed(directory + str((step+1 - 1) // 300), 
                        screens=screens[:step % 300 + 1], controls1=controls1[:step % 300 + 1], controls2=controls2[:step % 300 + 1], rooms=rooms[:step % 300 + 1])

def quit():
    control.update(0, 0)
    keyboard.tap('Escape')
    keyboard.tap('S')
    keyboard.tap('A')
    keyboard.tap('A')
    keyboard.tap('W')
    keyboard.tap('V')

def extract(directory, length, winner, p):
    ind = np.random.choice(np.arange(1, length), p=p)
    
    screens = np.zeros([10, 3, 240, 160], dtype=np.float)
    key1s = np.zeros([9, 6], dtype=np.float)
    key2s = np.zeros([9, 6], dtype=np.float)
    action1 = 0
    action2 = 0
    tmp_rooms = [5, 5]
    rew1 = 0
    rews = 0
    terminal = (winner != 0 and ind == length-1)
    #print(terminal)
    
    num_files = (length - 1) // 300 + 1
    for i in range(num_files):
        arrs = np.load(directory + str(i) + '.npz')
        scrs = arrs['screens']
        ctrls1 = arrs['controls1']
        ctrls2 = arrs['controls2']
        rms = arrs['rooms']
        for k in range(10):
            loc = ind-9+k
            if 300 * i <= loc < 300 * (i+1):
                screens[k] = scrs[loc % 300] / 256
        for k in range(9):
            loc = ind-9+k
            if 300 * i <= loc < 300 * (i+1):
                tmp_act1 = ctrls1[loc % 300]
                tmp_act2 = ctrls2[loc % 300]
                key1s[k] = control.act_to_key(tmp_act1)
                key2s[k] = control.act_to_key(tmp_act2)
        loc = ind-1
        if 300 * i <= loc < 300 * (i+1):
            action1 = ctrls1[loc % 300]
            action2 = ctrls2[loc % 300]
            tmp_rooms[0] = rms[loc % 300]
        loc = ind
        if 300 * i <= loc < 300 * (i+1):
            tmp_rooms[1] = rms[loc % 300]
        arrs.close()
    if tmp_rooms[1] == 0:
        rew1 = -5
        rew2 = 5
    elif tmp_rooms[1] == 10:
        rew1 = 5
        rew2 = -5        
    elif tmp_rooms[0] < tmp_rooms[1]:
        rew1 = 1
        rew2 = -1        
    elif tmp_rooms[0] > tmp_rooms[1]:
        rew1 = -1
        rew2 = 1        
    else:
        rew1 = 0
        rew2 = 0
    return ind-1, (screens, key1s, key2s, action1, action2, rew1, rew2, terminal)
    
def sample():
    while True:
        sel = np.random.randint(8)
        directory = './training_data/{}/'.format(sel)
        if os.path.exists(directory):
            break
    info = torch.load(directory + 'info.pt')
    winner = info['winner']
    length = info['length']
    p = info['td_errors']
    #print('p:', p)
    p /= p.sum()
    ind, rtr = extract(directory, length, winner, p)
    return sel, ind, rtr

def update_tdE(sel, ind, tdE):
    directory = './training_data/{}/'.format(sel)
    info = torch.load(directory + 'info.pt')
    td_errors = info['td_errors']
    td_errors[ind] = tdE
    torch.save(info, directory + 'info.pt')

if __name__ == '__main__':
    state1as, state2as, state1bs, state2bs, action1s, action2s, reward1s, reward2s, terminal = training_data()
    print(state1as.shape)
