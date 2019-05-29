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
    ext_ind = np.random.choice(np.arange(1, length), p=p)
    
    ext_screens = np.zeros([10, 3, 240, 160], dtype=np.float)
    ext_key1s = np.zeros([9, 6], dtype=np.float)
    ext_key2s = np.zeros([9, 6], dtype=np.float)
    ext_action1 = 0
    ext_action2 = 0
    ext_tmp_rooms = [5, 5]
    ext_rew1 = 0
    ext_rews = 0
    ext_terminal = (winner != 0 and ext_ind == length-1)
    #print(terminal)
    
    num_files = (length - 1) // 300 + 1
    for i in range(num_files):
        ext_arrs = np.load(directory + str(i) + '.npz')
        ext_scrs = ext_arrs['screens']
        ext_ctrls1 = ext_arrs['controls1']
        ext_ctrls2 = ext_arrs['controls2']
        ext_rms = ext_arrs['rooms']
        for k in range(10):
            loc = ext_ind-9+k
            if 300 * i <= loc < 300 * (i+1):
                ext_screens[k] = ext_scrs[loc % 300] / 256
        for k in range(9):
            loc = ext_ind-9+k
            if 300 * i <= loc < 300 * (i+1):
                ext_tmp_act1 = ext_ctrls1[loc % 300]
                ext_tmp_act2 = ext_ctrls2[loc % 300]
                ext_key1s[k] = control.act_to_key(ext_tmp_act1)
                ext_key2s[k] = control.act_to_key(ext_tmp_act2)
        loc = ext_ind-1
        if 300 * i <= loc < 300 * (i+1):
            ext_action1 = ext_ctrls1[loc % 300]
            ext_action2 = ext_ctrls2[loc % 300]
            ext_tmp_rooms[0] = ext_rms[loc % 300]
        loc = ext_ind
        if 300 * i <= loc < 300 * (i+1):
            ext_tmp_rooms[1] = ext_rms[loc % 300]
        ext_arrs.close()
    if ext_tmp_rooms[1] == 0:
        ext_rew1 = -5
        ext_rew2 = 5
    elif ext_tmp_rooms[1] == 10:
        ext_rew1 = 5
        ext_rew2 = -5        
    elif ext_tmp_rooms[0] < ext_tmp_rooms[1]:
        ext_rew1 = 1
        ext_rew2 = -1        
    elif ext_tmp_rooms[0] > ext_tmp_rooms[1]:
        ext_rew1 = -1
        ext_rew2 = 1        
    else:
        ext_rew1 = 0
        ext_rew2 = 0
    return ext_ind-1, (ext_screens, ext_key1s, ext_key2s, ext_action1, ext_action2, ext_rew1, ext_rew2, ext_terminal)
    
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
