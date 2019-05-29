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
    global step, preroom, screens, controls1, controls2, rooms
    screens[step % 300] = np.transpose(screen.capture(), (2, 1, 0))
    room, mode = memory.read()
    rooms[step % 300] = room
    state1 = np.zeros([76, 240, 160], dtype=np.float)
    state2 = np.zeros([76, 240, 160], dtype=np.float)
    for i in range(9):
        state1[3 * i : 3 * (i+1)] = screens[(step - 8 + i + 300) % 300] / 256
        state2[3 * i : 3 * (i+1)] = screens[(step - 8 + i + 300) % 300] / 256
    for i in range(8):
        control1 = np.unravel_index(controls1[(step - 8 + i + 300) % 300], (3, 3, 2, 2))
        state1[27 + 6 * i + 0] = control1[0] & 1
        state1[27 + 6 * i + 1] = (control1[0] & 2) >> 1
        state1[27 + 6 * i + 2] = control1[1] & 1
        state1[27 + 6 * i + 3] = (control1[1] & 2) >> 1
        state1[27 + 6 * i + 4] = control1[2]
        state1[27 + 6 * i + 5] = control1[3]
        control2 = np.unravel_index(controls2[(step - 8 + i + 300) % 300], (3, 3, 2, 2))
        state2[27 + 6 * i + 0] = control2[0] & 1
        state2[27 + 6 * i + 1] = (control2[0] & 2) >> 1
        state2[27 + 6 * i + 2] = control2[1] & 1
        state2[27 + 6 * i + 3] = (control2[1] & 2) >> 1
        state2[27 + 6 * i + 4] = control2[2]
        state2[27 + 6 * i + 5] = control2[3]
    state1[75] = 0
    state2[75] = 1
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
    return state1, state2, winner, rew1, rew2

def save_tdE(tdE1, tdE2):
    td_errors[step-1] = tdE1 + tdE2

def act(control1, control2):
    global step, preroom, screens, controls1, controls2, rooms
    controls1[step % 300] = control1
    controls2[step % 300] = control2
    control.update(control1, control2)
    step += 1

def checkpoint(episode):
    global step, preroom, screens, controls1, controls2, rooms
    directory = './training_data/{}/'.format(episode % 8)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save({'winner': 0, 'length': step, 'episode': episode, 'td_errors': td_errors[:step-1]}, directory + 'info.pt')
    np.savez_compressed(directory + str(step // 300 - 1), screens=screens, controls1=controls1, controls2=controls2, rooms=rooms)

def finish(episode, winner):
    global step, preroom, screens, controls1, controls2, rooms
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
    #inds = np.random.choice(np.arange(1, length), 4)
    inds = np.random.choice(np.arange(1, length), 4, p=p)
    
    state1as = np.zeros([4, 76, 240, 160], dtype=np.float)
    state2as = np.zeros([4, 76, 240, 160], dtype=np.float)
    state1bs = np.zeros([4, 76, 240, 160], dtype=np.float)
    state2bs = np.zeros([4, 76, 240, 160], dtype=np.float)
    action1s = np.zeros(4, dtype=np.int)
    action2s = np.zeros(4, dtype=np.int)
    tmp_rooms = np.full([4, 2], 5, dtype=np.int)
    reward1s = np.zeros(4, dtype=np.float)
    reward2s = np.zeros(4, dtype=np.float)
    #terminal = np.logical_and(inds == length-1, winner != 0)
    terminal = (inds == length-1)
    #print(terminal)
    
    num_files = (length - 1) // 300 + 1
    for i in range(num_files):
        arrs = np.load(directory + str(i) + '.npz')
        scrs = arrs['screens']
        ctrls1 = arrs['controls1']
        ctrls2 = arrs['controls2']
        rms = arrs['rooms']
        for j in range(4):
            for k in range(9):
                loc = inds[j]-1 - 8 + k
                if 300 * i <= loc < 300 * (i+1):
                    state1as[j][3 * k : 3 * (k+1)] = scrs[loc % 300] / 256
                    state2as[j][3 * k : 3 * (k+1)] = scrs[loc % 300] / 256
            for k in range(9):
                loc = inds[j] - 8 + k
                if 300 * i <= loc < 300 * (i+1):
                    state1bs[j][3 * k : 3 * (k+1)] = scrs[loc % 300] / 256
                    state2bs[j][3 * k : 3 * (k+1)] = scrs[loc % 300] / 256
            for k in range(8):
                loc = inds[j]-1 - 8 + k
                if 300 * i <= loc < 300 * (i+1):
                    control1 = np.unravel_index(ctrls1[loc % 300], (3, 3, 2, 2))
                    state1as[j][27 + 6 * k + 0] = control1[0] & 1
                    state1as[j][27 + 6 * k + 1] = (control1[0] & 2) >> 1
                    state1as[j][27 + 6 * k + 2] = control1[1] & 1
                    state1as[j][27 + 6 * k + 3] = (control1[1] & 2) >> 1
                    state1as[j][27 + 6 * k + 4] = control1[2]
                    state1as[j][27 + 6 * k + 5] = control1[3]
                    control2 = np.unravel_index(ctrls2[loc % 300], (3, 3, 2, 2))
                    state2as[j][27 + 6 * k + 0] = control2[0] & 1
                    state2as[j][27 + 6 * k + 1] = (control2[0] & 2) >> 1
                    state2as[j][27 + 6 * k + 2] = control2[1] & 1
                    state2as[j][27 + 6 * k + 3] = (control2[1] & 2) >> 1
                    state2as[j][27 + 6 * k + 4] = control2[2]
                    state2as[j][27 + 6 * k + 5] = control2[3]
            for k in range(8):
                loc = inds[j] - 8 + k
                if 300 * i <= loc < 300 * (i+1):
                    control1 = np.unravel_index(ctrls1[loc % 300], (3, 3, 2, 2))
                    state1bs[j][27 + 6 * k + 0] = control1[0] & 1
                    state1bs[j][27 + 6 * k + 1] = (control1[0] & 2) >> 1
                    state1bs[j][27 + 6 * k + 2] = control1[1] & 1
                    state1bs[j][27 + 6 * k + 3] = (control1[1] & 2) >> 1
                    state1bs[j][27 + 6 * k + 4] = control1[2]
                    state1bs[j][27 + 6 * k + 5] = control1[3]
                    control2 = np.unravel_index(ctrls2[loc % 300], (3, 3, 2, 2))
                    state2bs[j][27 + 6 * k + 0] = control2[0] & 1
                    state2bs[j][27 + 6 * k + 1] = (control2[0] & 2) >> 1
                    state2bs[j][27 + 6 * k + 2] = control2[1] & 1
                    state2bs[j][27 + 6 * k + 3] = (control2[1] & 2) >> 1
                    state2bs[j][27 + 6 * k + 4] = control2[2]
                    state2bs[j][27 + 6 * k + 5] = control2[3]
            loc = inds[j]-1
            if 300 * i <= loc < 300 * (i+1):
                action1s[j] = ctrls1[loc % 300]
                action2s[j] = ctrls2[loc % 300]
                tmp_rooms[j, 0] = rms[loc % 300]
            loc = inds[j]
            if 300 * i <= loc < 300 * (i+1):
                tmp_rooms[j, 1] = rms[loc % 300]
        arrs.close()
    for j in range(4):
        if tmp_rooms[j, 1] == 0:
            reward1s[j] = -5
            reward2s[j] = 5
        elif tmp_rooms[j, 1] == 10:
            reward1s[j] = 5
            reward2s[j] = -5        
        elif tmp_rooms[j, 0] < tmp_rooms[j, 1]:
            reward1s[j] = 1
            reward2s[j] = -1        
        elif tmp_rooms[j, 0] > tmp_rooms[j, 1]:
            reward1s[j] = -1
            reward2s[j] = 1        
        else:
            reward1s[j] = 0
            reward2s[j] = 0        
    state1as[:, 75] = 0
    state2as[:, 75] = 1
    state1bs[:, 75] = 0
    state2bs[:, 75] = 1
    return inds-1, (state1as, state2as, state1bs, state2bs, action1s, action2s, reward1s, reward2s, terminal)
    
def training_data():
    state1as = []
    state2as = []
    state1bs = []
    state2bs = []
    action1s = []
    action2s = []
    reward1s = []
    reward2s = []
    terminal = []
    for i in range(8):
        directory = './training_data/{}/'.format(i)
        if os.path.exists(directory):
            info = torch.load(directory + 'info.pt')
            winner = info['winner']
            length = info['length']
            tmp = extract(directory, length, winner) 
            state1as.append(tmp[0])
            state2as.append(tmp[1])
            state1bs.append(tmp[2])
            state2bs.append(tmp[3])
            action1s.append(tmp[4])
            action2s.append(tmp[5])
            reward1s.append(tmp[6])
            reward2s.append(tmp[7])
            terminal.append(tmp[8])
    state1as = np.concatenate(state1as)
    state2as = np.concatenate(state2as)
    state1bs = np.concatenate(state1bs)
    state2bs = np.concatenate(state2bs)
    action1s = np.concatenate(action1s)
    action2s = np.concatenate(action2s)
    reward1s = np.concatenate(reward1s)
    reward2s = np.concatenate(reward2s)
    terminal = np.concatenate(terminal)
    return state1as, state2as, state1bs, state2bs, action1s, action2s, reward1s, reward2s, terminal
    
def training_data_new():
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
    inds, rtr = extract(directory, length, winner, p)
    return sel, inds, rtr
    
def update_tdE(sel, inds, tdE):
    directory = './training_data/{}/'.format(sel)
    info = torch.load(directory + 'info.pt')
    td_errors = info['td_errors']
    td_errors[inds] = tdE
    torch.save(info, directory + 'info.pt')

if __name__ == '__main__':
    state1as, state2as, state1bs, state2bs, action1s, action2s, reward1s, reward2s, terminal = training_data()
    print(state1as.shape)
