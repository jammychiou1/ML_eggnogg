import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

import keyboard
import model as md
import environment

gamma = 0.95

model = md.Model()
model_tar = md.Model()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)
trained_episode = 0

if os.path.isfile('checkpoint.pt'):
    checkpoint = torch.load('checkpoint.pt')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    trained_episode = checkpoint['episode']

environment.init()

for episode in range(trained_episode, trained_episode+1000):
    print('Episode {} start'.format(episode))
    step = 0
    
    model_tar.load_state_dict(model.state_dict())
    
    environment.reset()
    
    last = time.time()
    while True:
        now = time.time()
        if now - last > 1/3:
            #print('FPS: {}'.format(1/(now - last)))
            last = now
            
            state1, state2, winner, rew1, rew2 = environment.observe()
            q1 = model(torch.Tensor(state1[np.newaxis, :]))[0]
            q2 = model(torch.Tensor(state2[np.newaxis, :]))[0]
            print('q1:', q1)
            print('q2:', q2)
            if step != 0:
                q_tar1 = 0
                q_tar2 = 0
                if winner == 0:
                    q_tar1 = model_tar(torch.Tensor(state1[np.newaxis, :]))[0, torch.argmax(q1)].item() * gamma
                    q_tar2 = model_tar(torch.Tensor(state2[np.newaxis, :]))[0, torch.argmax(q2)].item() * gamma
                q_tar1 += rew1
                q_tar2 += rew2
                environment.save_tdE((q_tar1 - q_old1) ** 2, (q_tar2 - q_old2) ** 2)
            if winner != 0:
                environment.finish(episode, winner)
                environment.quit()
                break
            if np.random.rand() < 0.75:
                act1 = np.random.randint(36)
            else:
                act1 = torch.argmax(q1).item()
            if np.random.rand() < 0.75:
                act2 = np.random.randint(36)
            else:
                act2 = torch.argmax(q2).item()
            #print(act1, act2)
            environment.act(act1, act2)
            q_old1 = q1[act1]
            q_old2 = q2[act2]
            step += 1
            
            if step % 300 == 0: #100 sec 300
                tmp = time.time()
                keyboard.tap('Escape')
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'episode': episode}, 'checkpoint.pt')
                environment.checkpoint(episode)
                #avg_loss = 0
                print('Training')
                for i_train in range(50):
                    sel, inds, data = environment.training_data_new()
                    state1as = data[0]
                    state2as = data[1]
                    state1bs = data[2]
                    state2bs = data[3]
                    action1s = data[4]
                    action2s = data[5]
                    reward1s = data[6]
                    reward2s = data[7]
                    terminal = data[8]
                    batch_size = state1as.shape[0]
                    q_tar1 = np.zeros(batch_size, dtype=np.float)
                    q_tar2 = np.zeros(batch_size, dtype=np.float)
                    #print(state1as.shape, state2as.shape, state1bs.shape, state2bs.shape)
                    if not np.all(terminal):
                        state1bs_nt = torch.Tensor(state1bs[np.flatnonzero(np.logical_not(terminal))])
                        state2bs_nt = torch.Tensor(state2bs[np.flatnonzero(np.logical_not(terminal))])
                        argmax1 = torch.argmax(model(state1bs_nt), 1)
                        argmax2 = torch.argmax(model(state2bs_nt), 1)
                        tar_out1 = model_tar(state1bs_nt)
                        tar_out2 = model_tar(state2bs_nt)
                        for idx, loc in enumerate(np.flatnonzero(np.logical_not(terminal))):
                            q_tar1[loc] = tar_out1[idx][argmax1[idx]].item() * gamma
                            q_tar2[loc] = tar_out2[idx][argmax2[idx]].item() * gamma
                    q_tar1 += reward1s
                    q_tar2 += reward2s
                    #print(model(torch.Tensor(state1as))[0].shape)
                    q_now1 = [model(torch.Tensor(state1as))[i][action1s[i]] for i in range(batch_size)]
                    q_now2 = [model(torch.Tensor(state2as))[i][action2s[i]] for i in range(batch_size)]
                    #print(q_tar1.shape)
                    #print(q_now1[0].shape)
                    loss = 0
                    for i in range(batch_size):
                        loss += ((q_tar1[i] - q_now1[i]) ** 2 + (q_tar2[i] - q_now2[i]) ** 2)
                        #print(loss.shape)
                    loss /= 2 * batch_size
                    print(i_train, loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    q_tar1 = np.zeros(batch_size, dtype=np.float)
                    q_tar2 = np.zeros(batch_size, dtype=np.float)
                    if not np.all(terminal):
                        state1bs_nt = torch.Tensor(state1bs[np.flatnonzero(np.logical_not(terminal))])
                        state2bs_nt = torch.Tensor(state2bs[np.flatnonzero(np.logical_not(terminal))])
                        #print(model(state1bs_nt))
                        argmax1 = torch.argmax(model(state1bs_nt), 1)
                        #print(argmax1)
                        argmax2 = torch.argmax(model(state2bs_nt), 1)
                        tar_out1 = model_tar(state1bs_nt)
                        #print(tar_out1)
                        tar_out2 = model_tar(state2bs_nt)
                        #print(type(np.flatnonzero(np.logical_not(terminal))))
                        #print(np.flatnonzero(np.logical_not(terminal)))
                        for idx, loc in enumerate(np.flatnonzero(np.logical_not(terminal))):
                            #print(idx, loc)
                            q_tar1[loc] = tar_out1[idx][argmax1[idx]].item() * gamma
                            q_tar2[loc] = tar_out2[idx][argmax2[idx]].item() * gamma
                    q_tar1 += reward1s
                    q_tar2 += reward2s
                    #print(model(torch.Tensor(state1as))[0].shape)
                    q_now1 = np.array([model(torch.Tensor(state1as))[i][action1s[i]].item() for i in range(batch_size)], dtype=np.float)
                    q_now2 = np.array([model(torch.Tensor(state2as))[i][action2s[i]].item() for i in range(batch_size)], dtype=np.float)
                    print('qn1', q_now1)
                    print('qt1', q_tar1)
                    print('qn2', q_now2)
                    print('qt2', q_tar2)
                    td_E = np.zeros(batch_size, dtype=np.float)
                    td_E = (q_tar1 - q_now1) ** 2 + (q_tar2 - q_now2) ** 2 
                    environment.update_tdE(sel, inds, td_E)
                keyboard.tap('Escape')
                last += time.time() - tmp
            if step == 1800: #10 min
                environment.quit()
                break
        
