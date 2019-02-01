import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import keyboard
import model
import environment

gamma = 0.999

model = model.Model()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)

if os.file.exists('checkpoint.pt'):
    checkpoint = torch.load('checkpoint.pt')
    model.state_dict = checkpoint['model']
    optimizer.state_dict = checkpoint['optimizer']
    trained_episode = checkpoint['episode']

environment.init()

for episode in range(trained_episode, trained_episode+1000):
    step = 0
    
    environment.reset()
    
    last = time.time()
    while True:
        now = time.time()
        if now - last > 1/3:
            print('FPS: {}'.format(1/(now - last)))
            last = now
            
            state1, state2, winner = environment.observe()
            if winner != 0:
                environment.finish(episode, winner)
                environment.quit()
                break
            if np.random.rand() < 0.25:
                act1 = np.random.randint(36)
            else:
                q1 = model(torch.Tensor(state1[np.newaxis, :]))
                act1 = torch.argmax(q1, 1).item()
            if np.random.rand() < 0.25:
                act2 = np.random.randint(36)
            else:
                q2 = model(torch.Tensor(state2[np.newaxis, :]))
                act2 = torch.argmax(q2, 1).item()
            #print(act1, act2)
            environment.act(act1, act2)
            step += 1
            
            if step % 300 == 0: #100 sec 300
                tmp = time.time()
                keyboard.tap('Escape')
                torch.save('checkpoint.pt', {'model': model.state_dict, 'optimizer': optimizer.state_dict, 'episode': episode})
                environment.checkpoint(episode)
                #avg_loss = 0
                print('training')
                for i_train in range(50):
                    data = environment.training_data()
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
                    q_tar1 = np.zeros([batch_size, 36])
                    q_tar2 = np.zeros([batch_size, 36])
                    q_tar1[np.where(np.logical_not(terminal))] = model(torch.Tensor(state1bs[np.where(np.logical_not(terminal))])).detach() * gamma
                    q_tar2[np.where(np.logical_not(terminal))] = model(torch.Tensor(state2bs[np.where(np.logical_not(terminal))])).detach() * gamma
                    q_tar1 = np.array([q_tar1[i][action1s[i]] for i in range(batch_size)])
                    q_tar2 = np.array([q_tar2[i][action2s[i]] for i in range(batch_size)])
                    q_tar1 += reward1s
                    q_tar2 += reward2s
                    #print(model(torch.Tensor(state1as))[0].shape)
                    q_now1 = [model(torch.Tensor(state1as))[i][action1s[i]] for i in range(batch_size)]
                    q_now2 = [model(torch.Tensor(state2as))[i][action2s[i]] for i in range(batch_size)]
                    #print(q_tar1.shape)
                    #print(q_now1[0].shape)
                    loss = 0
                    for i in range(batch_size):
                        loss += ((q_tar1[i] - q_now1[i]) ** 2 + (q_tar2[i] - q_now2[i]) ** 2) / 2 / batch_size
                        #print(loss.shape)
                    print(i_train, loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                keyboard.tap('Escape')
                last += time.time() - tmp
            if step == 2700: #15 min
                environment.quit()
                break
        
