import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

import keyboard
import control
import model as md
import environment
import autoencoder as ae

import gc

def train(i_train):
    global model, model_tar, encoder, encoder_tar, decoder, optimizer, gamma
    train_sel, train_ind, train_data = environment.sample()
    #print('sampling done')
    train_screens, train_key1s, train_key2s, train_action1, train_action2, train_rew1, train_rew2, train_terminal = train_data
    train_ima = torch.Tensor(train_screens)
    train_ima_ = decoder(encoder(train_ima))
    
    train_ae_loss = nn.L1Loss(reduction='mean')(train_ima, train_ima_)
    
    train_codes = encoder(train_ima).detach().numpy()
    train_codes_tar = encoder_tar(train_ima).detach().numpy()
    
    train_q_tar1 = 0
    train_q_tar2 = 0
    #print(codes.shape)
    #print(key1s.shape)
    #print(key1s.shape)
    #print(np.concatenate([codes[1:].flatten(), keys1[1:].flatten(), [0]])[np.newaxis, :].shape)
    
    if not train_terminal:
        train_state1b = torch.Tensor(np.concatenate([train_codes[1:].flatten(), train_key1s[1:].flatten(), [0]]))
        train_state2b = torch.Tensor(np.concatenate([train_codes[1:].flatten(), train_key2s[1:].flatten(), [1]]))
        train_argmax1 = torch.argmax(model(train_state1b)).item()
        train_argmax2 = torch.argmax(model(train_state2b)).item()
        train_state1b_tar = torch.Tensor(np.concatenate([train_codes_tar[1:].flatten(), train_key1s[1:].flatten(), [0]]))
        train_state2b_tar = torch.Tensor(np.concatenate([train_codes_tar[1:].flatten(), train_key2s[1:].flatten(), [1]]))
        train_tar_out1 = model_tar(train_state1b_tar).detach().numpy()
        train_tar_out2 = model_tar(train_state2b_tar).detach().numpy()
        train_q_tar1 = train_tar_out1[train_argmax1] * gamma
        train_q_tar2 = train_tar_out2[train_argmax2] * gamma
        #del state1b, state2b, argmax1, argmax2, state1b_tar, state2b_tar, tar_out1, tar_out2
    
    train_q_tar1 += train_rew1
    train_q_tar2 += train_rew2
    train_state1a = torch.Tensor(np.concatenate([train_codes[:-1].flatten(), train_key1s[:-1].flatten(), [0]]))
    train_state2a = torch.Tensor(np.concatenate([train_codes[:-1].flatten(), train_key2s[:-1].flatten(), [1]]))
    train_q_now1 = model(torch.Tensor(train_state1a))[train_action1]
    train_q_now2 = model(torch.Tensor(train_state2a))[train_action2]
    train_td_loss = ((train_q_tar1 - train_q_now1) ** 2 + (train_q_tar2 - train_q_now2) ** 2)
    environment.update_tdE(train_sel, train_ind, train_td_loss.item())
    
    train_loss = train_ae_loss + train_td_loss
    #print(i_train, train_ae_loss.item(), train_td_loss.item())
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    #del q_tar1, q_tar2, state1a, state2a, q_now1, q_now2, td_loss, loss, ae_loss, ima, ima_, codes, codes_tar, screens, key1s, key2s, action1, action2, rew1, rew2, terminal, data, sel, ind
    

gamma = 0.80

model = md.Model()
model_tar = md.Model()
encoder = ae.Encoder()
encoder_tar = ae.Encoder()
decoder = ae.Decoder()
optimizer = optim.Adam(model.parameters(), lr = 0.01)
trained_episode = 0

if os.path.isfile('checkpoint.pt'):
    checkpoint = torch.load('checkpoint.pt')
    model.load_state_dict(checkpoint['model'])
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    trained_episode = checkpoint['episode']

environment.init()

for episode in range(trained_episode, trained_episode+1000):
    print('Episode {} start'.format(episode))
    step = 0
    
    model_tar.load_state_dict(model.state_dict())
    encoder_tar.load_state_dict(encoder.state_dict())
    
    environment.reset()
    
    last = time.time()
    
    codes = np.zeros([8, 20], dtype=np.float)
    keys1 = np.zeros([8, 6], dtype=np.float)
    keys2 = np.zeros([8, 6], dtype=np.float)
    while True:
        now = time.time()
        if now - last > 1/3:
            #print('FPS: {}'.format(1/(now - last)))
            last = now
            
            scr, winner, rew1, rew2 = environment.observe()
            
            scr = scr / 256
            code = encoder(torch.Tensor(scr[np.newaxis, :])).detach().numpy()
            state1 = np.concatenate([codes.flatten(), code.flatten(), keys1.flatten(), [0]])
            state2 = np.concatenate([codes.flatten(), code.flatten(), keys2.flatten(), [1]])
            q1 = model(torch.Tensor(state1))
            q2 = model(torch.Tensor(state2))
            #print('q1:', q1.detach().numpy())
            #print('q2:', q2.detach().numpy())
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
                keyboard.tap('Escape')
                environment.finish(episode, winner)
                keyboard.tap('Escape')
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
            
            codes[:-1] = codes[1:]
            keys1[:-1] = keys1[1:]
            keys2[:-1] = keys2[1:]
            codes[-1] = code
            keys1[-1] = control.act_to_key(act1)
            keys2[-1] = control.act_to_key(act2)
            
            step += 1
            
            if step % 300 == 0: #100 sec 300
                tmp = time.time()
                keyboard.tap('Escape')
                
                environment.checkpoint(episode)
                
                '''
                cnt = 0
                for obj in gc.get_objects():
                    try:
                        if type(obj).__module__ == torch.__name__:
                            cnt += 1
                    except:
                        pass
                '''
                print("before {}".format(len(gc.get_objects())))
                
                print('Training')
                for i_train in range(100):
                    train(i_train)
                    
                print(len(gc.get_objects()))
                
                '''
                cnt = 0
                for obj in gc.get_objects():
                    try:
                        if type(obj).__module__ == torch.__name__:
                            cnt += 1
                    except:
                        pass
                '''
                print("after {}".format(len(gc.get_objects())))
                
                torch.save({'model': model.state_dict(), 'encoder': encoder.state_dict(), 'decoder': decoder.state_dict(),
                            'optimizer': optimizer.state_dict(), 'episode': episode}, 'checkpoint.pt')
                
                keyboard.tap('Escape')
                last += time.time() - tmp
            if step == 1800: #10 min
                environment.quit()
                break
        
