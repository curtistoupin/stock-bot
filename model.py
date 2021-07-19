# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 20:50:15 2021

@author: matt_
"""

import torch
import torch.nn as nn
import numpy as np
from util import csv_to_dataset
import matplotlib.pyplot as plt


# input_dim = 6

# hidden_dim = 10

# n_layers = 1


# lstm_layer=nn.LSTM(input_dim,hidden_dim, n_layers, batch_first=True)


# batch_size = 1

# seq_len = 3

# inp=torch.randn(batch_size,seq_len,input_dim)

# hidden_state = torch.randn(n_layers,batch_size,hidden_dim)

# cell_state = torch.randn(n_layers, batch_size, hidden_dim)

# hidden = (hidden_state, cell_state)


# out,hidden = lstm_layer(inp,hidden)
# print('output shape: ', out.shape)
# print('hidden: ', hidden)

is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
device = torch.device('cpu')

ohlcv_histories, _, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset(r'D:\stonks\data\dataBB_daily.csv')

ohlcv_histories = np.float32(ohlcv_histories)
next_day_open_values = np.float32(next_day_open_values)


test_split = 0.9
n = int(ohlcv_histories.shape[0] * test_split)

ohlcv_train = torch.from_numpy(ohlcv_histories[:n]).to(device)
y_train = torch.from_numpy(next_day_open_values[:n]).to(device)

ohlcv_test = torch.from_numpy(ohlcv_histories[n:]).to(device)
y_test = torch.from_numpy(next_day_open_values[n:]).to(device)

print(ohlcv_train.shape)
print(ohlcv_test.shape)



class StonkNet(nn.Module):
    def __init__(self, input_size, output_size, embedding_dim, hidden_dim, n_layers, batch_size, drop_prob=0.5):
        super(StonkNet,self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        #self.embedding = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc  = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, hidden):
        #batch_size = x.size(0)
        #x=x.long()
        #embeds = self.embedding(x)
        lstm_out , hidden = self.lstm(x,hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)
        
        #out = out.view(batch_size,-1)
        #out = out[:,-1]
        return out, hidden_dim
    
    def init_hidden(self,batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden
        
        
        
input_dim = 5
output_size = 1
embedding_dim=400
hidden_dim = 50
n_layers = 2
batch_size = 4770
seq_len = 50


#lstm_layer=nn.LSTM(input_dim,hidden_dim, n_layers, batch_first=True)

    


model = StonkNet(input_dim, output_size, embedding_dim, hidden_dim, n_layers, batch_size)
model.to(device)

lr=0.005
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

epochs = 100

counter = 0

print_every = 10000
clip = 5
valid_loss_min=1e-5

model.train()

for i in range(epochs):
    h = model.init_hidden(batch_size)
    
    #make train loader later with stonk data
    #for inputs, ground_truth in train_loader:
    counter += 1
    #h = tuple([e.data for e in h])
    model.zero_grad()
    output, h = model(ohlcv_train, h)
    output = output.view(batch_size,-1)
    output = output[:,-1]
    loss = criterion(output, y_train.squeeze())
    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(),clip)
    optimizer.step()
        
        
    # if counter % print_every == 0:
    #     val_h = model. init_hidden(batch_size)
    #     val_losses = []
    #     model.eval()
            
    #     for inp, lab in val_loader:
    #         val_h = tuple([each.data for each in val_h])
    #         out, val_h = model(inp, val_h)
    #         val_loss = criterion(out.squeeze(),lab.float())
    #         val_losses.append(val_loss.item())
                
    model.train()
    print('epoch: ', epochs)
    print('step: ', counter)
    print('loss: ', loss.item())
        #print('val loss: ', val_losses)

        # if np.mean(val_losses) <= valid_loss_min:
        #     torch.save(model.state_dict(), './state_dict.pt')
        #     print('validation loss low enough, saving')
        #     valid_loss_min = np.mean(val_losses)

    model.eval()
    
    
    
h = model.init_hidden(530)    
pred,h=model(ohlcv_test,h)

pred = pred.view(530,-1)
pred = pred[:,-1]

pred_np=pred.detach().numpy()
y_test_np=y_test.numpy()
    
plt.plot(y_test_np)        
plt.plot(pred_np)        







