#!/usr/bin/env python
# coding: utf-8

# In[73]:


import pandas as pd
from matplotlib import pyplot as plt
import torch
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
np.seterr(divide='ignore',invalid='ignore')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from math import floor, ceil


# In[74]:



MidtermProjectData=open(r'C:\Users\reidy\Downloads\MidtermProjectData.csv')
df = pd.read_csv(MidtermProjectData,index_col=0)
pd.set_option('display.max_columns', None)


# In[32]:


print(df.shape)
print(df.describe())


# In[33]:


index = ["Pressure, mbar", "Temperature, *C", "RH, %",
         "Dew Point, *C", "Solar Radiation, W/m^2", "Wind Speed, m/s",
         "Gust Speed, m/s, ", "Wind Direction, *", "Meter Reading, W"]


# In[34]:



df.dropna(inplace=True)

data_array = np.array(df, dtype='float64')

# using mean of hour data
new_data_array = np.empty([0, 9])

i = 0
while i < data_array.shape[0]:
    hour_data = np.zeros((12, 9))
    for j in range(12):
        hour_data[j, :] = data_array[i, :]
        i += 1
    hour_mean = hour_data.mean(axis=0)
    hour_mean = hour_mean.reshape(1, -1)
    new_data_array = np.append(new_data_array, hour_mean, axis=0)


# In[35]:


new_data_array_0 = new_data_array[:, 8]

scaler = MinMaxScaler(feature_range=(-1, 1))
new_data_array_0 = scaler.fit_transform(new_data_array_0.reshape(-1, 1))
target =new_data_array_0

weekday = new_data_array_0[:96]
weekend = new_data_array_0[96:144]
j = 0
for i in range(144, len(new_data_array_0)):
    a = new_data_array_0[i]
    a = a.reshape(1, -1)
    if j <120:
        weekday = np.append(weekday, a)
    else:
        weekend = np.append(weekend, a)
    j += 1
    if j > 167:
        j = 0
        

target_w = new_data_array_0[:1848]

target_s = new_data_array_0[1848:]

target_weekday = weekday.reshape(-1, 1)



target_weekend = weekend.reshape(-1, 1)

target_train_weekday = target_weekday[:2700]
target_test_weekday = target_weekday[2700:]

target_train_weekend = target_weekend[:1000]
target_test_weekend = target_weekend[1000:]


target_weekday_w = target_weekday[:1300]


target_weekend_w = target_weekend[:570]


target_weekday_s = target_weekday[1300:]


target_weekend_s = target_weekend[570:]

target_train_weekday_w = target_weekday_w[:1150]
target_test_weekday_w = target_weekday_w[1150:]

target_train_weekend_w = target_weekend_w[:500]
target_test_weekend_w = target_weekend_w[500:]

target_train_weekday_s = target_weekday_s[:1400]
target_test_weekday_s = target_weekday_s[1400:]

target_train_weekend_s = target_weekend_s[:500]
target_test_weekend_s = target_weekend_s[500:]


# In[41]:


def get_max(input_data):
    l=len(input_data)
    days = int(l/24)
    pow_h = np.zeros(days)
    hour_h = np.zeros(days)
    for i in range(days) :
        temp = input_data[i*24:i*24+24]
        pow_h[i] = max(temp)
        for j in range(24):
            if temp[j]== max(temp):
                hour_h[i] = j
    return days,pow_h,hour_h


# In[42]:


target_train=target[:3600] 
target_test=target[3600:] 

target_train_w = np.array(target_w[:1752])
target_test_w = np.array(target_w[1752:])

target_train_s = np.array(target_s[:1944])
target_test_s = np.array(target_s[1944:])


# In[43]:



target_w_days,target_w_pow,target_w_hour = get_max(target_w)
target_s_days,target_s_pow,target_s_hour = get_max(target_s)


# In[44]:


target_train_norm = torch.FloatTensor(target_train.reshape(-1, 1))
target_test_norm = torch.FloatTensor(target_test.reshape(-1, 1))

target_train_norm_w = torch.FloatTensor(target_train_w)
target_train_norm_s = torch.FloatTensor(target_train_s)

target_test_norm_w = torch.FloatTensor(target_test_w)
target_test_norm_s = torch.FloatTensor(target_test_s)


target_train_norm_weekday = torch.FloatTensor(target_train_weekday)
target_train_norm_weekend = torch.FloatTensor(target_train_weekend)

target_test_norm_weekday = torch.FloatTensor(target_test_weekday)
target_test_norm_weekend = torch.FloatTensor(target_test_weekend)



target_train_norm_weekday_w = torch.FloatTensor(target_train_weekday_w)
target_train_norm_weekend_w = torch.FloatTensor(target_train_weekend_w)

target_test_norm_weekday_w = torch.FloatTensor(target_test_weekday_w)
target_test_norm_weekend_w = torch.FloatTensor(target_test_weekend_w)

target_train_norm_weekday_s = torch.FloatTensor(target_train_weekday_s)
target_train_norm_weekend_s = torch.FloatTensor(target_train_weekend_s)


target_test_norm_weekday_s = torch.FloatTensor(target_test_weekday_s)
target_test_norm_weekend_s = torch.FloatTensor(target_test_weekend_s)


# In[45]:


def make_sequences(input_data, sl):
    inout_seq = []
    L = len(input_data)
    for i in range(L-sl):
        train_seq = input_data[i:i+sl]
        train_label = input_data[i+sl:i+sl+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


# In[46]:


seq_len = 96
training_sequences = make_sequences(target_train_norm, seq_len)
training_sequences_w = make_sequences(target_train_norm_w, seq_len)
training_sequences_s = make_sequences(target_train_norm_s, seq_len)
training_sequences_weekday = make_sequences(target_train_norm_weekday, seq_len)
training_sequences_weekend = make_sequences(target_train_norm_weekend, seq_len)
training_sequences_weekday_w = make_sequences(target_train_norm_weekday_w, seq_len)
training_sequences_weekend_w = make_sequences(target_train_norm_weekend_w, seq_len)
training_sequences_weekday_s = make_sequences(target_train_norm_weekday_s, seq_len)
training_sequences_weekend_s = make_sequences(target_train_norm_weekend_s, seq_len)


# In[47]:


class GRU(torch.nn.Module):
    def __init__(self, D_in, D_hidden, D_out, nLayers=1):
        super().__init__()
        self.D_hidden = D_hidden
        self.nLayers = nLayers
        self.gru = torch.nn.GRU(D_in, D_hidden, num_layers=nLayers)
        self.linear = torch.nn.Linear(D_hidden, D_out)
        self.hidden = torch.zeros(nLayers, 1, D_hidden)

    def forward(self, input_seq):
        gru_out, self.hidden = self.gru(input_seq.view(len(input_seq), 1, -1), self.hidden)
        y_pred = self.linear(gru_out.view(len(input_seq), 1, -1))
        return y_pred[-1]


# In[48]:


class LSTM(torch.nn.Module):
    def __init__(self, D_in, D_hidden, D_out, nLayers=1):
        super().__init__()
        self.D_hidden = D_hidden
        self.nLayers = nLayers
        self.lstm = torch.nn.LSTM(D_in, D_hidden, num_layers=nLayers)
        self.linear = torch.nn.Linear(D_hidden, D_out)
        self.hidden = (torch.zeros(nLayers, 1, D_hidden),
                       torch.zeros(nLayers, 1, D_hidden))

    def forward(self, input_seq):
        lstm_out, self.hidden = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden)
        y_pred = self.linear(lstm_out.view(len(input_seq), 1, -1))
        return y_pred[-1]


# In[49]:


def train(training_sequences, rnn_type, num_layers, learning_rate, epochs, D_hidden):
    D_in = training_sequences[0][0].shape[1]
    D_out = training_sequences[0][1].shape[1]
    D_hidden = D_hidden
    learning_rate = learning_rate
    epochs= epochs
    l = np.zeros(epochs)
    
    if rnn_type.upper() == 'GRU':
        model = GRU(D_in, D_hidden, D_out, nLayers=num_layers)
    elif rnn_type.upper() == 'LSTM':
        model = LSTM(D_in, D_hidden, D_out, nLayers=num_layers)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for seq, labels in training_sequences:

            optimizer.zero_grad()

            if rnn_type.upper() == 'LSTM':
                model.hidden = (torch.zeros(model.nLayers, 1, model.D_hidden),
                                torch.zeros(model.nLayers, 1, model.D_hidden))
            else:
                model.hidden = torch.zeros(model.nLayers, 1, model.D_hidden)


            outputs = model(seq)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
        print('epoch {}: loss = {}'.format(epoch, loss.item()))
        l[epoch]=loss.item()


    return model,l


# In[62]:


def Cross_Validation(target, K, rnn_type, number_layers, learning_rate, epochs, D_hidden, tune_sq):

    Y = target
    idx = np.arange(len(Y))
    splits = np.split(idx, K)

    rmse_cv, mape_cv, r2_cv = np.empty(0), np.empty(0), np.empty(0)

    for k in range(2, K):
        trn_idx = splits[0]
        trn_idx = np.append(trn_idx, splits[1])
        tst_idx = splits[k]
        for i in range(2, k):
            trn_idx = np.append(trn_idx, splits[i])
        Y_train, Y_test = Y[trn_idx], Y[tst_idx]
        scaler = MinMaxScaler(feature_range=(0, 1))
        Y_train_norm = scaler.fit_transform(Y_train.reshape(-1, 1))

        Y_train_norm = torch.FloatTensor(Y_train_norm)
        training_sequences = make_sequences(Y_train_norm, tune_sq)


        modelK,l= train(training_sequences, rnn_type, number_layers, learning_rate, epochs, D_hidden)
        pred_len = len(Y_test)
        test_inputs = Y_train_norm[-pred_len:].tolist()
        y_pred_unscaled = test(modelK, rnn_type, test_inputs, pred_len)

        # Inverse scaling
        y_pred = scaler.inverse_transform(np.array(y_pred_unscaled).reshape(-1, 1))

        rmse = np.sqrt(np.mean((y_pred - Y_test) ** 2))
        r2 = np.corrcoef(y_pred.squeeze(), Y_test.squeeze())[0, 1]**2
        mape = np.mean(np.abs(np.abs(y_pred - Y_test) / Y_test))

        rmse_cv = np.append(rmse_cv, rmse)
        mape_cv = np.append(mape_cv, mape)
        r2_cv = np.append(r2_cv, r2)

    return rmse_cv.mean(), mape_cv.mean(), np.nanmean(r2_cv)


# In[63]:


epochs =150
num_layers_list = list(range(1, 4))
lr_list = [1e-4, 1e-3]
D_hidden_list = np.arange(25,80,25)


# In[64]:



mape_tst = np.zeros((len(num_layers_list), len(lr_list),len(D_hidden_list)))


# In[65]:


def test(model, rnn_type, test_inputs, pred_len):
    model.eval()
    for i in range(pred_len):
        test_seq = torch.FloatTensor(test_inputs[-seq_len:])
        with torch.no_grad():
            if rnn_type.upper() == 'LSTM':
                model.hidden = (torch.zeros(model.nLayers, 1, model.D_hidden),
                                torch.zeros(model.nLayers, 1, model.D_hidden))
            else:
                model.hidden = torch.zeros(model.nLayers, 1, model.D_hidden)

            test_inputs.append(model(test_seq))

    # print(np.array(test_inputs[pred_len:]))
    return test_inputs[pred_len:]


# In[66]:


def tuning(tuning_set , rnn_type, num_layers_list, lr_list, epochs, D_hidden_list, test_sq):
    for h1, H1 in enumerate(num_layers_list):
        for l, lr in enumerate(lr_list):
            for d_h, D_H in enumerate(D_hidden_list):
                rmse, mape, r2 = Cross_Validation(tuning_set, 6, rnn_type, H1, lr, epochs, D_H, test_sq)
                mape_tst[h1, l, d_h] = mape
                print('Number Layers = {} lr = {} D_hidden = {} RMSE = {} MAPE = {} r2 = {}'.format(H1, lr, D_H, rmse, mape, r2))
    i, j, k = np.argwhere(mape_tst == np.min(mape_tst))[0]
    num_layers_best, lr_best, D_hidden_best = num_layers_list[i], lr_list[j] , D_hidden_list[k]
    return num_layers_best, lr_best, D_hidden_best
                


# In[67]:


tuning_set = target[:144]

tuning_set_s = target_s[:144]
tuning_set_w = target_w[:144]

tuning_set_weekday = target_weekday[:144]
tuning_set_weekend = target_weekend[:144]

tuning_set_weekday_w = target_weekday_w[:144]
tuning_set_weekend_w = target_weekend_w[:144]

tuning_set_weekday_s = target_weekday_s[:144]
tuning_set_weekend_s = target_weekend_s[:144]


# In[68]:


def pred(target_test,target_train_norm,model,rnn_type):
    pred_len = len(target_test)
    test_inputs = target_train_norm[-pred_len:].tolist()
    y_pred_unscaled = test(model, rnn_type, test_inputs, pred_len)
    y_pred = scaler.inverse_transform(np.array(y_pred_unscaled).reshape(-1, 1))
    print(y_pred)
    target_test = target_test.reshape(-1, 1)
    rmse = np.sqrt(np.mean((y_pred - target_test) ** 2))
    mape = np.mean(np.abs((y_pred - target_test) / target_test))
    r2 = np.corrcoef(y_pred.squeeze(), target_test.squeeze())[0, 1] ** 2
    print('rmse = ', rmse)
    print('mape = ', mape)
    print('r2 = ', r2)
    return y_pred


# In[69]:


def training_result(target_test, target_train_norm, model,rnn_type):
    pred_len = len(target_test)
    
    test_inputs = target_train_norm[-pred_len:].tolist()
    y_pred_unscaled = test(model, rnn_type, test_inputs, pred_len)
    y_pred = scaler.inverse_transform(np.array(y_pred_unscaled).reshape(-1, 1))
    target_test = scaler.inverse_transform(target_test.reshape(-1, 1))
    rmse = np.sqrt(np.mean((y_pred - target_test) ** 2))
    mape = np.mean(np.abs((y_pred - target_test) / target_test))
    r2 = np.corrcoef(y_pred.squeeze(), target_test.squeeze())[0, 1] ** 2

    print('rmse = ', rmse)
    print('mape = ', mape)
    print('r2 = ', r2)
    
    figx = plt.figure()
    plt.plot(range(len(target_test)), target_test, 'go', label='True data', alpha=0.5)
    plt.plot(range(len(target_test)), y_pred, '--', label='Predictions', alpha=0.5)
    plt.legend(loc='best')
    plt.show()
    return rmse,mape,r2,figx


# In[70]:


rnn_type = 'gru'


# In[71]:


num_layers_best_gru,lr_best_gru,D_hidden_best_gru = tuning(tuning_set,'gru',num_layers_list,lr_list, epochs,D_hidden_list,24)
print(num_layers_best_gru, lr_best_gru, D_hidden_best_gru)


# In[72]:


model_gru, l_r= train(training_sequences,rnn_type,num_layers_best_gru,lr_best_gru, epochs,D_hidden_best_gru)
plt.plot(l_r)
plt.show()


# In[76]:



rmse_gru,mape_gru,r2_gru,figx_gru = training_result(target_test,target_train_norm,model_gru,rnn_type)


# In[ ]:





# In[ ]:


num_layers_best_gru_weekday,lr_best_gru_weekday,D_hidden_best_gru_weekday=tuning(tuning_set_weekday,rnn_type,num_layers_list,lr_list,epochs,D_hidden_list,24)
print(num_layers_best_gru_weekday,lr_best_gru_weekday,D_hidden_best_gru_weekday)
model_gru_weekday,l_gru_weekday=train(training_sequences_weekday,rnn_type,num_layers_best_gru_weekday,lr_best_gru_weekday,epochs,D_hidden_best_gru_weekday)
plt.plot(l_gru_weekday)
plt.show()
rmse_gru_weekday,mape_gru_weekday,r2_gru_weekday,figx_gru_weekday=training_result(target_test_weekday,target_train_norm_weekday,model_gru_weekday,rnn_type)


# In[ ]:


num_layers_best_gru_weekend,lr_best_gru_weekend,D_hidden_best_gru_weekend=tuning(tuning_set_weekend,rnn_type,num_layers_list,lr_list,epochs,D_hidden_list,24)
print(num_layers_best_gru_weekend,lr_best_gru_weekend,D_hidden_best_gru_weekend)
model_gru_weekend,l_gru_weekend=train(training_sequences_weekend,rnn_type,num_layers_best_gru_weekend,lr_best_gru_weekend,epochs,D_hidden_best_gru_weekend)
plt.plot(l_gru_weekend)
plt.show()
rmse_gru_weekend,mape_gru_weekend,r2_gru_weekend,figx_gru_weekend=training_result(target_test_weekend,target_train_norm_weekend,model_gru_weekend,rnn_type)


# In[ ]:



num_layers_best_gru_weekday_w,lr_best_gru_weekday_w,D_hidden_best_gru_weekday_w=tuning(tuning_set_weekday_w,rnn_type,num_layers_list,lr_list,epochs,D_hidden_list,24)
print(num_layers_best_gru_weekday_w,lr_best_gru_weekday_w,D_hidden_best_gru_weekday_w)
model_gru_weekday_w,l_gru_weekday_w=train(training_sequences_weekday_w,rnn_type,num_layers_best_gru_weekday_w,lr_best_gru_weekday_w,epochs,D_hidden_best_gru_weekday_w)
plt.plot(l_gru_weekday_w)
plt.show()

rmse_gru_weekday_w,mape_gru_weekday_w,r2_gru_weekday_w,figx_gru_weekday_w=training_result(target_test_weekday_w,target_train_norm_weekday_w,model_gru_weekday_w,rnn_type)


# In[ ]:


num_layers_best_gru_weekday_s,lr_best_gru_weekday_s,D_hidden_best_gru_weekday_s=tuning(tuning_set_weekday_s,rnn_type,num_layers_list,lr_list,epochs,D_hidden_list,24)
print(num_layers_best_gru_weekday_s,lr_best_gru_weekday_s,D_hidden_best_gru_weekday_s)
model_gru_weekday_s,l_gru_weekday_s=train(training_sequences_weekday_s,rnn_type,num_layers_best_gru_weekday_s,lr_best_gru_weekday_s,epochs,D_hidden_best_gru_weekday_s)
plt.plot(l_gru_weekday_s)
plt.show()
rmse_gru_weekday_s,mape_gru_weekday_s,r2_gru_weekday_s,figx_gru_weekday_s=training_result(target_test_weekday_s,target_train_norm_weekday_s,model_gru_weekday_s,rnn_type)


# In[ ]:


num_layers_best_gru_weekend_w,lr_best_gru_weekend_w,D_hidden_best_gru_weekend_w=tuning(tuning_set_weekend_w,rnn_type,num_layers_list,lr_list,epochs,D_hidden_list,24)
print(num_layers_best_gru_weekend_w,lr_best_gru_weekend_w,D_hidden_best_gru_weekend_w)
model_gru_weekend_w,l_gru_weekend_w=train(training_sequences_weekend_w,rnn_type,num_layers_best_gru_weekend_w,lr_best_gru_weekend_w,epochs,D_hidden_best_gru_weekend_w)
plt.plot(l_gru_weekend_w)
plt.show()
rmse_gru_weekend_w,mape_gru_weekend_w,r2_gru_weekend_w,figx_gru_weekend_w=training_result(target_test_weekend_w,target_train_norm_weekend_w,model_gru_weekend_w,rnn_type)


# In[ ]:



num_layers_best_gru_weekend_s,lr_best_gru_weekend_s,D_hidden_best_gru_weekend_s=tuning(tuning_set_weekend_s,rnn_type,num_layers_list,lr_list,epochs,D_hidden_list,24)
print(num_layers_best_gru_weekend_s,lr_best_gru_weekend_s,D_hidden_best_gru_weekend_s)
model_gru_weekend_s,l_gru_weekend_s=train(training_sequences_weekend_s,rnn_type,num_layers_best_gru_weekend_s,lr_best_gru_weekend_s,epochs,D_hidden_best_gru_weekend_s)
plt.plot(l_gru_weekend_s)
plt.show()
rmse_gru_weekend_s,mape_gru_weekend_s,r2_gru_weekend_s,figx_gru_weekend_s=training_result(target_test_weekend_s,target_train_norm_weekend_s,model_gru_weekend_s,rnn_type)


# In[ ]:


num_layers_best_gru_w,lr_best_gru_w,D_hidden_best_gru_w=tuning(tuning_set_w,rnn_type,num_layers_list,lr_list,epochs,D_hidden_list,24)
print(num_layers_best_gru_w,lr_best_gru_w,D_hidden_best_gru_w)
model_gru_w,l_gru_w=train(training_sequences_w,rnn_type,num_layers_best_gru_w,lr_best_gru_w,epochs,D_hidden_best_gru_w)
plt.plot(l_gru_w)
plt.show()
rmse_gru_w,mape_gru_w,r2_gru_w,figx_gru_w=training_result(target_test_w,target_train_norm_w,model_gru_w,rnn_type)


# In[ ]:


num_layers_best_gru_s,lr_best_gru_s,D_hidden_best_gru_s=tuning(tuning_set_s,rnn_type,num_layers_list,lr_list,epochs,D_hidden_list,24)
print(num_layers_best_gru_s,lr_best_gru_s,D_hidden_best_gru_s)
model_gru_s,l_gru_s=train(training_sequences_s,rnn_type,num_layers_best_gru_s,lr_best_gru_s,epochs,D_hidden_best_gru_s)
plt.plot(l_gru_s)
plt.show()
rmse_gru_s,mape_gru_s,r2_gru_s,figx_gru_s=training_result(target_test_s,target_train_norm_s,model_gru_s,rnn_type)

