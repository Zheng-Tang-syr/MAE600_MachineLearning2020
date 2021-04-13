#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from math import floor, ceil


# In[2]:


MidtermProjectData=open(r'MidtermProjectData.csv')
df = pd.read_csv(MidtermProjectData,index_col=0)


# In[4]:


print(df.shape)
print(df.describe())


# In[5]:


index = ["Pressure, mbar, SATC Rooftop", "Temperature, *C, SATC Rooftop", "RH, %, SATC Rooftop",
         "Dew Point, *C, SATC Rooftop", "Solar Radiation, W/m^2, SATC Rooftop", "Wind Speed, m/s, SATC Rooftop",
         "Gust Speed, m/s, SATC Rooftop", "Wind Direction, *, SATC Rooftop", "Meter Reading, W"]

# for i in range(9):
#     sta = (df[index[i]] - df[index[i]].mean()) / df[index[i]].std()
#     df.drop(df[sta.abs() > 3].index, inplace=True)
#     print(df.shape)

df.dropna(inplace=True)

data_array = np.array(df, dtype='float64')

time_index = np.arange(0, len(data_array), 12)
new_data_array = data_array[time_index]



data = np.array(new_data_array[:, 0:8])
target = np.array(new_data_array[:, 8])
target = target.reshape(-1, 1)


# In[6]:


remove = []
for i in range(8):
    print(index[i])
    r2 = np.corrcoef(data[:, i], target[:, 0])[0, 1]
    print(r2)

    if abs(r2)<0.1:
        remove.append(i)


# In[8]:


new_data_array = np.delete(new_data_array, remove, axis=1)
# index.remove("Temperature, *C, SATC Rooftop")
# index.remove( "Dew Point, *C, SATC Rooftop")
# index.remove("Wind Direction, *, SATC Rooftop")
# print(index)


# In[9]:



#split data into weekdays and weekends
weekday = new_data_array[:96]
weekend = new_data_array[96:144]
j = 0
for i in range(144, len(new_data_array)):
    a = new_data_array[i]
    a = a.reshape(1, -1)
    if j <120:
        weekday = np.append(weekday, a, axis=0)
    else:
        weekend = np.append(weekend, a, axis=0)
    j += 1
    if j > 167:
        j = 0


# In[10]:


test_size_weekday = 2700
test_size_weekend = 1000


# In[11]:



target_weekday = np.array(weekday[:, 5])
target_weekday = target_weekday.reshape(-1, 1)

target_weekend = np.array(weekend[:, 5])
target_weekend = target_weekend.reshape(-1, 1)

target_train_weekday = target_weekday[:test_size_weekday]
target_test_weekday = target_weekday[test_size_weekday:]

target_train_weekend = target_weekend[:test_size_weekend]
target_test_weekend = target_weekend[test_size_weekend:]


# In[12]:


target_test_weekend.shape


# In[ ]:


scaler1 = MinMaxScaler(feature_range=(-1, 1))
scaler2 = MinMaxScaler(feature_range=(-1, 1))
target_train_norm_weekday = scaler1.fit_transform(target_train_weekday.reshape(-1, 1))
target_train_norm_weekend = scaler2.fit_transform(target_train_weekend.reshape(-1, 1))


# In[36]:


target_train_norm_weekday = torch.FloatTensor(target_train_norm_weekday)
target_train_norm_weekend = torch.FloatTensor(target_train_norm_weekend)


# In[37]:


def make_sequences(input_data, sl):
    inout_seq = []
    L = len(input_data)
    for i in range(L-sl):
        train_seq = input_data[i:i+sl]
        train_label = input_data[i+sl:i+sl+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


# In[38]:


seq_len = 96
training_sequences_weekday = make_sequences(target_train_norm_weekday, seq_len)
training_sequences_weekend = make_sequences(target_train_norm_weekend, seq_len)


# In[25]:


class RNN(torch.nn.Module):
    def __init__(self, D_in, D_hidden, D_out, nLayers=1):
        super().__init__()
        self.D_hidden = D_hidden
        self.nLayers = nLayers
        self.rnn = torch.nn.RNN(D_in, D_hidden, num_layers=nLayers)
        self.linear = torch.nn.Linear(D_hidden, D_out)
        self.hidden = torch.zeros(nLayers, 1, D_hidden)

    def forward(self, input_seq):
        rnn_out, self.hidden = self.rnn(input_seq.view(len(input_seq), 1, -1), self.hidden)
        y_pred = self.linear(rnn_out.view(len(input_seq), 1, -1))
        return y_pred[-1]


# In[26]:


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


# In[27]:


def train(training_sequences, rnn_type, num_layers, learning_rate=1e-3, epochs=200):
    D_in = training_sequences[0][0].shape[1]
    D_out = training_sequences[0][1].shape[1]
    D_hidden = 50

    if rnn_type.upper() == 'RNN':
        model = RNN(D_in, D_hidden, D_out, nLayers=num_layers)
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

    return model


# In[28]:


def Cross_Validation(target, K, rnn_type, number_layers, learning_rate):

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
        scaler = MinMaxScaler(feature_range=(-1, 1))
        Y_train_norm = scaler.fit_transform(Y_train.reshape(-1, 1))

        Y_train_norm = torch.FloatTensor(Y_train_norm)
        training_sequences = make_sequences(Y_train_norm, 24)


        modelK = train(training_sequences, rnn_type, number_layers, learning_rate)
        pred_len = len(Y_test)
        test_inputs = Y_train_norm[-pred_len:].tolist()
        y_pred_unscaled = test(modelK, rnn_type, test_inputs, pred_len)

        # Inverse scaling
        y_pred = scaler.inverse_transform(np.array(y_pred_unscaled).reshape(-1, 1))

        rmse = np.sqrt(np.mean((y_pred - Y_test) ** 2))
        r2 = np.corrcoef(y_pred.squeeze(), Y_test.squeeze())[0, 1]**2
        mape = np.mean(np.abs(y_pred - Y_test) / Y_test)

        rmse_cv = np.append(rmse_cv, rmse)
        mape_cv = np.append(mape_cv, mape)
        r2_cv = np.append(r2_cv, r2)

    return rmse_cv.mean(), mape_cv.mean(), np.nanmean(r2_cv)


# In[29]:


tuning_set = target[:144]
num_layers_list = list(range(1, 4))
lr_list = [1e-4, 1e-3, 1e-2]


# In[30]:


mape_tst = np.zeros((len(num_layers_list), len(lr_list)))


# In[31]:


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


# In[32]:


for h1, H1 in enumerate(num_layers_list):
    for l, lr in enumerate(lr_list):
        rmse, mape, r2 = Cross_Validation(tuning_set, 6, 'rnn', H1, lr)
        mape_tst[h1, l] = mape
        print('Number Layers = {}, lr = {}: RMSE = {} MAPE = {} r2 = {}'.format(H1, lr, rmse, mape, r2))


# In[33]:


i, j = np.argwhere(mape_tst == np.min(mape_tst))[0]
num_layers_best, lr_best = num_layers_list[i], lr_list[j]


# In[34]:


print(num_layers_best, lr_best)


# In[39]:


rnn_type = 'rnn'
model_weekday = train(training_sequences_weekday, rnn_type, num_layers_best, learning_rate=lr_best)
model_weekend = train(training_sequences_weekend, rnn_type, num_layers_best, learning_rate=lr_best)


# In[45]:


pred_len = len(target_test_weekday)

test_inputs_weekday = target_train_norm_weekday[-pred_len:].tolist()
y_pred_unscaled_weekday = test(model_weekday, rnn_type, test_inputs_weekday, pred_len)

pred_len = len(target_test_weekend)
test_inputs_weekend = target_train_norm_weekend[-pred_len:].tolist()
y_pred_unscaled_weekend = test(model_weekend, rnn_type, test_inputs_weekend, pred_len)


# In[46]:


y_pred_weekday = scaler1.inverse_transform(np.array(y_pred_unscaled_weekday).reshape(-1, 1))
print(y_pred_weekday)

y_pred_weekend = scaler2.inverse_transform(np.array(y_pred_unscaled_weekend).reshape(-1, 1))
print(y_pred_weekend)


# In[48]:
target_test_weekday = target_test_weekday.reshape(-1, 1)
rmse = np.sqrt(np.mean((y_pred_weekday - target_test_weekday) ** 2))
mape = np.mean(np.abs((y_pred_weekday - target_test_weekday) / target_test_weekday))
r2 = np.corrcoef(y_pred_weekday.squeeze(), target_test_weekday.squeeze())[0, 1] ** 2

print('rmse = ', rmse)
print('mape = ', mape)
print('r2 = ', r2)


target_test_weekend = target_test_weekend.reshape(-1, 1)
rmse = np.sqrt(np.mean((y_pred_weekend - target_test_weekend) ** 2))
mape = np.mean(np.abs((y_pred_weekend - target_test_weekend) / target_test_weekend))
r2 = np.corrcoef(y_pred_weekend.squeeze(), target_test_weekend.squeeze())[0, 1] ** 2

print('rmse = ', rmse)
print('mape = ', mape)
print('r2 = ', r2)


# In[49]:


def plot_results(Y_test, y_predicted):
    figx = plt.figure()
    plt.plot(range(len(Y_test)), Y_test, 'go', label='True data', alpha=0.5)
    plt.plot(range(len(Y_test)), y_predicted, '--', label='Predictions', alpha=0.5)
    plt.legend(loc='best')
    plt.show()
    return figx


# In[50]:

fig_RNN_weekday = plot_results(target_test_weekday, y_pred_weekday)
fig_RNN_weekend = plot_results(target_test_weekend, y_pred_weekend)


# In[51]:


tuning_set = target_test_weekend[:144]
num_layers_list = list(range(1, 4))
lr_list = [1e-4, 1e-3, 1e-2]


# In[52]:


mape_tst = np.zeros((len(num_layers_list), len(lr_list)))


# In[55]:


for h1, H1 in enumerate(num_layers_list):
    for l, lr in enumerate(lr_list):
        rmse, mape, r2 = Cross_Validation(tuning_set, 6, 'lstm', H1, lr)
        mape_tst[h1, l] = mape

        print('Number Layers = {}, lr = {}: RMSE = {} MAPE = {} r2 = {}'.format(H1, lr, rmse, mape, r2))


# In[56]:


i, j = np.argwhere(mape_tst == np.min(mape_tst))[0]
num_layers_best, lr_best = num_layers_list[i], lr_list[j]

print(num_layers_best, lr_best)


# In[58]:


rnn_type = 'lstm'
model_weekday = train(training_sequences_weekday, rnn_type, num_layers_best, learning_rate=lr_best)
model_weekend = train(training_sequences_weekend, rnn_type, num_layers_best, learning_rate=lr_best)



pred_len = len(target_test_weekday)
test_inputs = target_train_norm_weekday[-pred_len:].tolist()
y_pred_unscaled_weekday = test(model_weekday, rnn_type, test_inputs, pred_len)

pred_len = len(target_test_weekend)
test_inputs = target_train_norm_weekend[-pred_len:].tolist()
y_pred_unscaled_weekend = test(model_weekend, rnn_type, test_inputs, pred_len)
# In[59]:
y_pred_weekday = scaler1.inverse_transform(np.array(y_pred_unscaled_weekday).reshape(-1, 1))
print(y_pred_weekday)


y_pred_weekend = scaler2.inverse_transform(np.array(y_pred_unscaled_weekend).reshape(-1, 1))
print(y_pred_weekend)


# In[60]:
target_test_weekday = target_test_weekday.reshape(-1, 1)
rmse = np.sqrt(np.mean((y_pred_weekday - target_test_weekday) ** 2))
mape = np.mean(np.abs((y_pred_weekday - target_test_weekday) / target_test_weekday))
r2 = np.corrcoef(y_pred_weekday.squeeze(), target_test_weekday.squeeze())[0, 1] ** 2

print('rmse = ', rmse)
print('mape = ', mape)
print('r2 = ', r2)

target_test_weekend = target_test_weekend.reshape(-1, 1)
rmse = np.sqrt(np.mean((y_pred_weekend - target_test_weekend) ** 2))
mape = np.mean(np.abs((y_pred_weekend - target_test_weekend) / target_test_weekend))
r2 = np.corrcoef(y_pred_weekend.squeeze(), target_test_weekend.squeeze())[0, 1] ** 2


# In[61]:


print('rmse = ', rmse)
print('mape = ', mape)
print('r2 = ', r2)


# In[62]:
fig_LSTM_weekday = plot_results(target_test_weekday, y_pred_weekday)

fig_LSTM_weekend = plot_results(target_test_weekend, y_pred_weekend)


# In[63]:


len(target_train_weekend)


# In[ ]:





# In[ ]:





# In[ ]:




