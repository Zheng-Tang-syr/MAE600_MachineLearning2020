import pandas as pd
from matplotlib import pyplot as plt
import torch
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('MidtermProjectData.csv', index_col=0)
pd.set_option('display.max_columns', None)
print(df.shape)
print(df.describe())

# data cleaning
index = ["Pressure, mbar", "Temperature, *C", "RH, %",
         "Dew Point, *C", "Solar Radiation, W/m^2", "Wind Speed, m/s",
         "Gust Speed, m/s, ", "Wind Direction, *", "Meter Reading, W"]
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

data = np.array(new_data_array[:, 0:8])
target = np.array(new_data_array[:, 8])
target = target.reshape(-1, 1)

# Relationship between features and the target: Before splitting data
fig_x_y = plt.figure()
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.plot(data[:, i], target[:, 0], 'o')
    plt.xlabel(index[i])
    plt.ylabel('target')
plt.show()

# Splitting data into weekdays and weekends
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

# Plotting the energy consumption difference between weekdays and weekends
target_weekday = weekday[:, 8]
target_weekday = target_weekday.reshape(-1, 1)
target_weekend = weekend[:, 8]
target_weekend = target_weekend.reshape(-1, 1)
target_weekday_new = np.delete(target_weekday, slice(0, 96), 0)
target_weekend_new = np.delete(target_weekend, slice(0, 48), 0)
x=len(target_weekday_new)/120
x = int(x)
mean_weekday = np.zeros(x)
y = len(target_weekend_new)/48
y = int(y)
mean_weekend = np.zeros(y)
j = 0
for i in range(x):
    a = 0
    for k in range(120):
        a += target_weekday_new[j]
        j += 1
    mean_weekday[i] = a/120
j = 0
for i in range(y):
    a = 0
    for k in range(48):
        a += target_weekend_new[j]
        j += 1
    mean_weekend[i] = a/48
fig_day_end = plt.figure()
plt.plot(range(len(mean_weekday)), mean_weekday, '--', label='weekday', alpha=0.5)
plt.plot(range(len(mean_weekend)), mean_weekend, '--', label='weekend', alpha=0.5)
plt.legend(loc='best')
plt.show()

# Splitting into winter and summer
w_day = weekday[:1416]      # winter weekday
s_day = weekday[1416:]      # summer weekday
w_end = weekend[:576]      # winter weekend
s_end = weekend[576:]      # summer weekend

data_w_day = w_day[:, 0:8]
target_w_day = w_day[:, 8]
target_w_day = target_w_day.reshape(-1, 1)

data_s_day = s_day[:, 0:8]
target_s_day = s_day[:, 8]
target_s_day = target_s_day.reshape(-1, 1)

data_w_end = w_end[:, 0:8]
target_w_end = w_end[:, 8]
target_w_end = target_w_end.reshape(-1, 1)

data_s_end = s_end[:, 0:8]
target_s_end = s_end[:, 8]
target_s_end = target_s_end.reshape(-1, 1)

# cleaning outliers
def remove_outlier(data, target):
    outlier_idex = []
    for i in range(target.shape[0]):
        sta = (target[i] - np.mean(target)) / np.std(target)
        if abs(sta) > 3:
            outlier_idex.append(i)
    new_data = np.delete(data, outlier_idex, axis=0)
    new_target = np.delete(target, outlier_idex, axis=0)
    return new_data, new_target

data_w_day, target_w_day = remove_outlier(data_w_day, target_w_day)
data_s_day, target_s_day = remove_outlier(data_s_day, target_s_day)
data_w_end, target_w_end = remove_outlier(data_w_end, target_w_end)
data_s_end, target_s_end = remove_outlier(data_s_end, target_s_end)

# Relationship between features and the target: After splitting data
fig_x_y_w_day = plt.figure()
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.plot(data_w_day[:, i], target_w_day[:, 0], 'o')
    plt.xlabel(index[i])
    plt.ylabel('target')
plt.show()

fig_x_y_s_day = plt.figure()
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.plot(data_s_day[:, i], target_s_day[:, 0], 'o')
    plt.xlabel(index[i])
    plt.ylabel('target')
plt.show()

fig_x_y_w_end = plt.figure()
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.plot(data_w_end[:, i], target_w_end[:, 0], 'o')
    plt.xlabel(index[i])
    plt.ylabel('target')
plt.show()

fig_x_y_s_end = plt.figure()
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.plot(data_s_end[:, i], target_s_end[:, 0], 'o')
    plt.xlabel(index[i])
    plt.ylabel('target')
plt.show()

# Calculating correlation coefficients and neglecting features with very low correlation coefficient
remove_w_day = []
remove_s_day = []
remove_w_end = []
remove_s_end = []
print('winter weekday correlation coefficient between features and target')
for i in range(8):
    print(index[i], end="  ")
    r = np.corrcoef(data_w_day[:, i], target_w_day[:, 0])[0, 1]
    if abs(r) < 0.1:
        remove_w_day.append(i)
    print(r)
print("")

print('summer weekday correlation coefficient between features and target')
for i in range(8):
    print(index[i], end="  ")
    r = np.corrcoef(data_s_day[:, i], target_s_day[:, 0])[0, 1]
    if abs(r) < 0.1:
        remove_s_day.append(i)
    print(r)
print("")

print('winter weekend correlation coefficient between features and target')
for i in range(8):
    print(index[i], end="  ")
    r = np.corrcoef(data_w_end[:, i], target_w_end[:, 0])[0, 1]
    if abs(r) < 0.1:
        remove_w_end.append(i)
    print(r)
print("")

print('summer weekend correlation coefficient between features and target')
for i in range(8):
    print(index[i], end="  ")
    r = np.corrcoef(data_s_end[:, i], target_s_end[:, 0])[0, 1]
    if abs(r) < 0.1:
        remove_s_end.append(i)
    print(r)
print("")



data_w_day = np.delete(data_w_day, remove_w_day, axis=1)
data_s_day = np.delete(data_s_day, remove_s_day, axis=1)
data_w_end = np.delete(data_w_end, remove_w_end, axis=1)
data_s_end = np.delete(data_s_end, remove_s_end, axis=1)

# Splitting into training and testing set
test_size_day = 120
test_size_end = 48
X_train_w_day = data_w_day[:-test_size_day]
Y_train_w_day = target_w_day[:-test_size_day]
X_test_w_day = data_w_day[-test_size_day:]
Y_test_w_day = target_w_day[-test_size_day:]

X_train_s_day = data_s_day[:-test_size_day]
Y_train_s_day = target_s_day[:-test_size_day]
X_test_s_day = data_s_day[-test_size_day:]
Y_test_s_day = target_s_day[-test_size_day:]

X_train_w_end = data_w_end[:-test_size_end]
Y_train_w_end = target_w_end[:-test_size_end]
X_test_w_end = data_w_end[-test_size_end:]
Y_test_w_end = target_w_end[-test_size_end:]

X_train_s_end = data_s_end[:-test_size_end]
Y_train_s_end = target_s_end[:-test_size_end]
X_test_s_end = data_s_end[-test_size_end:]
Y_test_s_end = target_s_end[-test_size_end:]

# Normalize data
# scaler = MinMaxScaler(feature_range=(0, 1))
# X_train_w_day = scaler.fit_transform(X_train_w_day)
# X_test_w_day = scaler.fit_transform(X_test_w_day)
# X_train_s_day = scaler.fit_transform(X_train_s_day)
# X_test_s_day = scaler.fit_transform(X_test_s_day)
#
# X_train_w_end = scaler.fit_transform(X_train_w_end)
# X_test_w_end = scaler.fit_transform(X_test_w_end)
# X_train_s_end = scaler.fit_transform(X_train_s_end)
# X_test_s_end = scaler.fit_transform(X_test_s_end)

# Define a network
class ThreeLayerNet(torch.nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(ThreeLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, D_out)


    def forward(self, x):
        h1_relu = self.linear1(x).clamp(min=0)
        h2_relu = self.linear2(h1_relu).clamp(min=0)
        y_pred = self.linear3(h2_relu)
        return y_pred


# Training function
def train(X_train, Y_train, H1, H2, learning_rate, epochs):
    model = ThreeLayerNet(X_train.shape[1], H1, H2, Y_train.shape[1])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        inputs = torch.from_numpy(X_train).float()
        labels = torch.from_numpy(Y_train).float()

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

    return model

def kfold_CV(X, Y, K, H1, H2, learning_rate, epochs):
    hidden1 = H1
    hidden2 = H2
    lr = learning_rate

    kf = KFold(n_splits=K, shuffle=False)
    mape_trn_cv, mape_tst_cv, r2_cv, mbe_cv= np.empty(0), np.empty(0), np.empty(0), np.empty(0)

    for trn_idx, tst_idx in kf.split(X):
        X_train, X_test = X[trn_idx, :], X[tst_idx, :]
        Y_train, Y_test = Y[trn_idx, :], Y[tst_idx, :]

        modelK = train(X_train=X_train, Y_train=Y_train, H1=hidden1, H2=hidden2, learning_rate=lr, epochs=epochs)
        with torch.no_grad():
            yhat_trn = modelK(torch.from_numpy(X_train).float()).numpy()
            yhat_tst = modelK(torch.from_numpy(X_test).float()).numpy()

        # rmse_trn = np.sqrt(np.mean((yhat_trn - Y_train) ** 2))
        # rmse_tst = np.sqrt(np.mean((yhat_tst - Y_test) ** 2))
        mape_trn = np.mean(np.abs(yhat_trn - Y_train) / Y_train)
        mape_tst = np.mean(np.abs(yhat_tst - Y_test) / Y_test)
        mbe = np.mean(yhat_tst - Y_test)
        r2 = np.corrcoef(yhat_tst.squeeze(), Y_test.squeeze())[0, 1]**2


        # print(yhat_tst.squeeze(), Y_test.squeeze())
        # print(np.corrcoef(yhat_tst.squeeze(), Y_test.squeeze()))
        # print(rmse_trn, rmse_tst, r2)

        mape_trn_cv = np.append(mape_trn_cv, mape_trn)
        mape_tst_cv = np.append(mape_tst_cv, mape_tst)
        r2_cv = np.append(r2_cv, r2)
        mbe_cv = np.append(mbe_cv, mbe)

    return mape_trn_cv.mean(), mape_tst_cv.mean(), np.nanmean(r2_cv), mbe_cv.mean()

def tuning(X_tuning, Y_tuning, epochs, H1_list, H2_list, lr_list):
    mape_trn = np.zeros((len(H1_list), len(H2_list), len(lr_list)))
    mape_tst = np.zeros_like(mape_trn)
    r2_t = np.zeros_like(mape_trn)
    mbe_t = np.zeros_like(mape_trn)
    for h1, H1 in enumerate(H1_list):
        for l, lr in enumerate(lr_list):
            for h2, H2 in enumerate(H2_list):
                trn_val, tst_val, r2, mbe = kfold_CV(X_tuning, Y_tuning, 5, H1, H2, lr, epochs)
                mape_trn[h1, h2, l] = trn_val
                mape_tst[h1, h2, l] = tst_val
                r2_t[h1, h2, l] = r2
                mbe_t[h1, h2, l] = mbe
                # print('H1 = {}, H2 = {}, lr = {}: Training MAPE = {}, Testing MAPE = {} R2 = {} MBE = {}'.format(H1, H2, lr, trn_val, tst_val, r2, mbe))

    i, j, k = np.argwhere(mape_tst == np.min(mape_tst))[0]
    H1_best, H2_best, lr_best = H1_list[i], H2_list[j], lr_list[k]
    mape_b, r2_b, mbe_b = mape_tst[i, j, k], r2_t[i, j, k], mbe_t[i, j, k]
    print('H1_best = {}, H2_best = {}, lr_best = {}'.format(H1_best, H2_best, lr_best))
    print('MAPE = {}, R2 = {}, MBE = {}'.format(mape_b, r2_b, mbe_b))

    return H1_best, H2_best, lr_best, mape_trn, mape_tst, i, j, k

def plot_results(Y_test, y_predicted, title):
    figx = plt.figure()
    plt.plot(range(len(Y_test)), Y_test, 'go', label='True data', alpha=0.5)
    plt.plot(range(len(Y_test)), y_predicted, '--', label='Predictions', alpha=0.5)
    plt.legend(loc='best')
    plt.title(title)
    plt.show()
    return figx

def error_metrics(Y_test, y_pred):
    # RMSE, MAPE, MAE, MBE, R2
    rmse = np.sqrt(np.mean((y_pred - Y_test) ** 2))
    mape = np.mean(np.abs(y_pred - Y_test) / Y_test)
    mae = np.mean(np.abs(y_pred - Y_test))
    mbe = np.mean(y_pred - Y_test)
    r2 = np.corrcoef(y_pred.squeeze(), Y_test.squeeze())[0, 1] ** 2
    return rmse, mape, mae, mbe, r2

def tuning_results(mape_trn, mape_tst, i, j, k, H1_list, H2_list):
    lr_list = list(range(-3, 0))
    H1_mape_trn = mape_trn[:, j, k]
    H1_mape_tst = mape_tst[:, j, k]
    H2_mape_trn = mape_trn[i, :, k]
    H2_mape_tst = mape_tst[i, :, k]
    lr_mape_trn = mape_trn[i, j, :]
    lr_mape_tst = mape_tst[i, j, :]

    fig_tuning = plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(H1_list, H1_mape_trn, label='Training MAPE')
    plt.plot(H1_list, H1_mape_tst, label='Validation MAPE')
    plt.xlabel('H1')
    plt.ylabel('MAPE')
    plt.legend()
    plt.title('H1 neurons')

    plt.subplot(1, 3, 2)
    plt.plot(H2_list, H2_mape_trn, label='Training MAPE')
    plt.plot(H2_list, H2_mape_tst, label='Validation MAPE')
    plt.xlabel('H2')
    plt.ylabel('MAPE')
    plt.legend()
    plt.title('H2 neurons')

    plt.subplot(1, 3, 3)
    plt.plot(lr_list, lr_mape_trn, label='Training MAPE')
    plt.plot(lr_list, lr_mape_tst, label='Validation MAPE')
    plt.xlabel('learning rate (log)')
    plt.ylabel('MAPE')
    plt.legend()
    plt.title('Learning rate')
    plt.show()
    return fig_tuning


epochs_weekday = 5000
epochs_weekend = 20000
print('Cross validation before tuning')
trn_val_w_day1, tst_val_w_day1, r2_w_day1, mbe_w_day1 = kfold_CV(X_train_w_day, Y_train_w_day, 5, 5, 5, 0.01, epochs_weekday)
print('Winter weekdays: MAPE = {}, R2 = {}, MBE = {}'.format(tst_val_w_day1, r2_w_day1, mbe_w_day1))
trn_val_s_day1, tst_val_s_day1, r2_s_day1, mbe_s_day1 = kfold_CV(X_train_s_day, Y_train_s_day, 5, 5, 5, 0.01, epochs_weekday)
print('Summer weekdays: MAPE = {}, R2 = {}, MBE = {}'.format(tst_val_s_day1, r2_s_day1, mbe_s_day1))

trn_val_w_end1, tst_val_w_end1, r2_w_end1, mbe_w_end1 = kfold_CV(X_train_w_end, Y_train_w_end, 5, 5, 5, 0.01, epochs_weekend)
print('Winter weekends: MAPE = {}, R2 = {}, MBE = {}'.format(tst_val_w_end1, r2_w_end1, mbe_w_end1))
trn_val_s_end1, tst_val_s_end1, r2_s_end1, mbe_s_end1 = kfold_CV(X_train_s_end, Y_train_s_end, 5, 5, 5, 0.01, epochs_weekend)
print('Summer weekends: MAPE = {}, R2 = {}, MBE = {}'.format(tst_val_s_end1, r2_s_end1, mbe_s_end1))





model1_w_day = train(X_train_w_day, Y_train_w_day, 5, 5, 0.01, epochs_weekday)
with torch.no_grad():
    y_predicted1_w_day = model1_w_day(torch.from_numpy(X_test_w_day).float())
fig_w_day1 = plot_results(Y_test_w_day, y_predicted1_w_day, title='winter weekdays')
# rmse_w_day, mape_w_day, mae_w_day, mbe_w_day, r2_w_day = error_metrics(Y_test_w_day, y_predicted1_w_day.numpy())
# print('rmse = ', rmse_w_day)
# print('mape = ', mape_w_day)
# print('mae = ', mae_w_day)
# print('mbe = ', mbe_w_day)
# print('r2 = ', r2_w_day)

model1_s_day = train(X_train_s_day, Y_train_s_day, 5, 5, 0.01, epochs_weekday)
with torch.no_grad():
    y_predicted1_s_day = model1_s_day(torch.from_numpy(X_test_s_day).float())
fig_s_day1 = plot_results(Y_test_s_day, y_predicted1_s_day, title='summer weekdays')
# rmse_s_day, mape_s_day, mae_s_day, mbe_s_day, r2_s_day = error_metrics(Y_test_s_day, y_predicted1_s_day.numpy())
# print('rmse = ', rmse_s_day)
# print('mape = ', mape_s_day)
# print('mae = ', mae_s_day)
# print('mbe = ', mbe_s_day)
# print('r2 = ', r2_s_day)

model1_w_end = train(X_train_w_end, Y_train_w_end, 5, 5, 0.01, epochs_weekend)
with torch.no_grad():
    y_predicted1_w_end = model1_w_end(torch.from_numpy(X_test_w_end).float())
fig_w_end1 = plot_results(Y_test_w_end, y_predicted1_w_end, title='winter weekends')
# rmse_w_end, mape_w_end, mae_w_end, mbe_w_end, r2_w_end = error_metrics(Y_test_w_end, y_predicted1_w_end.numpy())
# print('rmse = ', rmse_w_end)
# print('mape = ', mape_w_end)
# print('mae = ', mae_w_end)
# print('mbe = ', mbe_w_end)
# print('r2 = ', r2_w_end)

model1_s_end = train(X_train_s_end, Y_train_s_end, 5, 5, 0.01, epochs_weekend)
with torch.no_grad():
    y_predicted1_s_end = model1_s_end(torch.from_numpy(X_test_s_end).float())
fig_s_end1 = plot_results(Y_test_s_end, y_predicted1_s_end, title='summer weekends')
# rmse_s_end, mape_s_end, mae_s_end, mbe_s_end, r2_s_end = error_metrics(Y_test_s_end, y_predicted1_s_end.numpy())
# print('rmse = ', rmse_s_end)
# print('mape = ', mape_s_end)
# print('mae = ', mae_s_end)
# print('mbe = ', mbe_s_end)
# print('r2 = ', r2_s_end)

# Hyperparameter tuning: Grid Search
tuning_len = 200
X_tuning_w_day = X_train_w_day[:tuning_len]
Y_tuning_w_day = Y_train_w_day[:tuning_len]
X_tuning_s_day = X_train_s_day[:tuning_len]
Y_tuning_s_day = Y_train_s_day[:tuning_len]

X_tuning_w_end = X_train_w_end[:tuning_len]
Y_tuning_w_end = Y_train_w_end[:tuning_len]
X_tuning_s_end = X_train_s_end[:tuning_len]
Y_tuning_s_end = Y_train_s_end[:tuning_len]

print('hyperparameters tuning: ')
H1_list_w_day = list(range(3, 16))
H2_list_w_day = list(range(3, 16))
lr_list = [1e-3, 1e-2, 1e-1]

print('winter weekdays')
H1_best_w_day, H2_best_w_day, lr_best_w_day, mape_trn_w_day, mape_tst_w_day, i_w_day, j_w_day, k_w_day = tuning(X_tuning_w_day, Y_tuning_w_day, epochs_weekday, H1_list_w_day, H2_list_w_day, lr_list)

H1_list_s_day = list(range(2, 16))
H2_list_s_day = list(range(2, 16))
print('summer weekdays')
H1_best_s_day, H2_best_s_day, lr_best_s_day, mape_trn_s_day, mape_tst_s_day, i_s_day, j_s_day, k_s_day = tuning(X_tuning_s_day, Y_tuning_s_day, epochs_weekday, H1_list_s_day, H2_list_s_day, lr_list)

H1_list_w_end = list(range(5, 16))
H2_list_w_end = list(range(5, 16))
print('winter weekends')
H1_best_w_end, H2_best_w_end, lr_best_w_end, mape_trn_w_end, mape_tst_w_end, i_w_end, j_w_end, k_w_end = tuning(X_tuning_w_end, Y_tuning_w_end, epochs_weekend, H1_list_w_end, H2_list_w_end, lr_list)

H1_list_s_end = list(range(2, 12))
H2_list_s_end = list(range(2, 12))
print('summer weekends')
H1_best_s_end, H2_best_s_end, lr_best_s_end, mape_trn_s_end, mape_tst_s_end, i_s_end, j_s_end, k_s_end = tuning(X_tuning_s_end, Y_tuning_s_end, epochs_weekend, H1_list_s_end, H2_list_s_end, lr_list)

fig_tuning_w_day = tuning_results(mape_trn_w_day, mape_tst_w_day, i_w_day, j_w_day, k_w_day, H1_list_w_day, H2_list_w_day)
fig_tuning_s_day = tuning_results(mape_trn_s_day, mape_tst_s_day, i_s_day, j_s_day, k_s_day, H1_list_s_day, H2_list_s_day)
fig_tuning_w_end = tuning_results(mape_trn_w_end, mape_tst_w_end, i_w_end, j_w_end, k_w_end, H1_list_w_end, H2_list_w_end)
fig_tuning_s_end = tuning_results(mape_trn_s_end, mape_tst_s_end, i_s_end, j_s_end, k_s_end, H1_list_s_end, H2_list_s_end)

# Cross validation after tuning
print('Cross validation after tuning')
trn_val_w_day2, tst_val_w_day2, r2_w_day2, mbe_w_day2 = kfold_CV(X_train_w_day, Y_train_w_day, 5, H1_best_w_day, H2_best_w_day, lr_best_w_day, epochs_weekday)
print('Winter weekdays: MAPE = {}, R2 = {}, MBE = {}'.format(tst_val_w_day2, r2_w_day2, mbe_w_day2))
trn_val_s_day2, tst_val_s_day2, r2_s_day2, mbe_s_day2 = kfold_CV(X_train_s_day, Y_train_s_day, 5, H1_best_s_day, H2_best_s_day, lr_best_s_day, epochs_weekday)
print('Summer weekdays: MAPE = {}, R2 = {}, MBE = {}'.format(tst_val_s_day2, r2_s_day2, mbe_s_day2))

trn_val_w_end2, tst_val_w_end2, r2_w_end2, mbe_w_end2 = kfold_CV(X_train_w_end, Y_train_w_end, 5, H1_best_w_end, H2_best_w_end, lr_best_w_end, epochs_weekend)
print('Winter weekends: MAPE = {}, R2 = {}, MBE = {}'.format(tst_val_w_end2, r2_w_end2, mbe_w_end2))
trn_val_s_end2, tst_val_s_end2, r2_s_end2, mbe_s_end2 = kfold_CV(X_train_s_end, Y_train_s_end, 5, H1_best_s_end, H2_best_s_end, lr_best_s_end, epochs_weekend)
print('Summer weekends: MAPE = {}, R2 = {}, MBE = {}'.format(tst_val_s_end2, r2_s_end2, mbe_s_end2))

# Train and test after tuning
# winter weekday
model_w_day2 = train(X_train_w_day, Y_train_w_day, H1_best_w_day, H2_best_w_day, lr_best_w_day, epochs_weekday)
with torch.no_grad():
    y_predicted_w_day2 = model_w_day2(torch.from_numpy(X_test_w_day).float())
fig_w_day2 = plot_results(Y_test_w_day, y_predicted_w_day2, title='winter weekdays')
# rmse_w_day2, mape_w_day2, mae_w_day2, mbe_w_day2, r2_w_day2 = error_metrics(Y_test_w_day, y_predicted_w_day2.numpy())
# print('rmse_w_day = ', rmse_w_day2)
# print('mape_w_day = ', mape_w_day2)
# print('mae_w_day = ', mae_w_day2)
# print('mbe_w_day = ', mbe_w_day2)
# print('r2_w_day = ', r2_w_day2)

# summer weekday
model_s_day2 = train(X_train_s_day, Y_train_s_day, H1_best_s_day, H2_best_s_day, lr_best_s_day, epochs_weekday)
with torch.no_grad():
    y_predicted_s_day2 = model_s_day2(torch.from_numpy(X_test_s_day).float())
fig_s_day2 = plot_results(Y_test_s_day, y_predicted_s_day2, title='summer weekdays')
# rmse_s_day2, mape_s_day2, mae_s_day2, mbe_s_day2, r2_s_day2 = error_metrics(Y_test_s_day, y_predicted_s_day2.numpy())
# print('rmse_s_day = ', rmse_s_day2)
# print('mape_s_day = ', mape_s_day2)
# print('mae_s_day = ', mae_s_day2)
# print('mbe_s_day = ', mbe_s_day2)
# print('r2_s_day = ', r2_s_day2)

# winter weekend
model_w_end2 = train(X_train_w_end, Y_train_w_end, H1_best_w_end, H2_best_w_end, lr_best_w_end, epochs_weekend)
with torch.no_grad():
    y_predicted_w_end2 = model_w_end2(torch.from_numpy(X_test_w_end).float())
fig_w_end2 = plot_results(Y_test_w_end, y_predicted_w_end2, title='winter weekends')
# rmse_w_end2, mape_w_end2, mae_w_end2, mbe_w_end2, r2_w_end2 = error_metrics(Y_test_w_end, y_predicted_w_end2.numpy())
# print('rmse_w_end = ', rmse_w_end2)
# print('mape_w_end = ', mape_w_end2)
# print('mae_w_end = ', mae_w_end2)
# print('mbe_w_end = ', mbe_w_end2)
# print('r2_w_end = ', r2_w_end2)

# summer weekend
model_s_end2 = train(X_train_s_end, Y_train_s_end, H1_best_s_end, H2_best_s_end, lr_best_s_end, epochs_weekend)
with torch.no_grad():
    y_predicted_s_end2 = model_s_end2(torch.from_numpy(X_test_s_end).float())
fig_s_end2 = plot_results(Y_test_s_end, y_predicted_s_end2, title='summer weekends')
# rmse_s_end2, mape_s_end2, mae_s_end2, mbe_s_end2, r2_s_end2 = error_metrics(Y_test_s_end, y_predicted_s_end2.numpy())
# print('rmse_s_end = ', rmse_s_end2)
# print('mape_s_end = ', mape_s_end2)
# print('mae_s_end = ', mae_s_end2)
# print('mbe_s_end = ', mbe_s_end2)
# print('r2_s_end = ', r2_s_end2)




