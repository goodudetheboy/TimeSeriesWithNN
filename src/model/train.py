import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

def split_data(stock, lookback):
    data_raw = stock# convert to numpy array
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback+1])
    
    data = np.array(data);
    test_set_size = int(np.round(0.2*data.shape[0]));
    train_set_size = data.shape[0] - (test_set_size);
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]
    
    return [x_train, y_train, x_test, y_test]

def train(train_loader, test_loader, learn_rate, network, model_type, batch_size=32, hidden_dim=64, EPOCHS=20, loss_func="mse"):
    # Setting common hyperparameters
    input_dim = 1
    output_dim = 1
    n_layers = 1
    
    # Instantiating the models
    model = network(input_dim, hidden_dim, output_dim, n_layers)
    
    # Defining loss function and optimizer
    train_loss = []
    test_loss = []
    if (loss_func == "l1"):
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    model.train()
    print("Starting Training of {} model".format(model_type))
    epoch_times = []
    
    # Start training loop
    for epoch in range(1,EPOCHS+1):
        start_time = time.perf_counter()
        h = model.init_hidden(batch_size)
        avg_loss = 0.
        for x, label in train_loader:
            h = h.data
            model.zero_grad()
            
            out, h = model.forward(x.float(), h)
            loss = criterion(out, label.float())
            loss.backward()
            
            optimizer.step()
            avg_loss += loss.item()
    
        length = len(train_loader)
        avg_loss /= length
        train_loss.append(avg_loss)
        print("Epoch {}/{} Done, Total Training {} Loss: {}".format(epoch, EPOCHS, loss_func.upper(), avg_loss))
        
        predictions = []
        values = []
        
        h = model.init_hidden(1)
        
        #define loss function criterion
        if(loss_func == "l1"):
            criterion = nn.L1Loss()
        else:
            #default
            criterion = nn.MSELoss()
        
        for x, label in test_loader:
            length = len(test_loader)
            avg_loss2 = 0.
            h = h.data
            model.zero_grad()
            
            out, h = model.forward(x.float(), h)
            loss2 = criterion(out, label.float()).item()
            avg_loss2 += loss2
            
            
            predictions.append(out.detach().numpy().reshape(-1))
            values.append(label.numpy().reshape(-1))
            
        test_loss.append(avg_loss2)
        print("Total Testing {} Loss: {}".format(loss_func.upper(), avg_loss2))
        
        current_time = time.perf_counter()
        epoch_times.append(current_time-start_time)
        print("Total Time Elapsed: {} seconds".format(str(current_time-start_time)))
        print()
    
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    return model, train_loss, test_loss, predictions, values

def evaluate(model, test_loader):
    with torch.no_grad():
        predictions = []
        values = []
        h = model.init_hidden(1)
        criterion1 = nn.MSELoss()
        criterion2 = nn.L1Loss()
        
        loss1 = 0
        loss2 = 0
        for x, label in test_loader:
            h = h.data
            model.zero_grad()
    
            out, h = model.forward(x.float(), h)
            loss1 = criterion1(out, label.float()).item()
            loss2 = criterion2(out, label.float()).item()
            predictions.append(out.numpy().reshape(-1))
            values.append(label.numpy().reshape(-1))
            
        print("Total MSELoss: {}".format(loss1))
        print("Total L1Loss: {}".format(loss2))
        
            
    return predictions, values, loss1,loss2

def plot(predictions, values, dataset_name):
    plt.figure(figsize=(14,10))
    plt.plot(np.asarray(predictions).reshape(-1,1), "-o", color="g", label="Predicted")
    plt.plot(np.asarray(values).reshape(-1,1), color="b", label="Actual")
    plt.title('Predicted and actual values on {}'.format(dataset_name))
    plt.ylabel(dataset_name)
    plt.xlabel('Time Points')
    plt.legend()

def plot_loss(loss_train, loss_test, dataset_name, isL1):
    loss_name = "L1" if isL1 else "MSE"
    plt.figure(figsize=(6, 4))
    plt.plot(loss_train, color="g", label="Training")
    plt.plot(loss_test, color="b", label="Testing")
    plt.title('Plot of train and test {} loss vs iterations for {}'.format(loss_name, dataset_name))
    plt.xlabel('Epochs')
    plt.ylabel('L1 Loss' if isL1 else 'MSE Loss')
    plt.legend()

def run(dataset_df, column_i, dataset_name, date_column, network, model_type, batch_size=32):
    dataset_df[date_column] = pd.to_datetime(dataset_df[date_column])
    dataset_df.set_index(date_column,inplace=True)
    dataset_df = pd.DataFrame(dataset_df[dataset_df.columns[column_i-1]])
    dataset_df.dropna(inplace=True)

    # Scaling the input data
    sc = MinMaxScaler()
    #label_sc = MinMaxScaler()
    scaled_data = sc.fit_transform(dataset_df.values)
    # Obtaining the Scale for the labels(usage data) so that output can be re-scaled to actual value during evaluation
    #label_sc.fit(microsoft_df.iloc[:,0].values.reshape(-1,1))
    lookback = 5
    train_x,train_y,test_x,test_y = split_data(scaled_data, lookback)

    # Print data shape
    print('X_train.shape: ', train_x.shape)
    print('y_train.shape: ', train_y.shape)
    print('X_test.shape: ', test_x.shape) 
    print('y_test.shape: ', test_y.shape)

    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, drop_last=True)
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
    test_loader = DataLoader(test_data, shuffle=False, batch_size=1, drop_last=True)

    lr = 0.001
    model, mseloss_train, mseloss_test, predictions, values = train(train_loader, test_loader, lr, network, model_type, batch_size=batch_size, EPOCHS=50, loss_func="mse")
    model, l1loss_train, l1loss_test, predictions_l1, values_l1 = train(train_loader, test_loader, lr, network, model_type, batch_size=batch_size, EPOCHS=50, loss_func="l1")

    plot(predictions, values, dataset_name)
    plot_loss(mseloss_train, mseloss_test, dataset_name, False)
    plot_loss(l1loss_train, l1loss_test, dataset_name, True)
