import os
import copy
import pickle
import datetime
import pytz
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms

import optuna

exec_date = datetime.datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m%d-%H%M-%S')
spec = 'baseline'
run_name = exec_date + '-' + spec
Path("./result/{}".format(run_name)).mkdir(parents=True,exist_ok=True) #

X_ = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_val.csv')
Y_ = pd.read_csv('data/Y_train.csv')
Y_test = pd.read_csv('data/Y_val.csv')

X_train, X_val, Y_train, Y_val = train_test_split(X_, Y_)


DEVICE = torch.device("cuda:0") #torch.device("cpu")#
BATCHSIZE = 16
DIR = os.getcwd()
EPOCHS = 2000
LOG_INTERVAL = 20
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10
N_TRIAL = 500

def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 2, 5)
    layers = []

    in_features = X_train.shape[1]
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 16, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.Sigmoid())
        p = trial.suggest_float("dropout_l{}".format(i), 0.0, 0.5)
        layers.append(nn.Dropout(p))
        in_features = out_features
    layers.append(nn.Linear(in_features, 2))

    return nn.Sequential(*layers)



def objective(trial):
    # Generate the model.
    model = define_model(trial).to(DEVICE)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the data loader
    train_target = torch.tensor(Y_train['Outcome'].values.astype(np.int32)).type(torch.LongTensor)
    train = torch.tensor(X_train.values.astype(np.float32))
    train_tensor = torch.utils.data.TensorDataset(train, train_target)
    train_loader = torch.utils.data.DataLoader(dataset = train_tensor, batch_size = BATCHSIZE, shuffle = True)

    valid_target = torch.tensor(Y_val['Outcome'].values.astype(np.int32)).type(torch.LongTensor)
    valid = torch.tensor(X_val.values.astype(np.float32))
    valid_tensor = torch.utils.data.TensorDataset(valid, valid_target)
    valid_loader = torch.utils.data.DataLoader(dataset = valid_tensor, batch_size = BATCHSIZE, shuffle = True)

    # Training of the model.
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break

            data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data).squeeze()
            #loss = F.nll_loss(output, target)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        #correct = 0
        total_val_loss = 0.
        n_val = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                # Limiting validation data.
                if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                    break
                data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                output = model(data).squeeze()
                # Get the index of the max log-probability.
                #pred = output.argmax(dim=1, keepdim=True)
                #correct += pred.eq(target.view_as(pred)).sum().item()
                total_val_loss +=loss_func(output, target) * data.size(0)
                n_val += data.size(0)
        loss_val = total_val_loss/n_val
        #accuracy = correct / min(len(valid_loader.dataset), N_VALID_EXAMPLES)
        #trial.report(accuracy, epoch)
        trial.report(loss_val, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    #return accuracy
    return loss_val

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=N_TRIAL)#, timeout=600

pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
best_trial = study.best_trial
#best_nn = define_model(trial)
with open('result/{}/BestCostFunctionSpecification.pickle'.format(run_name),'wb') as f:
    pickle.dump( best_trial, f )

print("  Value: ", best_trial.value)

print("  Params: ")
for key, value in best_trial.params.items():
    print("    {}: {}".format(key, value))


'''
2. Train the best model with the full dataset.
'''
with open('result/{}/BestCostFunctionSpecification.pickle'.format(run_name),'rb') as f:
    best_trial = pickle.load( f )

def train_on_full_sample(trial):
    model = define_model(trial).to(DEVICE)

    # Generate the optimizers.
    optimizer_name = trial.params['optimizer']
    lr = trial.params['lr']
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the data loader
    target = torch.tensor(Y_['Outcome'].values.astype(np.float32)).type(torch.LongTensor)
    X = torch.tensor(X_.values.astype(np.float32))
    full_tensor = torch.utils.data.TensorDataset(X, target)
    full_loader = torch.utils.data.DataLoader(dataset = full_tensor, batch_size = BATCHSIZE, shuffle = True)

    # Training of the model.
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(full_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break

            data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data).squeeze()
            #loss = F.nll_loss(output, target)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()

    model.eval()
    return model

best_model_trained = train_on_full_sample(best_trial)
torch.save(best_model_trained, 'result/{}/best_model_trained.pt'.format(run_name))

target = torch.tensor(Y_test['Outcome'].values.astype(np.float32)).type(torch.LongTensor).to(DEVICE)
X = torch.tensor(X_test.values.astype(np.float32)).to(DEVICE)
outputs = best_model_trained(X)
_, predicted = torch.max(outputs.data, 1)
predicted = predicted.
correct = (predicted == target).sum().item()

print('Test accuracy: {}'.format(correct/len(predicted)))
