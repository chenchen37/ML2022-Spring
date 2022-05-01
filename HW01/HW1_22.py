import math
import numpy as np
import pandas as pd
import os
import csv
from tqdm import tqdm
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

train_df = pd.read_csv('./covid.train.csv')
#train_df.iloc[:, [38, 39, 54, 55, 70, 71, 86, 87, 102, 103]] = train_df.iloc[:, [38, 39, 54, 55, 70, 71, 86, 87, 102, 103]]*100  # cil and ili
train_df.iloc[:, [38, 54, 70]] = train_df.iloc[:, [38, 54, 70]]*85.48  # cil (0-2)
train_df.iloc[:, [86, 102]] = train_df.iloc[:, [86, 102]]*86.12 # cli (3-4)
train_df.iloc[:, [39, 55, 71]] = train_df.iloc[:, [39, 55, 71]]*85.55  # ili (0-2)
train_df.iloc[:, [87, 103]] = train_df.iloc[:, [87, 103]]*86.17  # ili (3-4)
#train_df.iloc[:, [40, 56, 72, 88, 104]] = train_df.iloc[:, [40, 56, 72, 88, 104]]*6.67   # hh_cmnty_cli
train_df.iloc[:, [40, 56, 72]] = train_df.iloc[:, [40, 56, 72]]*5.91   # hh_cmnty_cli (0-2)
train_df.iloc[:, [88, 104]] = train_df.iloc[:, [88, 104]]*5.96   # hh_cmnty_cli (3-4)
#train_df.iloc[:, [41, 57, 73, 89, 105]] = train_df.iloc[:, [41, 57, 73, 89, 105]]*9.1   # nohh_comnty_cli
train_df.iloc[:, [41, 57, 73]] = train_df.iloc[:, [41, 57, 73]]*8   # nohh_comnty_cli (0-2)
train_df.iloc[:, [89, 105]] = train_df.iloc[:, [89, 105]]*8.1   # nohh_comnty_cli (3-4)
#train_df.iloc[:, [44, 60, 76, 92, 108]] = train_df.iloc[:, [44, 60, 76, 92, 108]]*3   # work_outside_home
train_df.iloc[:, [44, 60, 76]] = train_df.iloc[:, [44, 60, 76]]*1.75   # work_outside_home (0-2)
train_df.iloc[:, [92, 108]] = train_df.iloc[:, [92, 108]]*1.8   # work_outside_home (3-4)
#train_df.iloc[:, [50, 66, 82, 98, 114]] = train_df.iloc[:, [50, 66, 82, 98, 114]]*9.1   # anxious
train_df.iloc[:, [50, 66, 82]] = train_df.iloc[:, [50, 66, 82]]*5.61   # anxious (0-2)
train_df.iloc[:, [98, 114]] = train_df.iloc[:, [98, 114]]*5.72   # anxious (3-4)
#train_df.iloc[:, [53, 69, 85, 101]] = train_df.iloc[:, [53, 69, 85, 101]]*10   # tested positive
train_df.iloc[:, [53]] = train_df.iloc[:, [53]]*9.355*5   # tested positive (0)
train_df.iloc[:, [69]] = train_df.iloc[:, [69]]*9.53*5   # tested positive (1)
train_df.iloc[:, [85]] = train_df.iloc[:, [85]]*9.697*5   # tested positive (2)
train_df.iloc[:, [101]] = train_df.iloc[:, [101]]*9.854*5   # tested positive (3)

test_df = pd.read_csv('./covid.test.csv')
#test_df.iloc[:, [38, 39, 54, 55, 70, 71, 86, 87, 102, 103]] = test_df.iloc[:, [38, 39, 54, 55, 70, 71, 86, 87, 102, 103]]*100  # cil and ili
test_df.iloc[:, [38, 54, 70]] = test_df.iloc[:, [38, 54, 70]]*85.48  # cil (0-2)
test_df.iloc[:, [86, 102]] = test_df.iloc[:, [86, 102]]*86.12 # cli (3-4)
test_df.iloc[:, [39, 55, 71]] = test_df.iloc[:, [39, 55, 71]]*85.55  # ili (0-2)
test_df.iloc[:, [87, 103]] = test_df.iloc[:, [87, 103]]*86.17  # ili (3-4)
#test_df.iloc[:, [40, 56, 72, 88, 104]] = test_df.iloc[:, [40, 56, 72, 88, 104]]*6.67   # hh_cmnty_cli
test_df.iloc[:, [40, 56, 72]] = test_df.iloc[:, [40, 56, 72]]*5.91   # hh_cmnty_cli (0-2)
test_df.iloc[:, [88, 104]] = test_df.iloc[:, [88, 104]]*5.96   # hh_cmnty_cli (3-4)
#test_df.iloc[:, [41, 57, 73, 89, 105]] = test_df.iloc[:, [41, 57, 73, 89, 105]]*9.1   # nohh_comnty_cli
test_df.iloc[:, [41, 57, 73]] = test_df.iloc[:, [41, 57, 73]]*8   # nohh_comnty_cli (0-2)
test_df.iloc[:, [89, 105]] = test_df.iloc[:, [89, 105]]*8.1   # nohh_comnty_cli (3-4)
#test_df.iloc[:, [44, 60, 76, 92, 108]] = test_df.iloc[:, [44, 60, 76, 92, 108]]*3   # work_outside_home
test_df.iloc[:, [44, 60, 76]] = test_df.iloc[:, [44, 60, 76]]*1.75   # work_outside_home (0-2)
test_df.iloc[:, [92, 108]] = test_df.iloc[:, [92, 108]]*1.8   # work_outside_home (3-4)
#test_df.iloc[:, [50, 66, 82, 98, 114]] = test_df.iloc[:, [50, 66, 82, 98, 114]]*9.1   # anxious
test_df.iloc[:, [50, 66, 82]] = test_df.iloc[:, [50, 66, 82]]*5.61   # anxious (0-2)
test_df.iloc[:, [98, 114]] = test_df.iloc[:, [98, 114]]*5.72   # anxious (3-4)
#test_df.iloc[:, [53, 69, 85, 101]] = test_df.iloc[:, [53, 69, 85, 101]]*10   # tested positive
test_df.iloc[:, [53]] = test_df.iloc[:, [53]]*9.355*5   # tested positive (0)
test_df.iloc[:, [69]] = test_df.iloc[:, [69]]*9.53*5   # tested positive (1)
test_df.iloc[:, [85]] = test_df.iloc[:, [85]]*9.697*5   # tested positive (2)
test_df.iloc[:, [101]] = test_df.iloc[:, [101]]*9.854*5   # tested positive (3)


def same_seed(seed): 
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_valid_split(data_set, valid_ratio, seed):
    '''Split provided training data into training set and validation set'''
    valid_set_size = int(valid_ratio * len(data_set)) 
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)

def predict(test_loader, model, device):
    model.eval() # Set your model to evaluation mode.
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)                        
        with torch.no_grad():                   
            pred = model(x)                     
            preds.append(pred.detach().cpu())   
    preds = torch.cat(preds, dim=0).numpy()  
    return preds

class COVID19Dataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        # TODO: modify model's structure, be aware of dimensions. 
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 34),
            #nn.ReLU(),
            #nn.Linear(24, 16),
            #nn.ReLU(),
            #nn.Linear(16, 16),
            #nn.ReLU(),
            #nn.Linear(16, 16),
            nn.Linear(34, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1) # (B, 1) -> (B)
        return x
 
def select_feat(train_data, valid_data, test_data, select_all=True):
    '''Selects useful features to perform regression'''
    y_train, y_valid = train_data[:,-1], valid_data[:,-1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:,:-1], valid_data[:,:-1], test_data

    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        a = [38, 39, 40, 41, 44, 50]
        feat_idx = a + [i+16 for i in a] + [i+32 for i in a] + [i+48 for i in a] + [i+64 for i in a] + [53, 69, 85, 101] 
        # TODO: Select suitable feature columns. # TODO: Select suitable feature columns.
        
    return raw_x_train[:,feat_idx], raw_x_valid[:,feat_idx], raw_x_test[:,feat_idx], y_train, y_valid

def trainer(train_loader, valid_loader, model, config, device):

    criterion = nn.MSELoss(reduction='mean') # Define your loss function, do not modify this.

    # Define your optimization algorithm. 
    # TODO: Please check https://pytorch.org/docs/stable/optim.html to get more available algorithms.
    # TODO: L2 regularization (optimizer(weight decay...) or implement by your self).
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay = 1e-5) 
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999999995)

    if not os.path.isdir('./models'):
        os.mkdir('./models') # Create directory of saving models.

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    for epoch in range(n_epochs):
        model.train() # Set your model to train mode.
        loss_record = []

        # tqdm is a package to visualize your training progress.
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()               # Set gradient to zero.
            x, y = x.to(device), y.to(device)   # Move your data to device. 
            pred = model(x)             
            loss = criterion(pred, y)
            loss.backward()                     # Compute gradient(backpropagation).
            optimizer.step()                    # Update parameters.
            #scheduler.step()
            step += 1
            loss_record.append(loss.detach().item())
            
            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record)/len(loss_record)

        model.eval() # Set your model to evaluation mode.
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())
            
        mean_valid_loss = sum(loss_record)/len(loss_record)
        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}, Best loss: {best_loss:.4f}')

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path']) # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else: 
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            print("Best loss: %f" %best_loss)
            return

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 91,      # Your seed number, you can pick your lucky number. :)
    'select_all': False,   # Whether to use all features.
    'valid_ratio': 0.2,   # validation_size = train_size * valid_ratio
    'n_epochs': 3000,     # Number of epochs.            
    'batch_size': 512, 
    'learning_rate': 1e-4,              
    'early_stop': 400,    # If model has not improved for this many consecutive epochs, stop training.     
    'save_path': './models/model.ckpt'  # Your model will be saved here.
}

# Set seed for reproducibility
same_seed(config['seed'])


# train_data size: 2699 x 118 (id + 37 states + 16 features x 5 days) 
# test_data size: 1078 x 117 (without last day's positive rate)
train_data, test_data = train_df.values, test_df.values
train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])

# Print out the data size.
print(f"""train_data size: {train_data.shape} 
valid_data size: {valid_data.shape} 
test_data size: {test_data.shape}""")

# Select features
x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data, config['select_all'])

# Print out the number of features.
print(f'number of features: {x_train.shape[1]}')

train_dataset, valid_dataset, test_dataset = COVID19Dataset(x_train, y_train), \
                                            COVID19Dataset(x_valid, y_valid), \
                                            COVID19Dataset(x_test)

# Pytorch data loader loads pytorch dataset into batches.
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

model = My_Model(input_dim=x_train.shape[1]).to(device) # put your model and data on the same computation device.
trainer(train_loader, valid_loader, model, config, device)

def save_pred(preds, file):
    ''' Save predictions to specified file '''
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

model = My_Model(input_dim=x_train.shape[1]).to(device)
model.load_state_dict(torch.load(config['save_path']))
preds = predict(test_loader, model, device) 
save_pred(preds, 'pred_22.csv')         

