''' Preprocess and load data '''
import torch
import math
 
# Notes for Sam:
# If you want to use gpu, you will have to use this line:
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# then you will move everything onto the device, e.g.,
# train_x=torch.tensor(train_x).float.to(device)
 
# Training data is 100 points in [0,1] inclusive regularly spaced
train_x = torch.linspace(0, 1, 100)
# True function is sin(2*pi*x) with Gaussian noise
train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)