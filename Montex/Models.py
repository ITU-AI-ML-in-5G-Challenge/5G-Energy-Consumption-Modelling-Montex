import torch
import torch.nn as nn
import torch.nn.functional as F


# class MLP(nn.Module):
#     def __init__(self, input_dim, hidden_size):
#         super().__init__()
#         self.act_fun = nn.ReLU()

#         self.fc1 = nn.Linear(input_dim, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, 128)
#         self.fc3 = nn.Linear(128, 1)
    
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act_fun(x)
#         x = self.fc2(x)
#         x = self.act_fun(x)
#         x = self.fc3(x)
#         return x
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size) 
        self.fc2 = nn.Linear(hidden_size, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x    

class MLP_V2(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_size) 
        self.fc2 = nn.Linear(hidden_size, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x    

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        return h0

    def forward(self, x, h0):
        out, hn = self.rnn(x, h0)
        out = self.fc(out)
        return out

class GRU_V2(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True)
        middle_dim = 24
        self.fc1 = nn.Linear(hidden_dim, middle_dim)
        self.fc2 = nn.Linear(middle_dim, output_dim)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_dim) 
        self.bn2 = nn.BatchNorm1d(middle_dim) 

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        return h0

    def forward(self, x, h0):
        out, hn = self.rnn(x, h0)

        out = self.bn1(out[:, -1, :])
        out = self.bn2(self.relu(self.fc1(out)))
        out = self.fc2(out)
        return out

class GRU_V3(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True)
        middle_dim = 64
        self.fc1 = nn.Linear(hidden_dim, middle_dim)
        self.fc2 = nn.Linear(middle_dim, output_dim)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_dim) 

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        return h0

    def forward(self, x, h0):
        out, hn = self.rnn(x, h0)

        out = self.bn1(out[:, -1, :])
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        return h0

    def forward(self, x, h0):
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :]) 
        return out    
    
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.num_layers = layer_dim
        self.hidden_size = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def init_hidden(self, x):      
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        return h0, c0

    def forward(self, x, h0, c0):
        out, _ = self.lstm(x, (h0,c0))  
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class TINY_CONV(nn.Module):
    def __init__(self, ker_size, output_dim, input_dim, seq_len):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=160, kernel_size=ker_size[0], stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=160, out_channels=20, kernel_size=ker_size[1], stride=1, padding=0)

        self.bn1conv = nn.BatchNorm2d(160)
        self.bn2conv = nn.BatchNorm2d(20)
        
        self.fc1 = nn.Linear((input_dim-sum(ker_size)+len(ker_size))*(seq_len-sum(ker_size)+len(ker_size))*20, 64)
        self.fc2 = nn.Linear(64, output_dim)
        
        self.relu = nn.ReLU()        

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.bn1conv(out)
        out = self.relu((self.conv2(out)))
        out = self.bn2conv(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class CONV(nn.Module):
    def __init__(self, ker_size, output_dim, input_dim, seq_len):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=640, kernel_size=ker_size[0], stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=640, out_channels=256, kernel_size=ker_size[1], stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=ker_size[2], stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=ker_size[3], stride=1, padding=0)

        self.bn1conv = nn.BatchNorm2d(640)
        self.bn2conv = nn.BatchNorm2d(256)
        self.bn3conv = nn.BatchNorm2d(128)
        self.bn4conv = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear((input_dim-sum(ker_size)+len(ker_size))*(seq_len-sum(ker_size)+len(ker_size))*64, 1024)
        self.fc2 = nn.Linear(1024, output_dim)
        
        self.relu = nn.ReLU()        

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.bn1conv(out)
        out = self.relu((self.conv2(out)))
        out = self.bn2conv(out)
        out = self.relu(self.conv3(out))
        out = self.bn3conv(out)
        out = self.relu(self.conv4(out))
        out = self.bn4conv(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class BIG_CONV(nn.Module):
    def __init__(self, ker_size, output_dim, input_dim, seq_len):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=ker_size[0], stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=ker_size[1], stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=ker_size[2], stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=ker_size[3], stride=1, padding=0)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=ker_size[4], stride=1, padding=0)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=ker_size[5], stride=1, padding=0)
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=ker_size[6], stride=1, padding=0)

        self.bn4conv = nn.BatchNorm2d(512)
        self.bn3conv = nn.BatchNorm2d(256)
        self.bn2conv = nn.BatchNorm2d(128)
        self.bn1conv = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear((input_dim-sum(ker_size)+len(ker_size))*(seq_len-sum(ker_size)+len(ker_size))*512, 1024)
        self.fc2 = nn.Linear(1024, output_dim)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.bn1conv(out)
        out = self.relu((self.conv2(out)))
        out = self.bn1conv(out)
        out = self.relu(self.conv3(out))
        out = self.bn2conv(out)
        out = self.relu(self.conv4(out))
        out = self.bn3conv(out)
        out = self.relu(self.conv5(out))
        out = self.bn3conv(out)
        out = self.relu(self.conv6(out))
        out = self.bn4conv(out)
        out = self.relu(self.conv7(out))
        out = self.bn4conv(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
