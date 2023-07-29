
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForward, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size, bias =False)
        self.glu  = torch.nn.GLU()
        self.relu = nn.LeakyReLU()
        self.l2 = nn.Linear(hidden_size, 2*output_size, bias =False)
        
    def forward(self, x):
        x_l1 = self.l1(x) 
        x_relu = self.relu(x_l1)
        x_l2 = self.l2(x_relu + x_l1)
        x_glu = self.glu(x_l2)#self.glu(x_l2.repeat(1,1,2))
        return x_glu
    
class GRN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRN, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size, bias =False)
        self.glu  = torch.nn.GLU()
        self.relu = nn.LeakyReLU()
        self.ln = nn.LayerNorm(input_size)
        self.l2 = nn.Linear(hidden_size, 2*input_size, bias =False)
        
    def forward(self, x):
        x_l1 = self.l1(x) 
        x_l2 = self.relu(self.l2(self.relu(x_l1)))
        x_glu = self.ln(self.glu(x_l2) + x)
        return x_glu
    
class Dense(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Dense, self).__init__()
        self.hidden_size = hidden_size
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.fc3 = nn.Linear(hidden_size//2, hidden_size//2)
        self.fc4 = nn.Linear(hidden_size//2, hidden_size//4)
        self.fc5 = nn.Linear(hidden_size//4, output_size)
        
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)
    

    def forward(self, x):

        x = self.fc1(x)
        x = (self.relu(self.fc2(x)))
        x = (self.relu(self.fc3(x)))
        x = (self.relu(self.fc4(x)))
        x = self.fc5(x)
        return x
