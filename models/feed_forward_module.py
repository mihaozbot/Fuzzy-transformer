
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForward, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size, bias =False)
        self.glu  = torch.nn.GLU()
        self.relu = nn.LeakyReLU()
        self.l2 = nn.Linear(hidden_size, 2*output_size, bias =False)
        self.ln = nn.LayerNorm(input_size)

    def forward(self, x):
        x_l1 = self.l1(x) 
        x_relu = self.relu(x_l1)
        x_l2 = self.l2(x_relu + x_l1)
        x_glu = self.ln(self.glu(x_l2) + x_l1) #self.glu(x_l2.repeat(1,1,2))
        return x_glu
    
class GRN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRN, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size, bias =False)
        self.glu  = torch.nn.GLU()
        self.relu = nn.LeakyReLU()
        self.ln = nn.LayerNorm(input_size)
        self.l2 = nn.Linear(hidden_size, 2*hidden_size, bias =False)
        self.dropout = nn.Dropout(0.1)
        gain = torch.nn.init.calculate_gain('leaky_relu', param=0.01)
        torch.nn.init.xavier_uniform_(self.l1.weight, gain=gain)
        torch.nn.init.xavier_uniform_(self.l2.weight, gain=gain)


    def forward(self, x):
        x_l1 = self.dropout(self.relu(self.l1(x)))
        x_l2 = self.dropout(self.relu(self.l2(x_l1)))
        x_glu = self.ln(self.glu(x_l2) + x)
        return x_glu
    
class Dense(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Dense, self).__init__()
        self.hidden_size = hidden_size
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        self.relu = nn.LeakyReLU()

        gain = torch.nn.init.calculate_gain('leaky_relu', param=0.01)
        #gain = torch.nn.init.calculate_gain('relu')
        torch.nn.init.xavier_uniform_(self.fc1.weight, gain=gain)
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=gain)
        torch.nn.init.xavier_uniform_(self.fc3.weight, gain=gain)

        # Initialize weights with Kaiming (He) initialization
        #torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='leaky_relu')
        #torch.nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='leaky_relu')
        #torch.nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='leaky_relu')

        # Initialize biases to zero
        #torch.nn.init.zeros_(self.fc1.bias)
        #torch.nn.init.zeros_(self.fc2.bias)
        #torch.nn.init.zeros_(self.fc3.bias)
        self.dropout = nn.Dropout(0)

    def forward(self, x):
        x = (self.relu(self.fc1(x)))
        x = (self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
