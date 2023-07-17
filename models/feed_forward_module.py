
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(FeedForward, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size, bias =False)
        self.glu  = torch.nn.GLU()
        self.relu = nn.ELU()
        self.l2 = nn.Linear(hidden_size, output_size, bias =False)
        
    def forward(self, x):
        x_l1 = self.l1(x) 
        x_glu = self.glu(x_l1.repeat(1,1,2))
        x_relu = self.relu(x_glu)
        output = self.l2(x_relu)
        return output