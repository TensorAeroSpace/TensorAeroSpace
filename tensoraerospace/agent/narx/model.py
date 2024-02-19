import torch
import torch.nn as nn

class NARX(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NARX, self).__init__()
        self.hidden_size = hidden_size
        self.input_layer = nn.Linear(input_size + output_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)


    def forward(self, input_tensor, last_output):
        combined = torch.cat((input_tensor, last_output), 0)
        hidden = torch.tanh(self.input_layer(combined))
        output = self.output_layer(hidden)
        return output
    
    def train(self, predcit_tensor, target_tensor):
        loss = self.criterion(predcit_tensor, target_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
