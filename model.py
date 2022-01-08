import torch 
import torch.nn as nn

class CNN(torch.nn.Module):

    def __init__(self, args):
        super(CNN, self).__init__()
        self.fc1 = torch.nn.Linear(4, 4, bias=args.bias)
        torch.nn.init.xavier_uniform(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.Sigmoid(),
            torch.nn.Dropout(p=args.dropout))

    def forward(self, args, x):
        out = self.layer4(x)
        return out



def train_para_vec(args, para_vec, model, cost, optimizer):
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()