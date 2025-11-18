import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class SAGERecommender(nn.Module):
    def __init__(self, in_feats=1, hidden_size=16, out_feats=1):
        super().__init__()
        self.conv1 = SAGEConv(in_feats, hidden_size)
        self.conv2 = SAGEConv(hidden_size, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
