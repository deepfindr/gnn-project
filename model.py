import torch
import torch.nn.functional as F 
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_geometric.nn import TransformerConv, GATConv, TopKPooling, BatchNorm
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn.conv.x_conv import XConv
torch.manual_seed(42)

class GNN(torch.nn.Module):
    def __init__(self, feature_size):
        super(GNN, self).__init__()
        num_classes = 2
        embedding_size = 64
    
        # GNN layers
        self.conv1 = TransformerConv(feature_size, 
                                    embedding_size, 
                                    heads=4, 
                                    dropout=0.2,
                                    edge_dim=11)
        self.head_transform1 = Linear(embedding_size*4, embedding_size)
        self.pool1 = TopKPooling(embedding_size, ratio=0.8)
        self.conv2 = TransformerConv(embedding_size, 
                                     embedding_size, 
                                     heads=4, 
                                     dropout=0.2,
                                    edge_dim=11)
        self.head_transform2 = Linear(embedding_size*4, embedding_size)
        self.pool2 = TopKPooling(embedding_size, ratio=0.8)
        self.conv3 = TransformerConv(embedding_size, 
                                     embedding_size, 
                                     heads=4, 
                                     dropout=0.8,
                                    edge_dim=11)
        self.head_transform3 = Linear(embedding_size*4, embedding_size)
        self.conv4 = TransformerConv(embedding_size, 
                                     embedding_size, 
                                     heads=4, 
                                     dropout=0.2,
                                    edge_dim=11)
        self.head_transform4 = Linear(embedding_size*4, embedding_size)
        self.conv5 = TransformerConv(embedding_size, 
                                     embedding_size, 
                                     heads=4, 
                                     dropout=0.2,
                                    edge_dim=11)
        self.head_transform5 = Linear(embedding_size*4, embedding_size)
        self.pool3 = TopKPooling(embedding_size, ratio=0.8)

        # Linear layers
        self.linear1 = Linear(embedding_size*2, 256)
        self.linear2 = Linear(256, num_classes)  

    def forward(self, x, edge_attr, edge_index, batch_index):
        # First block
        x = self.conv1(x, edge_index, edge_attr)
        x = self.head_transform1(x)

        # x, edge_index, edge_attr, batch_index, _, _ = self.pool1(x, 
        #                                                 edge_index, 
        #                                                 None, 
        #                                                 batch_index)
        x1 = torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1)

        # Second block
        x = self.conv2(x, edge_index, edge_attr)
        x = self.head_transform2(x)
        # x, edge_index, edge_attr, batch_index, _, _ = self.pool2(x, 
        #                                                 edge_index, 
        #                                                 None, 
        #                                                 batch_index)
        x2 = torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1)

        # Third block
        x = self.conv3(x, edge_index, edge_attr)
        x = self.head_transform3(x)
        # x, edge_index, edge_attr, batch_index, _, _ = self.pool3(x, 
        #                                                 edge_index, 
        #                                                 None, 
        #                                                 batch_index)
        x3 = torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1)

        x = self.conv4(x, edge_index, edge_attr)
        x = self.head_transform4(x)
        # x, edge_index, edge_attr, batch_index, _, _ = self.pool3(x, 
        #                                                 edge_index, 
        #                                                 None, 
        #                                                 batch_index)
        x4 = torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1)

        x = self.conv5(x, edge_index, edge_attr)
        x = self.head_transform5(x)
        # x, edge_index, edge_attr, batch_index, _, _ = self.pool3(x, 
        #                                                 edge_index, 
        #                                                 None, 
        #                                                 batch_index)
        x5 = torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1)
        
        # Concat pooled vectors
        x = x1 + x2 + x3 + x4 + x5

        # Output block
        x = self.linear1(x).relu()
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.linear2(x)

        return x

