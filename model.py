import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LayerNorm, Linear, GraphSizeNorm, BatchNorm
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gdp
from gatedgcn import GatedGCNLayer
  
class BRM(torch.nn.Module):
    def __init__(self, node_feature_size, num_classes, edge_feature_size, embedding_size=64):
        super(BRM, self).__init__()
        # self.conv1 = GATv2Conv(embedding_size, embedding_size, heads=n_heads, add_self_loops=hierarchy, concat=False, dropout=dropout_GNN)#, dropout=0.3)
        self.edge_emb = Linear(edge_feature_size, embedding_size)
        self.conv1 = GatedGCNLayer(in_dim=embedding_size,
                                    out_dim=embedding_size,
                                    dropout=0.,  # Dropout is handled by GraphGym's `GeneralLayer` wrapper
                                    residual=True,  # Residual connections are handled by GraphGym's `GNNStackStage` wrapper
                                    equivstable_pe=False)
        self.bn11 = LayerNorm(in_channels=embedding_size)
        self.bn12 = LayerNorm(in_channels=embedding_size)
        self.bn13 = LayerNorm(in_channels=embedding_size)

        self.linear_trans11 = Linear(node_feature_size, embedding_size)
        self.linear_trans12 = Linear(embedding_size, embedding_size)

        # self.conv2 = GATv2Conv(embedding_size, embedding_size, heads=n_heads, add_self_loops=hierarchy, concat=False, dropout=dropout_GNN)#, dropout=0.3)
        self.conv2 = GatedGCNLayer(in_dim=embedding_size,
                                    out_dim=embedding_size,
                                    dropout=0.,  # Dropout is handled by GraphGym's `GeneralLayer` wrapper
                                    residual=True,  # Residual connections are handled by GraphGym's `GNNStackStage` wrapper
                                    equivstable_pe=False)
        self.bn21 = LayerNorm(in_channels=embedding_size)
        self.bn22 = LayerNorm(in_channels=embedding_size)
        self.bn23 = LayerNorm(in_channels=embedding_size)

        self.linear_trans21 = Linear(embedding_size, embedding_size)
        self.linear_trans22 = Linear(embedding_size, embedding_size)

        # self.conv3 = GATv2Conv(embedding_size, embedding_size, heads=n_heads, add_self_loops=hierarchy, concat=False, dropout=dropout_GNN)#, dropout=0.3)
        self.conv3 = GatedGCNLayer(in_dim=embedding_size,
                                    out_dim=embedding_size,
                                    dropout=0.,  # Dropout is handled by GraphGym's `GeneralLayer` wrapper
                                    residual=True,  # Residual connections are handled by GraphGym's `GNNStackStage` wrapper
                                    equivstable_pe=False)
        self.bn31 = LayerNorm(in_channels=embedding_size)
        self.bn32 = LayerNorm(in_channels=embedding_size)
        self.bn33 = LayerNorm(in_channels=embedding_size)

        self.linear_trans31 = Linear(embedding_size, embedding_size)
        self.linear_trans32 = Linear(embedding_size, embedding_size)


        self.class1 = Linear(embedding_size*2, 256)
        self.bnl = LayerNorm(in_channels=256)
        self.class2 = Linear(256, num_classes)

    def forward(self, x, edge_index, edge_attr, batch_index):
        #First Block
        edge_attr = self.edge_emb(edge_attr)
    
        x = self.linear_trans11(x)
        x = self.bn11(x) 
        # x = F.relu(self.conv1(x, edge_index))
        x, edge_attr = self.conv1(x, edge_index, edge_attr)
        x = self.bn12(x)
        x = self.linear_trans12(x)
        x = self.bn13(x)
        # x1 =x [2046::2047]
        x1 = torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1)

        # Second block
        x = self.linear_trans21(x)
        x = self.bn21(x) 
        # x = F.relu(self.conv2(x, edge_index))
        x, edge_attr = self.conv2(x, edge_index, edge_attr)
        x = self.bn22(x)
        x = self.linear_trans22(x)
        x = self.bn23(x)
        # x2 = x[2046::2047]
        x2 = torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1)
    
        #Third block
        x = self.linear_trans31(x)
        x = self.bn31(x) 
        # x = F.relu(self.conv3(x, edge_index))
        x, edge_attr = self.conv3(x, edge_index, edge_attr)
        x = self.bn32(x)
        x = self.linear_trans32(x)
        x = self.bn33(x)
        # x3 = x[2046::2047]
        x3 = torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1)

        x = x1 + x2 + x3 
        x = F.relu(self.class1(x))
        x = self.bnl(x)
        x = self.class2(x)
        return x    

class HIGSI(torch.nn.Module):
    def __init__(self, node_feature_size, num_classes, edge_feature_size, embedding_size=64):
        super(HIGSI, self).__init__()
        
        self.edge_emb = Linear(edge_feature_size, embedding_size)
        self.conv1 = GatedGCNLayer(in_dim=embedding_size,
                                    out_dim=embedding_size,
                                    dropout=0.,  # Dropout is handled by GraphGym's `GeneralLayer` wrapper
                                    residual=True,  # Residual connections are handled by GraphGym's `GNNStackStage` wrapper
                                    equivstable_pe=False)
        self.bn11 = LayerNorm(in_channels=embedding_size)
        self.bn12 = LayerNorm(in_channels=embedding_size)
        self.bn13 = LayerNorm(in_channels=embedding_size)

        self.linear_trans11 = Linear(node_feature_size, embedding_size)
        self.linear_trans12 = Linear(embedding_size, embedding_size)
     
        self.conv2 = GatedGCNLayer(in_dim=embedding_size,
                                    out_dim=embedding_size,
                                    dropout=0.,  # Dropout is handled by GraphGym's `GeneralLayer` wrapper
                                    residual=True,  # Residual connections are handled by GraphGym's `GNNStackStage` wrapper
                                    equivstable_pe=False)
        self.bn21 = LayerNorm(in_channels=embedding_size)
        self.bn22 = LayerNorm(in_channels=embedding_size)
        self.bn23 = LayerNorm(in_channels=embedding_size)

        self.linear_trans21 = Linear(embedding_size, embedding_size)
        self.linear_trans22 = Linear(embedding_size, embedding_size)

        self.conv3 = GatedGCNLayer(in_dim=embedding_size,
                                    out_dim=embedding_size,
                                    dropout=0.,  # Dropout is handled by GraphGym's `GeneralLayer` wrapper
                                    residual=True,  # Residual connections are handled by GraphGym's `GNNStackStage` wrapper
                                    equivstable_pe=False)
        self.bn31 = LayerNorm(in_channels=embedding_size)
        self.bn32 = LayerNorm(in_channels=embedding_size)
        self.bn33 = LayerNorm(in_channels=embedding_size)

        self.linear_trans31 = Linear(embedding_size, embedding_size)
        self.linear_trans32 = Linear(embedding_size, embedding_size)

        self.graph_size_norm = GraphSizeNorm()
        self.class1 = Linear(embedding_size, 256)
        self.class2 = Linear(256, num_classes)
        
    def forward(self, x, edge_index, edge_attr, batch_index, graph_index, reduced_index):
        #First Block
        edge_attr = self.edge_emb(edge_attr)
    
        x = self.linear_trans11(x)
        x = self.bn11(x) 
        x, edge_attr = self.conv1(x, edge_index, edge_attr)
        x = self.bn12(x)
        x = self.linear_trans12(x)
        x = self.bn13(x)

        # Second block
        x = self.linear_trans21(x)
        x = self.bn21(x) 
        x, edge_attr = self.conv2(x, edge_index, edge_attr)
        x = self.bn22(x)
        x = self.linear_trans22(x)
        x = self.bn23(x)
    
        #Third block
        x = self.linear_trans31(x)
        x = self.bn31(x) 
        x, edge_attr = self.conv3(x, edge_index, edge_attr)
        x = self.bn32(x)
        x = self.linear_trans32(x)
        x = self.bn33(x)

        x = F.relu(self.class1(x))
        
        x = self.graph_size_norm(x,graph_index)
        x=gdp(x,graph_index)
        exp=torch.exp(x)
        numerator=exp.sum(dim=1, keepdim=True)
        denominator=gdp(numerator,reduced_index).sum(dim=1, keepdim=True)
        exp_denominator=denominator.index_select(dim=0, index=reduced_index)
        alpha=numerator/exp_denominator
        x=x*alpha

        x = gdp(x, reduced_index)
            
        x = self.class2(x)
        return x