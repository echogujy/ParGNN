import torch, sys
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv
import torch.distributed as dist
from managers import GraphManager
from typing import List, Tuple, Optional
import dgl, sys

class FusedLayerNormReLU_dropout(nn.Module):
    def __init__(self, normalized_shape, dropout_prob):
        super(FusedLayerNormReLU_dropout, self).__init__()
        self.layernorm = nn.LayerNorm(normalized_shape)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.layernorm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x

class Featrecv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, graph: GraphManager ,index: int, input: torch.Tensor, buff: int):
        out = graph._recv_rt(index,input,buff)
        ctx.save_for_backward(graph.nodes_mask[index])
        return out
    @staticmethod
    def backward(ctx, grad_out):
        input_mask, = ctx.saved_tensors
        grad_input = grad_out[input_mask]
        return None, None,grad_input,None
    
# Define a new method with a fixed signature
def feat_recv_fn(graph, index, input, buff):
    return Featrecv.apply(graph, index, input, buff)



class GCN_model(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes, nlayers:int = 3, dropout:float = 0.5,device:torch.device=torch.device('cpu')):
        super(GCN_model, self).__init__()
        
        self.in_conv =  GraphConv(in_feats=in_features, out_feats=hidden_features,allow_zero_in_degree=True).to(device)
        # self.in_conv_tail = torch.compile(FusedLayerNormReLU_dropout(hidden_features,dropout).to(device))
        self.in_conv_tail = FusedLayerNormReLU_dropout(hidden_features,dropout).to(device)

        self.hidden_conv_layers = nn.ModuleList()
        self.hidden_conv_layers_retail = nn.ModuleList()
        for _ in range(nlayers-2):
            conv_layer = GraphConv(in_feats=hidden_features, out_feats=hidden_features,allow_zero_in_degree=True).to(device)
            # retail = torch.compile(FusedLayerNormReLU_dropout(hidden_features,dropout).to(device))
            retail = FusedLayerNormReLU_dropout(hidden_features,dropout).to(device)
            self.hidden_conv_layers.append(conv_layer)
            self.hidden_conv_layers_retail.append(retail)

        self.out_conv = GraphConv(in_feats=hidden_features, out_feats=num_classes,allow_zero_in_degree=True).to(device)

    def forward(self,graph: GraphManager, graphs:List[dgl.DGLGraph],feats:List[torch.Tensor],feats2:List[torch.Tensor]):

        feats_b = [None for i in range(graph.graph_num)] 
        feats_tmp = [None for i in range(graph.graph_num)]
        
        for i in range(graph.graph_num):
            feats_b[i] = self.in_conv(graphs[i],(feats[i],feats2[i]))
            feats_b[i] = self.in_conv_tail(feats_b[i])
            graph.feat_send(i,feats_b[i],0)
        t = 0
        for gc,retail in zip(self.hidden_conv_layers,self.hidden_conv_layers_retail):
            for i in range(graph.graph_num):
                feats_tmp[i] = feat_recv_fn(graph,i,feats_b[i],t)
                feats_b[i] = gc(graphs[i],(feats_tmp[i],feats_b[i]))
                feats_b[i] = retail(feats_b[i])
                graph.feat_send(i,feats_b[i],t+1)
            t += 1
        for i in range(graph.graph_num):
            feats_tmp[i] = feat_recv_fn(graph,i,feats_b[i],t)
            feats_b[i] = self.out_conv(graphs[i],(feats_tmp[i],feats_b[i]))
        feats_ = torch.vstack(feats_b)
        return F.log_softmax(feats_, dim=1)

from dgl.nn.pytorch.conv import GATConv
class GAT_model(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes, nlayers:int = 3, dropout:float = 0.5, num_heads:int= 4,device:torch.device=torch.device('cpu')):
        super(GAT_model, self).__init__()

        self.in_conv =  GATConv(in_feats=in_features, out_feats=hidden_features, num_heads=num_heads, allow_zero_in_degree=True).to(device)

        # self.in_conv =  torch.compile(GATConv(in_feats=in_features, out_feats=hidden_features, num_heads=num_heads, allow_zero_in_degree=True).to(device))
        self.in_conv_tail = torch.compile(FusedLayerNormReLU_dropout(hidden_features * num_heads,dropout).to(device))
        # self.in_conv_tail = FusedLayerNormReLU_dropout(hidden_features,dropout).to(device)
        self.hidden_conv_layers = nn.ModuleList()
        self.hidden_conv_layers_retail = nn.ModuleList()
        for _ in range(nlayers-2):
            conv_layer =  GATConv(in_feats=hidden_features * num_heads , out_feats=hidden_features, num_heads=num_heads ,allow_zero_in_degree=True).to(device)
            retail = torch.compile(FusedLayerNormReLU_dropout(hidden_features * num_heads,dropout).to(device))
            # retail = FusedLayerNormReLU_dropout(hidden_features,dropout).to(device)
            
            self.hidden_conv_layers.append(conv_layer)
            self.hidden_conv_layers_retail.append(retail)

        self.out_conv =  GATConv(in_feats=hidden_features * num_heads , out_feats=num_classes, num_heads=1, allow_zero_in_degree=True).to(device)

        
    def forward(self,graph: GraphManager, graphs:List[dgl.DGLGraph],feats:List[torch.Tensor],feats2:List[torch.Tensor]):

        feats_b = [None for i in range(graph.graph_num)] 
        feats_tmp = [None for i in range(graph.graph_num)]
        for i in range(graph.graph_num):
            feats_b[i] = self.in_conv(graphs[i],(feats[i],feats2[i])).flatten(1)
            feats_b[i] = self.in_conv_tail(feats_b[i])
            graph.feat_send(i,feats_b[i],0)
        t = 0
        for gc,retail in zip(self.hidden_conv_layers,self.hidden_conv_layers_retail):
            for i in range(graph.graph_num):
                feats_tmp[i] = feat_recv_fn(graph,i,feats_b[i],t)               
                feats_b[i] = gc(graphs[i],(feats_tmp[i],feats_b[i])).flatten(1)
                feats_b[i] = retail(feats_b[i])
                graph.feat_send(i,feats_b[i],t+1)
            t += 1
        for i in range(graph.graph_num):
            feats_tmp[i] = feat_recv_fn(graph,i,feats_b[i],t)
            feats_b[i] = self.out_conv(graphs[i],(feats_tmp[i],feats_b[i])).mean(dim=-2)
        feats_ = torch.vstack(feats_b)
        return F.log_softmax(feats_, dim=1)

