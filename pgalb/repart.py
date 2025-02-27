# -*- coding: utf-8 -*- #
 
import numpy as np
import time
import torch
from typing import Optional, List
from utils_plus import MetisPy, RefinePy  # type: ignore

import os.path as osp
import dgl

from dgl.nn.pytorch import GATConv,GraphConv
import psutil
import sys

def print_memory_usage():
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        print(f"GPU {current_device} Memory Usage:")
        print("GPU Allocated:", round(torch.cuda.memory_allocated(current_device) / 1024**3, 2), "GB")
        print("GPU Cached:   ", round(torch.cuda.memory_reserved(current_device) / 1024**3, 2), "GB")
    process = psutil.Process()
    mem_info = process.memory_info()
    print("CPU Memory Usage:")
    print("RSS (Resident Set Size):", round(mem_info.rss / 1024**3, 2), "GB")  # 物理内存使用量
    print("VM (Virtual Memory):   ", round(mem_info.vms / 1024**3, 2), "GB")    # 虚拟内存使用量
    sys.stdout.flush()
    
 
   
class Graph_profile_model(object):
    def __init__(self,
                 in_feats:int,
                 out_feats:int,
                 
                 heads:int,
                 device: torch.device,
                 iters: int,
                 hidden: int=256,
                 ):
        self.gcn1 = GraphConv(in_feats=in_feats,out_feats=out_feats,allow_zero_in_degree=True).to(device)
        # self.gcn2 = GraphConv(in_feats=hidden,out_feats=out_feats,allow_zero_in_degree=True).to(device)
        
        # self.linear = torch.nn.Sequential(
        #     torch.nn.Linear(in_feats,hidden),
        #     torch.nn.ReLU(),
        #     torch.nn.LayerNorm(hidden)).to(device)
        
        self.gat1 = GATConv(in_feats=in_feats,out_feats=out_feats,num_heads=heads,allow_zero_in_degree=True).to(device)
        # self.gat2 = GATConv(in_feats=hidden,out_feats=out_feats,num_heads=heads,allow_zero_in_degree=True).to(device)
        
        self.iters = iters
        self.gcn_times = []
        self.gat_times = []
        self.data_comm_l = []
    def call_eval(self,graph:dgl.DGLGraph,feat1:torch.Tensor,feat2:torch.Tensor):
        with torch.no_grad():
            for i in range(5):
                self.gcn1(graph,(feat1,feat2))
                self.gat1(graph,(feat1,feat2))
                # tmp1 = self.linear(feat1)
                # tmp2 = self.linear(feat2)
                # self.gcn2(graph,(tmp1,tmp2))
                # self.gat2(graph,(tmp1,tmp2))
            torch.cuda.synchronize()
            t0 = time.time()
            for i in range(self.iters):
                self.gcn1(graph,(feat1,feat2))
                # tmp1 = self.linear(feat1)
                # tmp2 = self.linear(feat2)
                # self.gcn2(graph,(tmp1,tmp2))
        
            torch.cuda.synchronize()
            t1 = time.time()
            for i in range(self.iters):
                self.gat1(graph,(feat1,feat2))
                # tmp1 = self.linear(feat1)
                # tmp2 = self.linear(feat2)
                # self.gat2(graph,(tmp1,tmp2))
            torch.cuda.synchronize()
            t2 = time.time()
            self.gcn_times.append(t1-t0)
            self.gat_times.append(t2-t1)
    def cal_comm(self, src_g:torch.Tensor,parts:torch.Tensor,world_size:int):
        nodes_from = parts[src_g]
        ranks_, counts_ = torch.unique(nodes_from, return_counts=True)
        data_comm = np.zeros(shape=(world_size,), dtype=np.int64)
        data_comm[ranks_.cpu().numpy()] = counts_.cpu().numpy()
        self.data_comm_l.append(data_comm)   
    
    def save_data(self,path):
        data_comm = np.array(self.data_comm_l)
        data_gcn = np.array(self.gcn_times)
        data_gat = np.array(self.gat_times)
        np.savez(path,data_comm=data_comm,data_gcn=data_gcn,data_gat=data_gat)
    
    def return_data(self):
        data_comm = np.array(self.data_comm_l)
        data_gcn = np.array(self.gcn_times)
        data_gat = np.array(self.gat_times)
        self.gcn_times = []
        self.gat_times = []
        self.data_comm_l = []
        return data_comm,data_gcn,data_gat

def normalize_clamp(arr, a: int = 2, b: int = 10) -> np.ndarray:
    # Get the minimum and maximum values of the array
    min_val = np.min(arr)
    max_val = np.max(arr)

    # Normalize the array using linear mapping
    normalized_arr = ((arr - min_val) / (max_val - min_val + 0.001)) * (b - a) + a
    normalized_arr = normalized_arr.astype(np.int64)
    return normalized_arr.flatten(order='C')

def node_work_numeric(arr:np.ndarray,up_stage:int = 1_000) -> np.ndarray:
    return (arr * up_stage).astype(np.int64).flatten(order='C')
from scipy.sparse import csr_matrix
class adapt_graph(object):
    def __init__(self,data_comm:np.ndarray,data_gcn:np.ndarray,data_gat:np.ndarray,com_clamp:Optional[float]=0.2) -> None:
        # data_comm = data_comm + data_comm.T
        np.fill_diagonal(data_comm, 0)
        self.data_comm = data_comm.copy()
        # threshold = data_comm.max() * com_clamp 
        # data_comm = np.where(data_comm < threshold, 0, data_comm)
        graph = csr_matrix(data_comm)
        
        self.indptr = graph.indptr.astype(np.int64).flatten(order='C')
        self.indices = graph.indices.astype(np.int64).flatten(order='C')
        eweight = graph.data
        
        mult_times = eweight.max() // (eweight.min() + 1)
        self.data_gcn = data_gcn
        self.data_gat = data_gat
        
        self.eweight = eweight
        # self.eweight = normalize_clamp(eweight, a=1, b=mult_times)
        self.gcn_w = normalize_clamp(data_gcn,a=200,b=1000)
        self.gat_w = normalize_clamp(data_gat,a=200,b=1000)
        # self.gcn_w = node_work_numeric(data_gcn)
        # self.gat_w = node_work_numeric(data_gat)
        # self.nweight = gcn_w + gat_w     
        
        self.N = int(self.indptr.shape[0] - 1)
    #     # assert self.N == self.nweight.shape[0]
    def compute_comm(self,mapping:np.ndarray):
        from itertools import combinations
        k :int = mapping.max() + 1
        comm = self.data_comm.copy().astype(np.int64)
        np.fill_diagonal(comm, 0)
        for i in range(k):
            a = np.where(mapping == i)[0].tolist()
            if len(a) > 1:
                comb = combinations(a, 2)
                for j0, j1 in comb:
                    comm[j0, j1] = 0
                    comm[j1, j0] = 0
        # print(comm)
        return comm.sum()

    def mapping(self,k,profile:str='gat',ori_mapping:Optional[np.ndarray]=None):
        if profile == 'both':
            nweight = self.gcn_w + 2 *  self.gat_w
        elif profile == 'gcn':
            nweight = self.gcn_w
        elif profile == 'gat':
            nweight = self.gat_w
        else:
            raise ValueError('profile must be one of both, gcn, gat')
        
        if ori_mapping is None:
            # ori_mapping = RefinePy(k,self.N, 1, self.indptr, self.indices, nweight, self.eweight)
            ori_mapping = MetisPy(k,self.N, 1, self.indptr, self.indices, nweight, self.eweight)
            return ori_mapping
        else:
            assert ori_mapping.shape[0] == self.N
            assert ori_mapping.max() == k - 1
        _, a , b= self.get_imbalance(k,ori_mapping)
        my_dict = {}
        my_dict[a] = ori_mapping.copy()
        for i in range(4):
            ori_mapping = RefinePy(k,self.N, 1, self.indptr, self.indices, nweight, self.eweight,ori_mapping.copy())
            # ori_mapping = MetisPy(k,self.N, 1, self.indptr, self.indices, nweight)
            _, a, b = self.get_imbalance(k,ori_mapping)
            my_dict[a] = ori_mapping.copy()
            ori_mapping = my_dict[min(my_dict.keys())]
            print(a,b)
        min_load = min(my_dict.keys())    
        # print(f"The min {profile} load: ",min_load)
        print(my_dict.keys())
        return my_dict[min_load]
    
    def get_imbalance(self,k,map_arr:np.ndarray):
        gcn_wl = torch.zeros(size=(k,),dtype=torch.float64).index_add_(0,index=torch.from_numpy(map_arr),source=torch.from_numpy(self.data_gcn))
        gat_wl = torch.zeros(size=(k,),dtype=torch.float64).index_add_(0,index=torch.from_numpy(map_arr),source=torch.from_numpy(self.data_gat))
        gcn_im = gcn_wl.max()/(gcn_wl.mean() + 1e-6)
        gat_im = gat_wl.max()/(gat_wl.mean() + 1e-6)

        return gcn_im.item(),gat_im.item(), self.compute_comm(map_arr)
        
def graph_eval(
    name:str = "ogbn-arxiv",
    dir_path:str = "../data/dataset",
    num_part : int = 16,
    flag: str = '',
    iters = 30,
    reused = True,
):
    """
    :param name: 
    :param dir_path:
    :param algo:
    :param num_part:
    :return:
    """
    
    
    graph_path = osp.join(dir_path, name, f"part_{num_part}")
    eval_path = osp.join(graph_path,f'{flag}_eval.npz')
    print(f"Eval path: {eval_path}")
    if osp.exists(eval_path) and reused:
        print("data exits.")
        data = np.load(eval_path)
        data_comm = data['data_comm']
        data_gcn = data['data_gcn']
        data_gat = data['data_gat']
        return data_comm, data_gcn, data_gat
    
    device = torch.device('cuda:2')
    torch.cuda.set_device(device)
    if name in ['ogbn-arxiv']:
        feat_dim = 128
        class_dim = 40
    elif name in ['ogbn-products']:
        feat_dim = 100
        class_dim = 47
    elif name in ['ogbn-proteins']:
        feat_dim = 8
        class_dim = 112
    elif name in ['ogbn-papers100M']:
        feat_dim = 128
        class_dim = 127
    elif name in ['reddit']:
        feat_dim = 602
        class_dim = 41
    elif name in ['yelp']:
        feat_dim = 300
        class_dim = 41
    elif 'igb' in name:
        feat_dim = 1024
        class_dim = 19
    else:
        raise ValueError(f'{name} is not supported')
    hidden = 256
    heads = 4 
    eval_model = Graph_profile_model(in_feats=feat_dim,out_feats=class_dim,heads=heads,hidden=hidden,device=device,iters=iters)
    part_ids = torch.load(f"{graph_path}/part_ids.pt",map_location=device) 
    
    for i in range(num_part):
        graph = torch.load(osp.join(graph_path,f'graph_{i}.pt'),map_location=device)
        nfeat = torch.load(osp.join(graph_path,f'nfeat_{i}.pt'),map_location=device)
        src_g,dst = graph['src_nodes'], graph['dst_nodes']
        node_id = graph['node_id']
        gid2lid = torch.ones((int(node_id.max().item()+1),),device=device,dtype=torch.int64).mul_(-1)
        gid2lid[node_id] = torch.arange(node_id.size(0),device=device)
        src, dst = gid2lid[src_g], gid2lid[dst]
        N = int(node_id.size(0))
        nodes = torch.unique(dst)
        loca_map = torch.ones(size=(N,),device=device,dtype=torch.int64).mul_(-1)
        loca_map[nodes] = torch.arange(nodes.size(0),device=device)
        dst_m = loca_map[dst]
        num_nodes_dict = {'U_':N, 'V_': nodes.size(0)}
        edges_dict = {
            ('U_', 'U_to_V_', 'V_'): (src, dst_m)
        }
        hg = dgl.heterograph(edges_dict, num_nodes_dict=num_nodes_dict) # type: ignore
        feat1 = nfeat['feat']
        feat2 = feat1[nodes].clone()
        
        eval_model.call_eval(hg, feat1,feat2)
        eval_model.cal_comm(src_g, part_ids, num_part)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
    eval_model.save_data(eval_path)
    data_comm,data_gcn,data_gat = eval_model.return_data()
    return data_comm,data_gcn,data_gat


def evaluation():
    datasets : List[str] = ['ogbn-arxiv', 'ogbn-products','yelp','reddit','ogbn-proteins',
                        'igb-tiny','igb-small','igb-medium']
    dir_path:str = "../data/DistData"

    num_parts : List[int] = [2,4,8]

    for name in datasets:
        for num_part in num_parts:
            data_comm,data_gcn,data_gat = graph_eval(name=name,dir_path=dir_path,num_part=num_part, reused=False)
if __name__ == '__main__':
    import sys

    datasets : List[str] = ['ogbn-products','yelp','reddit','ogbn-proteins']
    dir_path:str = "../data/DistData"

    num_parts : List[int] = [16,16,16,16]
    
    # k = 4
    profile_mode = 'both'
    only_eval = False
    for name, num_part in zip(datasets,num_parts):

        data_comm,data_gcn,data_gat = graph_eval(name=name,dir_path=dir_path,num_part=num_part, reused=False)
        if only_eval:
            continue
        adapt_alog = adapt_graph(data_comm=data_comm,data_gcn=data_gcn,data_gat=data_gat,com_clamp=0.1)
        k = num_part // 4
        t0 = time.time()
        reparts = adapt_alog.mapping(k,profile=profile_mode)
        t1 = time.time()
        file_path = osp.join(dir_path, name, f"part_{num_part}", f'adapt_repart_to_{k}.pt')
        torch.save(torch.tensor(reparts), file_path)
        imB_gcn,imB_gat, _ = adapt_alog.get_imbalance(k,reparts)
        print(f'{name} {profile_mode}-profile {num_part} --> {k}: Time:{t1-t0:.4f}')
        print(f'{name} {num_part}: GAT: {imB_gat:.4f}, GCN: {imB_gcn:.4f}')

        sys.stdout.flush()
