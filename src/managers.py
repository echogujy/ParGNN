import torch, argparse, os.path as osp
from typing import Dict, List, Optional, Tuple
import torch.distributed as dist
import time, sys
import torch.nn.functional as F
import dgl.function as dglfn
import json
# import numpy as np
# import pickle
# from copy import deepcopy
# from loguru import logger
import dgl, os
from utils import print_memory_usage
import dgl.sparse as dglsp

class GraphManager(object):
    def __init__(self,args:argparse.Namespace,
                 device : torch.device=torch.device('cpu'),
                 addselfloop:bool=False,
                 async_comm_mode:bool=False,
                 hidden_dim:int = 256, nlayers:int = 3):
        assert dist.is_initialized(),"GraphManager object is used for distributed training environment"
        self.device = device
        self.addselfloop = addselfloop
        self.cpu_device = torch.device('cpu')
        self.rank = args.rank
        self.world_size = args.world_size
        self.path = osp.join(args.data_path,args.dataset,
                        f'part_{args.total_parts}')

        self.total_parts = int(args.total_parts)

        self.current_step = 0

        reparter: str = args.reparter
        
        # assert type(repart_poiter) == Optional[torch.Tensor]
        # repar
        # reparter_path = osp.join(args.data_path,args.dataset,f'part_{args.total_parts}',f"{reparter}-{args.model}_repart.pt")
        if reparter == 'adapt':
            reparter_path = osp.join(args.data_path,args.dataset,f'part_{args.total_parts}',f"{reparter}_repart_to_{self.world_size}.pt")
            repart_poiter = torch.load(reparter_path,map_location=self.device)
        elif reparter == 'base':
            if self.total_parts == self.world_size:
                repart_poiter = torch.arange(self.total_parts,dtype=torch.int64,device=self.device)
            else:
                repart_poiter = torch.arange(self.total_parts,dtype=torch.int64,device=self.device)
                repart_poiter = repart_poiter.remainder(self.world_size)[torch.randperm(self.total_parts,device=self.device)]
                rand_p = torch.randint(low=0,high=self.total_parts,size=(2,),dtype=torch.int64).tolist()
                a ,b = rand_p[0], rand_p[1]
                repart_poiter[a] = b % self.world_size
                # torch.randperm(self.total_parts)
                # repart_poiter = torch.randint(low=0,high=10_000,size=(self.total_parts,),dtype=torch.int64,device=self.device)  + torch.randint(low=0,high=100_000,size=(self.total_parts,),dtype=torch.int64,device=self.device) 
                # print(repart_poiter)
                # repart_poiter = repart_poiter.remainder(self.world_size)
                # print(repart_poiter)
        else:
            raise NotImplementedError(f"reparter: {reparter} is not supoort or not generater")
        
        self.repart_poiter = repart_poiter

        if self.rank == 0:
            print(self.repart_poiter)
        self.dim = hidden_dim
        self.n_buff = nlayers -1
        self.comm_ops = []
        self.gpu_group = dist.GroupMember.WORLD
        self._load_datasets(reparter)
        self.main_rank  = 0
        self.comm_t : float = 0.0
        self.async_op : bool = async_comm_mode

        # print_memory_usage()
        # self.th_pool = ThreadPool(processes=self.graph_num + 5)

    def _save_and_load(self,reparter:str="adapt",reused:bool= False):

        dir_path = osp.join(self.path,f"comm_files_{reparter}_to_{self.world_size}")
        if not osp.exists(dir_path) or not osp.isfile(osp.join(dir_path,f"comm_0.pt")) or not reused:
            os.makedirs(dir_path,exist_ok=True)
            self.flush_comm_pipe_info()
            # save
            data_dic = {
                "local_maps": self.local_maps,
                "node_mask": self.nodes_mask,
                "fetch_data_to_stable": self.fetch_data_to_stable,
                "fetch_data_from_rtable": self.fetch_data_from_rtable,
                "fit_to_data": self.fit_to_data,
                "send_nums_list": self.send_nums_list,
                "recv_nums_list": self.recv_nums_list,
                "recv_offset_index": self.recv_offset_index,
                "recv_len": self.recv_len,
                "max_graph_num": self.max_graph_num,
            }
            torch.save(data_dic,f=osp.join(dir_path,f"comm_{self.rank}.pt"))
        else:
            data_dic = torch.load(f=osp.join(dir_path,f"comm_{self.rank}.pt"),map_location=self.device)
            self.local_maps = data_dic["local_maps"]
            self.nodes_mask = data_dic["node_mask"]
            self.fetch_data_to_stable = data_dic["fetch_data_to_stable"]
            self.fetch_data_from_rtable = data_dic["fetch_data_from_rtable"]
            self.fit_to_data = data_dic["fit_to_data"]
            self.send_nums_list = data_dic["send_nums_list"]
            self.recv_nums_list = data_dic["recv_nums_list"]
            self.recv_offset_index = data_dic["recv_offset_index"]
            self.recv_len = data_dic["recv_len"]
            self.max_graph_num = data_dic["max_graph_num"]
            self.recv_buffs = [torch.zeros(size=(self.recv_len,self.dim),dtype=torch.float32,device=self.device) for i in range(self.n_buff)]
            self.grad_buff = [None for i in range(self.graph_num)]
        # print_memory_usage()
    def _load_datasets(self,reparter:str="adapt"):
        my_graphs_id  = torch.where(self.repart_poiter == self.rank)[0].tolist()
        graph_list = []
        nfeat_list = []
        for rank in my_graphs_id:
            graph = torch.load(osp.join(self.path,f'graph_{rank}.pt'),map_location=self.device)
            nfeat = torch.load(osp.join(self.path,f'nfeat_{rank}.pt'),map_location=self.device)
            graph_list.append(graph)
            nfeat_list.append(nfeat)
        self.part_ids = torch.load(f"{self.path}/part_ids.pt") #,map_location=self.device) 
        # self.part_ids = torch.load(f"{self.path}/part_ids.pt",map_location=self.device) 
        self.graph_num = len(my_graphs_id)

        self.gpu_graphs = {}
        # print("Start load")
        # import sys
        sys.stdout.flush()
        self._init_build_gpuGraphs(graph_list,nfeat_list,my_graphs_id)
        # print("Finish load")
        # import sys
        sys.stdout.flush()
        self.my_graphs_id = my_graphs_id
        self._save_and_load(reparter)
        # self.flush_comm_pipe_info()

    def _init_build_gpuGraphs(self,graph_list:List[Dict[str,torch.Tensor]],nfeat_list:List[Dict[str,torch.Tensor]],my_graphs_id:List[int]):
        for i in range(self.graph_num):
            src, dst = graph_list[i]['src_nodes'], graph_list[i]['dst_nodes']

            node_id = graph_list[i]['node_id']
            gid2lid = torch.ones((int(node_id.max().item()+1),),device=self.device,dtype=torch.int64).mul_(-1)
            gid2lid[node_id] = torch.arange(node_id.size(0),device=self.device)

            recv_nodes = torch.unique(src)
            out_index = torch.logical_not(self.part_ids[recv_nodes.cpu()] == my_graphs_id[i]).to(self.device)

            recv_nodes = recv_nodes[out_index]  #
            # inner_mask = torch.where(out_index == False)[0]
            # node_mask = torch.ones_like(node_id,dtype=torch.int32,device=self.device).mul_(-1)
            
            # node_mask = self.data_reader.read_node_data(key='node_mask',ind=node_id)
            # feat = self.data_reader.read_node_data(key='feat',ind=node_id)
            # label = self.data_reader.read_node_data(key='label',ind=node_id)
            node_mask = nfeat_list[i]['mask']
            feat = nfeat_list[i]['feat']
            label = nfeat_list[i]['label']
            
            graph = dgl.graph((gid2lid[src], gid2lid[dst]),num_nodes=node_mask.size(0))
            graph.ndata['feat'] = feat
            graph.ndata['label'] = label.to(torch.int64)
            graph.ndata['node_mask'] = node_mask

            gdata_comm = {
                'recv_nodes' : recv_nodes, # global node id
                'gid2lid'   : gid2lid
            }
            graph.gdata = gdata_comm   # type: ignore
            if self.addselfloop:
                graph = dgl.remove_self_loop(graph)
                graph = dgl.add_self_loop(graph)
            self.gpu_graphs[my_graphs_id[i]] = graph
        
    
    def flush_comm_pipe_info(self):
 
        send_list = [list() for i in range(self.world_size)]

        for i,repart in enumerate(self.repart_poiter.tolist()):
            send_list[repart].append(i)
        
        max_graph_num = max([len(send_list[i]) for i in range(self.world_size)])
        


        recv_nodes_comp = torch.cat([graph.gdata['recv_nodes'] for graph in self.gpu_graphs.values()]) 
        recv_nodes = torch.unique(recv_nodes_comp)

        # graph_part_id = self.part_ids[recv_nodes]        #
        graph_part_id = self.part_ids[recv_nodes.cpu()].to(self.device)        
        gpu_part_id = self.repart_poiter[graph_part_id]     

        recv_gpu_ids, counts = torch.unique(gpu_part_id,return_counts=True)
        sorted_ids = torch.argsort(gpu_part_id)
        recv_nodes_sorted_by_gpu = recv_nodes[sorted_ids]
        recv_nums = torch.zeros((self.world_size),dtype=torch.int64,device=self.device)
        send_nums = recv_nums.clone()
        recv_nums[recv_gpu_ids] = counts
        
        
        dist.all_to_all_single(output=send_nums, input=recv_nums, group=self.gpu_group)
        send_nodes = torch.zeros(size=(send_nums.sum().item(),), # type: ignore
                                 dtype=recv_nodes_sorted_by_gpu.dtype,device=self.device)
        dist.all_to_all_single(output=send_nodes, input=recv_nodes_sorted_by_gpu,
                               output_split_sizes=send_nums.tolist(),input_split_sizes=recv_nums.tolist(),
                               group=self.gpu_group)


        kpp  = self.part_ids[send_nodes.cpu()].to(self.device)
        send_nodes_list = torch.split(send_nodes,split_size_or_sections=send_nums.tolist(),dim=0)
        kpp_list = torch.split(kpp,split_size_or_sections=send_nums.tolist(),dim=0)
        
        send_tables = [list() for i in range(self.world_size)] #
        fetch_data_to_stable = [ list() for i in range(self.world_size)]
 
        self.local_maps = []
        self.nodes_mask = []
        for i in self.my_graphs_id:
            graph = self.gpu_graphs[i]
            src, dst = graph.edges()
            len_s = graph.number_of_nodes()
            nodes = torch.unique(dst)
            local_map = torch.ones(size=(len_s,),device=self.device,dtype=torch.int64).mul_(-1)
            local_map[nodes] = torch.arange(nodes.size(0),device=self.device)
            # self.gpu_graphs[i].ndata['local_map'] = local_map
            # self.gpu_graphs[i].gdata['nodes'] = nodes
            self.local_maps.append(local_map)
            self.nodes_mask.append(nodes.long())
            for j in range(self.world_size):
                
                pos_of_gid_mysend = torch.where(kpp_list[j]==i)[0]
                
                send_tables[j].append(send_nodes_list[j][pos_of_gid_mysend])
                pos_from = graph.gdata['gid2lid'][send_tables[j][-1]]
                
                pos_from = local_map[pos_from]
                fetch_data_to_stable[j].append(pos_from)
        for _ in range(max_graph_num - self.graph_num):
            for j in range(self.world_size):
                send_tables[j].append(torch.zeros(size=(0,),dtype=torch.int64,device=self.device))
                fetch_data_to_stable[j].append(torch.zeros(size=(0,),dtype=torch.int64,device=self.device))
        
        self.fetch_data_to_stable = []
        recv_tables = []
        
        send_nums_list = []
        recv_nums_list = []
        for i in range(max_graph_num):
            self.fetch_data_to_stable.append(
                torch.cat([fetch_data_to_stable[j][i] for j in range(self.world_size)])
            )
            send_table = torch.cat([send_tables[j][i] for j in range(self.world_size)])
            
            send_nums = torch.tensor([send_tables[j][i].size(0) for j in range(self.world_size)],dtype=torch.int64,device=self.device)
            recv_nums  = send_nums.clone()
            dist.all_to_all_single(output=recv_nums, input=send_nums, group=self.gpu_group)
            recv_table = torch.zeros(size=(recv_nums.sum().item(),),dtype=torch.int64,device=self.device) # type: ignore
            dist.all_to_all_single(output=recv_table, input=send_table, output_split_sizes=recv_nums.tolist(),input_split_sizes=send_nums.tolist(),
                                   group=self.gpu_group)
            
            recv_tables.append(recv_table)
            
            send_nums_list.append(send_nums.tolist())
            recv_nums_list.append(recv_nums.tolist())
        
    
        recv_table = torch.cat(recv_tables)
        search_table = torch.zeros(size=(recv_table.max().item()+ 1,),dtype=torch.int64,device=self.device) # type: ignore
        search_table[recv_table] = torch.arange(recv_table.size(0),device=self.device)
        
        self.fetch_data_from_rtable = []
        self.fit_to_data = []
        for graph_id in self.my_graphs_id:
            pos_of_gid_myrecv = search_table[self.gpu_graphs[graph_id].gdata['recv_nodes']]

            self.fit_to_data.append(self.gpu_graphs[graph_id].gdata['gid2lid'][recv_table[pos_of_gid_myrecv]])
            self.fetch_data_from_rtable.append(pos_of_gid_myrecv)

        self.send_nums_list = send_nums_list
        self.recv_nums_list = recv_nums_list
        self.recv_offset_index = []
        demo_ind = [0,0]
        for i, len_list in enumerate(self.recv_nums_list):
            demo_ind[0] = demo_ind[1]
            demo_ind[1] = demo_ind[0] + sum(len_list)
            self.recv_offset_index.append(demo_ind.copy())
        
        self.recv_len = self.recv_offset_index[-1][-1]
        self.max_graph_num = max_graph_num
        send_len = torch.tensor(self.send_nums_list,dtype=torch.int64,device=self.device)
        com_data = send_len.sum() - send_len[:,self.rank].sum()
        dist.all_reduce(com_data, op=dist.ReduceOp.SUM)
        if self.rank == 0:
            print("Recving size:", self.recv_len, com_data)
        ## debug print info
        # for i in range(max_graph_num):
        # logger.debug(f'Rank:[{self.rank}] \t reparter: {self.repart_poiter}')
        # logger.debug(f'Rank:[{self.rank}] \t send_nums_list:{self.send_nums_list} \t \
        #     recv_nums_list:{self.recv_nums_list} \t send_len:{self.send_len} \t recv_len:{self.recv_len}')
        # logger.debug(f'Rank:[{self.rank}], recv_offset_index:{self.recv_offset_index}')
        # if self.rank == 0:
        #     print(f"The communication size is {com_data}")
        #     print(self.recv_len,self.dim)
        # logger.debug(f'Rank:[{self.rank}], fetch_data_to_table: {self.fetch_data_to_stable}')
        # logger.debug(f'Rank:[{self.rank}], fetch_data_from_rtable: {self.fetch_data_from_rtable}')
        # logger.debug(f'Rank:[{self.rank}], fit_to_data: {self.fit_to_data}')
        self.recv_buffs = [torch.zeros(size=(self.recv_len,self.dim),dtype=torch.float32,device=self.device) for i in range(self.n_buff)]
        self.grad_buff = [None for i in range(self.graph_num)]
        # dist.barrier()
        # exit()

    ## some properties
    def get_feats(self) -> List[torch.Tensor]: 
        feats = []
        for graph_id in self.my_graphs_id:
            feats.append(self.gpu_graphs[graph_id].ndata['feat'])
        return feats
        # return [graph.ndata['feat'] for graph in self.gpu_graphs.values()]

    def get_matrices(self):
        res = []
        for i,graph_id in enumerate(self.my_graphs_id):
            graph = self.gpu_graphs[graph_id]
            src, dst = graph.edges()

            # nodes = self.nodes_mask[i]
            local_map = self.local_maps[i]
            dst_m = local_map[dst]
            hg = dglsp.from_coo(dst_m,src)
            # num_nodes_dict = {'U_': graph.number_of_nodes(), 'V_': nodes.size(0)}
            # edges_dict = {
            #     ('U_', 'U_to_V_', 'V_'): (src, dst_m)
            # }
            # hg = dgl.heterograph(edges_dict, num_nodes_dict=num_nodes_dict) # type: ignore
            res.append(hg)
        return res, self.nodes_mask
    def get_graphs(self,to_hetegraph:bool = False):
        if to_hetegraph:
            res = []
            for i,graph_id in enumerate(self.my_graphs_id):
                graph = self.gpu_graphs[graph_id]
                src, dst = graph.edges()

                nodes = self.nodes_mask[i]
                local_map = self.local_maps[i]
                dst_m = local_map[dst]
                num_nodes_dict = {'U_': graph.number_of_nodes(), 'V_': nodes.size(0)}
                edges_dict = {
                    ('U_', 'U_to_V_', 'V_'): (src, dst_m)
                }
                hg = dgl.heterograph(edges_dict, num_nodes_dict=num_nodes_dict) # type: ignore
                res.append(hg)
            return res, self.nodes_mask
        else:
            return [self.gpu_graphs[graph_id] for graph_id in self.my_graphs_id], self.nodes_mask
        
    def nfeats(self,index:int) -> torch.Tensor:
        
        return self.gpu_graphs[self.my_graphs_id[index]].ndata['feat']
    def get_feats_dim(self) -> int:
        # keys_list = list(self.gpu_graphs.keys())
        return self.gpu_graphs[self.my_graphs_id[0]].ndata['feat'].size(1)
    def get_labels(self) -> torch.Tensor:
        labels = []
        for i,graph_id in enumerate(self.my_graphs_id):
            label = self.gpu_graphs[graph_id].ndata['label'].view(self.gpu_graphs[graph_id].num_nodes(),-1)
            label = label[self.nodes_mask[i]]
            labels.append(label)
        labels = torch.vstack(labels)
        return labels
        # return [graph.ndata['label'] for graph in self.gpu_graphs.values()]
    def get_mask(self,type_s:str='train') -> torch.Tensor:
        key = -1
        if type_s == 'train':
            key = 0
        elif type_s == 'valid':
            key = 1
        elif type_s == 'test':
            key = 2
        else:
            raise ValueError(f"Only 'train','valid' or 'test' are supoorted, Please check {type_s}")
        masks = []
        for i,graph_id in enumerate(self.my_graphs_id):
            graph = self.gpu_graphs[graph_id]
            a = (graph.ndata['node_mask'] == key).flatten()
            masks.append(a[self.nodes_mask[i]])
        masks = torch.hstack(masks)
        return masks
        # return [(graph.ndata['node_mask'] == key).flatten() for graph in self.gpu_graphs.values()]

    def get_com_time(self):
        t = self.comm_t
        self.comm_t = 0.0
        return t
    
    @torch.no_grad()
    def feat_send(self,index:int,feat:torch.Tensor,buff:int=0):
        if self.async_op:
            t0 = time.time()
            feat = self._send_rt(index,feat,buff)
            self.comm_t += time.time() - t0
        else:
            torch.cuda.synchronize(self.device)
            t0 = time.time()
            feat = self._send_rt_sync(index,feat,buff)
            torch.cuda.synchronize(self.device)
            self.comm_t += time.time() - t0
    
    
    @torch.no_grad()
    def _send_rt_sync(self,index:int,feat:torch.Tensor,buff:int=0):
        data = feat
        output = self.recv_buffs[buff][self.recv_offset_index[index][0]:self.recv_offset_index[index][1]]
        input = data[self.fetch_data_to_stable[index]]        
        work = dist.all_to_all_single(
            output = output, input = input,
            output_split_sizes = self.recv_nums_list[index], input_split_sizes = self.send_nums_list[index],
            async_op=False, group=self.gpu_group
        )
        # work.wait() # type: ignore
        if index == self.graph_num-1:
            for ind in range(index+1,self.max_graph_num):
                output = self.recv_buffs[buff][self.recv_offset_index[ind][0]:self.recv_offset_index[ind][1]]
                input = torch.zeros(size=(0,self.dim),dtype=data.dtype,device=self.device)
                works = dist.all_to_all_single(
                    output = output, input = input,
                    output_split_sizes = self.recv_nums_list[ind], input_split_sizes = self.send_nums_list[ind],
                    async_op=False, group=self.gpu_group
                )
                # works.wait() # type: ignore
        return feat.contiguous()
    
    @torch.no_grad()
    def _send_rt(self,index:int,feat:torch.Tensor,buff:int=0):
        data = feat
        output = self.recv_buffs[buff][self.recv_offset_index[index][0]:self.recv_offset_index[index][1]]
        input = data[self.fetch_data_to_stable[index]]
        self.comm_ops.append(
            dist.all_to_all_single(
                output = output, input = input,
                output_split_sizes = self.recv_nums_list[index], input_split_sizes = self.send_nums_list[index],
                async_op=True, group=self.gpu_group
            )
        )
        if index == self.graph_num-1:
            for ind in range(index+1,self.max_graph_num):
                output = self.recv_buffs[buff][self.recv_offset_index[ind][0]:self.recv_offset_index[ind][1]]
                input = torch.zeros(size=(0,self.dim),dtype=data.dtype,device=self.device)
                self.comm_ops.append(
                    dist.all_to_all_single(
                        output = output, input = input,
                        output_split_sizes = self.recv_nums_list[ind], input_split_sizes = self.send_nums_list[ind],
                        async_op=True, group=self.gpu_group
                    )
                )
    
    @torch.no_grad()
    def communicate_rt(self):
        t0 = time.time()
        for op in self.comm_ops:
            op.wait()
        self.comm_ops = []
        self.comm_t += time.time() - t0

    @torch.no_grad()
    def _recv_rt(self,index:int,feat:torch.Tensor,buff:int=0):
        
        len_s = self.local_maps[index].size(0)
        data = torch.zeros(size=(len_s,self.dim),device=self.device,dtype=torch.float32,requires_grad=True)
        data[self.nodes_mask[index]] = feat
        if index == 0:
            self.communicate_rt()  ## wait before to feat data from communication buff
        data[self.fit_to_data[index],:] = self.recv_buffs[buff][self.fetch_data_from_rtable[index],:]
        return data
