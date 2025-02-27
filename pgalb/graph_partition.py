import dgl
import numpy as np
import torch, json
import os
import os.path as osp
import torch.multiprocessing as mp
from typing import List, Dict
import time
from ogb.nodeproppred import DglNodePropPredDataset
def call_back(res):
    return None

def err_call_back(err):
    print(f'!!! ~ error: {str(err)}')


def graph_partition_dgl_metis(
    name:str = "ogbn-arxiv",
    dir_path:str = "/work/gujy/data",
    save_path:str = "/work/gujy/data/DistData",
    algo_obj : str = 'cut',
    num_part : int = 16,
    num_processes : int = 4,
):
    """
    :param name: 
    :param dir_path:
    :param algo:
    :param num_part:
    :param to_undirected:
    :param num_processes:
    :return:
    """
    t_p = time.time()
    if name in ['ogbn-arxiv','ogbn-products']:
        dataset = DglNodePropPredDataset(name=name,root=dir_path)
        graph : dgl.DGLGraph = dataset[0][0]

        feat = graph.ndata['feat']
        label : torch.Tensor = dataset[0][1].reshape(graph.num_nodes(),-1) # type: ignore
        split_idx  = dataset.get_idx_split()
        mask = torch.ones(size=(graph.num_nodes(),), dtype=torch.int32).mul_(-1)
        mask[split_idx["train"]] = 0 # type: ignore
        mask[split_idx["valid"]] = 1 # type: ignore
        mask[split_idx["test"]] = 2  # type: ignore
    elif name in ['ogbn-proteins']:
        dataset = DglNodePropPredDataset(name='ogbn-proteins',root=dir_path)
        # print(dataset[0][0])
        graph : dgl.DGLGraph = dataset[0][0]
        label : torch.Tensor = dataset[0][1].reshape(graph.num_nodes(),-1) # type: ignore
        A = graph.adj()
        A = dgl.sparse.val_like(A,graph.edata['feat'])
        feat = A.smean(1)
        split_idx  = dataset.get_idx_split()
        mask = torch.ones(size=(graph.num_nodes(),), dtype=torch.int32).mul_(-1)
        mask[split_idx["train"]] = 0 # type: ignore
        mask[split_idx["valid"]] = 1 # type: ignore
        mask[split_idx["test"]] = 2  # type: ignore
    elif name in ['yelp','reddit', 'flickr']:
        from dgl.data import YelpDataset, RedditDataset
        dataset = YelpDataset(raw_dir=dir_path) if name == 'yelp' else RedditDataset(raw_dir=dir_path)
        if name == 'flickr':
            from dgl.data import FlickrDataset
            dataset = FlickrDataset(raw_dir=dir_path)
        
        dataset.num_classes
        graph = dataset[0]
        # print(graph)
        # get node feature
        feat = graph.ndata['feat']
        # get node labels
        label = graph.ndata['label']
        # get data split
        train_mask = graph.ndata['train_mask'].to(torch.bool)
        val_mask = graph.ndata['val_mask'].to(torch.bool)
        test_mask = graph.ndata['test_mask'].to(torch.bool)

        mask = torch.ones(size=(graph.num_nodes(),), dtype=torch.int32).mul_(-1)
        mask[train_mask] = 0
        mask[val_mask] = 1
        mask[test_mask] = 2

    elif 'igb' in name:
        _ , dataset_size = name.split('-')
        from igb.dataloader import IGB260M
        dir_path = osp.join(dir_path,"IGB260M")
        ## need to download first
        dataset = IGB260M(root=dir_path, size=dataset_size, in_memory=1, classes=19, synthetic=0)
        feat = torch.from_numpy(dataset.paper_feat)
        node_edges = torch.from_numpy(dataset.paper_edge)
        label = torch.from_numpy(dataset.paper_label).to(torch.long)

        graph = dgl.graph((node_edges[:, 0],node_edges[:, 1]), num_nodes=feat.shape[0])

        mask = torch.ones(size=(graph.num_nodes(),), dtype=torch.int32).mul_(-1)
        n_nodes = feat.shape[0]
        n_train = int(n_nodes * 0.6)
        n_val   = int(n_nodes * 0.2)

        mask[:n_train] = 0
        mask[n_train:n_train + n_val] = 1
        mask[n_train + n_val:] = 2 

    else:
        
        print(f"Data: {name} is not support now, please check it!")
        raise 

    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
    graph = dgl.to_bidirected(graph)

    t0 = time.time()
    parts_re : torch.Tensor = dgl.metis_partition_assignment(graph, k = num_part, objtype=algo_obj) # type: ignore
    t1 = time.time()
    assert num_part == parts_re.max() + 1, "The number of partitions should be equal to the maximum partition id plus 1."
    assert graph.num_nodes() == parts_re.shape[0], "The number of nodes in the graph should be equal to the length of partition ids." # type: ignore
    
    part_metadata = {}
    print(save_path,name,f"part_{num_part}")
    save_path = osp.join(save_path,name, f"part_{num_part}")
    os.makedirs(save_path,exist_ok=True)
    torch.save(parts_re,osp.join(save_path,'part_ids.pt'))

    pool = mp.Pool(processes=num_processes)
    for i in range(num_part):
        part_nodes = torch.where(parts_re == i)[0]
        sub_graphs = dgl.in_subgraph(graph, part_nodes)

        src, dst = sub_graphs.edges()
        node_id_all = torch.unique(torch.cat([src,dst])) # 原图的节点id
        # N_all = node_id_all.shape[0]
        graph_t = {
            'node_id' : node_id_all,  
            'edge_id' : sub_graphs.edata[dgl.EID], # type: ignore
            'src_nodes'   : src,
            'dst_nodes'   : dst,
            'inner_node_num' : part_nodes.size(0)
        }
 
        parts_info = {
            "part_id" : i,
            "num_inner_nodes": graph_t['inner_node_num'],
            "num_nodes" : graph_t['node_id'].shape[0],
            "num_edges" : graph_t['edge_id'].shape[0],
            "part_graph" : osp.join(f'part_{i}','graph.pt'),
        }
        
        ndata = {
            'feat': feat[node_id_all],
            'label': label[node_id_all],
            'mask': mask[node_id_all]
        }
        
        part_metadata[i] = parts_info
        # torch.save(graph_t, os.path.join(save_path,f'graph_{i}.pt'))
        
        pool.apply_async(torch.save, args=(graph_t, os.path.join(save_path,f'graph_{i}.pt')), callback=call_back, error_callback=err_call_back)
        pool.apply_async(torch.save, args=(ndata, os.path.join(save_path,f'nfeat_{i}.pt')), callback=call_back, error_callback=err_call_back)
        
    pool.close()
    pool.join()
    # save metadata
    with open(osp.join(save_path,'metadata.json'), 'w') as f:
        json.dump(part_metadata, f, sort_keys=False, indent=4)
    t2 = time.time()
    print(f"Data: {name} nparts: {num_part} load: {t0-t_p:.2f} part: {t1-t0:.2f} after_part: {t2-t1:.2f}\n\n\n")


if __name__ == '__main__':

    datasets : List[str] = ['yelp','reddit',"ogbn-arxiv",'ogbn-products','ogbn-proteins']
    dir_path : str = "../data"
    save_path: str = "../data/DistData"
    algo : str = 'metis'
    num_part  = 16
    for name in datasets:
        graph_partition_dgl_metis(name=name, dir_path= dir_path, save_path = save_path ,num_part = num_part)