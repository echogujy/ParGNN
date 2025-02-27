
import torch,sys
import torch.distributed as dist
import os
import os.path as osp
import pandas as pd
import numpy as np
import time
from datetime import timedelta
import torch.multiprocessing as mp
from typing import List
from torch.nn.parallel import DistributedDataParallel as DDP
sys.path.append(osp.dirname(osp.abspath(__file__)))
import argparse
from utils import ddp_set_env_variables,argment_parser,rand_seed
from managers import GraphManager
from utils import print_memory_usage, My_loss, EvalModel
import torch._dynamo.config as dynamo_config
torch._dynamo.reset()
def acc_dist(pred:torch.Tensor, true:torch.Tensor):
    with torch.no_grad():
        data = torch.zeros(size=(2,),dtype=torch.int64,device=pred.device)
        output_k = pred.argmax(1)
        right = torch.sum(output_k == true)
        data[0] += right
        data[1] += output_k.shape[0]
        dist.all_reduce(data, op=dist.ReduceOp.SUM)
    return data[0] / (data[1] + 1e-10)
from torchmetrics import AUROC
def roc_auc_dist(preds, true):
    preds_tensor = preds.clone().detach()
    true_tensor = true.clone().detach()
    # num_classes = preds_tensor.shape[1]
    roc_aucs = torch.zeros(1, dtype=torch.float32, device=preds_tensor.device)
    auroc = AUROC(task='binary')


        # 计算每个进程上的 AUC
    auc = auroc(preds_tensor,true_tensor)

    # 收集所有进程上的 AUC
    dist.all_reduce(auc, op=dist.ReduceOp.SUM)
    roc_aucs= auc / dist.get_world_size()
    return roc_aucs

def main(args:argparse.Namespace):
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_id = torch.cuda.current_device()
    device = torch.device(f"cuda:{device_id}")
    # device = torch.device(f"cpu")
    enable_pipeline = True if args.use_pipeline == 1 else False
    hidden_dim_comm = args.hidden if args.model == 'gcn' else args.hidden  * args.num_heads
    graph_man = GraphManager(args,device=device,
                             addselfloop=False,async_comm_mode=enable_pipeline,
                             hidden_dim=hidden_dim_comm,nlayers=args.num_layers)
    from models import GCN_model, GAT_model

    labels = graph_man.get_labels()
    labels = labels.squeeze(1)
    # labels = torch.remainder(labels,40)
    train_masks = graph_man.get_mask('train')
    valid_masks = graph_man.get_mask('valid')
    test_masks = graph_man.get_mask('test')
    # print(train_masks,valid_masks,test_masks)
    # print(train_masks.sum(),valid_masks.sum(),test_masks.sum())

    if args.model == 'gcn':
        model = GCN_model(
            in_features=graph_man.get_feats_dim(),
            hidden_features=args.hidden,
            num_classes=args.num_class,
            nlayers=args.num_layers,
            dropout=args.dropout,
            device = device
        )
    elif args.model == 'gat':
        model = GAT_model(
            in_features=graph_man.get_feats_dim(),
            hidden_features=args.hidden,
            num_classes=args.num_class,
            nlayers=args.num_layers,
            dropout=args.dropout,
            num_heads=args.num_heads,
            device = device
        )

    if args.dataset == 'yelp' or args.dataset=='ogbn-proteins':
        criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
        labels = labels.float()
    else:
        criterion = torch.nn.CrossEntropyLoss()
        labels = labels.flatten()
    # nll_loss_type = ['ogbn-arxiv','ogbn-products','ogbn-papers100M']
    # if args.dataset in nll_loss_type:
    #     criterion = My_loss(type='nll_loss') 
    #     evaliation = EvalModel(device=device)
    # model = torch.compile(model)
    model = DDP(model,device_ids=[device_id],output_device=device_id,gradient_as_bucket_view=True)
    # model = DDP(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    model.train()
    dist.barrier()
    if rank == 0:
        if enable_pipeline:
            print(f'Training on {args.dataset} with  {args.reparter} partiioning and pipeline')
        else:
            print(f'Training on {args.dataset} with  {args.reparter} partiioning')
        print(f"== print model structure ==")
        print(args)
        print(model)
        print(f"=======================")
 
    used_matrix = False
    feats = graph_man.get_feats()
    if used_matrix:
        maxtrices, n_masks = graph_man.get_matrices()
    else:
        graphs,n_masks = graph_man.get_graphs(to_hetegraph=True)
        feats2 = [feats[i][n_masks[i]] for i in range(graph_man.graph_num)]
    train_t, comm_t  = [], []
    train_acc_l, val_acc_l, test_acc_l = [], [], []
    train_loss  = []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()   
    # torch.cuda.empty_cache()
    for epoch in range(args.epochs):
        t0 = time.time()

        if used_matrix:
            output = model(graph_man,maxtrices,feats)
        else:
            output = model(graph_man,graphs,feats,feats2)
        loss = criterion(output[train_masks], labels[train_masks]) # type:ignore
        # t_f = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

        t_end = time.time()
        if args.dataset == 'yelp' or args.dataset=='ogbn-proteins':
            train_acc = roc_auc_dist(output[train_masks], labels[train_masks])
            val_acc = roc_auc_dist(output[valid_masks], labels[valid_masks])
            test_acc = roc_auc_dist(output[test_masks], labels[test_masks])
        else:
            train_acc = acc_dist(output[train_masks], labels[train_masks]) # type:ignore
            val_acc = acc_dist(output[valid_masks], labels[valid_masks]) # type:ignore
            test_acc = acc_dist(output[test_masks], labels[test_masks]) # type:ignore
        # train_acc = acc_dist(output[train_masks], labels[train_masks]) # type:ignore
        # val_acc = acc_dist(output[valid_masks], labels[valid_masks]) # type:ignore
        # test_acc = acc_dist(output[test_masks], labels[test_masks]) # type:ignore
        comm_it = graph_man.get_com_time()
        train_acc_l.append(train_acc.item())
        val_acc_l.append(val_acc.item())
        test_acc_l.append(test_acc.item())
        
        train_t.append(t_end - t0)
        comm_t.append(comm_it)

        train_loss.append(loss.item())
        if rank == 0 and (epoch+1) % args.print_freq == 0: 
            print(f"rank:{rank},Epoch:{epoch},loss:{loss.item():.4f}, time:{np.mean(train_t[5:]):.4f}, comm time:{np.mean(comm_t[5:]):.4f} ,acc:{train_acc:.4f}, val_acc:{val_acc:.4f}, test_acc:{test_acc:.4f}")
            sys.stdout.flush()

    if rank == 0:
        print_memory_usage()
    train_dur = torch.tensor(train_t,dtype=torch.float32)
    comm_dur = torch.tensor(comm_t,dtype=torch.float32)
    
    train_acc = torch.tensor(train_acc_l,dtype=torch.float32)
    val_acc = torch.tensor(val_acc_l,dtype=torch.float32)
    test_acc = torch.tensor(test_acc_l,dtype=torch.float32)
    train_loss = torch.tensor(train_loss,dtype=torch.float32) 
    
    data_t = torch.vstack([train_dur,comm_dur,train_acc,val_acc,test_acc,train_loss]).to(device)
    dist.all_reduce(data_t,op=dist.ReduceOp.SUM)
    data_t = data_t/dist.get_world_size()

    # frame: pipe
    frame = '_pipe' if enable_pipeline else ''
    filename = f'../results/{args.dataset}_{args.model}_{args.total_parts}to{world_size}_{args.reparter}{frame}_{args.hidden}.pt'
    if rank == 0:
        print_memory_usage()
        os.makedirs("../results/",exist_ok=True)
        print(f'save to {filename}')
        torch.save(data_t,filename)
    dist.barrier()
 
def main_prof(args:argparse.Namespace):
    
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_id = torch.cuda.current_device()
    device = torch.device(f"cuda:{device_id}")

    enable_pipeline = True if args.use_pipeline == 1 else False
    graph_man = GraphManager(args,device=device,
                             addselfloop=False,async_comm_mode=enable_pipeline,
                             hidden_dim=args.hidden,nlayers=args.num_layers)

    from models import GCN_model, GAT_model

    labels = graph_man.get_labels()
    labels = labels.squeeze(1)
    train_masks = graph_man.get_mask('train')
    valid_masks = graph_man.get_mask('valid')
    test_masks = graph_man.get_mask('test')
    # print(train_masks,valid_masks,test_masks)
    # print(train_masks.sum(),valid_masks.sum(),test_masks.sum())
    
    if args.model == 'gcn':
        model = GCN_model(
            in_features=graph_man.get_feats_dim(),
            hidden_features=args.hidden,
            num_classes=args.num_class,
            nlayers=args.num_layers,
            dropout=args.dropout,
            device = device
        )
    elif args.model == 'gat':
        model = GAT_model(
            in_features=graph_man.get_feats_dim(),
            hidden_features=args.hidden,
            num_classes=args.num_class,
            nlayers=args.num_layers,
            dropout=args.dropout,
            num_heads=args.num_heads,
            device = device
        )
    if args.dataset == 'yelp' or args.dataset=='ogbn-proteins':
        criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
        labels = labels.float()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    # nll_loss_type = ['ogbn-arxiv','ogbn-products','ogbn-papers100M']
    # if args.dataset in nll_loss_type:
    #     criterion = My_loss(type='nll_loss') 
    #     evaliation = EvalModel(device=device)
    # model = torch.compile(model)
    model = DDP(model,device_ids=[device_id],output_device=device_id,find_unused_parameters=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    model.train()
    if rank == 0:
        if enable_pipeline:
            print(f'Training on {args.dataset} with {args.partitioner} base partiioning and pipeline')
        else:
            print(f'Training on {args.dataset} with {args.partitioner} base partiioning')
        print(f"== print model structure ==")
        print(args)
        print(model)
        print(f"=======================")
    graphs,n_masks = graph_man.get_graphs(to_hetegraph=True)
    feats = graph_man.get_feats()  
    feats2 = [feats[i][n_masks[i]].clone() for i in range(graph_man.graph_num)]
    train_t, comm_t  = [], []
    train_acc_l, val_acc_l, test_acc_l = [], [], []
    train_loss  = []
    prof =  torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./'),
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )
    

    for epoch in range(5):
        t0 = time.time()

        output = model(graph_man,graphs,feats,feats2)
        # print("forward-0")
        # sys.stdout.flush()
        loss = criterion(output[train_masks], labels[train_masks]) # type:ignore
        # t_f = time.time()
        optimizer.zero_grad()
        # print("backward-0")
        # sys.stdout.flush()
        loss.backward()
        # print("backward-1")
        # sys.stdout.flush()
        optimizer.step() 

        t_end = time.time()
        # if args.dataset == 'yelp' or args.dataset=='ogbn-proteins':
        #     train_acc = roc_auc_dist(output[train_masks], labels[train_masks])
        #     val_acc = roc_auc_dist(output[valid_masks], labels[valid_masks])
        #     test_acc = roc_auc_dist(output[test_masks], labels[test_masks])
        # else:
        #     train_acc = acc_dist(output[train_masks], labels[train_masks]) # type:ignore
        #     val_acc = acc_dist(output[valid_masks], labels[valid_masks]) # type:ignore
        #     test_acc = acc_dist(output[test_masks], labels[test_masks]) # type:ignore
        # comm_it = graph_man.get_com_time()
        # train_acc_l.append(train_acc.item())
        # val_acc_l.append(val_acc.item())
        # test_acc_l.append(test_acc.item())
        
        # train_t.append(t_end - t0)
        # comm_t.append(comm_it)

        # train_loss.append(loss.item())
        # if rank == 0 and (epoch+1):
        #     print(f"rank:{rank},Epoch:{epoch},loss:{loss.item():.4f}, time:{np.mean(train_t[5:]):.4f}, comm time:{np.mean(comm_t[5:]):.4f} ,acc:{train_acc:.4f}, val_acc:{val_acc:.4f}, test_acc:{test_acc:.4f}")
        #     sys.stdout.flush()

    prof.start()
    for epoch in range(3):
        t0 = time.time()

        output = model(graph_man,graphs,feats,feats2)
        loss = criterion(output[train_masks], labels[train_masks]) # type:ignore
        # t_f = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

        t_end = time.time()
        
        # train_acc = acc_dist(output[train_masks], labels[train_masks]) # type:ignore
        # val_acc = acc_dist(output[valid_masks], labels[valid_masks]) # type:ignore
        # test_acc = acc_dist(output[test_masks], labels[test_masks]) # type:ignore
        # comm_it = graph_man.get_com_time()
        # train_acc_l.append(train_acc.item())
        # val_acc_l.append(val_acc.item())
        # test_acc_l.append(test_acc.item())
        
        # train_t.append(t_end - t0)
        # comm_t.append(comm_it)

        # train_loss.append(loss.item())
        # if rank == 0 and (epoch+1) % args.print_freq == 0:
        #     print(f"rank:{rank},Epoch:{epoch},loss:{loss.item():.4f}, time:{np.mean(train_t[5:]):.4f}, comm time:{np.mean(comm_t[5:]):.4f} ,acc:{train_acc:.4f}, val_acc:{val_acc:.4f}, test_acc:{test_acc:.4f}")
        #     sys.stdout.flush()

        prof.step()
    prof.stop()
    # print_memory_usage()
    # train_dur = torch.tensor(train_t,dtype=torch.float32)
    # comm_dur = torch.tensor(comm_t,dtype=torch.float32)
    
    # train_acc = torch.tensor(train_acc_l,dtype=torch.float32)
    # val_acc = torch.tensor(val_acc_l,dtype=torch.float32)
    # test_acc = torch.tensor(test_acc_l,dtype=torch.float32)
    # train_loss = torch.tensor(train_loss,dtype=torch.float32) 
    
    # data_t = torch.vstack([train_dur,comm_dur,train_acc,val_acc,test_acc,train_loss]).to(device)
    # dist.all_reduce(data_t,op=dist.ReduceOp.SUM)
    # data_t = data_t/dist.get_world_size()

    # # frame: pipe
    # frame = '_pipe' if enable_pipeline else ''
    # filename = f'../results/{args.dataset}_{args.model}_{args.total_parts}to{world_size}_{args.reparter}{frame}.pt'
    # if rank == 0:
    #     print_memory_usage()
    #     os.makedirs("../results/",exist_ok=True)
    #     print(f'save to {filename}')
    #     torch.save(data_t,filename)
    # dist.barrier()
 

if __name__ == '__main__':
    rank,world_size,local_rank = ddp_set_env_variables()
    torch.set_float32_matmul_precision('high')
    if torch.cuda.is_available():
        if torch.cuda.device_count() == 1:
            device = torch.cuda.current_device()
        else:
            device = torch.device(f'cuda:{local_rank + 4}')
        torch.cuda.set_device(device)
    else:
        raise RuntimeError('GPU is not available,please check!!')
    args = argment_parser(rank,world_size,local_rank)
    rand_seed(args.seed)
    dist.init_process_group(backend=args.backend, init_method=args.dist_url)
    main(args)
    # main_prof(args)
    dist.destroy_process_group()

    
