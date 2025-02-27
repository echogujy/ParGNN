import torch, numpy as np, dgl
import torch.distributed as dist
import os, random, sys
import os.path as osp
from datetime import timedelta
import argparse
from typing import List, Optional,Dict
import scipy.sparse as sps
import psutil
import torch.nn as nn
import torch.nn.functional as F
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

class My_loss(nn.Module):
    def __init__(self,type:str='nll_loss'):
        super(My_loss,self).__init__()
        if type == 'nll_loss':
            self.fn  = F.nll_loss
        elif 'roc_auc':
            self.fn = F.binary_cross_entropy_with_logits
        else:
            self.fn = F.nll_loss
    def forward(self,x,y,mask):
        loss = self.fn(x[mask],y[mask])
        if torch.isnan(loss):
            with torch.no_grad():
                loss.data = torch.tensor(0,dtype=torch.float32,device=y.device).data
        return loss

class EvalModel:
    def __init__(self, type: str = 'acc', device: torch.device = torch.device('cpu')):
        self.is_roc_auc = False
        if type == "roc_auc":
            self.is_roc_auc = True
        self.device = device
        
    def acc(self, pred:torch.Tensor, true:torch.Tensor):
        with torch.no_grad():
            data = torch.zeros(size=(2,),dtype=torch.int64,device=self.device)
            print(5)
            sys.stdout.flush()
            print(51)
            sys.stdout.flush()
            output_k = pred.argmax(1)
            print(6)
            sys.stdout.flush()
            right = torch.sum(output_k == true)
            data[0] += right
            data[1] += output_k.shape[0]
            print(7)
            sys.stdout.flush()
            dist.all_reduce(data, op=dist.ReduceOp.SUM)
            print(8)
        return data[0] / (data[1] + 1e-10)

def ddp_set_env_variables(only_view_one:int=False):
    import os
    if "OMPI_COMM_WORLD_RANK" in os.environ:
        # lrank=$OMPI_COMM_WORLD_LOCAL_RANK
        # RANK=$OMPI_COMM_WORLD_RANK
        # WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        os.environ['MASTER_PORT'] = str(int(os.environ['MASTER_PORT']) +  world_size)
    else: 
        try:
            os.environ["RANK"] = os.environ["SLURM_PROCID"]
            os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
            os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
            rank = int(os.environ["SLURM_PROCID"])
            world_size = int(os.environ["SLURM_NTASKS"])  ## num_gpus * N
            local_rank = int(os.environ['SLURM_LOCALID'])

            os.environ['OMP_NUM_THREADS'] = os.environ['SLURM_CPUS_PER_TASK']
            torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
        except:
            print("Please use mpi or slurm to run this code")
            exit(100)
    if only_view_one:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)
    return rank,world_size,local_rank

def rand_seed(seed:int = 2024):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    dgl.seed(seed)
    if torch.cuda.is_available():
        # torch.backends.cudnn.benchmark = False # if useless for HIP
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True # type: ignore
    # torch.autograd.set_detect_anomaly(False)
    # torch.autograd.profiler.profile(False)
    # torch.autograd.profiler.emit_nvtx(False)


def argment_parser(rank:int=0,world_size:int=0,local_rank:int=0):
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--activateion',
                        default='relu', type= str,
                        help='the activation you want')
    parser.add_argument('--epochs',
                        default=30, type=int, metavar='N',
                        help='number of total epochs to run (default:30)')
    parser.add_argument('--lr',
                        '--learning-rate',
                        default=0.001, type=float, metavar='LR',
                        help='initial learning rate (default: 1e-3)',
                        dest='lr')
    parser.add_argument('--momentum',
                        default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd',
                        '--weight-decay',
                        default=1e-5, type=float, metavar='W',
                        help='weight decay (default: 1e-5)',
                        dest='weight_decay')
    parser.add_argument('--print_freq',
                        default=10,
                        type=int,
                        metavar='N',
                        help='print frequency (default: 10)')
    parser.add_argument('-e',
                        '--evaluate',
                        dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--seed',
                        default=2024,
                        type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--hidden',
                        default=128, \
                        type=int,
                        help='The dimension of hidden layers. ')
    parser.add_argument('--num_class',
                        default=40, \
                        type=int,
                        help='The dimension of hidden layers. ')
    parser.add_argument('--num_layers',
                        default=4,
                        type=int,
                        help='The number of layers. ')
    parser.add_argument('--backend',
                        default='nccl',
                        type=str,
                        help='file used to initial distributed training')
    parser.add_argument('--world_size', 
                        default=world_size, 
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', 
                        default=rank, 
                        type=int,
                        help='node rank for distributed training')
    parser.add_argument('--local_rank', 
                        default=local_rank, 
                        type=int,
                        help='node rank for distributed training in local')
    parser.add_argument('--dist_url', 
                        default='env://', 
                        type=str,
                         help='url used to set up distributed training')   
    parser.add_argument('--dropout',
                        default=0.5,
                        type=float, 
                        help='the dropout rate')    

    
    ## ======================  ##
    parser.add_argument('--partitioner',
                        default='metis', type=str,
                        help='partitioner for graph: ("metis", "hash", ..)') 
    parser.add_argument('--dataset',
                        default='ogbn-arxiv', type= str,
                        help='the name of dataset to train')  
    parser.add_argument('--data_path',
                        default='/public/home/liufang395/gujy/DirGragh/data', type = str,
                        help='path to dataset')

    parser.add_argument('--total_parts',
                        default=1, type = int,
                        help='total number of sub graph')     
    parser.add_argument('--model',
                        default='gcn', type = str)
    parser.add_argument('--num_heads',
                        default=2, type = int)
    parser.add_argument('--use_pipeline',
                        default=1, type = int,
                        help='If use pipeline communication: 1-Y, 0-N')
    parser.add_argument('--reparter',
                        default='base', type = str,
                        help = " The repart methon : base or adapt ")
    
    
    parser.add_argument('--con_adapt',
                        default='lh-lp', type = str,
                        help = " The repart methon : base or adapt ")
    parser.add_argument('--update_epoch',
                        default=20, type=int, metavar='N',
                        help='number of total epochs to run (default:20)')
    args = parser.parse_args()
    return args
