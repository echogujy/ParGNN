
current_datetime=$(date +"%Y-%m-%d %H:%M:%S")
echo "=============================="
echo "start time: $current_datetime"
which mpirun
which python
echo "=============================="
export MASTER_PORT=25310
export MASTER_ADDR=localhost
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.9,max_split_size_mb:512"

export TORCH_DISTRIBUTED_DEBUG=OFF

# echo end on $(date)
backend="nccl"
cd ../src/
apps="python main_train.py \
    --dataset ogbn-products \
    --epochs 300 \
    --hidden 256 \
    --num_class 47 \
    --model gcn \
    --num_layers 3 \
    --total_parts 16 \
    --seed 2024 \
    --use_pipeline 0 \
    --reparter base \
    --data_path ../data/DistData \
    --backend $backend"
echo $apps

mpirun -n 4 $apps 

apps="python main_train.py \
    --dataset ogbn-products \
    --epochs 300 \
    --hidden 256 \
    --num_class 47 \
    --model gcn \
    --num_layers 3 \
    --total_parts 16 \
    --seed 2024 \
    --use_pipeline 1 \
    --reparter base \
    --data_path ../data/DistData \
    --backend $backend"
echo $apps

mpirun -n 4 $apps 


apps="python main_train.py \
    --dataset ogbn-products \
    --epochs 300 \
    --hidden 256 \
    --num_class 47 \
    --model gcn \
    --num_layers 3 \
    --total_parts 16 \
    --seed 2024 \
    --use_pipeline 0 \
    --reparter adapt \
    --data_path ../data/DistData \
    --backend $backend"
echo $apps

mpirun -n 4 $apps 

apps="python main_train.py \
    --dataset ogbn-products \
    --epochs 300 \
    --hidden 256 \
    --num_class 47 \
    --model gcn \
    --num_layers 3 \
    --total_parts 16 \
    --seed 2024 \
    --use_pipeline 1 \
    --reparter adapt \
    --data_path ../data/DistData \
    --backend $backend"
echo $apps

mpirun -n 4 $apps 


## ========================================

apps="python main_train.py \
    --dataset ogbn-products \
    --epochs 300 \
    --hidden 256 \
    --num_class 47 \
    --model gat \
    --num_heads 3 \
    --num_layers 4 \
    --total_parts 16 \
    --seed 2024 \
    --use_pipeline 0 \
    --reparter base \
    --data_path ../data/DistData \
    --backend $backend"
echo $apps

mpirun -n 4 $apps 

apps="python main_train.py \
    --dataset ogbn-products \
    --epochs 300 \
    --hidden 256 \
    --num_class 47 \
    --model gat \
    --num_heads 3 \
    --num_layers 4 \
    --total_parts 16 \
    --seed 2024 \
    --use_pipeline 1 \
    --reparter base \
    --data_path ../data/DistData \
    --backend $backend"
echo $apps

mpirun -n 4 $apps 


apps="python main_train.py \
    --dataset ogbn-products \
    --epochs 300 \
    --hidden 256 \
    --num_class 47 \
    --model gat \
    --num_heads 4 \
    --num_layers 3 \
    --total_parts 16 \
    --seed 2024 \
    --use_pipeline 0 \
    --reparter adapt \
    --data_path ../data/DistData \
    --backend $backend"
echo $apps

mpirun -n 4 $apps 

apps="python main_train.py \
    --dataset ogbn-products \
    --epochs 300 \
    --hidden 256 \
    --num_class 47 \
    --model gat \
    --num_heads 4 \
    --num_layers 3 \
    --total_parts 16 \
    --seed 2024 \
    --use_pipeline 1 \
    --reparter adapt \
    --data_path ../data/DistData \
    --backend $backend"
echo $apps

mpirun -n 4 $apps 


current_datetime=$(date +"%Y-%m-%d %H:%M:%S")
echo "=============================="
echo "end time: $current_datetime"
echo "=============================="
## watch -n 0.5 nvidia-smi