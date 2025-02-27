# ParGNN: A Scalable Graph Neural Network Training Framework on multi-GPUs

ParGNN is accepted by DAC 2025.

ParGNN, an efficient full-batch training system for GNNs, which adopts a profiler-guided adaptive load balancing partition method(PGALB) and a subgraph pipeline algorithm to overlap communication and computation.

### 1. Clone this project and Setup environment

### 2. Install PGALB and partition graph data

###### 2.1. setup pgalb

```shell
cd pgalb  ## go into the pgalb dirctionary
python setup.py build_ext --inplace  ## install pgalb
python test_adpat.py   ## check the install of C extension
```

###### 2.2. parition and reparition

* ParGNN use two stage partition to deal with the load unbalance question, and provide the 3 functions:
    * graph_partition_dgl_metis:  load graph from local path(or download online automately), and create DGL -format graph. Then run the intial patition.
    * graph_eval: profile graph the subgraph from the inital parition.
    * mapping: map the subgraph to a cluster for running in the same GPUs.
* we provide two scripts to test the partition and repartition process, just run:
    ```shell
    python graph_partition.py
    python repart.py
    ```
* data used in the evaluation on paper: (script will download them automately when they are needed)
    * ogb graph dataset: ogbn-products, ogbn-proteins from https://ogb.stanford.edu/docs/nodeprop/
    * yelp dataset: https://www.dgl.ai/dgl_docs/generated/dgl.data.YelpDataset.html#dgl.data.YelpDataset
    * reddit dataset: https://www.dgl.ai/dgl_docs/generated/dgl.data.RedditDataset.html

### 3. Run distributed GNN trainning
    The scripts dirctionary has the example scripts to run ParGNN. 
    ```shell
    cd scripts
    sh train_all.sh
    ```









































































*
