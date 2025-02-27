from utils_plus import MetisPy, RefinePy  # type: ignore

import numpy as np
from scipy.sparse import csr_matrix


num_nodes = 100


num_edges = 800


np.random.seed(0) 
edges = np.random.randint(0, num_nodes, size=(2, num_edges))
edges = np.unique(edges, axis=1)


edges = edges[:, edges[0] != edges[1]]


edges = np.sort(edges, axis=0)
edges, _ = np.unique(edges, return_index=True, axis=1)


num_edges = edges.shape[1]


row = edges[0]
col = edges[1]
data = np.ones(num_edges)  


A = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))

vwgt = np.random.randint(1, 10, size=num_nodes).astype(np.int64)
ewgt = np.random.randint(1, 10, size=num_edges).astype(np.int64)

indptr = A.indptr.astype(np.int64)
indices = A.indices.astype(np.int64)
a = MetisPy(4,num_nodes,1,indptr,indices,vwgt,ewgt)

b = RefinePy(4,num_nodes,1,indptr,indices,vwgt,ewgt,a)
print("Checking metis: ",a)
print("Checking adapt: ",b)
if (sum(a) > 0 and sum(b) > 0):
    print("Checking Pass without error calls.")
