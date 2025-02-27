ctypedef long idx_t
ctypedef double real_t

cimport numpy as np
import numpy as np

## default using 64-bit intergal and float number
cdef extern from "metis.h":
    int METIS_PartGraphRecursive(idx_t *nvtxs, idx_t *ncon, idx_t *xadj,idx_t *adjncy, idx_t *vwgt, idx_t *vsize, idx_t *adjwgt,idx_t *nparts, real_t *tpwgts, real_t *ubvec, idx_t *options,idx_t *edgecut, idx_t *part)

    int METIS_PartGraphKway(idx_t *nvtxs, idx_t *ncon, idx_t *xadj,idx_t *adjncy, idx_t *vwgt, idx_t *vsize, idx_t *adjwgt,idx_t *nparts, real_t *tpwgts, real_t *ubvec, idx_t *options,idx_t *edgecut, idx_t *part)

    int newrefine(idx_t *nvtxs, idx_t *ncon, idx_t *xadj, idx_t *adjncy, idx_t *vwgt, idx_t *vsize, idx_t *adjwgt, idx_t *nparts, real_t *tpwgts, real_t *ubvec, idx_t *options, idx_t *objval, idx_t *part, idx_t *new_part);    
cdef int METIS_NOPTIONS = 40  # 根据实际情况调整这个值
cdef int METIS_OPTION_CTYPE = 2
cdef int METIS_OPTION_IPTYPE = 3
cdef int METIS_OPTION_RTYPE = 4
cdef int METIS_OPTION_NO2HOP = 20
cdef int METIS_OPTION_ONDISK = 10
cdef int METIS_OPTION_DROPEDGES = 19
cdef int METIS_OPTION_MINCONN = 11
cdef int METIS_OPTION_CONTIG = 12
cdef int METIS_OPTION_SEED = 9
cdef int METIS_OPTION_NIPARTS = 6
cdef int METIS_OPTION_NITER = 7
cdef int METIS_OPTION_NCUTS = 8
cdef int METIS_OPTION_UFACTOR = 17
cdef int METIS_OPTION_DBGLVL = 5


def MetisPy(idx_t nparts, idx_t nvtxs, idx_t ncon, idx_t[:] xadj, idx_t[:] adjncy,\
        idx_t[:] vwgt_, idx_t[:] adjwgt_,str method='kway'):

    """
        wrapper of the metis partition
    """
    # tmp = np.zeros(shape=(nparts),dtype=np.int64)
    # cdef idx_t[:] part = tmp.view()
    cdef:
        idx_t[:] part = np.zeros(shape=(nvtxs),dtype=np.int64)
        idx_t ncut 
        idx_t vwgt_size = vwgt_.shape[0]
        idx_t adjwgt_size = adjwgt_.shape[0]
        
    xadj = np.ascontiguousarray(xadj)
    adjncy = np.ascontiguousarray(adjncy)

    #cdef str k = &adjncy
    #print(k)
    #print(&adjncy)
    if method == 'kway':
        if vwgt_size == 0 and adjwgt_size == 0:
            METIS_PartGraphKway(&nvtxs,&ncon,<idx_t *> &xadj[0],<idx_t *> &adjncy[0],\
                NULL,NULL,NULL ,&nparts,\
                NULL,NULL,NULL,\
                &ncut, <idx_t *> &part[0])
        elif vwgt_size == 0 and adjwgt_size != 0:
            adjwgt_ = np.ascontiguousarray(adjwgt_)
            METIS_PartGraphKway(&nvtxs,&ncon,<idx_t *> &xadj[0],<idx_t *> &adjncy[0],\
                NULL,NULL,<idx_t *> &adjwgt_[0] ,&nparts,\
                NULL,NULL,NULL,\
                &ncut, <idx_t *> &part[0])    
        elif vwgt_size != 0 and adjwgt_size == 0: 
            vwgt_ = np.ascontiguousarray(vwgt_) 
            METIS_PartGraphKway(&nvtxs,&ncon,<idx_t *> &xadj[0],<idx_t *> &adjncy[0],\
                <idx_t *> &vwgt_[0],NULL,NULL ,&nparts,\
                NULL,NULL,NULL,\
                &ncut, <idx_t *> &part[0])   
        else:
            vwgt_ = np.ascontiguousarray(vwgt_) 
            adjwgt_ = np.ascontiguousarray(adjwgt_)
            METIS_PartGraphKway(&nvtxs,&ncon,<idx_t *> &xadj[0],<idx_t *> &adjncy[0],\
                <idx_t *> &vwgt_[0],NULL,<idx_t *> &adjwgt_[0] ,&nparts,\
                NULL,NULL,NULL,\
                &ncut, <idx_t *> &part[0])                 
    elif method == 'recursive':
        if vwgt_size == 0 and adjwgt_size == 0:
            METIS_PartGraphRecursive(&nvtxs,&ncon,<idx_t *> &xadj[0],<idx_t *> &adjncy[0],\
                NULL,NULL,NULL ,&nparts,\
                NULL,NULL,NULL,\
                &ncut, <idx_t *> &part[0])
        elif vwgt_size == 0 and adjwgt_size != 0:
            adjwgt_ = np.ascontiguousarray(adjwgt_)
            METIS_PartGraphRecursive(&nvtxs,&ncon,<idx_t *> &xadj[0],<idx_t *> &adjncy[0],\
                NULL,NULL,<idx_t *> &adjwgt_[0] ,&nparts,\
                NULL,NULL,NULL,\
                &ncut, <idx_t *> &part[0])    
        elif vwgt_size != 0 and adjwgt_size == 0: 
            vwgt_ = np.ascontiguousarray(vwgt_) 
            METIS_PartGraphRecursive(&nvtxs,&ncon,<idx_t *> &xadj[0],<idx_t *> &adjncy[0],\
                <idx_t *> &vwgt_[0],NULL,NULL ,&nparts,\
                NULL,NULL,NULL,\
                &ncut, <idx_t *> &part[0])   
        else:
            vwgt_ = np.ascontiguousarray(vwgt_) 
            adjwgt_ = np.ascontiguousarray(adjwgt_)
            METIS_PartGraphRecursive(&nvtxs,&ncon,<idx_t *> &xadj[0],<idx_t *> &adjncy[0],\
                <idx_t *> &vwgt_[0],NULL,<idx_t *> &adjwgt_[0] ,&nparts,\
                NULL,NULL,NULL,\
                &ncut, <idx_t *> &part[0])    
    else:
        print("Metis Only support the 'Kway' and 'Recursive' partitions!!")
        raise
    # print("nucts: ",ncut)
    return np.frombuffer(part,dtype=np.int64)
     

def RefinePy(idx_t nparts, idx_t nvtxs, idx_t ncon, idx_t[:] xadj, idx_t[:] adjncy,\
        idx_t[:] vwgt_, idx_t[:] adjwgt_, idx_t[:] old_part = np.zeros(shape=(0),dtype=np.int64)):

    """
        wrapper of the Refine Partition
    """
    


    # tmp = np.zeros(shape=(nparts),dtype=np.int64)
    # cdef idx_t[:] part = tmp.view()
    assert vwgt_.shape[0] == nvtxs
    assert adjwgt_.shape[0] == adjncy.shape[0]
    old_part = np.ascontiguousarray(old_part) if old_part.shape[0] != 0 else np.zeros(shape=(nvtxs),dtype=np.int64)
    spart = old_part.copy()
    cdef:
        idx_t[:] new_part = np.zeros(shape=(nvtxs),dtype=np.int64)
        # idx_t[:] spart = np.zeros(shape=(nvtxs),dtype=np.int64)
        idx_t ncut 
        idx_t options[40]
    for i in range(40):
        options[i] = -1
        
    # Set specific options if needed
    options[1] = 1  # C-style numbering
    options[METIS_OPTION_CTYPE] = 1
    options[METIS_OPTION_IPTYPE] = 4
    options[METIS_OPTION_RTYPE] = 1
    options[METIS_OPTION_NO2HOP] = 0
    options[METIS_OPTION_ONDISK] = 0
    options[METIS_OPTION_DROPEDGES] = 0
    options[METIS_OPTION_MINCONN] = 0
    options[METIS_OPTION_CONTIG] = 0
    options[METIS_OPTION_SEED] = -1
    options[METIS_OPTION_NIPARTS] = -1
    options[METIS_OPTION_NITER] = 10
    options[METIS_OPTION_NCUTS] = 1
    options[METIS_OPTION_UFACTOR] = 1
    options[METIS_OPTION_DBGLVL] = 0
    
    vwgt_ = np.ascontiguousarray(vwgt_)
    adjwgt_ = np.ascontiguousarray(adjwgt_)
    xadj = np.ascontiguousarray(xadj)
    adjncy = np.ascontiguousarray(adjncy)
    
    newrefine(&nvtxs, &ncon, <idx_t *> &xadj[0], <idx_t *> &adjncy[0],\
        <idx_t *> &vwgt_[0], NULL, <idx_t *> &adjwgt_[0], &nparts,\
        NULL, NULL, <idx_t *> &options,\
        &ncut, <idx_t *> &spart[0], <idx_t *> &new_part[0])
    return np.frombuffer(new_part, dtype=np.int64)
