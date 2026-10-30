"""Algebraically equivalent aggregation without edge-by-hidden allocations."""
import torch


class _SparseValueMM(torch.autograd.Function):
    """Avoid native value-backward's dense owner-by-counterpart product."""
    @staticmethod
    def forward(ctx,owners,counterparts,values,weight,entities,chunk):
        matrix=torch.sparse_coo_tensor(torch.stack((owners,counterparts)),values,
            (entities,len(weight)),device=values.device).coalesce()
        ctx.save_for_backward(matrix,weight,owners,counterparts)
        ctx.chunk=int(chunk)
        return torch.sparse.mm(matrix,weight)

    @staticmethod
    def backward(ctx,gradient):
        matrix,weight,owners,counterparts=ctx.saved_tensors
        dv=(pair_dot(gradient,weight,owners,counterparts,ctx.chunk)
            if ctx.needs_input_grad[2] else None)
        dw=(torch.sparse.mm(matrix.transpose(0,1).coalesce(),gradient)
            if ctx.needs_input_grad[3] else None)
        return None,None,dv,dw,None,None


def sparse_value_mm(owners,counterparts,values,weight,entities,chunk=32768):
    if chunk<1:raise ValueError('Positive sparse-backward chunk required')
    return _SparseValueMM.apply(owners,counterparts,values,weight,entities,chunk)


def sparse_encoder_forward(self, owners, counterparts, residual, count, entities, *, return_hidden=False):
    shape=(entities,self.value.num_embeddings)
    indices=torch.stack((owners,counterparts))
    masks=torch.sparse_coo_tensor(indices,torch.ones_like(residual),shape,device=residual.device).coalesce()
    pooled=(sparse_value_mm(owners,counterparts,residual,self.value.weight,entities,
                          getattr(self,'_v7_edge_chunk',32768))
            +torch.sparse.mm(masks,self.mask.weight))
    pooled=pooled/count.clamp_min(1).pow(getattr(self,"pooling_power",.5))[:,None]
    hidden=self.network[1](self.network[0](pooled))
    code=self.network[2](hidden)
    if getattr(self,"residual_skip",False):
        local=residual.new_zeros(entities).index_add(0,owners,residual)/count.clamp_min(1)
        code=code+self.statistics(torch.stack((local,count.log1p()/10),-1))
    if getattr(self,"degree_gate",False):
        code=code*(count/(count+self.count_tau)).sqrt()[:,None]
    return (code,hidden) if return_hidden else code


class _PairDot(torch.autograd.Function):
    @staticmethod
    def forward(ctx,left,right,users,items,chunk):
        ctx.save_for_backward(left,right,users,items)
        ctx.chunk=int(chunk)
        out=left.new_empty(len(users))
        for start in range(0,len(users),ctx.chunk):
            sl=slice(start,start+ctx.chunk)
            out[sl]=(left[users[sl]]*right[items[sl]]).sum(-1)
        return out

    @staticmethod
    def backward(ctx,gradient):
        left,right,users,items=ctx.saved_tensors
        gl=torch.zeros_like(left) if ctx.needs_input_grad[0] else None
        gr=torch.zeros_like(right) if ctx.needs_input_grad[1] else None
        for start in range(0,len(users),ctx.chunk):
            sl=slice(start,start+ctx.chunk);g=gradient[sl,None]
            if gl is not None:gl.index_add_(0,users[sl],g*right[items[sl]])
            if gr is not None:gr.index_add_(0,items[sl],g*left[users[sl]])
        return gl,gr,None,None,None


def pair_dot(left,right,users,items,chunk=32768):
    """Same first-order dot-product objective, one logical optimizer update."""
    if chunk<1:raise ValueError("Positive computational chunk required")
    if len(users)<=chunk:
        return (left[users]*right[items]).sum(-1)
    return _PairDot.apply(left,right,users,items,chunk)
