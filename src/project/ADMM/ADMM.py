import numpy as np
from scipy.stats import ortho_group
import time

def generate(n=100,s=0.05,rank=25):
    m=ortho_group.rvs(n)
    m=m[:rank]
    for i in range(rank):
        m[i]=m[i]*np.random.rand()*100
    low_rank=np.matmul(m.T,m)
    position=np.random.randint(0,n**2,(int)(s*(n**2)*1.5))
    position=np.unique(position)[:(int)(s*(n**2))]
    for i in position:
        low_rank[i//n][i % n]+=(np.random.rand()-0.5)*100
    return low_rank


def shrinkage(Matrix, tau):
    return np.sign(Matrix)*(np.maximum(np.abs(Matrix)-tau,np.zeros(Matrix.shape)))
def decomposition(Matrix, tau):
    U,S,V=np.linalg.svd(Matrix)
    return np.matmul(U,np.matmul(np.diag(shrinkage(S,tau)),V))


max_iter=1e5
iter=0
n=1000
s_ratio=0.05
rank=100
M=generate(n,s_ratio,rank)
max_iter=1e5
iter=0
L=np.zeros(M.shape)
S=np.zeros(M.shape)
Y=np.zeros(M.shape)
lbda=1/np.sqrt(max(M.shape))
miu=10*lbda
error=np.linalg.norm(M-L-S,ord="fro")
limit=np.linalg.norm(M,ord="fro")* 1e-7
begin_time=time.time()
while error>limit and iter < max_iter:
    L=decomposition(M-S+1/miu *Y,1/miu)
    S=shrinkage(M-L+1/miu * Y,lbda/miu)
    Y=Y+miu*(M-L-S)
    iter+=1
    error=np.linalg.norm(M-L-S,ord="fro")
            # if iter%10==0:
            #     print(iter,error)
time_end=time.time()
print(time_end-begin_time)