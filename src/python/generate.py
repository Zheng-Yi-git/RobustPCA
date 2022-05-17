import numpy as np
from scipy.stats import ortho_group

def generate(n=100,s=200,rank=5):
    m=ortho_group.rvs(n)
    m=m[:rank]
    for i in range(rank):
        m[i]=m[i]*np.random.rand()*100
    low_rank=np.matmul(m.T,m)
    position=np.random.randint(0,n**2,s)
    for i in position:
        low_rank[i//100][i % 100]+=(np.random.rand()-0.5)*100
    return low_rank
    
generate()