import numpy as np
import os

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

N=100000 #100000
dummy_data_multiplier=3
N_queries = 1000
d=16   #16
K=5    #5

np.random.seed(1)

print("Generating data...")
batches_dummy= [normalized(np.float32(np.random.random( (N,d)))) for _ in range(dummy_data_multiplier)]
batch_final = normalized (np.float32(np.random.random( (N,d))))
queries = normalized(np.float32(np.random.random( (N_queries,d))))
print("Computing distances...")
dist=np.dot(queries,batch_final.T)
topk=np.argsort(-dist)[:,:K]
print("Saving...")

# dist_1=np.dot(queries,batches_dummy[1].T)
# topk_1=np.argsort(-dist_1)[:,:K]
# np.int32(topk_1).tofile('data/gt_1.bin')

# dist_2=np.dot(queries,batches_dummy[2].T)
# topk_2=np.argsort(-dist_2)[:,:K]
# np.int32(topk_2).tofile('data/gt_2.bin')

try:
    os.mkdir("data")
except OSError as e:
    pass

for idx, batch_dummy in enumerate(batches_dummy):
    dist_dummy=np.dot(queries,batch_dummy.T)
    topk_dummy=np.argsort(-dist_dummy)[:,:K]
    np.int32(topk_dummy).tofile('data/gt_%02d.bin' %idx)    
    batch_dummy.tofile('data/batch_dummy_%02d.bin' %idx)
batch_final.tofile('data/batch_final.bin')
queries.tofile('data/queries.bin')
np.int32(topk).tofile('data/gt.bin')
with open("data/config.txt", "w") as file:
    file.write("%d %d %d %d %d" %(N, dummy_data_multiplier, N_queries, d, K))
