import numpy as np



def sim(pred,target):
    pred /=np.sum(pred)
    target /=np.sum(target)
    s=0
    for x in range(pred.shape[0]):
	for y in range(pred.shape[1]):
	    s += min (pred[x,y],target[x,y])
    return s

def cc(pred,target):
    ts = np.std(target)
    ps = np.std(pred)
    tu = np.mean(target)
    pu = np.mean(pred)
    s=0
    for x in range(pred.shape[0]):
        for y in range(pred.shape[1]):
            s += (pred[x,y]-pu)*(target[x,y]-tu)
    s/=ts
    s/=ps
    return s

