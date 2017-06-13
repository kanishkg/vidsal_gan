import numpy as np
import scipy.ndimage
import cv2

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
    target-=tu
    pred-=pu
    target/=ts
    pred/=ps
    #s=0
    #for x in range(pred.shape[0]):
    #    for y in range(pred.shape[1]):
    #        s += (pred[x,y]-pu)*(target[x,y]-tu)
    #s/=pred.shape[0]*pred.shape[1]
    #s/=ts
    #s/=ps
    s = np.corrcoef(pred.reshape(-1), target.reshape(-1))[0][1]
    return s

def nss2(pred,gt,h,w):
    mh,mw = np.shape(pred)
    salMap = pred
    salMap = scipy.ndimage.zoom(pred, (float(h)/mh, float(w)/mw), order=3)
    salMap = salMap - np.mean(salMap)
    salMap = salMap / np.std(salMap)
 #   target = np.zeros((h,w))
    #for y,x in gt:
#	target[y-1,x-1]+=1
    #target = cv2.resize(target,(224,224))
    #score = np.mean(np.multiply(target,salMap))
    score = np.mean([ salMap[y-1][x-1] for y,x in gt ])
    return score

def aucb(pred,gt,h,w):
    mh,mw = np.shape(pred)
    pred = scipy.ndimage.zoom(pred, (float(h)/mh, float(w)/mw), order=3)
    pred-= np.min(pred)
    pred/=np.max(pred)
    P = pred.reshape(-1)
    s = np.asarray([ pred[y-1][x-1] for y,x in gt ])

    Nfixations = len(gt)
    Npixels = len(P)

    # sal map values at random locations
    randfix = P[np.random.randint(Npixels, size=100000)]

    allthreshes = np.arange(0,np.max(np.concatenate((s, randfix), axis=0)),0.01)
    allthreshes = allthreshes[::-1]
    tp = np.zeros(len(allthreshes)+2)
    fp = np.zeros(len(allthreshes)+2)
    tp[-1]=1.0
    fp[-1]=1.0
    tp[1:-1]=[float(np.sum(s >= thresh))/Nfixations for thresh in allthreshes]
    fp[1:-1]=[float(np.sum(randfix >= thresh))/100000 for thresh in allthreshes]
    auc = np.trapz(tp,fp)
    return auc
