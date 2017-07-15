import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

def build_map(fixations,sigma = 19)
    sal_map = np.zeros((224,224))

    for y,x in fixations:
        sal_map[y-1][x-1] = 1
    
    sal_map = ndimage.filters.gaussian_filter(sal_map, sigma)
    sal_map -= np.min(sal_map)
    sal_map /= np.max(sal_map)
    return sal_map


def gmm(y,x,mean,params_v,weights)
    s = 0
    for i in range(len(mean)):
	s+= weights[i] /  (2*3.14) /(params_v[i][0]*params_v[i][1])^0.5 * exp(-0.5*((x-mean[i][0]/params_v[i][0])^2+(y-mean[i][1]/params_v[i][1])^2))
    return s


def build_prediction(mean,params_v,weights):
    sal_map = np.zeros((224,224))
    for x in range(224):
	for y in range(224):
	    sal_map[y][x] = gmm(y,x,mean,params_v,weights)

	
