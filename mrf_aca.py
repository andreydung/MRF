# Image segmentation using MRF model
from PIL import Image
import numpy
from pylab import *
from scipy.cluster.vq import *
from scipy.signal import *
import cv
import scipy

def main():
	# Read in image
	im=Image.open('7.png')
	im=numpy.array(im)
	
	# If grayscale add one dimension for easy processing
	if (im.ndim==2):
		im=im[:,:,newaxis]

	# Initial kmean segmentation
	nlevels=4	
	lev=getinitkmean(im, nlevels)

	# MRF ICM
	win_dim=256
	while (win_dim>7):
		print win_dim
		locav=local_average(im,lev,nlevels,win_dim)
		lev=MRF(im, lev, locav,nlevels)
		win_dim=win_dim/2
	scipy.misc.imsave('lev.png',lev*20)

	# Get the color average based on locav
	out=ACAreconstruction(lev,locav)
	scipy.misc.imsave('locav.png',out)


def ACAreconstruction(lev,locav):
	out=0
	for i in range(locav.shape[3]):
		mask=(lev==i)
		out+=mask[:,:,newaxis]*locav[:,:,:,i]
	return out

def getinitkmean(im, nlevels):
	obs=reshape(im,(im.shape[0]*im.shape[1],-1))	
	obs=whiten(obs)

	(centroids, lev)=kmeans2(obs,nlevels)
	lev=lev.reshape(im.shape[0],im.shape[1])
	return lev

def delta(a,b):
	if (a==b):
		return -1
	else:
		return 1	

def MRF(obs, lev, locav, nlevels):
	(M,N)=obs.shape[0:2]
	for i in range(M):
		for j in range(N):
			# Find segmentation level which has min energy (highest posterior)
			cost=[energy(k,i,j, obs, lev, locav) for k in range(nlevels)]
			lev[i,j]=cost.index(min(cost))
	return lev
			
def energy(pix_lev,i, j, obs,lev,locav):
	beta=0.5
	std=7
	cl=clique(pix_lev,i,j,lev)
	closeness=numpy.linalg.norm(locav[i,j,:,pix_lev]-obs[i,j,:])
	return beta*cl+closeness/std**2

def local_average(obs, lev, nlevels, win_dim):
	# Use correlation to perform averaging
	mask=numpy.ones((win_dim,win_dim))/win_dim**2

	# 4d array (512, 512, ncolors, nlevels)
	locav=ones((obs.shape+(nlevels,)))
		
	for i in range(obs.shape[2]):	# loop through image channels
		for j in range(nlevels):	# loop through segmentation levels
			temp=(obs[:,:,i]*(lev==j))
			locav[:,:,i,j]=fftconvolve(temp,mask,mode='same')
	return locav

def clique(pix_lev, i, j, lev):
	(M,N)=lev.shape[0:2]

	#find correct neighbors
	if (i==0 and j==0):
		neighbor=[(0,1), (1,0)]
	elif i==0 and j==N-1:
		neighbor=[(0,N-2), (1,N-1)]
	elif i==M-1 and j==0:
		neighbor=[(M-1,1), (M-2,0)]
	elif i==M-1 and j==N-1:
		neighbor=[(M-1,N-2), (M-2,N-1)]
	elif i==0:
		neighbor=[(0,j-1), (0,j+1), (1,j)]
	elif i==M-1:
		neighbor=[(M-1,j-1), (M-1,j+1), (M-2,j)]
	elif j==0:
		neighbor=[(i-1,0), (i+1,0), (i,1)]
	elif j==N-1:
		neighbor=[(i-1,N-1), (i+1,N-1), (i,N-2)]
	else:
		neighbor=[(i-1,j), (i+1,j), (i,j-1), (i,j+1),\
				  (i-1,j-1), (i-1,j+1), (i+1,j-1), (i+1,j+1)]
	
	return sum(delta(pix_lev,lev[i]) for i in neighbor)

if __name__=="__main__":
	main()	