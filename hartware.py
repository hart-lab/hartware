#!/usr/bin/python

import os
import sys
import numpy as np
import pandas as pd
import matplotlib as pl


def ut_as_list( dframe, diag=1, cols=['Row','Column','Value'] ):
	"""
	for a symmetric dataframe, where cols=rows, get the upper triangle as a list of row/column pairs
	diag = 1 (default): ignore diagonal
	diag = 0: include diagonal
	"""
	if (dframe.index.name == dframe.columns.name):
		dframe.index.name = dframe.index.name + '.1'
		dframe.index.name = dframe.index.name + '.2'
	d = dframe.where( np.triu( np.ones( dframe.shape ), k=diag).astype(np.bool))
	d = d.stack().reset_index()
	d.columns=cols
	return d

def qnorm_dataframe( data ):
	"""
	quantile normalize a dataframe with numeric values only!
	Normalizes to rank mean
	Does not deal with ties
	"""
	rank_mean = data.stack().groupby(data.rank(method='first').stack().astype(int)).mean()
	qnormed_data    = data.rank(method='min').stack().astype(int).map(rank_mean).unstack()
	return qnormed_data


def qnorm_array(anarray):
	"""
	anarray where rows=genes and columns=samples
	"""
	anarray.dtype = np.float64 
	A=anarray
	AA = np.float64( np.zeros_like(A) )
	I = np.argsort(A,axis=0)
	AA[I,np.arange(A.shape[1])] = np.float64( np.mean(A[I,np.arange(A.shape[1])],axis=1)[:,np.newaxis] ) 
	return AA

def generate_correlation_map_df(x, y):
    """Correlate each row of X with each row of Y.

    Parameters
    ----------
    x : pandas.dataframe
      Shape N X T.

    y : pandas.dataframe
      Shape M X T.

    Returns
    -------
    pandas.dataframe
      N X M frame in which each element is a Pearson correlation coefficient. 
      Index = X index labels
      Colums = Y index labels.

    """
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError('X and Y must have the same number of columns.')
    s_x = x.std(1, ddof=n - 1)
    s_y = y.std(1, ddof=n - 1)
    cov = np.dot(x,
                 y.T) - n * np.dot(mu_x[:, np.newaxis],
                                  mu_y[np.newaxis, :])
    corrmatrix = cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])
    outframe = pd.DataFrame( index=x.index.values, columns=y.index.values, data=corrmatrix)
    return outframe

def clean_axis(ax):
	#
	# for use, e.g., when plotting a dendrogram alongside a heatmap. removes border on a frame.
	#
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


###############################
# SOME USEFUL COLOR GRADIENTS #
###############################

def getYellowCyanGradientCmap():
	cdict = {'red': ((0.0, 0.0, 0.0),
			   (0.5, 0.0, 0.0),
			   (1.0, 1.0, 1.0)),
		 'green': ((0.0, 1.0, 1.0),
			   (0.5, 0.0, 0.0),
			   (1.0, 1.0, 1.0)),
		 'blue':  ((0.0, 0.1, 1.0),
			   (0.5, 0.0, 0.0),
			   (1.0, 0.0, 0.0))
		 }
	YellowCyanGradient = LinearSegmentedColormap('mycmap', cdict)
	YellowCyanGradient.set_bad('gray',1.)
	return YellowCyanGradient

def getBrickGradientCmap():
    cdict = {'red':   ((0.0, 1.0, 1.0),
                       (0.7, 0.70, 0.70),
                       (1.0, 0.70, 0.70)),
             'green': ((0.0, 1.0, 1.0),
                       (0.7, 0.15, 0.15),
                       (1.0, 0.15, 0.15)),
             'blue':  ((0.0, 1.0, 1.0),
                       (0.7, 0.07, 0.07),
                       (1.0, 0.07, 0.07))
             }
    BrickGradientCmap = LinearSegmentedColormap('mycmap', cdict)
    BrickGradientCmap.set_bad('gray',1.)
    return BrickGradientCmap

def getTealGradientCmap():
    cdict = {'red':   ((0.0, 1.0, 1.0),
                       (0.167, 1.0, 1.0),
                       (1.0, 0.0, 0.0)),
             'green': ((0.0, 1.0, 1.0),
                       (0.167, 1.0, 1.0),
                       (1.0, 0.4, 0.4)),
             'blue':  ((0.0, 1.0, 1.0),
                       (0.167, 1.0, 1.0),
                       (1.0, 0.4, 0.4))
             }
    GradientCmap = LinearSegmentedColormap('mycmap', cdict)
    GradientCmap.set_bad('lightgray',1.)
    return GradientCmap

