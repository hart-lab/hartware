#!/usr/bin/python

import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


def ut_as_list( dframe, diag=1, cols=['Row','Column','Value'] ):
  """
  for a symmetric dataframe, where cols=rows, get the upper triangle as a list of row/column pairs
  diag = 1 (default): ignore diagonal
  diag = 0: include diagonal
  """
  #if (dframe.index.name == dframe.columns.name):
  dframe.index.name = cols[0]
  dframe.columns.name = cols[1]
  #		dframe.index.name = dframe.index.name + '.1'
  #		dframe.index.name = dframe.index.name + '.2'
  d = dframe.where( np.triu( np.ones( dframe.shape ), k=diag).astype(np.bool))
  d = d.stack().reset_index()
  d.columns=cols
  return d

def ut_as_values( dframe, diag=1):
  array_of_ut_values = dframe.where( np.triu( np.ones( dframe.shape ), k=1).astype(np.bool)).stack().values
  return array_of_ut_values

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
    cov = np.dot(x, y.T) - n * np.dot(mu_x[:, np.newaxis], mu_y[np.newaxis, :])
    corrmatrix = cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])
    outframe = pd.DataFrame( index=x.index.values, columns=y.index.values, data=corrmatrix)
    return outframe

###############################################################
# tools for calculating and manipulating partial correlations #
###############################################################

def partial(x,y,z,cc):
    #
    # x, y, z = gene (row/column) names
    # cc = dataframe; symmetric matrix of pearson correlations
    #
    pxy = cc.loc[x,y]
    pxz = cc.loc[x,z]
    pyz = cc.loc[y,z]
    pxy_z = (pxy - pxz*pyz) / (np.sqrt(1-pxz**2) * np.sqrt(1-pyz**2) )
    return pxy_z

def get_all_partials( g1, g2, cc):
    pxy = cc.loc[g1][g2]
    pxy_vect = np.array( list([pxy])*(cc.shape[0]) ) #vector
    pxz = cc.loc[g1]                              #vector
    pyz = cc.loc[g2]                              #vector
    pxy_all = (pxy_vect -  np.multiply(pxz, pyz)) / ( np.sqrt( 1-pxz**2) * np.sqrt( 1-pyz**2) )
    framename = 'pc_' + g1 + '_' + g2
    pxy_all = pxy_all.to_frame(framename)
    pxy_all.drop( [g1, g2], axis=0, inplace=True) # don't include these!
    pxy_all['ratio'] = pxy_all[framename]**2 / pxy**2
    pxy_all.sort_values('ratio', ascending=True, inplace=True)
    return pxy_all

########################################################
# plotting tools for connecting and viewing dataframes #
########################################################


def violin_by_group( data, data_label, group, group_label, figsize=(4,4), rot=0):
    #
    # pass slice (data, group; can be from two different dfs with same index), build df, and build sns.violin
    #
    # eg
    # violin_by_group( bf['KRAS'], 'KRAS BF', gof['KRAS'], 'KRAS mutation')
    # where bf = (genes x cells) BF matrix and gof = (genes x cells) category matrix (GOF or WT)
    #
    mydf = data.to_frame(name=data_label).join( group.to_frame(name=group_label), how='inner')
    fig, ax=plt.subplots()
    fig.set_size_inches(figsize)
    g = sns.violinplot(data=mydf, y=data_label, x=group_label, ax=ax)
    g.set_xticklabels(g.get_xticklabels(), rotation=rot)



def violin_by_quantile( data, data_label, group, group_label, num_quantiles=4, figsize=(4,4), rot=0):
    #
    # pass slice (data, group), build df, label quantiles, groupby quantile labels, and plot
    #
    # eg
    # violin_by_quantile( bf['ERBB2'], 'ERBB2 BF', expr['ERBB2'], 'ERBB2 expr', num_quantiles=12, figsize=(8,4), rot=0)
    #
    q = np.linspace(0,1,num_quantiles+1)
    mydf = data.to_frame(name=data_label).join( group.to_frame(name=group_label), how='inner')
    mydf[group_label + ' quantiles'] = pd.qcut( mydf[group_label], q, labels=range(1,num_quantiles+1) )
    fig, ax=plt.subplots()
    fig.set_size_inches(figsize)
    g = sns.violinplot(data=mydf, y=data_label, x=group_label + ' quantiles', ax=ax)
    g.set_xticklabels(g.get_xticklabels(), rotation=rot)


def scatter_with_density( x, y):
    xy = np.vstack([x,y])
    z = stats.gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    fig, ax = plt.subplots( figsize=(6,6))
    ax.scatter(x, y, c=z, s=50, edgecolor='', cmap='jet')
    plt.show()


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

