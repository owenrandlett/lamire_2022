#%%
import os, subprocess, sys, fnmatch, glob,  h5py, time, tifffile, colorsys, napari
from turtle import position
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from numpy.core.fromnumeric import size
from scipy import stats, cluster
from scipy.signal import medfilt, medfilt2d
from tifffile import imsave, imread
from scipy.stats import pearsonr, spearmanr, median_abs_deviation, zscore,  kruskal, chisquare
from scipy.cluster import hierarchy
from platform import uname
import colorcet as cc
from skimage.registration import phase_cross_correlation
from skimage.measure import block_reduce
from sklearn.cluster import AffinityPropagation, KMeans, SpectralClustering, AgglomerativeClustering
from suite2p.io import BinaryFile
from scipy.ndimage import zoom, morphology, gaussian_filter, convolve, median_filter, sum_labels
from sklearn.decomposition import PCA
from scipy.spatial import distance
from scipy.signal import savgol_filter
from matplotlib import colors
from natsort import natsorted
from tqdm.notebook import tqdm
from numba import jit, njit, prange
from statannot import add_stat_annotation
from sklearn.linear_model import LinearRegression, SGDRegressor
from distinctipy import get_colors, distinct_color
        
plt.rcParams.update({'font.size': 22})

cell_thresh = 0.3 # classifier probability threshold


suite2p_dir = os.path.realpath('/mnt/md0/suite2p_output/')
os.chdir(suite2p_dir)
analysis_out = r'/media/BigBoy/Owen/2020_HabScreenPaper/data/GCaMPAnalysis/'

zbrain = tifffile.imread('/media/BigBoy/ciqle/ref_brains/ZBrain2_0.tif')
zbrain_LUT = pd.read_excel('/media/BigBoy/ciqle/ref_brains/ZBrain2_0_LUT.xls', header=None)
Zs, height, width = zbrain.shape


def rolling_window(a, window):
    pad = np.ones(len(a.shape), dtype=np.int32)
    pad[-1] = window-1
    pad = list(zip(pad, np.zeros(len(a.shape), dtype=np.int32)))
    a = np.pad(a, pad,mode='reflect')
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def GCaMPConvolve(trace, ker):
    if np.sum(trace) == 0:
        return trace
    else:
        trace_conv = np.convolve(trace, ker, 'full')
        trace_conv = trace_conv[1:trace.shape[0]+1] 
        trace_conv[np.logical_not(np.isfinite(trace_conv))] = 0
        trace_conv = trace_conv/max(trace_conv)
        return trace_conv



def pearsonr_2D(x, y):
    """computes pearson correlation coefficient
       where x is a 1D and y a 2D array
       from https://stackoverflow.com/questions/30143417/computing-the-correlation-coefficient-between-two-multi-dimensional-arrays """

    upper = np.sum((x - np.mean(x)) * (y - np.mean(y, axis=1)[:,None]), axis=1)
    lower = np.sqrt(np.sum(np.power(x - np.mean(x), 2)) * np.sum(np.power(y - np.mean(y, axis=1)[:,None], 2), axis=1))
    
    rho = upper / lower
    
    return rho

@njit
def pearsonr_numba2(x, y):
    """computes pearson correlation coefficient
       where x is a 1D and y a 2D array
       from https://stackoverflow.com/questions/30143417/computing-the-correlation-coefficient-between-two-multi-dimensional-arrays """
    n_var = y.shape[1]
    y_mean = np.sum(y, axis=1) / n_var
    y_mean = y_mean.repeat(n_var).reshape((-1, n_var))

    upper = np.sum((x - np.mean(x)) * (y - y_mean), axis=1)
    

    lower = np.sqrt(np.sum(np.power(x - np.mean(x), 2)) * np.sum(np.power(y - y_mean, 2), axis=1))
    
    rho = upper / lower
    
    return rho

@njit
def pearsonr_vec_2Dnumb(x,y):
    # computes the pearson correlation coefficient between a a vector (x) and each row in 2d matrix (y), using numba acceleration
    
    n_rows_y = int(y.shape[0])
    corr = np.zeros((n_rows_y))
    for row_y in prange(n_rows_y):
        corr[row_y] = np.corrcoef(x, y[row_y,:])[0,1]
    return corr


@njit
def pearsonr_2Dnumb(x,y, print_progress = False):

    # computes the pearson correlation coefficient between a each row in 2d matrix (x) and each row in 2d matrix (y), using numba acceleration

    n_rows_y = int(y.shape[0])
    n_rows_x = x.shape[0]
    corr = np.zeros((n_rows_x, n_rows_y))

    for row_x in prange(n_rows_x):
        for row_y in prange(n_rows_y):
            y[row_y,:]
            x[row_x, :]
            corr[row_x, row_y] = np.corrcoef(x[row_x, :], y[row_y,:])[0,1]
        if print_progress:
            print('done correlations on row ' + str(row_x) + ' in x, out of ' + str(n_rows_x))

    return corr

def plot_cluster_proportions(hits_IDs, clust_IDs, use_boots = False, n_boots=1000, save_name=None):
    col_vec = color_fish 
    n_hits = len(hits_IDs)
    n_clusters = np.max(clust_IDs)+1
    categories = fish_data[hits_IDs, 1]
    fish_ID =  fish_data[hits_IDs, 0]
    n_fish = np.max(fish_ID)
    unique, counts_all = np.unique(categories, return_counts=True)
    n_categories = len(unique)

    fish_category = np.zeros(n_fish, dtype=int)
    for i in range(n_fish):
        fish_category[i] = categories[np.where(fish_ID== i)[0][0]]

    counts_per_fish = np.zeros((n_fish, n_clusters))
    for i in range(n_fish):
        clust_IDs_fish = clust_IDs[fish_ID==i]
        unique, counts = np.unique(clust_IDs_fish, return_counts=True)
        counts_per_fish[i, unique] = counts/np.sum(counts)

    with plt.rc_context({'font.size':30}):
        fig = plt.figure(figsize=(10,7))
        if use_boots:
            counts_clust = np.zeros((n_categories, n_clusters, n_boots))
            couts_clust_norm = np.zeros((n_categories, n_clusters, n_boots))
            for boot in range(n_boots):
                rand_IDs = np.random.randint(0, n_hits, n_hits)
                hits_IDs_boot =  hits_IDs[rand_IDs]
                clust_IDs_boot = clust_IDs[rand_IDs]
                categories = fish_data[hits_IDs_boot, 1] 
                unique, counts_all = np.unique(categories, return_counts=True)
                for i in range(n_clusters):
                    counts_in_clust, _ = np.histogram(categories[clust_IDs_boot == i], bins=np.arange(0.5, len(unique)+1.5))
                    counts_clust[:, i, boot] = counts_in_clust/counts_all
                couts_clust_norm[:,:,boot] = counts_clust[:,:,boot]/np.sum(counts_clust[:, i, boot], axis=0)

            
            
            mu = np.nanmedian(counts_clust, axis=2)
            # sigma = np.nanstd(counts_clust, axis=2)
            # CI = stats.norm.interval(0.95, loc=mu, scale=sigma)

            for i in range(n_categories):
                plt.plot(np.arange(n_clusters), mu[i,:], 'd', color=col_vec[i])

            #%
            for i in range(n_categories):
                sns.violinplot(data=counts_clust[i,:,:].T, width = 1.3, inner=None,  color=col_vec[i])
            plt.setp(plt.gca().collections, alpha=.5)
            # for i in range(n_categories):
            #     sns.stripplot(data=counts_per_fish[fish_category==i+1,:],  color=col_vec[i], alpha=0.5, jitter=0.1,)

        else:


            for i in range(n_categories):
                sns.pointplot(data=counts_per_fish[fish_category==i+1,:], estimator=np.nanmean, color=col_vec[i],  join=False, markers="_", scale=4, ci=None)        

            for i in range(n_categories):

                #sns.violinplot(data=counts_per_fish[fish_category==i+1,:], width = 1.5, inner=None,  color=col_vec[i])
                sns.stripplot(data=counts_per_fish[fish_category==i+1,:],  color=col_vec[i], alpha=0.5, jitter=0.1,)
            plt.ylim((0,0.3))
            # for i in range(n_categories):
            #     sns.boxplot(data=counts_per_fish[fish_category==i+1,:],color=col_vec[i])
        plt.xticks(np.arange(n_clusters), labels = cluster_names, rotation=90, fontweight='bold')  
        #plt.ylim((0.02, 0.22))
        plt.yticks((0.0, 0.1, 0.2))
        #plt.xlabel('Cluster Number')
        plt.ylabel('Proportion of treatment\ngroup in cluster')
        plt.legend(['DMSO', 'Picrotoxinin', 'Melatonin'])
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        xtick_labels = plt.gca().get_xticklabels()
        for i, tick_label in enumerate(xtick_labels):
            tick_label.set_color(clust_colors[i])
        if not save_name==None:
            plt.savefig(os.path.join(analysis_out, save_name+'_cluster_proportions.png'))
            plt.savefig(os.path.join(analysis_out, save_name+'_cluster_proportions.svg'))


#%
#%
def plot_cluster_means(zTrace_matrix, hits, hits_clust, median=False, plot_heat = False, cluster_heat = False, re_order = True, ymax=5, use_clustlabels = False, save_name = None):
    from scipy.cluster import hierarchy

    n_clusters = np.max(hits_clust)+1
    mean_vecs = np.zeros((n_clusters, F_norm.shape[1]))
    len_nanpad = int(len(hits) * 0.02)
    nan_pad = np.zeros((len_nanpad, zTrace_matrix.shape[1]))
    nan_pad[:] = -50
    for i in range(n_clusters):
        if median:
            mean_trace = np.median(zTrace_matrix[hits[hits_clust==i], :], axis=0)
        else:
            mean_trace = np.mean(zTrace_matrix[hits[hits_clust==i], :], axis=0)
        
        mean_vecs[i, :] = mean_trace
    
    met = 'correlation'
    meth = 'complete'
    if re_order:
        # reorder clusters based on linkage

        link =  hierarchy.linkage(mean_vecs, optimal_ordering=True, metric=met, method=meth)

        order = hierarchy.leaves_list(link)
        
        hits_clust_reorder = hits_clust.copy()
        mean_vec_reorder = mean_vecs.copy()

        for i in range(n_clusters):
            mean_vec_reorder[i,:] = mean_vecs[order[i], :]
            hits_clust_reorder[hits_clust == order[i]] = i
    else:
        mean_vec_reorder = mean_vecs.copy()
        hits_clust_reorder = hits_clust.copy()

    with plt.rc_context({'lines.linewidth':8}):
        plt.figure(figsize=(5,40))
        dend = hierarchy.dendrogram(
            hierarchy.linkage(mean_vec_reorder, optimal_ordering=True, metric=met, method=meth),
            orientation='left', labels=np.arange(n_clusters)+1,
            color_threshold=25, leaf_font_size=80)
        ax = plt.gca()
        
        xtick_labels = ax.get_yticklabels()
        for i, tick_label in enumerate(xtick_labels):
            tick_label.set_color(clust_colors[i])
            
        ax.invert_yaxis()
        if not save_name==None:
            plt.savefig(os.path.join(analysis_out, save_name+'_cluster_linkage.png'))
            plt.savefig(os.path.join(analysis_out, save_name+'_cluster_linkage.svg'))
        plt.show()

    n_bef = int(ops[fish_name]['fs']*5)
    n_aft = int(ops[fish_name]['fs']*35)
    n_df = len(df_start_inds)
    cmap = sns.color_palette("gnuplot", n_colors =n_df, as_cmap=False)
    plt.figure(figsize=(5,5))
    sns.palplot(cmap)
    if not save_name==None:
        plt.savefig(os.path.join(analysis_out, save_name+'_stimReps_cmap.svg'))
    plt.show()
    # plot mean traces
    fig, ax = plt.subplots(n_clusters,2, figsize=(20,40))
    for clust in range(n_clusters):

        y = mean_vec_reorder[clust,:]
        x = np.arange(len((y)))/(ops[fish_name]['fs']*60)
        x = x - 5 
        if clust==0:
            ax[clust,0].plot(x, (ymax-1)+stim_df_conv, 'k--')
        ax[clust,0].plot(x, y, 'k')
        if use_clustlabels:
            #ax[clust,0].set_title( str(clust+1)+': ' +  cluster_names[clust], {'fontsize':50, 'fontweight':'bold', 'va':'bottom'}, color=clust_colors[clust])
            ax[clust,0].text(30,3, str(clust+1)+': ' +  cluster_names[clust], {'fontsize':50, 'fontweight':'bold', 'ha':'center', 'color':clust_colors[clust]})
        else:
            ax[clust,0].set_title('cluster ' + str(clust+1), {'fontsize':35}, color=clust_colors[clust])
        
        ax[clust,0].set_xlim([-3, 61])
        
        ax[clust,0].spines['right'].set_visible(False)
        ax[clust,0].spines['top'].set_visible(False)

        response_mat = np.zeros((n_df, n_bef+n_aft))
        resp_vec = mean_vec_reorder[clust,:]
        x = np.arange(-n_bef, n_aft)/ops[fish_name]['fs']
        
        ax[clust,1].plot(x, ymax*stim_df_conv[df_start_inds[0]-n_bef:df_start_inds[0]+n_aft], 'k--')
        for i in range(n_df):
            response_mat[i,:] = resp_vec[df_start_inds[i]-n_bef:df_start_inds[i]+n_aft]
            ax[clust,1].plot(x, response_mat[i,:], color=cmap[i])
        
        ax[clust,1].set_xlim((x[0], x[-1]))
        ax[clust,1].spines['right'].set_visible(False)
        ax[clust,1].spines['top'].set_visible(False)

        if not ymax==None:
            ax[clust,0].set_ylim([-1, ymax])
            ax[clust,1].set_ylim([-1, ymax])
    
    plt.tight_layout()
    if not save_name==None:
        plt.savefig(os.path.join(analysis_out, save_name+'_cluster_vecs.png'))
        plt.savefig(os.path.join(analysis_out, save_name+'_cluster_vecs.svg'))
    plt.show()

    # make heatmap of data
    if plot_heat:
        for i in range(n_clusters):
            zTrace_subset = zTrace_matrix[hits[hits_clust_reorder==i], :]
            if cluster_heat:
                # link =  hierarchy.linkage(zTrace_subset, optimal_ordering=False, metric='correlation', method='average')
                # order = hierarchy.leaves_list(link)
                kmeans = KMeans(n_clusters=10, random_state=1).fit(zTrace_subset)
                order = np.argsort(kmeans.labels_)
                zTrace_subset = zTrace_subset[order, :]
            if i == 0:
                im_clust = zTrace_subset
            else:
                im_clust = np.vstack((im_clust, nan_pad, zTrace_subset))

        im_clust = block_reduce(im_clust, block_size=(50,1), func=np.mean)
        with plt.rc_context({'font.size':20}):
            plt.figure(figsize=(100,50))
            sns.heatmap(
                im_clust,
                vmin=-1,
                vmax=3,
                robust=True,
                cmap='gray_r',
                cbar_kws = {'shrink':0.3, 'aspect':3},
                xticklabels = False,
                yticklabels = False,
            )
            # plt.imshow(im_clust, vmin=-2, vmax=2, cmap=col_map)
            # plt.axis('off')
            if not save_name==None:
                plt.savefig(os.path.join(analysis_out, save_name+'_cluster_heatmap.png'))
                #plt.savefig(os.path.join(analysis_out, save_name+'_cluster_heatmap.svg'))
        plt.show()

    # plot proportions of different cluster assignments

    return mean_vec_reorder, hits_clust_reorder


#%
def plot_cluster_rois(hits, hits_clust, save_name = None):

    n_clusters = np.max(hits_clust)+1
    IM_clust = np.zeros([Zs, height, width,  n_clusters], dtype=np.int8)
    IM_clust_centroid = np.zeros([Zs, height, width,  n_clusters], dtype=np.int8)
    fig, ax = plt.subplots(nrows=2, ncols=np.max((int(np.ceil(n_clusters/2)), 2)), figsize = (40,17))
    k = 0
    l = 0
    for i in range(n_clusters): # loop through the different correlation vectors
        hits_clust_i = hits[hits_clust==i]
        IM_clust[:,:,:, i], im_show = draw_hit_volume(hits_clust_i, draw_outline=True)
        IM_clust_centroid[:,:,:, i], _ = draw_hit_volume(hits_clust_i, draw_centroid=True)
        
        ax[k, l].imshow(im_show, vmin=0, vmax=0.7, cmap='inferno')
        ax[k, l].set_axis_off()
        ax[k, l].text(im_show.shape[1]/2, -10,str(i+1)+':' +  cluster_names[i], {'fontsize':55, 'fontweight':'bold', 'ha':'center'}, color=clust_colors[i])

        l +=1 
        if l == ax.shape[1]:
            k+=1
            l=0
    #plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.15)
    if not save_name == None:
        plt.savefig(os.path.join(analysis_out, save_name+'_cluster_ROI_Projections.png'))
        plt.savefig(os.path.join(analysis_out, save_name+'_cluster_ROI_Projections.svg'))
    plt.show()

    return IM_clust, IM_clust_centroid

#%
#% 20210201 data has artifacts in it (perhaps related to uncalibraiton, or gremlins)
dmso_data_paths = [
    r'/mnt/md0/suite2p_output/20220202_HuCGCaMP7f_5d_DMSO_fish-002/',
    r'/mnt/md0/suite2p_output/20220208_HuCGCaMP7f_5d_DMSO_fish-003/',
    r'/mnt/md0/suite2p_output/20220209_HuCGCaMP7f_5d_DMSO_fish-002',
    r'/mnt/md0/suite2p_output/20220131_HuCGCaMP7f_5d_DMSO_fish-002',
    # fish is moving, not useabler'/mnt/md0/suite2p_output/20220214_HuCGCaMP7f_5d_DMSO_fish-001',
    r'/mnt/md0/suite2p_output/20220215_HuCGCaMP7f_5d_DMSO_fish-003',
    # data has sacanning atrificats in it r'/mnt/md0/suite2p_output/20220216_HuCGCaMP7f_5d_DMSO_fish-002',
    r'/mnt/md0/suite2p_output/20220221_HuCGCaMP7f_5d_DMSO_fish-002',
    r'/mnt/md0/suite2p_output/20220222_HuCGCaMP7f_5d_DMSO_fish-003',
    # fis is moving, not useable r'/mnt/md0/suite2p_output/2022023_HuCGCaMP7f_5d_DMSO_fish-001',

    r'/mnt/md0/suite2p_output/20220301_HuCGCaMP7f_5d_DMSO_fish-002',
    r'/mnt/md0/suite2p_output/20220302_HuCGCaMP7f_5d_DMSO_fish-003',
    r'/mnt/md0/suite2p_output/20220307_HuCGCaMP7f_5d_DMSO_fish-002',
    # fish is moving, not useable: r'/mnt/md0/suite2p_output/20220308_HuCGCaMP7f_5d_DMSO_fish-001',
    # fish is moving, not useable: r'/mnt/md0/suite2p_output/20220309_HuCGCaMP7f_5d_DMSO_fish-003'
    r'/mnt/md0/suite2p_output/20220314_HuCGCaMP7f_5d_DMSO_fish-003',
    r'/mnt/md0/suite2p_output/20220315_HuCGCaMP7f_5d_DMSO_fish-002',
    r'/mnt/md0/suite2p_output/20220330_HuCGCaMP7f_6d_DMSO_fish-003',
    # fish moving, not useable: '/mnt/md0/suite2p_output/20220329_HuCGCaMP7f_5d_DMSO_fish-001'
    r'/mnt/md0/suite2p_output/20220328_HuCGCaMP7f_5d_DMSO_fish-002'
]

picro_data_paths = [ 
    r'/mnt/md0/suite2p_output/20220209_HuCGCaMP7f_5d_picrotoxinin_fish-003',
    # fish looks like it is moving... r'/mn t/md0/suite2p_output/20220208_HuCGCaMP7f_5d_picrotoxinin_fish-001'
    r'/mnt/md0/suite2p_output/20220207_HuCGCaMP7f_5d_picrotoxinin_fish-002',
    r'/mnt/md0/suite2p_output/20220202_HuCGCaMP7f_5d_picrotoxinin_fish-003',
    r'/mnt/md0/suite2p_output/20220214_HuCGCaMP7f_5d_picrotoxinin_fish-002',
    # fish is moving, not useable r'/mnt/md0/suite2p_output/20220215_HuCGCaMP7f_5d_picrotoxinin_fish-001',
    # data has scanning artifacts... r'/mnt/md0/suite2p_output/20220216_HuCGCaMP7f_5d_picrotoxinin_fish-003'
    r'/mnt/md0/suite2p_output/20220221_HuCGCaMP7f_5d_picrotoxinin_fish-003',
    # fish moving, not useable r'/mnt/md0/suite2p_output/20220222_HuCGCaMP7f_5d_picrotoxinin_fish-001',
    r'/mnt/md0/suite2p_output/2022023_HuCGCaMP7f_5d_picrotoxinin_fish-002',

    r'/mnt/md0/suite2p_output/20220301_HuCGCaMP7f_5d_picrotoxinin_fish-003',
    # fish moving, not useable r'/mnt/md0/suite2p_output/20220302_HuCGCaMP7f_5d_picrotoxinin_fish-001'
    # fish moving, not useable r'/mnt/md0/suite2p_output/20220307_HuCGCaMP7f_5d_picrotoxinin_fish-001',
    # data looks corrupted, or fish is seizing? not useable: '/mnt/md0/suite2p_output/20220308_HuCGCaMP7f_5d_picrotoxinin_fish-003', 
    r'/mnt/md0/suite2p_output/20220308_HuCGCaMP7f_5d_picrotoxinin_fish-003',
    #fish moving, not useable: '/mnt/md0/suite2p_output/20220314_HuCGCaMP7f_5d_picrotoxinin_fish-001'
    # fish moving, not useable: '/mnt/md0/suite2p_output/20220330_HuCGCaMP7f_6d_picrotoxinin_fish-001'
    r'/mnt/md0/suite2p_output/20220329_HuCGCaMP7f_5d_picrotoxinin_fish-002',
]

melatonin_data_paths = [
    # fish is moving, not useable r'/mnt/md0/suite2p_output/20220209_HuCGCaMP7f_5d_melatonin_fish-001'
    r'/mnt/md0/suite2p_output/20220208_HuCGCaMP7f_5d_melatonin_fish-002',
    r'/mnt/md0/suite2p_output/20220207_HuCGCaMP7f_5d_melatonin_fish-003',
    # fish is moving, not useable r'/mnt/md0/suite2p_output/20220202_HuCGCaMP7f_5d_melatonin_fish-001'
    r'/mnt/md0/suite2p_output/20220214_HuCGCaMP7f_5d_melatonin_fish-003',
    r'/mnt/md0/suite2p_output/20220215_HuCGCaMP7f_5d_melatonin_fish-002',
    # fish is moving, not useable. probably scanning artifiacts r'/mnt/md0/suite2p_output/20220216_HuCGCaMP7f_5d_melatonin_fish-001'
    # fish is moving, not useable. r'/mnt/md0/suite2p_output/20220221_HuCGCaMP7f_5d_melatonin_fish-001'
    r'/mnt/md0/suite2p_output/20220222_HuCGCaMP7f_5d_melatonin_fish-002',
    r'/mnt/md0/suite2p_output/2022023_HuCGCaMP7f_5d_melatonin_fish-003',

    r'/mnt/md0/suite2p_output/20220302_HuCGCaMP7f_5d_melatonin_fish-002',
    r'/mnt/md0/suite2p_output/20220307_HuCGCaMP7f_5d_melatonin_fish-003',
    # fish is moving, not useable. r'/mnt/md0/suite2p_output/20220309_HuCGCaMP7f_5d_melatonin_fish-001', 
    # data looks corrupted again -- laser issue? r'/mnt/md0/suite2p_output/20220308_HuCGCaMP7f_5d_melatonin_fish-002',
    r'/mnt/md0/suite2p_output/20220314_HuCGCaMP7f_5d_melatonin_fish-002',
    # fish is moving, not useable. r'/mnt/md0/suite2p_output/20220315_HuCGCaMP7f_5d_melatonin_fish-001'
    r'/mnt/md0/suite2p_output/20220330_HuCGCaMP7f_6d_melatonin_fish-002',
    r'/mnt/md0/suite2p_output/20220328_HuCGCaMP7f_5d_melatonin_fish-001', # this fish is borderline... decide to include. May have movement at the end...
] 

regions_of_interset_all = {
    'Tectum S.P.':[112],
    'Tectum Neuropil':[111],
    'Pretectum':[23],
    'Thalamus':[27,28],

    'Habenula': [2],
    'Hypothalamus':[1, 3, 4, 24,25,26 ],
    'Optic Tract/AFs':[5,6,7,8,9,10,11,12,13,14],
    'Pineal': [15],
    'Pituitary':[16],
    'Posterior Tuberculum':[17,18],
    'Preglomerular Complex':[19],
    'Preoptic Area':[20,21,22],
    
    'Olfactory Bulb':[29],
    'Pallium':[30],
    'Subpallium':[31],
    'Inferior Olive':[50],
    'Cerebellum': [56,57,58,59],
    'Rhombomere 1':[55,60,61, 62,63,64,65],
    'Rhombomere 2':[66,67,68,69,70,71,72,73],
    'Rhombomere 3:8':[74,75,76, 77,78,79,80,81,82, 83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,48,49,51,52,53,54],
  
    'Torus Longitudinalis':[116],
    'Torus Semicircularis':[117],
    'Tegmentum':[113,114,115],
    'Retina':[118],
    'Spinal Cord':[119],
}   



regions_of_interset_MAPMap = {
    'Retina':[118],
    
    # Telencephalon
    'Pallium':[30],
    'Subpallium':[31],

    # midbrain
    'Tectum S.P.':[112],
    'Tectum Neuropil':[111],
    'Torus Semicircularis':[117],
    'Torus Longitudinalis':[116],
    'Tegmentum':[113,114,115],
    
    #diencephalon
    'Optic Tract/AFs':[5,6,7,8,9,10,11,12,13,14],
    'Pretectum':[23],
    'Thalamus':[27,28],
    'Hypothalamus':[1, 3, 4, 24,25,26,20,21,22],
    'Habenula': [2],
    'Preglomerular Complex Area':[19],
    'Posterior Tuberculum':[17,18],
    'Pineal':[15],

    #hindbrain
    'Cerebellum': [56,57,58,59],
    'Rhombomere 1':[55,60,61, 62,63,64,65],
    'Rhombomere 2':[66,67,68,69,70,71,72,73],
    'Rhombomere 3:8':[74,75,76, 77,78,79,80,81,82, 83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,48,49,51,52,53,54],
    'Inferior Olive':[50],

    'Spinal Cord':[119],
}   


regions_of_interset = {
    # midbrain
    'Tectum S.P.':[112],
    'Tectum Neuropil':[111],
    'Torus Semicircularis':[117],
    'Torus Longitudinalis':[116],
    'Tegmentum':[113,114,115],
    
    #diencephalon
    'Optic Tract/AFs':[5,6,7,8,9,10,11,12,13,14],
    'Pretectum':[23],
    'Thalamus':[27,28],
    'Habenula': [2],
    'Preglomerular Complex Area':[19],

    #hindbrain
    'Cerebellum': [56,57,58,59],
    'Rhombomere 1':[55,60,61, 62,63,64,65],
    'Rhombomere 2':[66,67,68,69,70,71,72,73],
    'Rhombomere 3:8':[74,75,76, 77,78,79,80,81,82, 83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,48,49,51,52,53,54],
    'Inferior Olive':[50],

}   

data_paths = dmso_data_paths + picro_data_paths + melatonin_data_paths
#%
col_map = sns.diverging_palette(360,180, s=100, l=50, sep=30, as_cmap=True, center="dark")

start_analyze_frame = 200  # ignore the first frames for correlation analyses, scanning artifact should be done by then

# parameters for GCaMP kernel
DecCnst = 0.3
RiseCnst = 0.5
frame_rate = 1.976
DecCnst = DecCnst*frame_rate # now in frames
RiseCnst = RiseCnst*frame_rate

KerRise = np.power(2, (np.arange(0,5)*RiseCnst)) - 1
KerRise= KerRise[KerRise < 1]
KerRise = KerRise/max(KerRise)

KerDec = np.power(2, (np.arange(20, 0, -1)*DecCnst))
KerDec = (KerDec - min(KerDec))/(max(KerDec) - min(KerDec));

KerDec = KerDec[KerDec > 0]
KerDec = KerDec[1:]
KerTotal = np.concatenate([KerRise, KerDec])
plt.plot(np.arange(len(KerTotal))/frame_rate, KerTotal)
plt.xlabel('seconds')
plt.ylabel('predicted GCaMP\nresponse')
plt.show()

# z-brain dimensions
height = 1406
width = 621
Zs = 138

color_fish = ['#2258e0', '#22e061', '#e02222']
#%

n_first_last = 3 # number of stimuli to incorporate into correlaton analysis 


#%% Figure 4, Z-Brain analysis of pERK maps, 

n_regions = np.max(zbrain)
name_parts = zbrain_LUT.values
zbrain_names = []
for i in range(n_regions):
    name = ''
    for part in name_parts[i]:
        part = str(part)
        if part != 'nan':
            name = name + '-' + part
    zbrain_names.append(name)
#%
regions_labels = np.zeros(zbrain.shape, dtype=np.uint8)
for k, region in enumerate(regions_of_interset_MAPMap.keys()):
    ids = regions_of_interset_MAPMap[region]
    # print('...')
    # print('...')
    # print(k)
    # print(ids)
    # print(region)
    for id in ids:
        regions_labels[zbrain == id] = k+1

n_regions = np.max(regions_labels)
regions_orig = [*regions_of_interset_MAPMap.keys()]
regions = []
for reg in regions_orig:
    regions.append(reg.replace('Preglomerular', 'Pregl.'))

#%
mapmap_dir = r'/media/BigBoy/Owen/2020_HabScreenPaper/Figure4/MAP-MAPs'
mapmap_out = r'/media/BigBoy/Owen/2020_HabScreenPaper/Figure4/MAP-MAPs/analysis/'
os.chdir(mapmap_dir)

mapmaps = glob.glob('*SignificantDeltaMedians.tif')
#print(mapmaps)


mean_regions = np.zeros((len(mapmaps), n_regions, 2))
for n_map, mapmap in enumerate(mapmaps):
    print(mapmap)
    im = imread(mapmap)

    Zs_down, h_down, w_down, ch = im.shape
    Zs_zb, h_zb, w_zb = zbrain.shape
    im_pos = im[:,:,:,1]
    im_neg = im[:,:,:,0]

    im_pos = zoom(im_pos, (Zs_zb/Zs_down, h_zb/h_down, w_zb/w_down), order=0)
    im_neg = zoom(im_neg, (Zs_zb/Zs_down, h_zb/h_down, w_zb/w_down), order=0)


    for i in range(n_regions):
        mean_regions[n_map, i, 0] = np.mean(im_pos[regions_labels==i+1])
        mean_regions[n_map, i, 1] = np.mean(im_neg[regions_labels==i+1])

#%
n_print = 5
outline_fullframe = tifffile.imread('/media/BigBoy/ciqle/ref_brains/ZBrain2_0_outline_proj.tif')
max_val_mean = 6000

for i in range(len(mapmaps)):
    comp_name = mapmaps[i].replace('_SignificantDeltaMedians.tif', '').replace('_', ' ').replace('over', 'vs')
    plt.figure(figsize=(15,10))
    plt.plot(mean_regions[i,:,0], 'g')
    plt.plot(mean_regions[i,:,1], 'r')
    plt.title(comp_name)
    plt.xticks(np.arange(n_regions), labels=regions, rotation=90)
    plt.show()
    print(mapmaps[i])
    print('<<<<<<<<top positive>>>>>>>>>>>')
    top = np.argsort(mean_regions[i,:,0])[-n_print:]
    for id in top:
        print(regions[id])
    print('<<<<<<<<top negative>>>>>>>>>>>')
    top = np.argsort(mean_regions[i,:,1])[-n_print:]
    for id in top:
        print(regions[id])

    regions_image_pos = np.zeros((Zs_zb, h_zb, w_zb))
    regions_image_neg = np.zeros((Zs_zb, h_zb, w_zb))
    for id in range(n_regions):
        regions_image_pos[regions_labels == id+1] = mean_regions[i, id, 0]
        regions_image_neg[regions_labels == id+1] = mean_regions[i, id, 1]

    im_proj_z = np.mean(regions_image_pos, axis=0)
    im_proj_x = zoom(np.mean(regions_image_pos, axis=2).T, [1, 2/0.798])
    im_proj = np.hstack((im_proj_z, im_proj_x))
    im_proj[outline_fullframe == 1] = max_val_mean
    im_proj_negz = np.mean(regions_image_neg, axis=0)
    im_proj_negx = zoom(np.mean(regions_image_neg, axis=2).T, [1, 2/0.798])
    im_proj_neg = np.hstack((im_proj_negz, im_proj_negx))
    im_proj_neg[outline_fullframe == 1] = max_val_mean
    im_proj = np.stack((im_proj_neg, im_proj, im_proj_neg),axis=-1)
    plt.figure(figsize=(10,10))
    plt.imshow(im_proj/max_val_mean)
    plt.axis('off')
    plt.savefig(os.path.join(mapmap_out, 'regions_projection_'+comp_name+'.png'),bbox_inches='tight')
    #plt.title(np.max(im_proj))
# %
comp_names = []
for mapmap in mapmaps:
    comp_name = mapmap.replace('_SignificantDeltaMedians.tif', '').replace('_', ' ').replace('over', 'vs').replace('Control', 'DMSO').replace('Melatonin', 'Mel.').replace('Flash', 'DF').replace('NoStim', '0DF').replace('Picro', 'PTX')
    comp_names.append(comp_name)
# heatmap_data = np.zeros((n_regions, len(mapmaps)*2))

# for i in range(len(mapmaps)):
#     heatmap_data[:, 2*i] = mean_regions[i, :, 0]
#     heatmap_data[:, 2*i+1] = -mean_regions[i, :, 1]
heatmap_data = np.zeros((n_regions, len(mapmaps)))
for i in range(len(mapmaps)):
    heatmap_data[:, i] = mean_regions[i, :, 0] - mean_regions[i, :, 1]

col_map = sns.diverging_palette(300, 120, s=100, l=50, sep=1, as_cmap=True, center="dark")
plt.figure(figsize=(20,20))
with plt.rc_context({'font.size':50}):
    sns.heatmap(
        heatmap_data,
        cmap = col_map , 
        center=0,
        vmin=-3000,
        vmax=3000,
        yticklabels= regions,
        robust=True,
        square=True,
        cbar_kws = {'shrink':0.3, 'aspect':4},
        )
plt.xticks(np.arange(len(mapmaps))+0.5, labels=comp_names, rotation=90,) 

plt.savefig(os.path.join(mapmap_out, 'heatmap_regions.svg'),bbox_inches='tight')
plt.savefig(os.path.join(mapmap_out, 'heatmap_regions.png'),bbox_inches='tight')


#% difference between habituated conditions and WT

comps = (
    (6,2),
    (10,2)
    )
comp_vecs = np.zeros((n_regions, len(comps)))
for k, comp in enumerate(comps):
    comp_vecs[:,k] = (mean_regions[comp[0], :, 0] - mean_regions[comp[0], :, 1])  - (mean_regions[comp[1], :,0] - mean_regions[comp[1], :,1])
    
plt.figure(figsize=(7,20))
with plt.rc_context({'font.size':50}):
    sns.heatmap(
        comp_vecs,
        cmap = col_map , 
        center=0,
        xticklabels = ['Hab Δ pERK, Mel. vs DMSO', 'Hab Δ pERK, PTX vs DMSO'],
        yticklabels= regions,
        robust=True,
        cbar_kws = {'shrink':0.3, 'aspect':4},
        )


plt.savefig(os.path.join(mapmap_out, 'heatmap_drugVsControlHab.svg'),bbox_inches='tight')
plt.savefig(os.path.join(mapmap_out, 'heatmap_drugVsControlHab.png'),bbox_inches='tight')

#%% Figure 5, analyze Ca2+ data

reload_data = False
if reload_data: 
    ops = {}
    roi_stats = None
    F_norm = None
    fish_data = None
    motor_power_fish = []
    motor_power_fish_flash = []
    motor_power_fish_noflash = []


    fish_types = np.zeros(len(data_paths))
    for k, data_path in enumerate(tqdm(data_paths)):
        

        data_path =  os.path.realpath(data_path)
        os.chdir(data_path)
        fish_name = os.path.split(data_path)[1]
        planes = natsorted(glob.glob('*plane*_data.npy'))

        print('loading...')
        print(fish_name)
        
        if fnmatch.fnmatch(fish_name, '*DMSO*'):
            fish_type = 1
            print('dmso control fish')
        
        if fnmatch.fnmatch(fish_name, '*icro*'):
            fish_type = 2
            print('picrotoxin fish')
        if fnmatch.fnmatch(fish_name, '*elat*'):
            fish_type = 3
            print('melatonin fish')
        fish_types[k] = fish_type
        # load in planes for that fish
        for i in range(len(planes)):
            plane_data = np.load(planes[i], allow_pickle=True).item()
            #print(plane_data['plane'])
            roi_stats_temp = plane_data['roi_stats']
            iscell = plane_data['iscell']
            # cells = iscell[:,0] == 1
            cells = iscell[:,1] > cell_thresh
            n_cells = np.sum(cells)
            roi_stats_temp = roi_stats_temp[cells]
            F_temp = stats.zscore(plane_data['F'][cells,:], axis=1)
            F_temp[~np.isfinite(F_temp)] = 0
            fish_data_temp = np.stack((k*np.ones(n_cells), fish_type*np.ones(n_cells))).T.astype('uint8')

            for roi in range(len(roi_stats_temp)):
                roi_stats_temp[roi]['fish_name']=fish_name
            if i == 0 and k == 0: 
                roi_stats = roi_stats_temp
                F_norm = F_temp
                fish_data = fish_data_temp
            else:
                roi_stats = np.hstack((roi_stats, roi_stats_temp))
                F_norm = np.vstack((F_norm, F_temp))
                fish_data = np.vstack((fish_data, fish_data_temp))
            
            if i == 5:
                ops[fish_name] = plane_data['ops']

        # determine correlation with stim and motion_artifact

        stim_df = np.load('stimvec_mean.npy')

        df_start_inds = np.where(np.diff(stim_df)>0.1)[0]
        df_start_inds = np.delete(df_start_inds, np.where(np.diff(df_start_inds) == 1)[0]+1)
        stim_df_starts = np.zeros(stim_df.shape)

        motor_sig = ops[fish_name]['corrXY']
        motor_sig[ops[fish_name]['badframes']] = 0
        motor_pow = savgol_filter(np.std(rolling_window(motor_sig, 3), -1), 15, 2)
        motor_pow = motor_pow - np.median(motor_pow)
        motor_pow_flash = zscore(motor_pow.copy())
        motor_pow_flash[stim_df < 0.5] = 0
        motor_pow_flash[motor_pow_flash<3] = 0
        motor_pow_noflash = zscore(motor_pow.copy())
        motor_pow_noflash[motor_pow_noflash<3] = 0
        motor_pow_noflash[motor_pow_flash > 0] = 0

        motor_power_fish.append(motor_pow)
        motor_power_fish_flash.append(motor_pow_flash)
        motor_power_fish_noflash.append(motor_pow_noflash)
        stim_df_conv = GCaMPConvolve(stim_df, KerTotal)
        stim_df_starts_conv = GCaMPConvolve(stim_df_starts, KerTotal)
       
        motor_conv = GCaMPConvolve(motor_pow, KerTotal)
        motor_conv[motor_conv<0.2] = 0
        
        # inteaction between motor and df stimuli

        motor_df_conv = zscore(motor_conv*stim_df_conv)
        motor_nodf_conv = zscore(motor_conv*(1-stim_df_conv))
        motor_conv = zscore(motor_conv)
        
        plt.figure(figsize=(20,10))
        plt.plot(stim_df_conv, label='dark flash stim')
        plt.plot(motor_nodf_conv, label='motor')
        plt.plot(motor_df_conv, label='motor x df stim')
        plt.legend()
        plt.show()

        # %
        fish_IDs = np.where(fish_data[:,0] == k)[0]
        F_norm_fish = F_norm[fish_IDs, :]
        nROIs = len(fish_IDs)
        corrMat_temp = np.zeros([nROIs, 9])

        # # correlation between zscored activity traces and different variables, order is: 
        # 0 - all flashes
        # 1 - first 'n_first_last' flashes
        # 2 - last 'n_first_last' flashes
        # 3 - flash onset, all flashes
        # 4 - flash onset, first 'n_first_last' flashes
        # 5 - flash onset, last 'n_first_last' flashes
        # 6 - motor power derived from motion artifact in images
        # 7 - motor power x flashes
        # 8 - motor power x not flashes


        corrMat_temp[:, 0] = pearsonr_vec_2Dnumb(stim_df_conv[start_analyze_frame:], F_norm_fish[:, start_analyze_frame:])
        corrMat_temp[:, 1] = pearsonr_vec_2Dnumb(stim_df_conv[df_start_inds[0]-50:df_start_inds[n_first_last]], F_norm_fish[:, df_start_inds[0]-50:df_start_inds[n_first_last]])
        corrMat_temp[:, 2] = pearsonr_vec_2Dnumb(stim_df_conv[df_start_inds[-n_first_last]-50:], F_norm_fish[:, df_start_inds[-n_first_last]-50:])
        corrMat_temp[:, 3] = pearsonr_vec_2Dnumb(stim_df_starts_conv[start_analyze_frame:], F_norm_fish[:, start_analyze_frame:])
        corrMat_temp[:, 4] = pearsonr_vec_2Dnumb(stim_df_starts_conv[df_start_inds[0]-50:df_start_inds[n_first_last]], F_norm_fish[:, df_start_inds[0]-50:df_start_inds[n_first_last]])
        corrMat_temp[:, 5] = pearsonr_vec_2Dnumb(stim_df_starts_conv[df_start_inds[-n_first_last]-50:], F_norm_fish[:, df_start_inds[-n_first_last]-50:])
        corrMat_temp[:, 6] = pearsonr_vec_2Dnumb(motor_conv[start_analyze_frame:], F_norm_fish[:, start_analyze_frame:])
        corrMat_temp[:, 7] = pearsonr_vec_2Dnumb(motor_df_conv[start_analyze_frame:], F_norm_fish[:, start_analyze_frame:])
        
        corrMat_temp[:, 8] = pearsonr_vec_2Dnumb(motor_nodf_conv[start_analyze_frame:], F_norm_fish[:, start_analyze_frame:])
        


        corrMat_temp[np.isnan(corrMat_temp)] = 0 # set invalid correlations to 0

        # sns.jointplot(x=corrMat_temp[:,0], y=corrMat_temp[:,3], ylim=(-1,1), xlim=[-1,1])
        # plt.show()

        if k == 0:
            corrMat = np.copy(corrMat_temp)
        else:
            corrMat = np.vstack((corrMat, corrMat_temp))
    
    # peform multiple regression
    stim_df_conv = GCaMPConvolve(stim_df, KerTotal)
    n_blocks = 12
    df_stim_vecs = np.zeros((12, len(stim_df_conv)))
    stim_per_block = int(60/n_blocks)

    for i in range(n_blocks):
        inds = (df_start_inds[i*stim_per_block], df_start_inds[i*stim_per_block+(stim_per_block-1)]+50)
        vec = np.zeros(len(stim_df_conv))
        vec[inds[0]:inds[1]] = stim_df_conv[inds[0]:inds[1]]
        df_stim_vecs[i,:] = vec
    plt.plot(df_stim_vecs.T)
    plt.xlim((df_start_inds[0]-10, df_start_inds[6]))

    n_fish = len(motor_power_fish)

    motor_conv = np.zeros((n_fish, len(stim_df_conv)))

    for i in range(n_fish):
        motor_conv[i,:] = GCaMPConvolve(motor_power_fish[i], KerTotal)
    
    motor_conv[motor_conv<0.2] = 0

    # inteaction between motor and df stimuli
    motor_df_conv = motor_conv*stim_df_conv
    motor_nodf_conv = motor_conv*(1-stim_df_conv)
    # normalize with z-score
    motor_nodf_conv = zscore(motor_nodf_conv, axis=1)
    motor_df_conv = zscore(motor_df_conv, axis=1)
    df_stim_vecs = zscore(df_stim_vecs, axis=1)

    no_transients = np.sum(F_norm, axis=1) == 0
    n_rois = len(roi_stats)
    scores = np.zeros(n_rois)
    coeffs = np.zeros((n_rois, n_blocks+2))

    #%
    for i in tqdm(range(n_rois), mininterval=1):
        if not no_transients[i]:
            fish_id = fish_data[i, 1]

            roi = roi_stats[i]
            X = np.vstack([df_stim_vecs, motor_nodf_conv[fish_id,:], motor_df_conv[fish_id,:]]).T
            y = F_norm[i, start_analyze_frame:]
            X = X[start_analyze_frame:,:]
            
            reg = LinearRegression().fit(X,y)

            scores[i] = reg.score(X,y)

            coeffs[i,:] = reg.coef_

    os.chdir(suite2p_dir)
    h5_file = 'compiled_imaging_data.h5'
    try:
        f_h5 = h5py.File(h5_file, 'w')
    except:
        f_h5.close()
        f_h5 = h5py.File(h5_file, 'w')
    f_h5.create_dataset('F_norm', data=F_norm, chunks=True, dtype=np.float32)
    f_h5.create_dataset('fish_data', data=fish_data, chunks=True, dtype=np.uint8)
    f_h5.create_dataset('corrMat', data=corrMat, chunks=True, dtype=np.float32)
    f_h5.create_dataset('stim_df', data=stim_df, chunks=True, dtype=np.float32)
    f_h5.create_dataset('df_start_inds', data=df_start_inds, chunks=True)
    f_h5.create_dataset('motor_power_fish', data=np.array(motor_power_fish), chunks=True)
    f_h5.create_dataset('motor_power_fish_flash', data=np.array(motor_power_fish_flash), chunks=True)
    f_h5.create_dataset('motor_power_fish_noflash', data=np.array(motor_power_fish_noflash), chunks=True)
    f_h5.create_dataset('scores', data=scores, chunks=True)
    f_h5.create_dataset('coeffs', data=coeffs, chunks=True)
    f_h5.close()

    npy_file = 'compiled_imaging_data.npy'
    imaging_data = {
        'roi_stats' : roi_stats,
        'ops' : ops, 
        }
    np.save(npy_file, imaging_data, allow_pickle=True)






#%

os.chdir(suite2p_dir)

f_h5 = h5py.File('compiled_imaging_data.h5', 'r')
df_start_inds = f_h5['df_start_inds'][()]
stim_df = f_h5['stim_df'][()]
F_norm = f_h5['F_norm'][()]
fish_data = f_h5['fish_data'][()]
corrMat = f_h5['corrMat'][()]
motor_power_fish = f_h5['motor_power_fish'][()]
motor_power_fish_flash = f_h5['motor_power_fish_flash'][()]
motor_power_fish_noflash = f_h5['motor_power_fish_noflash'][()]
scores =  f_h5['scores'][()]
coeffs =  f_h5['coeffs'][()]
imaging_data = np.load('compiled_imaging_data.npy', allow_pickle=True).item()
ops = imaging_data['ops']
roi_stats = imaging_data['roi_stats']
stim_df_conv = GCaMPConvolve(stim_df, KerTotal)

#%

outline = tifffile.imread('/media/BigBoy/ciqle/ref_brains/ZBrain2_0_outline_proj.tif')


def draw_hit_volume(hits_inds, values = [1], draw_centroid=False, add_write=True, proj_mean=True, draw_outline=False, save_name = None, normalize=True):
    hits_inds_shuf = hits_inds.copy()
    np.random.shuffle(hits_inds_shuf)
    IM_roi = np.zeros((Zs, height, width))
    for j in range(len(hits_inds)):
        roi_coords_y = roi_stats[hits_inds[j]]['ypix_zbrain'].astype('int')
        roi_coords_x = roi_stats[hits_inds[j]]['xpix_zbrain'].astype('int')
        roi_coords_z = roi_stats[hits_inds[j]]['centroid_zbrain'][2].astype('int')
        roi_coords_y[roi_coords_y > height-1] = height-1
        roi_coords_x[roi_coords_x > width-1] = width-1
        if draw_centroid:
            roi_coords_y = np.mean(roi_coords_y).astype('int')
            roi_coords_x = np.mean(roi_coords_x).astype('int')
            roi_coords_z = np.mean(roi_coords_z).astype('int')
        if add_write:
            if len(values) == 1:  
                IM_roi[roi_coords_z, roi_coords_y, roi_coords_x]  += values
            else:
                IM_roi[roi_coords_z, roi_coords_y, roi_coords_x]  += values[j]
        else:
            if len(values) == 1:  
                IM_roi[roi_coords_z, roi_coords_y, roi_coords_x]  = values
            else:
                IM_roi[roi_coords_z, roi_coords_y, roi_coords_x]  = values[j]

    IM_roi_crop = IM_roi[zlims[0]:zlims[1], ylims[0]:ylims[1], xlims[0]:xlims[1]]
    if proj_mean:
        im_proj_z = np.mean(IM_roi_crop[:,:, :], axis=0)
        im_proj_x = zoom(np.mean(IM_roi_crop[:,:, :], axis=2).T, [1, 2/0.798])
    else:
        im_proj_z = np.max(IM_roi_crop[:,:, :], axis=0)
        im_proj_x = zoom(np.max(IM_roi_crop[:,:, :], axis=2).T, [1, 2/0.798])
    
    if normalize:
        im_proj = np.hstack((im_proj_z/np.max(im_proj_z), im_proj_x/np.max(im_proj_x)))
    else:
        im_proj = np.hstack((im_proj_z, im_proj_x))

    if draw_outline:
        im_proj[outline > 0.01] = np.max(im_proj)

    if not save_name==None:
        imsave(os.path.join(analysis_out, save_name+'_proj_image.tif'), im_proj)
    return IM_roi, im_proj


# make the outline image, cropped to area imaged with at least 5 fish

volume = np.zeros(zbrain.shape, dtype=np.uint16)

xlims = (0, width)
ylims = (0, height)
zlims = (0, Zs)

IM_rois, im_rois_proj = draw_hit_volume(np.arange(len(roi_stats)), draw_outline=True, save_name = 'roi_density')
plt.figure(figsize=(20,20))
plt.imshow(im_rois_proj, cmap='inferno')
plt.title('Density of ROIs detected')
plt.axis('off')
plt.show()

#% crop to imaged volume
n_rois_min = 10
x_trace = np.max(np.max(IM_rois, axis=0) , axis=0)>n_rois_min
y_trace = np.max(np.max(IM_rois, axis=0) , axis=1)>n_rois_min
z_trace =  np.max(np.max(IM_rois, axis=1) , axis=1)>n_rois_min
#plt.plot(z_trace)

xlims = np.where(x_trace)[0][0], np.where(x_trace)[0][-1]
print(xlims)

ylims = np.where(y_trace)[0][0], np.where(y_trace)[0][-1]
print(ylims)

zlims = np.where(z_trace)[0][0], np.where(z_trace)[0][-1]
print(zlims)

zbrain_crop = zbrain[zlims[0]:zlims[1], ylims[0]:ylims[1], xlims[0]:xlims[1]]
Zs_crop, height_crop, width_crop = zbrain_crop.shape


zbrain_outline_z = np.zeros((height_crop, width_crop))
zbrain_outline_x = zoom(np.zeros((Zs_crop, height_crop)).T, [1, 2/0.798])
mask_3d = np.zeros(zbrain_crop.shape)
mask_3d[:] = 0


IDs = [
    #np.where((zbrain_crop >=27) & (zbrain_crop <= 28)), # thalamus
    np.where((zbrain_crop >=29) & (zbrain_crop <= 31)), # telencephalon
    np.where((zbrain_crop >=48) & (zbrain_crop <= 110) | (zbrain_crop == 119)), # hindbrain
    np.where((zbrain_crop >=111) & (zbrain_crop <= 112)), # tectum
    np.where((zbrain_crop == 50) ), # inferior olive
    np.where((zbrain_crop == 23) ), # pretectum  
    #np.where((zbrain_crop == 114) ), # nucMLF
    #np.where((zbrain_crop == 70) |  (zbrain_crop == 71) | (zbrain_crop == 72)| (zbrain_crop == 77) | (zbrain_crop == 78)| (zbrain_crop == 79) | (zbrain_crop == 91)| (zbrain_crop == 94)| (zbrain_crop == 95)| (zbrain_crop == 101)| (zbrain_crop == 102) | (zbrain_crop == 107)| (zbrain_crop == 108)), # reticulospinal 
    #np.where((zbrain_crop >=84) & (zbrain_crop <= 89)), # more RS
    #np.where((zbrain_crop == 79) | (zbrain_crop == 89) | (zbrain_crop == 95) ), # v cells

]

for ids in IDs:
    mask_3d = np.zeros(zbrain_crop.shape)
    mask_3d[ids] = 1
    mask = np.max(mask_3d, axis=0)
    outline = morphology.distance_transform_edt(1-mask) == 1
    #outline = morphology.binary_dilation(outline, iterations=1)
    zbrain_outline_z[outline==1] =1

    mask = zoom(np.max(mask_3d, axis=2).T, [1, 2/0.798], order=0)
    outline = morphology.distance_transform_edt(1-mask) == 1
    #outline = morphology.binary_dilation(outline, iterations=1)
    zbrain_outline_x[outline==1] =1




proj = np.hstack((zbrain_outline_z, zbrain_outline_x))
proj = proj * 2
proj = proj.astype(np.uint8)
proj[proj>0] = 255
proj[proj < 255] = 0

tifffile.imsave('/media/BigBoy/ciqle/ref_brains/ZBrain2_0_outline_proj_areas_crop.tif', data=proj)
outline = proj
IM_rois, im_rois_proj = draw_hit_volume(np.arange(len(roi_stats)), draw_outline=True, save_name = 'roi_density_cropped')
plt.figure(figsize=(20,20))
plt.imshow(im_rois_proj, cmap='inferno')
plt.title('Density of ROIs detected')
plt.axis('off')
IM_rois=None


fish_names = []
for fish_name in ops.keys():
    fish_names.append(fish_name)

#%% correlations to stim vs motor 
corr_df = np.max(abs(corrMat[:,:6]), axis=1)
corr_motor = np.max(abs(corrMat[:,6:]), axis=1)
corr_motor_flash = abs(corrMat[:,7])
corr_motor_noflash = abs(corrMat[:,8])


sns.jointplot(x=corr_motor_flash, y=corr_motor_noflash, ylim=(-0.5,0.5), xlim=(-0.5,0.5), kind='hist')

# sns.jointplot(x=np.sqrt(scores), y=corr_df, ylim=(-0.5,0.5), xlim=(-0.5,0.5), kind='hist')
# plt.show()0
#%

min_color = 0.5
max_color = 0.91

# print(np.min(hue_norm))
# plt.hist(hue_norm, 500)
#%
fish_types = ['DMSO', 'Picrotoxinin', 'Melatonin']

# def plot_tuning_imges()
def plot_tuning_images(hue_norm, saturation, lut_title, sat_thresh = 0.2, sat_max = 0.7):
    single_fish_inds = [2]
    fish_types_plot = [
        'DMSO', 
        'Picrotoxinin',
        'Melatonin',
        'All', 
        os.path.split(data_paths[single_fish_inds[0]])[1]]
    fig,ax = plt.subplots(nrows=3, ncols=2, figsize=(50,30))
    for i in range(5):
        subplot_ind = np.unravel_index(i, (3,2))
        if i == 0:
            hits_sat = np.where((saturation > sat_thresh) & (fish_data[:,1] == 1))[0]
        elif i == 1:
            hits_sat = np.where((saturation > sat_thresh) & (fish_data[:,1] == 2))[0]
        elif i == 2:
            hits_sat = np.where((saturation > sat_thresh) & (fish_data[:,1] == 3))[0]
        elif i == 3: 
            hits_sat = np.where((saturation > sat_thresh))[0]
        else:
            hits_sat = np.where((saturation > sat_thresh) & (fish_data[:,0] == single_fish_inds[0]))[0]
        r = np.zeros(len(hits_sat))
        g = np.zeros(len(hits_sat))
        b = np.zeros(len(hits_sat))

        for k, roi in enumerate(hits_sat):
            
            r[k], g[k], b[k] = colorsys.hsv_to_rgb(hue_norm[roi],1, np.max(((saturation[roi]-sat_thresh)/(sat_max-sat_thresh), 0)))

        IM_roi_r, im_proj_r = draw_hit_volume(hits_sat, values=r, add_write=False, proj_mean=False, draw_outline=True, normalize=False)
        IM_roi_g, im_proj_g = draw_hit_volume(hits_sat, values=g, add_write=False, proj_mean=False, draw_outline=True, normalize=False)
        IM_roi_b, im_proj_b = draw_hit_volume(hits_sat, values=b, add_write=False, proj_mean=False, draw_outline=True, normalize=False)

        ax[subplot_ind].imshow(np.stack((im_proj_r, im_proj_g, im_proj_b), axis=2))
        ax[subplot_ind].set_title(fish_types_plot[i])
        ax[subplot_ind].set_axis_off()

    cbar_size = 100

    cbar = np.zeros((cbar_size+1,cbar_size+1,3))
    for k, h in enumerate(np.arange(0,1+1/cbar_size,1/cbar_size)):
        for j, v in enumerate(np.arange(0,1+1/cbar_size,1/cbar_size)):
            h_norm = max_color*(h + min_color)/(1+min_color)
            cbar[j,k,:] = colorsys.hsv_to_rgb(h_norm,1, np.max(((v-sat_thresh)/(sat_max-sat_thresh), 0)))
    non_hit_area = int(cbar_size*sat_thresh)
    
    cbar[:non_hit_area, :, :] = 0
    ax[2,1].imshow(cbar, origin='lower')
    ax[2,1].set_axis_off()
    ax[2,1].set_title(lut_title)
    plt.savefig(os.path.join(analysis_out,lut_title+'_tuningImages.svg'))
    plt.savefig(os.path.join(analysis_out,lut_title+'_tuningImages.png'))
    plt.show()

def plot_tuning_region(tuning, saturation, sat_thresh = 0.25):
    regions_labels = np.zeros(zbrain.shape, dtype=np.uint8)
    for k, region in enumerate(regions_of_interset.keys()):
        ids = regions_of_interset[region]
        for id in ids:
            regions_labels[zbrain == id] = k+1

    n_regions = np.max(regions_labels)
    regions = [*regions_of_interset.keys()]

    n_fish = np.max(fish_data[:,0] )+1
    treat_types = np.zeros(n_fish)
    tuning_regions = np.zeros((n_regions+1, n_fish))
    tuning_regions[:] = np.nan
    for fish in range(n_fish):
        
        fish_inds = np.where(fish_data[:,0] == fish)[0]
        treat_types[fish] = fish_data[fish_inds[0],1]
        hits_sat = np.where((saturation > sat_thresh) & (fish_data[:,0] == fish))[0]
        n_hits = len(hits_sat)
        rois_hits = roi_stats[hits_sat]
        tuning_hits = tuning[hits_sat]
        saturation_hits = saturation[hits_sat]
        centroid_regions = np.zeros(n_hits, dtype=int)
        

        for i in range(n_hits):
            centroid_x, centroid_y, centroid_z = rois_hits[i]['centroid_zbrain']
            centroid_regions[i] = regions_labels[centroid_z, centroid_y, centroid_x]
        #%
        unique, unique_inv, unique_counts = np.unique(centroid_regions, return_inverse=True, return_counts=True)


        k = 0
        j = 0
        for ind in range(n_regions+2):
            if np.any(unique == ind):
                tuning_regions[ind, fish] = np.sum(tuning_hits[unique_inv == ind] * saturation_hits[unique_inv == ind] )/np.sum(saturation_hits[unique_inv == ind] *np.ones(tuning_hits[unique_inv == ind].shape[0]))
                #tuning_regions[ind, fish] = np.mean(tuning_hits[unique_inv == ind])
    #%
    plt.figure(figsize=(7,5))
    for i in range(3):
        inds = np.where(treat_types == i+1)[0]
        mean_vec = np.nanmean(tuning_regions[1:, inds], axis=1)
        std_vec = np.nanstd(tuning_regions[1:, inds], axis=1)
        sterr_vec = std_vec/np.sqrt(len(inds))
        plt.fill_between(np.arange(n_regions),mean_vec+sterr_vec, mean_vec-sterr_vec, color = color_fish[i], alpha=0.5, label=fish_types[i])
    plt.xticks(np.arange(n_regions), labels = regions, rotation=90)
    plt.legend()

    for i in range(3):
        inds = np.where(treat_types == i+1)[0]
        mean_vec = np.nanmean(tuning_regions[1:, inds], axis=1)
        plt.plot(mean_vec, '.', color = color_fish[i])
    plt.show()


saturation = np.max(np.vstack((corr_df, corr_motor)), axis=0)
coeffs_zscore = zscore(coeffs, axis=0)
hue = np.mean(abs(coeffs_zscore[:,0:-2]), axis=1)/(np.mean(abs(coeffs_zscore[:,0:-2]), axis=1) + np.mean(abs(coeffs_zscore[:,-2:]), axis=1))
hue[np.isnan(hue)] = 0
hue_norm = max_color*(hue + min_color)/(1+min_color)
plot_tuning_images(hue_norm, saturation, 'tuning, df stim vs. motor_regression', sat_thresh =0.2)
plot_tuning_region(hue, saturation, sat_thresh = 0.2)
#%
# n_blocks = 2
# hue =np.mean(coeffs_zscore[:,:n_blocks], axis=1)/(np.mean(coeffs_zscore[:,:n_blocks], axis=1)+np.mean(coeffs_zscore[:,-(3+n_blocks):-3], axis=1))
first_resp = abs(corrMat[:,1])
rest_resp = abs(corrMat[:,2])

hue = first_resp/(first_resp+rest_resp)
hue[np.isnan(hue)] = 0
hue_norm = max_color*(hue + min_color)/(1+min_color)
saturation = np.max(abs(corrMat[:,:3]), axis=1)

plot_tuning_images(hue_norm, saturation, 'tuning, first vs last regressors', sat_thresh =0.2)
plot_tuning_region(hue, saturation, sat_thresh = 0.2)
# saturation = np.max(abs(corrMat[:,-2:]), axis=1)
# saturation
# hue = abs(coeffs[:,13])/(abs(coeffs[:,12])+abs(coeffs[:,13]))
# saturation[np.isnan(hue)] =0
# hue[np.isnan(hue)] = 0
# hue_norm = max_color*(hue + min_color)/(1+min_color)
# plot_tuning_images(hue_norm, saturation, 'tuning, motor flash vs noflash regressors', sat_thresh =0.02)

# saturation = np.max(np.vstack((corr_df, corr_motor)), axis=0)
# hue = corr_df/(corr_df + corr_motor)
# hue[np.isnan(hue)] = 0
# #hue = hue+color_rot
# hue_norm = max_color*(hue + min_color)/(1+min_color)
# plot_tuning_images(hue_norm, saturation, 'tuning, df stim vs. motor')


#%
# x = np.arange(12).reshape(-1, 1)
# slopes = np.zeros(len(scores))
# for roi in tqdm(range(len(scores))):
#     weights = coeffs[roi, :12]
#     reg = LinearRegression().fit(x,weights)
#     slopes[roi] = reg.coef_

# slope_span = 0.02
# hue = -slopes
# hue = (hue)/(slope_span)
# hue[hue > 1] = 1
# hue[hue<-1] = -1
# hue = (hue+1)/2
# plt.hist(hue, 500)
# #%

# hue_norm = max_color*(hue + min_color)/(1+min_color)
# saturation = np.max(abs(corrMat[:,:6]), axis=1)

# plot_tuning_images(hue_norm, saturation, 'tuning, slope of df regressors', sat_thresh =0.2)

#%

# #% plot correlations to first vs last stimuli

# hue_firstLast = abs(corrMat[:,1])/ (abs(corrMat[:,1]) + abs(corrMat[:,2]))
# hue_norm = max_color*(hue_firstLast + min_color)/(1+min_color)

# sat_firstLast = np.max(abs(corrMat[:,:3]), axis=1)
# #sat_firstLast[sat_firstLast < 0.1] = 0
# plot_tuning_images(hue_firstLast, sat_firstLast, 'first vs. last flashes')


#%



# plot_tuning_region(corr_df/(corr_df + corr_motor), np.max(np.vstack((corr_df, corr_motor)), axis=0))
# plot_tuning_region(abs(corrMat[:,1])/ (abs(corrMat[:,1]) + abs(corrMat[:,2])), np.max(abs(corrMat[:,:3]), axis=1))
# plot_tuning_region((corr_motor_flash)/(corr_motor_flash+ corr_motor_noflash), corr_motor)
#%

# #%
# sat_motor = corr_motor
# hue = (corr_motor_flash)/(corr_motor_flash+ corr_motor_noflash)
# #hue[np.isnan(hue)] = 0
# #hue = hue+color_rot
# hue_norm = max_color*(hue + min_color)/(1+min_color)
# plot_tuning_images(hue_norm, sat_motor, 'motor flash vs no-flash',sat_thresh = 0.25)
# #%
# sat_stim_motorFlash = np.max(np.vstack((corr_df, corr_motor_flash)), axis=0)
# hue = corr_df/(corr_df + corr_motor_flash)
# hue[np.isnan(hue)] = 0
# #hue = hue+color_rot
# hue_norm = max_color*(hue + min_color)/(1+min_color)
# plot_tuning_images(hue_norm, sat_stim_motorFlash, 'stim vs motor-stim',sat_thresh = 0.25)

#%


#%% figure 6, clustering of Ca2+ traces 

recalc_clusters = False
if recalc_clusters:
    corr_thresh = 0.25
    rescan_thresh = 0.3

    #hits_IDs = np.where(abs(corrMat[:,0])>corr_thresh)[0] # only use one criterion for hit
    #hits_IDs = np.where(np.max(abs(corrMat[:,:6]), axis=1)>=corr_thresh)[0] # any of the dark flash correlations will cause inclusion
    hits_IDs = np.where((np.max(abs(corrMat[:,:6]), axis=1)>=corr_thresh))[0] 
    #hits_IDs = hits_IDs[fish_data[hits_IDs,1] == 1] # only look at wt fish...
    n_hits = len(hits_IDs)
    perc_hits = np.round(n_hits/F_norm.shape[0] * 100)

    print('selected ' + str(n_hits) + ' ROIs, ' + str(perc_hits) + ' percent of ROIs based on correlation threshold')
    IM_roi, im_show = draw_hit_volume(hits_IDs, draw_outline=True, save_name = 'ROIs_Correlated_stim')
    plt.figure(figsize=(10,20))
    plt.imshow(im_show, vmin = 0, vmax=0.7, cmap='inferno')
    plt.axis('off')
    plt.title('units by correlation')

    plt.savefig(os.path.join(analysis_out, 'units_correlated_wStimOrMotor.svg'))
    #hits_IDs = np.where(np.max(abs(corrMat[:,0:3]), axis=1)>corr_thresh)[0]
    
    clustData = F_norm[hits_IDs, start_analyze_frame:]
    corr_m = np.corrcoef(clustData)


    other_fish_thresh = 0.3
    other_fish_number = 5
    fish_data_hits = fish_data[hits_IDs, :]

    hits_other_fish = []

    for roi in range(n_hits):
        fish = fish_data_hits[roi, 0]
        above_thresh = np.where((fish_data_hits[:, 0]!=fish) & (corr_m[:,roi] > other_fish_thresh))[0]
        n_other_fish = len(np.unique(fish_data_hits[above_thresh, 0]))
        if n_other_fish > other_fish_number:
            hits_other_fish.append(roi)

    n_hits_other = len(hits_other_fish)
    perc_hits_other = np.round(n_hits_other/F_norm.shape[0] * 100)
    print('selected ' + str(n_hits_other) + ' ROIs, ' + str(perc_hits_other) + ' percent of ROIs based on correlation to ROIs in other fish')
    hits_IDs = hits_IDs[hits_other_fish]
    clustData = clustData[hits_other_fish, :]
    corr_m = corr_m[np.ix_(hits_other_fish, hits_other_fish)]

    IM_roi, im_show = draw_hit_volume(hits_IDs, draw_outline=True, save_name='ROIs_clustering')
    plt.figure(figsize=(10,20))
    plt.imshow(im_show, vmin = 0, vmax=0.7, cmap='inferno')
    plt.axis('off')
    plt.title('units selected for clustering')
    plt.show()

    print('>>>>>>Affinity Propagation<<<<<<')

    #af = AffinityPropagation(preference=-9, damping=0.9, max_iter=500, random_state=1, affinity='precomputed', verbose=True).fit(corr_m)
    #af = AffinityPropagation(preference=-3.75, damping=0.9, max_iter=500, random_state=1, affinity='precomputed', verbose=True).fit(corr_m)
    af = AffinityPropagation(preference=-9, damping=0.9, max_iter=500, random_state=1, affinity='precomputed', verbose=True).fit(corr_m)
    #%

    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    n_clusters = len(cluster_centers_indices)
    print(n_clusters)
    #clust_colors = get_colors(n_clusters, pastel_factor=0, exclude_colors= [(1.0, 1.0, 0.0), (0,0,0)], rng=1,n_attempts=1000, colorblind_type = 'Deuteranomaly')
    clust_colors = get_colors(n_clusters, pastel_factor=0, rng=1,n_attempts=1000)
    clust_colors[9] = (0.8, 0.8, 0)
    sns.color_palette(clust_colors)
    mean_vecs, clust_IDs_reorder = plot_cluster_means(F_norm, hits_IDs, af.labels_, median=False, plot_heat=False, save_name='original_clusters_DMSO')
    IM_clust, IM_clust_centroid = plot_cluster_rois(hits_IDs, clust_IDs_reorder)  


# using multiple regression
    n_rois = F_norm.shape[0]
    scores_toClust = np.zeros(n_rois)
    coeffs_toClust = np.zeros((n_rois, n_clusters))
    X = mean_vecs[:,start_analyze_frame:].T
    potential_hits = np.where(scores>0.000)[0]
    for roi in tqdm(potential_hits):

        y = F_norm[roi,start_analyze_frame:]
        reg = LinearRegression(positive=True).fit(X, y)

        scores_toClust[roi]=reg.score(X,y)

        coeffs_toClust[roi, :]=reg.coef_

    #%
    corr_to_clusts_potential = pearsonr_2Dnumb(mean_vecs[:,start_analyze_frame:], F_norm[potential_hits,start_analyze_frame:])
    corr_to_clusts = np.zeros(coeffs_toClust.shape)
    corr_to_clusts[potential_hits,:] = corr_to_clusts_potential.T


    #hits_clusters = np.where((scores_toClust > 0.1))[0]
    #hits_clusters_IDs = np.argmax(coeffs_toClust[hits_clusters,: ], axis=1)

    hits_clusters = np.where((np.max(corr_to_clusts, axis=1) > rescan_thresh))[0]
    hits_clusters_IDs = np.argmax(corr_to_clusts[hits_clusters,: ], axis=1)
    


    n_hits_found = len(hits_clusters)
    perc_found = np.round(100*n_hits_found/n_rois, decimals=2)
    print('found '+ str(n_hits_found) + ' ROIs in functional clusters, ' + str(perc_found) + ' percent of all ROIs')
    #%


    mean_vecs_reorder, clust_IDs_reorder = plot_cluster_means(F_norm, hits_clusters, hits_clusters_IDs, median=True, plot_heat=True, re_order=True, cluster_heat=False, ymax=5.5,  save_name='rescan_clusters_all')
    
    save_clust_name = os.path.join(analysis_out, 'cluster_assignments.npz')
    np.savez(save_clust_name, mean_vecs_reorder=mean_vecs_reorder, clust_IDs_reorder=clust_IDs_reorder,hits_clusters=hits_clusters  )
#%
else:

    clust_data = np.load(os.path.join(analysis_out, 'cluster_assignments.npz'))
    mean_vecs_reorder = clust_data['mean_vecs_reorder']
    clust_IDs_reorder = clust_data['clust_IDs_reorder']
    hits_clusters = clust_data['hits_clusters']
    n_clusters = np.max(clust_IDs_reorder)+1
    clust_colors = get_colors(n_clusters, pastel_factor=0, rng=1,n_attempts=1000)
    clust_colors[9] = (0.8, 0.8, 0)

#%

cluster_names = [
    'noH,On',
    'medH,On',
    'medH,L',
    'strgH,L',
    'weakH,M1',
    'weakH,M2',
    'medH,S',
    'strgH,S',
    'noH,M',
    'noH,S',
    'weakH,S',
    'sensH,M',
]
    
#%
_ , _ = plot_cluster_means(F_norm, hits_clusters, clust_IDs_reorder, median=True, plot_heat=True, re_order=False, cluster_heat=False, ymax=5.5, use_clustlabels = True,  save_name='rescan_clusters_all')
IM_clust, IM_clust_centroid = plot_cluster_rois(hits_clusters, clust_IDs_reorder, save_name='rescan_clusters_all')  
#%% regional analyeses of clusters, Figure 6
x = np.arange(len(stim_df))
x = x/frame_rate/60
x = x-5
plt.figure(figsize=(20,1))
plt.axis('off')
plt.plot(x, stim_df, 'k')
plt.savefig(os.path.join(analysis_out, 'df_stim.svg'))
plt.show()


def zbrain_analysis_centroids(regions_of_interset, IM_clust_centroid, save_name=None, sort_regions=False):
    regions_labels = np.zeros(zbrain.shape, dtype=np.uint8)
    for k, region in enumerate(regions_of_interset.keys()):
        ids = regions_of_interset[region]
        for id in ids:
            regions_labels[zbrain == id] = k+1

    # viewer = napari.Viewer()
    # viewer.add_image(regions_labels)
    # viewer.add_image(zbrain)

    n_regions = np.max(regions_labels)
    n_clusters = IM_clust_centroid.shape[3]
    regions_sum = np.zeros((n_regions, n_clusters))
    regions_sum_norm = np.zeros((n_regions, n_clusters))
    regions = [*regions_of_interset.keys()]
    for i in range(n_clusters):
        sum_cluster = sum_labels(IM_clust_centroid[:,:,:,i], labels=regions_labels, index=np.arange(n_regions)+1)

        regions_sum[:,i] = sum_cluster
        regions_sum_norm[:,i]  = sum_cluster/np.sum(sum_cluster)

    #%
    regions_analysis = pd.DataFrame(regions_sum, index=regions)
    if sort_regions:
        sorted_ix = np.argsort(np.sum(regions_sum, axis=1))[-1::-1]
        nplot= np.where(np.sum(regions_sum, axis=1)[sorted_ix] < 10)[0][0] 

        inds = sorted_ix[:nplot]
    else: 
        nplot = len(regions)
        inds = np.arange(nplot)
    regions_plot = []
    for ind in inds:
        regions_plot.append(regions[ind])
    #%
    regions_plot[9] = 'Pregl. Complex Area'
    with plt.rc_context({'font.size':25}):
        plt.figure(figsize=(10,8))
        for i in range(n_clusters):
            plt.plot(regions_sum[inds, i], 'P--', linewidth=2, markersize=15, color = clust_colors[i], alpha=0.9)
        plt.xticks(ticks=np.arange(nplot), labels=regions_plot, rotation=90)
        plt.ylabel('counts in each region')
        plt.legend(cluster_names)
        if not save_name==None:
            plt.savefig(os.path.join(analysis_out, save_name+'_clusterCountsInRegion.svg'))
        plt.show()

        plt.figure(figsize=(7,9))
        g = sns.heatmap(regions_sum, yticklabels=regions_plot, xticklabels=cluster_names, cmap='magma', robust=True, cbar_kws = {'shrink':0.3, 'aspect':3},)
        for i, tick_label in enumerate(g.axes.get_xticklabels()):
            tick_text = tick_label.get_text()
            tick_label.set_color(clust_colors[i])
        if not save_name==None:
            plt.savefig(os.path.join(analysis_out, save_name+'_clusterCountsInRegion_heatmap.svg'))
        plt.show()

        plt.figure(figsize=(10,8))
        for i in range(n_clusters):
            plt.plot(regions_sum_norm[inds, i], 'P--', linewidth=2, markersize=15, color = clust_colors[i], alpha=0.9)
        plt.xticks(ticks=np.arange(nplot), labels=regions_plot, rotation=90)
        plt.ylabel('proportion normalized to cluster counts')
        plt.legend(cluster_names)
        if not save_name==None:
            plt.savefig(os.path.join(analysis_out, save_name+'_clusterCountsInRegion_norm1.svg'))
        plt.show()

        plt.figure(figsize=(7,9))
        g = sns.heatmap(regions_sum_norm, yticklabels=regions_plot, vmin=0, vmax=0.3, xticklabels=cluster_names, cmap='magma', robust=True, cbar_kws = {'shrink':0.3, 'aspect':3},)
        for i, tick_label in enumerate(g.axes.get_xticklabels()):
            tick_text = tick_label.get_text()
            tick_label.set_color(clust_colors[i])
        if not save_name==None:
            plt.savefig(os.path.join(analysis_out, save_name+'_clusterCountsInRegion_norm1_heatmap.svg'))
        plt.show()

        regions_sum_normregion = regions_sum.copy()
        for i in range(regions_sum.shape[0]):
            regions_sum_normregion[i,:] = regions_sum_normregion[i,:] / np.sum(regions_sum_normregion[i,:])

        plt.figure(figsize=(10,8))
        for i in range(n_clusters):
            plt.plot(regions_sum_normregion[inds, i], 'P--', linewidth=2, markersize=15, color = clust_colors[i], alpha=0.9)
        plt.xticks(ticks=np.arange(nplot), labels=regions_plot, rotation=90)
        plt.ylabel('Proportion normalized to area counts')
        plt.legend(cluster_names)
        if not save_name==None:
            plt.savefig(os.path.join(analysis_out, save_name+'_clusterCountsInRegion_norm2.svg'))
        plt.show()

        plt.figure(figsize=(7,9))
        g = sns.heatmap(regions_sum_normregion, yticklabels=regions_plot, xticklabels=cluster_names, cmap='magma', robust=True, cbar_kws = {'shrink':0.3, 'aspect':3},)
        for i, tick_label in enumerate(g.axes.get_xticklabels()):
            tick_text = tick_label.get_text()
            tick_label.set_color(clust_colors[i])
        if not save_name==None:
            plt.savefig(os.path.join(analysis_out, save_name+'_clusterCountsInRegion_norm2_heatmap.svg'))
        plt.show()

        regions_sum_normregion = regions_sum_norm.copy()
        for i in range(regions_sum.shape[0]):
            regions_sum_normregion[i,:] = regions_sum_normregion[i,:] / np.sum(regions_sum_normregion[i,:])

        plt.figure(figsize=(10,8))
        for i in range(n_clusters):
            plt.plot(regions_sum_normregion[inds, i], 'P--', linewidth=2, markersize=15, color = clust_colors[i], alpha=0.9)
        plt.xticks(ticks=np.arange(nplot), labels=regions_plot, rotation=90)
        plt.ylabel('Proportion normalized to\ncluster size and regional counts')
        plt.legend(cluster_names)
        if not save_name==None:
            plt.savefig(os.path.join(analysis_out, save_name+'_clusterCountsInRegion_norm3.svg'))
        plt.show()

        plt.figure(figsize=(7,9))
        g = sns.heatmap(regions_sum_normregion, yticklabels=regions_plot, xticklabels=cluster_names, cmap='magma', robust=True, cbar_kws = {'shrink':0.3, 'aspect':3},)
        for i, tick_label in enumerate(g.axes.get_xticklabels()):
            tick_text = tick_label.get_text()
            tick_label.set_color(clust_colors[i])
        if not save_name==None:
            plt.savefig(os.path.join(analysis_out, save_name+'_clusterCountsInRegion_norm3_heatmap.svg'))
        plt.show()
zbrain_analysis_centroids(regions_of_interset, IM_clust_centroid, save_name='all_fish')
#%% correlation to motor performance across clusters, Figure 6
corr_vals = corrMat[hits_clusters, :]
        # 0 - all flashes
        # 1 - first 'n_first_last' flashes
        # 2 - last 'n_first_last' flashes
        # 3 - flash onset, all flashes
        # 4 - flash onset, first 'n_first_last' flashes
        # 5 - flash onset, last 'n_first_last' flashes
        # 6 - motor power derived from motion artifact in images
        # 7 - motor power during flashes
        # 8 - motor power outside of flashes
corr_to_stim_motor =pd.DataFrame({
    'cluster':clust_IDs_reorder,
    'flash_all':corr_vals[:,0],
    'flash_first':corr_vals[:,1],
    'flash_last':corr_vals[:,2],
    'motor_all':corr_vals[:,6],
    'motor_flash':corr_vals[:,7],
    'motor_spont':corr_vals[:,8],
})
ysets = corr_to_stim_motor.keys()[1:]

for yset in ysets:
    fig = plt.figure(figsize=(9,6))
    #sns.violinplot(data = corr_to_stim_motor, x ='cluster', y=yset,)

    sns.stripplot(data = corr_to_stim_motor, x ='cluster', y=yset, palette=clust_colors, alpha=0.1, jitter=0.45, zorder=2)
    g = sns.pointplot(data = corr_to_stim_motor, ci=99.99999, color='k', x ='cluster', y=yset, estimator=np.median, linestyles = '', markers='', zorder=1)
    g.set_xticklabels(cluster_names, rotation=90)
    plt.xlabel('')
    plt.ylabel('corr. with\n'+yset)
    plt.hlines(0, xmin=-1, xmax=12, color='k', linestyles='--')
    plt.xlim((-0.5, 11.5))
    for i, tick_label in enumerate(g.axes.get_xticklabels()):
        tick_text = tick_label.get_text()
        tick_label.set_color(clust_colors[i])
    
    plt.savefig(os.path.join(analysis_out, 'ClusterCorrelationVals_w_'+yset+'.svg'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(analysis_out, 'ClusterCorrelationVals_w_'+yset+'.png'), bbox_inches='tight', dpi=300)
    plt.show()

#%% clustered correlogram for functional type in space, Figure 6

z_blocksize =5
xy_blocksize = int(z_blocksize*2/0.798)
for i in range(n_clusters):
    IM_centroid_downsample = IM_clust[zlims[0]:zlims[1], ylims[0]:ylims[1], xlims[0]:xlims[1], i].astype(float)
    IM_centroid_downsample = block_reduce(IM_centroid_downsample, block_size=(z_blocksize,xy_blocksize,xy_blocksize)).flatten()
    #IM_centroid_downsample = zoom(IM_centroid_downsample[:,:,:], [0.05, 0.02, 0.02])
    if i == 0:
        IM_clust_vecs = IM_centroid_downsample
    else:
        IM_clust_vecs = np.vstack((IM_clust_vecs, IM_centroid_downsample))
    # plt.plot(IM_centroid_downsample)
    # plt.show()
#%
# IM_clust_vecs = regions_sum.T
link =  hierarchy.linkage(IM_clust_vecs, optimal_ordering=True, metric='correlation', method='complete')

idx = hierarchy.leaves_list(link)+1
corr_im = np.corrcoef(IM_clust_vecs)

g = sns.clustermap(
    corr_im,row_linkage=link, 
    col_linkage=link, 
    vmin=0.3, vmax=0.7,
    cmap = 'magma',
    tree_kws=dict(linewidths=5, colors=(0.2, 0.2, 0.2)),
    )   


new_labels = []
for i, tick_label in enumerate(g.ax_heatmap.axes.get_yticklabels()):
    tick_text = tick_label.get_text()
    tick_label.set_color(clust_colors[int(tick_text)])
    new_labels.append(str(int(tick_text)+1) + ': ' + cluster_names[int(tick_text)])

for i, tick_label in enumerate(g.ax_heatmap.axes.get_xticklabels()):
    tick_text = tick_label.get_text()
    tick_label.set_color(clust_colors[int(tick_text)])
   

g.ax_heatmap.axes.set_yticklabels(new_labels, rotation=360, fontsize=35)
g.ax_heatmap.axes.set_xticklabels(new_labels, rotation=90, fontsize=35)
plt.savefig(os.path.join(analysis_out, '_clustered_correlogram_downsmapledVolumes.svg'))

#%% Figure 7, screen zbrain databases for similar datasets structures:

zbrain_anat = r'/media/BigBoy/ciqle/ref_brains/AnatomyDatabases/AnatomyLabelDatabase.hdf5'
zbb_anat = r'/media/BigBoy/ciqle/ref_brains/AnatomyDatabases/ZBBDatabase.hdf5'
mapzebrain_dir = r'/media/BigBoy/ciqle/ref_brains/AnatomyDatabases/mapzebrain_20220504_inzbrain/'

mapzebrain_tiffs = glob.glob(mapzebrain_dir+'*.tif')
f_h5 = h5py.File(zbrain_anat, 'r')
f_h52 =  h5py.File(zbb_anat, 'r')
anat_names = [*f_h5.keys()]
anat_names_2 = [*f_h52.keys()]
n_labels = len(anat_names) + len(anat_names_2) + len(mapzebrain_tiffs)
recalculate_correlations = False 
corr_database_file = os.path.join(analysis_out, 'corr_to_databases.npy')
if recalculate_correlations:
    corr_scores = np.zeros((n_clusters, n_labels))
    k = 0
    for i in tqdm(range(n_labels)):
        if i < len(anat_names):
            anat_name = anat_names[i]
            anat_dset = np.moveaxis(f_h5[anat_name], 2,1)
        elif i < len(anat_names) + len(anat_names_2):
            anat_name = anat_names_2[i-len(anat_names)]
            anat_dset = np.moveaxis(f_h52[anat_name], 2,1)
        else:
            anat_name = mapzebrain_tiffs[i-len(anat_names) - len(anat_names_2)] 
            anat_dset = imread(anat_name)
            anat_name = os.path.split(anat_name)[1]
        
        # anat_dset = np.moveaxis(f_h5['Gad1b-GFP_6dpf_MeanImageOf10Fish'], 2,1)
        z_blocksize =5
        xy_blocksize = int(z_blocksize*2/0.798)

        anat_dset= anat_dset[zlims[0]:zlims[1], ylims[0]:ylims[1], xlims[0]:xlims[1]].astype(float)
        print(anat_name)
        proj_out = os.path.join(analysis_out, 'anatomy_projections')
        im_proj_z = np.max(anat_dset[:,:, :], axis=0)
        im_proj_x = zoom(np.max(anat_dset[:,:, :], axis=2).T, [1, 2/0.798])

        im_proj = np.hstack((im_proj_z, im_proj_x))

        im_proj[outline > 0.01] = np.max(im_proj)

        anat_dset_downsample = block_reduce(anat_dset, block_size=(z_blocksize,xy_blocksize,xy_blocksize))
        

        for i in range(n_clusters):
            IM_centroid_downsample = IM_clust_centroid[zlims[0]:zlims[1], ylims[0]:ylims[1], xlims[0]:xlims[1], i].astype(float)
            IM_centroid_downsample = block_reduce(IM_centroid_downsample, block_size=(z_blocksize,xy_blocksize,xy_blocksize))

            corr, p = pearsonr(IM_centroid_downsample.flatten(), anat_dset_downsample.flatten())

            corr_scores[i,k] = corr
        k+=1

    
    np.save(corr_database_file, corr_scores)

corr_scores = np.load(corr_database_file)
#%
max_corr = np.max(corr_scores, axis=0)
potential_hits = np.where(abs(max_corr) >= 0.15)[0]
anat_names_total = anat_names+anat_names_2+mapzebrain_tiffs
anat_names_hits = []
for i in potential_hits:
    name = anat_names_total[i].replace('ZBB_', '').replace('-', ':').replace('T_AVG_', '').replace('.tif', '')
    name = name.replace(mapzebrain_dir, '')
    if name.find('_6dpf') > 0:
        name = name[:name.find('_6dpf')]
    anat_names_hits.append(name)
#%
scores_to_plot = corr_scores[:,potential_hits].T
scores_to_plot = zscore(scores_to_plot, axis=1)
with plt.rc_context({'font.size':100, 'font.weight':'heavy'}):
    #plt.figure(figsize=(50,75))
    g = sns.clustermap(   
        scores_to_plot , 
        yticklabels = anat_names_hits,
        xticklabels = cluster_names,
        col_cluster=False,
        row_cluster=True,
        metric='correlation',
        method='complete',
        vmin=-3.5,
        vmax=3.5,
        cmap=sns.diverging_palette(300, 120, s=100, l=50, sep=30, as_cmap=True, center="dark"),
        figsize = (70,150),
        )


    for i, tick_label in enumerate(g.ax_heatmap.axes.get_xticklabels()):
        tick_text = tick_label.get_text()
        tick_label.set_color(clust_colors[i])
        tick_label.set_fontsize(150)

    plt.savefig(os.path.join(analysis_out, 'cluster_correlation_zbrain.svg'))
    plt.savefig(os.path.join(analysis_out, 'cluster_correlation_zbrain.png'))
    plt.show()


#%
top_hits = np.argmax(scores_to_plot, axis=0)
for i in range(len(cluster_names)):
    print('......................')
    print(cluster_names[i])
    print(anat_names_hits[top_hits[i]])



#%% Figure 7, Analyze gad1b:dsred;HuC:GCaMP data
gad_thresh = 0.25

analysis_out = r'/media/BigBoy/Owen/2020_HabScreenPaper/data/GCaMPAnalysis/gad1b'
analysis_in = r'/media/BigBoy/Owen/2020_HabScreenPaper/data/GCaMPAnalysis'
parent_dirs = [
    os.path.realpath(r'/media/BigBoy/ciqle/MartinPhotonData/20180319-20180323_DarkFlashHabituation'),
    os.path.realpath(r'/media/BigBoy/ciqle/MartinPhotonData/20190513-17_DarkFalshHabituation')
]
im_files = []
for parent_dir in parent_dirs:
    im_files += glob.glob(parent_dir + r'/*/*ad1*_Z_0_0.tif')

dir_martinphoton = r'/mnt/md0/suite2p_output_martinphoton'
# folders = glob.glob(dir_martinphoton+r'/*')
im_files.pop(1)
# im_files = im_files[:-2]
print(im_files)
#%
gad1b_vals_all = []
hits_clusters_all = []
hits_clusters_IDs_all = []
for im_file in im_files:

    im_folder, im_name = os.path.split(im_file)
    print(im_name)
    im_basename = im_name[:im_name.find('_Z_0_0.tif')]

    stack_dir = os.path.join(im_folder,im_basename )
    gcamp_file = tifffile.imread(glob.glob(stack_dir + r'/*_0.tif')).astype('float')
    gad1b_file = tifffile.imread(glob.glob(stack_dir + r'/*_1.tif')).astype('float')
    #%
    Zs, height, width = gcamp_file.shape
    n_sum = 10
    Zs = int(Zs/n_sum)
    gcamp_stack = np.zeros((Zs,height,width))
    gad1b_stack = np.zeros((Zs,height,width))

    for i in range(Zs):
        gcamp_stack[i,:,:] = np.sum(gcamp_file[i*n_sum:i*n_sum+n_sum, :, :], axis=0)
        gad1b_stack[i,:,:] = np.median(gad1b_file[i*n_sum:i*n_sum+n_sum, :, :], axis=0)

    gcamp_stack = gcamp_stack/np.max(gcamp_stack)
    gad1b_stack = gad1b_stack/np.max(gad1b_stack)

    #%
    # z = 30
    # # plt.figure(figsize=(20,20))
    # # plt.imshow(np.stack((gad1b_stack[z,:,:],gcamp_stack[z,:,:], gad1b_stack[z,:,:]), axis=-1)*3)

    #%

    suite2p_dir = glob.glob(dir_martinphoton+'/*'+im_basename+'*')[0]
    ops = np.load(suite2p_dir+'/suite2p/plane0/ops.npy', allow_pickle=True).item()
    F = np.load(suite2p_dir+'/suite2p/plane0/F.npy', allow_pickle=True)
    iscell = np.load(suite2p_dir+'/suite2p/plane0/iscell.npy', allow_pickle=True)

    IM = ops['meanImg']
    #plt.imshow(IM)

    shifts = np.zeros((Zs, 2)) # order will be y, x, z
    errors = np.zeros(Zs)
    phasediff = np.zeros(Zs)
    anat_offset = np.zeros(3)
    for slice in range(Zs):
        shifts[slice,:], errors[slice], phasediff[slice] = phase_cross_correlation(gcamp_stack[slice, :,:],IM,  normalization=None)

    #%
    z_coord = np.nanargmin(errors)
    anat_offset[:2] = shifts[z_coord, :]
    anat_offset[2] = z_coord

    print(anat_offset)
    #ops['anat_stack_offsets'] = anat_offset

    plt.figure(figsize=(20,20))
    plt.imshow(np.stack((gad1b_stack[z_coord,:,:],gcamp_stack[z_coord,:,:], gad1b_stack[z_coord,:,:]), axis=-1)*3)
    plt.show()
    #%
    roi_stats = np.load(suite2p_dir+'/suite2p/plane0/stat.npy', allow_pickle=True)
    n_rois = len(roi_stats)
    gad1b_vals = np.zeros(n_rois)
    gcamp_vals = np.zeros(n_rois)
    gad1_enrich = np.zeros(n_rois)
    #%
    
    im_cents = np.copy(gcamp_stack[z_coord,:,:])
    im_gad1b = np.copy(gad1b_stack[z_coord,:,:])
    im_gcamp = np.copy(gcamp_stack[z_coord,:,:])
    im_gadinrois = np.zeros(im_gad1b.shape)
    for i in range(n_rois):
        # shift x/y appropriately
        x_coord = roi_stats[i]['med'][1] + anat_offset[1]
        y_coord = roi_stats[i]['med'][0] + anat_offset[0]
        x_coords = roi_stats[i]['xpix'] + anat_offset[1]
        y_coords = roi_stats[i]['ypix'] + anat_offset[0]
        x_coords[x_coords>width-1] = width-1
        y_coords[y_coords>height-1] = height-1
        gad1b_vals[i] = np.mean(im_gad1b[y_coords.astype(int), x_coords.astype(int)])
        gcamp_vals[i] = np.mean(im_gcamp[y_coords.astype(int), x_coords.astype(int)])
        gad1_enrich[i] = gad1b_vals[i]/(gad1b_vals[i] + gcamp_vals[i])
        if iscell[i, 1]> 0.3 and x_coord < width and y_coord < height:
            im_cents[y_coords.astype(int), x_coords.astype(int)] = np.random.random(1)*0.1
            im_gadinrois[y_coords.astype(int), x_coords.astype(int)] = gad1_enrich[i]
    gad1b_vals_all.append(gad1_enrich)
    plt.figure(figsize=(20,20))
    plt.imshow(3*np.stack((im_gad1b/np.max(im_gad1b), im_gcamp/np.max(im_gcamp),im_gad1b/np.max(im_gad1b)), axis=-1))
    plt.axis('off')
    plt.savefig(os.path.join(analysis_out, im_name.replace('.tif', '') + '_GCaMPandGad1b.svg'))
   
    # plt.figure(figsize=(20,20))
    # plt.imshow(np.stack((im_gad1b*5, im_cents*3, shift(IM/4, anat_offset[:2])), axis=-1))
    # #plt.imshow(im_cents)
    plt.show()
    plt.figure(figsize=(20,20))
    plt.imshow(im_gadinrois, cmap='magma')
    plt.colorbar(shrink=0.3, aspect=5)
    plt.axis('off')
    plt.savefig(os.path.join(analysis_out, im_name.replace('.tif', '') + '_gad1binregions.svg'))
    plt.show()
    # im_gadinrois_abovethresh = np.copy(im_gadinrois)
    # im_gadinrois_abovethresh[im_gadinrois_abovethresh<gad_thresh] = 0
    # plt.figure(figsize=(20,20))
    # plt.imshow(im_gadinrois_abovethresh, cmap='inferno')
    # plt.show()
    #%
    # gad1b_vals = gad1b_vals/np.mean(gad1b_vals)
    # gcamp_vals =  gcamp_vals/np.mean(gcamp_vals)
    #%
    gad1b_incells = gad1b_vals[iscell[:, 1]> 0.3]
    plt.hist(gad1b_incells, np.arange(0,0.5,0.01), alpha=0.5, label='gad1b signal in rois')
    gcamp_incells = gcamp_vals[iscell[:, 1]> 0.3]
    plt.hist(gcamp_incells, np.arange(0,0.5,0.01), alpha=0.5, label='gcamp signal in rois')
    plt.legend()
    plt.show()


    plt.hist(gad1_enrich, np.arange(0,1,0.01))
    plt.vlines(gad_thresh, ymin=0, ymax=200, colors = 'k')
    plt.show()
    #%
    #%
    n_frames = F.shape[1]
    stim_file = im_file.replace('.tif', '_stim.txt')
    stim = np.loadtxt(stim_file)

    stim_frames = stim[:,0]
    #stim_frames = stim_frames[:np.where(stim_frames == n_frames-2)[0][-1]]
    stim_df = 5-stim[:,2]
    #plt.plot(stim_df)

    stim_df_frames = np.zeros(n_frames)
    for i in range(n_frames):
        inds = stim_df[stim_frames==i]
        if len(inds) > 0:
            stim_df_frames[i] = np.max(stim_df[stim_frames==i])
    #%
    df_start_inds = np.where(np.diff(stim_df_frames) > 4)[0]

    frame_rate = 60/np.mean(np.diff(df_start_inds))


    # %
    os.chdir('/mnt/md0/suite2p_output/')

    f_h5 = h5py.File('compiled_imaging_data.h5', 'r')
    df_start_inds_bruker = f_h5['df_start_inds'][()]
    stim_df_bruker = f_h5['stim_df'][()]

    #%
    flash_frames_bruker = df_start_inds_bruker[-1] - df_start_inds_bruker[0]
    flash_frames_mphot = df_start_inds[-1] - df_start_inds[0]

    ration_bruk_to_mphot = flash_frames_bruker/flash_frames_mphot

    from scipy import signal, interpolate
    x_orig = np.arange(n_frames)
    # plt.plot(x_orig, stim_df_frames)
    interp = interpolate.interp1d(x_orig, stim_df_frames)
    x_upsamp = np.arange(n_frames*ration_bruk_to_mphot)
    x_upsamp = (n_frames-1) * x_upsamp/np.max(x_upsamp)
    stim_df_upsamp = interp(x_upsamp)
    # plt.plot(x_upsamp, stim_df_upsamp)
    #%
    df_start_inds_upsamp = np.where(np.diff(stim_df_upsamp) > 2.4)[0]
    # print(len(df_start_inds_upsamp))
    # print(df_start_inds_upsamp)
    st_frame = df_start_inds_upsamp[0] - df_start_inds_bruker[0] 
    end_frame = st_frame + len(stim_df_bruker) 
    stim_df_upsamp = stim_df_upsamp[st_frame:end_frame]
    #%


    # plt.figure(figsize=(10,10))
    # plt.plot(stim_df_upsamp[df_start_inds_bruker[-0]-5:df_start_inds_bruker[1]])
    # plt.plot(4.75*stim_df_bruker[df_start_inds_bruker[-0]-5:df_start_inds_bruker[1]])

    #%
    F_norm = zscore(F, axis=1)
    interp = interpolate.interp1d(x_orig, F_norm)
    F_upsamp = interp(x_upsamp)[:,st_frame:end_frame]
    #%
    # for i in range(5,3000, 200):
    #     plt.plot(F_upsamp[i,:])

    #%
    save_clust_name = os.path.join(analysis_in, 'cluster_assignments.npz')
    data = np.load(save_clust_name)
    mean_vecs_reorder = data['mean_vecs_reorder']
    # plt.plot(mean_vecs_reorder.T)

    #%
    from sklearn.linear_model import LinearRegression, SGDRegressor
    start_analyze_frame = 200 
    n_clusters = mean_vecs_reorder.shape[0]
    n_rois = F_upsamp.shape[0]
    scores_toClust = np.zeros(n_rois)
    coeffs_toClust = np.zeros((n_rois, n_clusters))
    X = mean_vecs_reorder[:,start_analyze_frame:].T
    F_upsamp[np.isnan(F_upsamp)] = 0
    for roi in range(n_rois):

        y = F_upsamp[roi,start_analyze_frame:]
        reg = LinearRegression(positive=True).fit(X, y)

        scores_toClust[roi]=reg.score(X,y)

        coeffs_toClust[roi, :]=reg.coef_
    
    corr_to_clusts = pearsonr_2Dnumb(mean_vecs_reorder[:,start_analyze_frame:], F_upsamp[:,start_analyze_frame:])

    hits_clusters = np.where((np.max(corr_to_clusts, axis=0) > 0.3))[0]

    hits_clusters_IDs = np.argmax(corr_to_clusts[:, hits_clusters], axis=0)
    gad1b_vals_clust = np.zeros(n_clusters)
    gad1b_prop = np.zeros(n_clusters)
    hits_clusters_all.append(hits_clusters)
    hits_clusters_IDs_all.append(hits_clusters_IDs)
    for i in range(n_clusters):

       
        gad1b_vals_clust[i] = np.mean(gad1_enrich[hits_clusters[hits_clusters_IDs==i]])
        gad1b_prop[i] = np.mean(gad1_enrich[hits_clusters[hits_clusters_IDs==i]]> gad_thresh)
        print(gad1b_vals_clust[i])

    plt.show()
    plt.plot(gad1b_vals_clust)
    plt.plot(gad1b_prop)
    plt.title(im_name)
    plt.show()


clust_colors = get_colors(n_clusters, pastel_factor=0, rng=1,n_attempts=1000)
clust_colors[9] = (0.8, 0.8, 0)
nboots = 5000
gad_vals_in_cluster = np.zeros((n_clusters, nboots))
gad_prop_in_cluster =  np.zeros((n_clusters, nboots))
gad1b_vals_allcells = []
total_cells = 0
for cluster in range(n_clusters):
    gad1b_vals = []
    for fish in range(len(hits_clusters_all)):
        ids = hits_clusters_all[fish][hits_clusters_IDs_all[fish] == cluster]
        gad1b_vals.append(gad1b_vals_all[fish][ids])
    gad1b_vals = np.hstack(np.array(gad1b_vals))
    gad1b_vals = gad1b_vals[~np.isnan(gad1b_vals)]
    n_cells = len(gad1b_vals)
    total_cells += n_cells
    print('cluster ' + str(cluster+1))
    print(str(n_cells)+ ' cells detected')
    for boot in range(nboots):
        ids = np.random.randint(0, high=n_cells, size=n_cells)
        gad_vals_in_cluster[cluster, boot] = np.mean(gad1b_vals[ids])
        gad_prop_in_cluster[cluster, boot] = np.mean(gad1b_vals[ids]>gad_thresh)
    gad1b_vals_allcells.append(gad1b_vals)
plt.plot(np.arange(n_clusters) + 1, np.mean(gad_vals_in_cluster, axis=1), 'x--')
plt.plot(np.arange(n_clusters) + 1, np.mean(gad_prop_in_cluster, axis=1), 'x--')
plt.title(gad_thresh)
plt.show()


gad1b_vals_allcellsvec = np.zeros(0)
for vals in gad1b_vals_allcells:
    gad1b_vals_allcellsvec = np.hstack((gad1b_vals_allcellsvec, vals))


expected_prop = np.mean(gad1b_vals_allcellsvec>gad_thresh)

with plt.rc_context({'font.size':30}):
    fig = plt.figure(figsize=(10,8))
    plt.plot(np.mean(gad_prop_in_cluster, axis=1), 'kp')
    plt.hlines(expected_prop, xmin=-1, xmax=12, colors = 'k', linestyles='dashed', alpha=0.7)
    sns.violinplot(data=gad_prop_in_cluster.T, palette =clust_colors, width=1, inner=None)
    plt.xticks(np.arange(n_clusters), labels=np.arange(n_clusters)+1)

    plt.xlim((-0.5, 11.5))
    
    plt.ylabel('proportion \nGad1b-positive')
    plt.xticks(np.arange(n_clusters), labels = cluster_names, rotation=90, fontweight='bold')  
    xtick_labels = plt.gca().get_xticklabels()
    
    for i, tick_label in enumerate(xtick_labels):
        tick_label.set_color(clust_colors[i])
    plt.savefig(os.path.join(analysis_out, 'violin_plots.svg'))
    plt.show()


print('.............. chi square test w bonferonni correction ...............')
print('...................... "significant" differences .....................')
print('gad threshold = ' + str(gad_thresh))
for cluster in range(n_clusters):
    obs = (len(np.where(gad1b_vals_allcells[cluster] > gad_thresh)[0]), len(np.where(gad1b_vals_allcells[cluster]  <= gad_thresh)[0]))
    exp = len(gad1b_vals_allcells[cluster]) * expected_prop, len(gad1b_vals_allcells[cluster]) * (1-expected_prop)
    chisq, p = chisquare(obs, f_exp=exp)
    p = p*n_clusters # mult comparisons correction
    if p<0.05:
        print('cluster '+ str(cluster+1)+ ': '+cluster_names[cluster] + '... p value = ' + str(p))

print('total cells = ' + str(total_cells))


#%% Figure 8, effects of drugs on functional clusters

# use behavioural data from motion artifact to look at habituation, Figure 8
n_fish = np.max(fish_data[:,0])+1
fish_exp_types  = np.zeros(n_fish, dtype=np.uint8)
fish_types_all = fish_data[hits_clusters,1]
for fish in range(n_fish):
    fish_inds = fish_data[hits_clusters,0] == fish

    fish_exp_types[fish] = fish_types_all[fish_inds][0]


df_responses = np.zeros((n_fish, 60))
for i in range(n_fish):
    for j in range(60):
        df_responses[i,j] = np.max(motor_power_fish_flash[i, df_start_inds[j]:df_start_inds[j]+int(20*frame_rate)])

df_responses[df_responses > 0] = 1
# plt.figure(figsize=(20,20))
# plt.yticks(np.arange(n_fish), labels=fish_names)
# plt.imshow(df_responses)
# plt.show()

first_half = np.mean(df_responses[:,:30], axis=1)
second_half = np.mean(df_responses[:,30:], axis=1)

perc_hab = 100*(1- second_half/(0.5*(first_half+second_half)))

ignore_fish = np.sum(df_responses[:,:], axis=1) < 0
perc_hab[ignore_fish] = np.nan
fish_groups = []
fish_group_types = ['DMSO', 'Picrotoxinin','Melatonin']
for i in range(len(perc_hab)):
    if fish_exp_types[i]==1:
        fish_groups.append('DMSO')
    if fish_exp_types[i]==2:
        fish_groups.append('Picrotoxinin')
    if fish_exp_types[i]==3:
        fish_groups.append('Melatonin')
#%
hab_data = pd.DataFrame({
    'data': perc_hab,
    'groups': fish_groups
}
)
hab_data = hab_data.dropna()

plt.figure(figsize=(7,7))
ax = sns.stripplot(data=hab_data, x = 'groups', y='data', alpha=0.75, size=15, jitter=0.1,palette = color_fish)
sns.pointplot(data=hab_data, x = 'groups', y='data', estimator=np.nanmedian, color='k', join=False, markers="_", scale=4, n_boots=500, ci=None)        
#
box_pairs = [
    [fish_group_types[0], fish_group_types[1]],
    [fish_group_types[0], fish_group_types[2]],
    [fish_group_types[1], fish_group_types[2]]
]

h,p_krusk= kruskal(
    perc_hab[fish_exp_types==1], 
    perc_hab[fish_exp_types==2], 
    perc_hab[fish_exp_types==3],
    nan_policy='omit')
if p_krusk < 0.05:
    test_results = add_stat_annotation(ax, data=hab_data, y ='data', x='groups', order=fish_group_types,
                                                    
                                                    box_pairs=box_pairs,


                                                    test='Mann-Whitney', text_format='simple',

                                                    verbose=2,
                                                    )

plt.xticks(np.arange(3), fish_group_types, rotation=0)
plt.ylabel(r'% Habituation')
plt.xlabel('')
plt.savefig(os.path.join(analysis_out, 'perc_hab_drugs.svg'))

# plot the different analyses for clusters separately for each treatment. 
for i in range(3):
    print(fish_types[i])
    type_hits = fish_data[hits_clusters, 1] == i+1
    hits_clusters_type = hits_clusters[type_hits]
    hits_clusters_IDs_type = clust_IDs_reorder[type_hits]
    mean_vecs_reorder_type, clust_IDs_reorder_type = plot_cluster_means(F_norm, hits_clusters_type, hits_clusters_IDs_type, median=True, plot_heat=True, cluster_heat=False, save_name='FishType_'+ fish_types[i], re_order=False, ymax=5.5)
    IM_clust_type, IM_clust_centroid_type = plot_cluster_rois(hits_clusters_type, hits_clusters_IDs_type, save_name='FishType_'+ fish_types[i])  
    zbrain_analysis_centroids(regions_of_interset, IM_clust_centroid_type, save_name='FishType_'+ fish_types[i])

plot_cluster_proportions(hits_clusters, clust_IDs_reorder, use_boots=True, n_boots=5000, save_name='rescan_clusters_all')

# compare cluster mean vectors

n_clusters=mean_vecs_reorder.shape[0]

fig, ax = plt.subplots(nrows=n_clusters, ncols=1, figsize=(20,80))

mean_vecs_fishtype = np.zeros((n_clusters,mean_vecs_reorder.shape[1], 3))

for clust in range(n_clusters):
    for fishtype in range(3):
        hits = hits_clusters[clust_IDs_reorder == clust]
        hits = hits[fish_data[hits, 1] == fishtype+1]
        n_fish = len(np.unique(fish_data[fish_data[:,1] == fishtype+1, :]))
        y = np.mean(F_norm[hits, :], axis=0)
        std = np.std(F_norm[hits, :], axis=0)/np.sqrt(n_fish-1)
        mean_vecs_fishtype[clust, :, fishtype] = y
        x = np.arange(len((y)))/(ops[fish_name]['fs']*60)
        x = x - 5 
        ax[clust].fill_between(x, y+std, y-std, color=color_fish[fishtype], alpha=0.33)
        ax[clust].set_title('cluster ' + str(clust+1))
        ax[clust].set_xlim([-3, 61])
        #ax[clust].set_xlim([59, 60])
        #ax[clust].set_ylim((-1,6))
    ax[clust].set_ylabel('mean F (z-score)')
plt.xlabel('time (min)')
ax[0].set_legend(['DMSO', 'Picrotoxinin', 'Melatonin'])



# %% calculated number of rois per fish

n_fish = fish_data[-1,0]+1
n_rois_fish = np.zeros(n_fish)

for i in range(n_fish):
    n_rois = np.sum(fish_data[:,0]==i)
    #print(n_rois)
    n_rois_fish[i] = n_rois
print('mean rois = ')
print(np.mean(n_rois_fish))
print('+/-')
print(np.std(n_rois_fish))

