#%%
import numpy as np
from scipy.signal import savgol_filter, medfilt
import matplotlib.pyplot as plt
import pandas as pd
import os, glob
from scipy.stats import zscore, pearsonr, spearmanr
lag_plot = 50
analysis_out = r'/media/BigBoy/Owen/2020_HabScreenPaper/data/GCaMPAnalysis/motion_analysis/'

tail_power_thresh = 0.1
folders = [
    r'/mnt/md0/suite2p_output/20220307_HuCGCaMP7f_5d_DMSO_fish-002',
    r'/mnt/md0/suite2p_output/20220302_HuCGCaMP7f_5d_DMSO_fish-003',
    r'/mnt/md0/suite2p_output/20220307_HuCGCaMP7f_5d_melatonin_fish-003',
    r'/mnt/md0/suite2p_output/20220314_HuCGCaMP7f_5d_DMSO_fish-003',
    r'/mnt/md0/suite2p_output/20220315_HuCGCaMP7f_5d_DMSO_fish-002',
    r'/mnt/md0/suite2p_output/20220314_HuCGCaMP7f_5d_melatonin_fish-002',
]
stats = np.zeros(len(folders))
ps = np.zeros(len(folders))
lags_all = np.zeros((len(folders), lag_plot*2))
for fish, folder in enumerate(folders):
    os.chdir(folder)
    fish_name = os.path.split(folder)[1]
    #%
    ops = np.load(os.path.join(folder, 'suite2p/plane5/ops.npy'), allow_pickle=True).item()

    #
    stim_df = np.load(os.path.join(folder, 'stimvec_mean.npy'))

    #%
    time_stamps_imaging = np.load(os.path.join(folder, 'time_stamps.npy'))
    time_stamps_imaging = np.mean(time_stamps_imaging, axis=0)/1000 # time stamp in the middle of the stack, in msec
    df_start_inds = np.where(np.diff(stim_df)>0.1)[0]
    df_start_inds = np.delete(df_start_inds, np.where(np.diff(df_start_inds) == 1)[0]+1)
    #%

    def rolling_window(a, window):
        pad = np.ones(len(a.shape), dtype=np.int32)
        pad[-1] = window-1
        pad = list(zip(pad, np.zeros(len(a.shape), dtype=np.int32)))
        a = np.pad(a, pad,mode='reflect')
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    #%

    motor_sig = ops['corrXY']
    motor_sig[ops['badframes']] = 0
    motor_pow = savgol_filter(np.std(rolling_window(motor_sig, 3), -1), 15, 2)
    motor_pow = motor_pow - np.median(motor_pow)
    motor_pow = motor_pow/np.max(motor_pow)
    # plt.plot(motor_pow)
    # %

    coords = pd.read_csv(glob.glob('*_coords.txt')[0], sep=',',header=None).values.astype(float)
    tstamps = pd.read_csv(glob.glob('*_tstamps.txt')[0], sep=',',header=None).values.astype(float)

    coords_x = coords[::2,:] 
    coords_y = coords[1::2,:]

    #%
    df_readings = tstamps[1::2].flatten()
    tstamps = tstamps[::2].flatten()

    # coords_x = coords_x[-len(tstamps):, :]# why are they different sizes?? something wrong here. Assume the end of the coordinates file are correct? 
    # coords_y = coords_y[-len(tstamps):, :]
    # coords_x.shape
    # tstamps.shape
    #%

    df_start = np.where(np.diff(df_readings.flatten())==-1)[0][0]
    t_df_start = tstamps[df_start]

    # %

    tstamp_imaging_df = time_stamps_imaging[df_start_inds[0]]
    tstamp_imaging_end = time_stamps_imaging[-1]

    # %
    t_imaging_start = t_df_start - tstamp_imaging_df
    ind_imagin_start = np.where(tstamps >= t_imaging_start-1.5)[0][0]

    t_imaging_end = t_imaging_start + tstamp_imaging_end

    ind_imaging_end = np.where(tstamps >= t_imaging_end)[0][0]
    #plt.plot(df_readings[ind_imagin_start:ind_imaging_end])
    coords_x = coords_x[ind_imagin_start:ind_imaging_end, :]
    coords_y = coords_y[ind_imagin_start:ind_imaging_end, :]
    # %
    keep_points = 31
    angles = np.unwrap(np.arctan2(np.diff(coords_x[:, :keep_points], axis=1), np.diff(coords_y[:,:keep_points], axis=1)))

    diffangles = np.diff(angles, axis=1)
    diffangles[np.isnan(diffangles)] = 0
    #diffangles = savgol_filter(diffangles, 11, 2, axis=0)
    # diffangles[abs(diffangles) > 1] = 0 # max of 1 radian per segment
    bend_amps = savgol_filter(np.nansum(diffangles, axis=1), 5, 2)
    tail_power = np.std(rolling_window(bend_amps, int(len(bend_amps)/len(motor_pow))), -1)
    tail_power = tail_power -np.median(tail_power)
 
    # inds = np.arange(15000, 16000)+10000*4

    # plt.plot(motor_power[inds])
    # plt.show()
    #%
    from scipy import signal, interpolate

    interp = interpolate.interp1d(tstamps[ind_imagin_start:ind_imaging_end], tail_power)
    tail_power_interp = interp(np.arange(tstamps[ind_imagin_start], tstamps[ind_imaging_end], 1/5))
    tail_power_frames = signal.resample(tail_power_interp, len(motor_pow))
    
    interp_bendAmp = interpolate.interp1d(tstamps[ind_imagin_start:ind_imaging_end], bend_amps)
    bend_amps_interp = interp_bendAmp(np.arange(tstamps[ind_imagin_start], tstamps[ind_imaging_end], 1/5))
    bend_amps_frames = signal.resample(bend_amps_interp, len(motor_pow))
    
    
    
    tail_power_frames[tail_power_frames<tail_power_thresh] = 0
    motor_pow[motor_pow<tail_power_thresh] = 0
    with plt.rc_context({'font.size':20}):
        max_y = np.max([tail_power_frames, motor_pow])
        fig, ax = plt.subplots(nrows=5, figsize=(13,20))
        
        for k,i in enumerate(range(0, 5)):
            inds = np.arange(1500) + 1500*i
            ax[k].plot(inds, tail_power_frames[inds])
            ax[k].plot(inds, motor_pow[inds])
            ax[k].set_ylabel('inferred\n movement')
            ax[k].set_ylim((0,max_y))
            #ax[k].set_ylim((-0.1, 1))
        ax[0].legend(['tail tracking', 'suite2p image\nmotion artifact'])
        
        plt.xlabel('time (frames)')
        ax[0].set_title(fish_name)
        plt.savefig(os.path.join(analysis_out, fish_name+'_tailVsImagePower.png'))
        plt.savefig(os.path.join(analysis_out, fish_name+'_tailVsImagePower.svg'))
        plt.show()
    #%
    xcorr = np.correlate(tail_power_frames,motor_pow, mode='same')
    [stats[fish], ps[fish]] = pearsonr(tail_power_frames,motor_pow)
    lags = signal.correlation_lags(len(tail_power_frames),len(motor_pow), mode='same')
    
    lag_0 = np.where(lags==0)[0][0]
    #%
    lags_all[fish,:] = xcorr[lag_0-lag_plot:lag_0+lag_plot]
    plt.plot(lags[lag_0-lag_plot:lag_0+lag_plot], xcorr[lag_0-lag_plot:lag_0+lag_plot])
    plt.vlines(0, ymin=0, ymax=np.max(xcorr), color='k')
    plt.savefig(os.path.join(analysis_out, fish_name+'_xcorr.png'))
    plt.savefig(os.path.join(analysis_out, fish_name+'_xcorr.svg'))
    plt.show()
    # lags = signal.correlation_lags(motor_power_frames.size, motor_pow.size, mode="full")
    print(stats[fish])
print(stats)
print(ps)


#%%
x = lags[lag_0-lag_plot:lag_0+lag_plot]/1.98
mean_xcorr = np.mean(lags_all, axis=0)
sterr_xcorr = np.std(lags_all, axis=0)/np.sqrt(len(folders))
with plt.rc_context({'font.size':20}):
    plt.figure(figsize=(7,5))
    plt.plot(x, mean_xcorr)
    plt.fill_between(x, mean_xcorr+sterr_xcorr, mean_xcorr-sterr_xcorr)
    plt.xlabel('lag (s)')
    plt.ylabel('cross-correlation')
    plt.title('mean correlation = ' + str(np.around(np.mean(stats), decimals=3)))
    plt.savefig(os.path.join(analysis_out, 'cross_correlation.png'))
    plt.savefig(os.path.join(analysis_out, 'cross_correlation.svg'))
#%%


