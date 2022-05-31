#%%


from scipy.signal.signaltools import medfilt2d
from tkinter import *
import os
import numpy as np
import glob
import warnings
warnings.simplefilter(action="default")
import gspread
import pandas as pd
import FishTrack
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm.notebook import tqdm
from scipy.signal import savgol_filter, find_peaks, medfilt
import pickle
import natsort

current_dir = os.path.dirname(__file__)

sys.path.append(current_dir)
sys.path.append(os.path.realpath(current_dir + r'/ExtraFunctions/glasbey-master/'))
from glasbey import Glasbey
#%
def ffill_cols(a, startfillval=0):
    mask = np.isnan(a)
    tmp = a[0].copy()
    a[0][mask[0]] = startfillval
    mask[0] = False
    idx = np.where(~mask,np.arange(mask.shape[0])[:,None],0)
    out = np.take_along_axis(a,np.maximum.accumulate(idx,axis=0),axis=0)
    a[0] = tmp
    return out

def subtract_angles(lhs, rhs):
    import math
    """Return the signed difference between angles lhs and rhs

    Return ``(lhs - rhs)``, the value will be within ``[-math.pi, math.pi)``.
    Both ``lhs`` and ``rhs`` may either be zero-based (within
    ``[0, 2*math.pi]``), or ``-pi``-based (within ``[-math.pi, math.pi]``).
    """

    return math.fmod((lhs - rhs) + math.pi * 3, 2 * math.pi) - math.pi

# single directory
# dirs = [r'Z:\GranatoLabData\NoIssues\170412']

# all subdirectories for screen data
dirs = sorted(glob.glob(r'Z:/GranatoLabData/NoIssues/*/'))
#dirs = sorted(glob.glob(r'/media/BigBoy/Owen/GranatoLabData/NoIssues/*/'))
# root_dir = r'/media/BigBoy/MultiTracker/'
# dirs = [r'/media/BigBoy/MultiTracker/20211130_130548/']
#%
plot_burst = False

plot_tracking_results = False

stimFrame = 1; # the frame that the stimulus was delivered in
nStimInBlocks = 60; # the number of stimuli in a block
ContGroup = 1;

RestMin = 62; # the number of minutes of rest, this is used if the timestamps on the tracked files are missing
TestRestMin = 62*4; # length of rest between training and re-test, this is used if the timestamps on the tracked files are missing

TrainLast = 240;
TapIndStart = 270; # the indexes of where the taps happen (0 indexed based on the tracked file naming, not the sequence of indexes, in case some tracked data is missing)
TapIndEnd = 300;

OMRStimStart = 300;   # the indexes of where the OMR happens and we will analyze the resulting changes in heading
OMRStimEnd = 359;

RetestStart = 360; # the indexes where the retest flashes happen. (0 indexed based on the tracked file naming, not the sequence of indexes, in case some tracked data is missing)
RetestEnd = 420;

FrameRate = 560; # speed camera is recording frames, frame rate of the burst recorded movies

OBendThresh = 3; # radians of curvature to call an O-bend
CBendThresh = 1; # radians of curvature to call an C-bend

sav_ord = 3; # parameters for the sgolayfilt of curvature
sav_sz = 15;
SpeedSmooth = 5; # the kernel of the medfilt1 filter for the speed

AreaMin = 100; # min size of an acceptable fish
AreaMax = 600; # max size of an acceptable fish

CurvatureStdThresh = 1.4; # max std in curvature trace that is acceptable. non-tracked fish have noisy traces

AngVelThresh = np.pi/2.5; # the max angular velocity per frame we will accept as not an artifact

XYCoorSmoothMulti = 2; # smoothing window for MultiTracker coordinate data moving average
SpeedSmoothMulti = 5; # smoothing window for the speed and orientation traces
SpeedThreshMulti = 1; # the speed threshold for the multitracker data to be considered movement

for exp_dir in tqdm(dirs):
    print(exp_dir)
    try: # lazy!!!
        for plate in range(2):
            #exp_dir = os.path.realpath(exp_dir)
            os.chdir(exp_dir)

            if plate == 0: # % load in the tracked file names for the relevant plate
                Trials = natsort.natsorted(glob.glob('*_plate_0_0*_0000_Track.mat'))
            else:
                Trials = natsort.natsorted(glob.glob('*_plate_1_0*_0000_Track.mat'))



            if len(Trials) == 0:
                print('\a')
                warnings.warn('Did not find any trials for plate ' + str(plate) + ' in folder ' + exp_dir)
                continue
            #%

            gc = gspread.oauth()

            sh = gc.open_by_key('1ZnZwNIv5oYVvDSKfzGMfq6jae5TcHiV5KymZvhkeJec')
            worksheet = sh.get_worksheet(0)
            df = pd.DataFrame(worksheet.get_all_records())
            #%
            path = os.path.normpath(exp_dir)
            ExpDate = '20'+ path.split(os.sep)[-1][:6]
            

            rows = df.loc[(df['Date Screened'] == int(ExpDate)) & (df['Plate'] == plate)]

            nTreat = rows.shape[0]

            if nTreat == 0:
                print('\a')
                warnings.warn('didnt find any entries for ' + ExpDate + ', plate number: ' + str(plate))
                break
            # note that ROIs are 1 indexed in the spreadsheet

            rois = []
            names = []

            for i in range(nTreat):
                rois.append(np.arange(rows['ROI Start'].iloc[i]-1, rows['ROI End'].iloc[i], 1 ))
                if len(rows["Other ROIs"].iloc[i]) > 0:
                    other_rois = FishTrack.convert_roi_str(rows["Other ROIs"].iloc[i])
                    rois[i] = np.hstack((rois[i], other_rois))

                names.append(str(rows['Product Name'].iloc[i]))
            #%

            names.insert(0, rows['Control Name'].iloc[0])
            rois.insert(0, FishTrack.convert_roi_str(rows['Vehicle ROIs'].iloc[0]))


            #%

            def load_track_burst_data(track_name):

                BurstData = loadmat(track_name)
                x_coors = np.array(BurstData["HeadX"])
                y_coors = np.array(BurstData["HeadY"])
                orient = np.array(BurstData["Orientation"])
                curve = np.array(BurstData["Curvature"])
                area = np.array(BurstData["Areas"])
                tiff_date = BurstData["TiffDate"]

                return x_coors, y_coors, orient, curve, area, tiff_date

            x_coors, y_coors, orient, curve, area, tiff_date = load_track_burst_data(Trials[0])


            #%

            n_fish = np.shape(x_coors)[1]
            n_trials = len(Trials)
            stim_given = np.zeros((n_trials))

            # set the stim indicies where taps and retests happen. 1 = training flash, 2 = tap, 3 = retest flash
            stim_given[:TrainLast] = 1
            stim_given[TapIndStart:TapIndEnd] = 2
            stim_given[RetestStart:RetestEnd] = 3
            stim_given
            

            # %
            n_blocks = np.floor(n_trials/nStimInBlocks)

            #%
            track_data = {
                "OBendEvents":np.zeros((n_trials, n_fish)),
                "OBendLatencies":np.zeros((n_trials, n_fish)),
                "DidASecondOBend":np.zeros((n_trials, n_fish)),
                "DeltaOrientPerOBend":np.zeros((n_trials, n_fish)),
                "DispPerOBend":np.zeros((n_trials, n_fish)),
                "OBendDurations":np.zeros((n_trials, n_fish)),
                "MaxCurvatureOBendEvents":np.zeros((n_trials, n_fish)),
                "DidAMultiBendOBend":np.zeros((n_trials, n_fish)),
                "C1LengthOBendEvents":np.zeros((n_trials, n_fish)),
                "C1AngVelOBendEvents":np.zeros((n_trials, n_fish)),
                "TiffTimeInds":[],
                "names":names,
                "rois":rois,
                "spreadsheet":rows,
                "stim_given":stim_given }


            #%

            for trial in range(n_trials):

                x_coors, y_coors, orient, curve, area, tiff_date = load_track_burst_data(Trials[trial])

                track_data["TiffTimeInds"].append(datetime.strptime(tiff_date[0], '%d-%b-%Y %H:%M:%S'))


                # %

                MeanArea = np.nanmean(area, axis=0)

                # % fish are considered not to be tracked properly if they are
                #             % not found in the first frame 'isnan(XCoors(1,:)', if the area
                #             % of the blob is too big or two small ' MeanArea < AreaMin |
                #             % MeanArea > AreaMax', or if the curvature trace is too noisy,
                #             % ' nanstd(Curvature, 1) > CurvatureStdThresh'

                fish_not_tracked = (np.isnan(x_coors[0,:])) | (MeanArea < AreaMin) | (MeanArea > AreaMax) | (np.nanstd(curve, axis=0) > CurvatureStdThresh)


                delta_orient_trace = np.vstack((np.full(n_fish, np.nan), np.diff(orient, axis=0)))

                # % remove single frame big jumps - these happen when the head and tail
                #             % get confused by the tracking program, or other noisy reasons

                delta_orient_trace[delta_orient_trace > np.pi] = 2*np.pi-delta_orient_trace[delta_orient_trace > np.pi]
                delta_orient_trace[delta_orient_trace < -np.pi] = delta_orient_trace[delta_orient_trace < -np.pi] + 2*np.pi
                delta_orient_trace[abs(delta_orient_trace) > AngVelThresh] = np.nan




                # remove nan values to avoid errors, assume 0 if not tracking from beginning, otherwise fill with previous value





                # savgol filter
                curve_smooth = savgol_filter(ffill_cols(curve), sav_sz, sav_ord, axis=0)
                diff_x = np.diff(savgol_filter(ffill_cols(x_coors), sav_sz, sav_ord, axis=0), axis=0)
                diff_y = np.diff(savgol_filter(ffill_cols(y_coors), sav_sz, sav_ord, axis=0), axis=0)

                # calculate speed
                speed = np.sqrt(np.square(diff_x) + np.square(diff_x))
                speed = np.vstack((np.zeros(n_fish), speed))

                #%

                obend_start = np.full([n_fish], np.nan)
                obend_happened =  np.full([n_fish], np.nan)
                obend_dorient =  np.full([n_fish], np.nan)
                obend_disp =  np.full([n_fish], np.nan)
                obend_dur =  np.full([n_fish], np.nan)
                obend_max_curve =  np.full([n_fish], np.nan)
                obend_second_counter =  np.full([n_fish], np.nan)
                obend_multibend =  np.full([n_fish], np.nan)
                obend_ang_vel =  np.full([n_fish], np.nan)
                obend_c1len = np.full([n_fish], np.nan)
                #%



                for fish in range(n_fish):
                    peakind_curve_pos = find_peaks(curve_smooth[:,fish], width=5)[0]
                    peak_curve_pos = curve[peakind_curve_pos,fish]

                    peakind_curve_neg = find_peaks(-curve_smooth[:,fish], width=5)[0]
                    peak_curve_neg = curve[peakind_curve_neg,fish]

                    peakinds_curve = np.hstack((peakind_curve_pos, peakind_curve_neg))
                    peaks_curve = abs(np.hstack((peak_curve_pos, peak_curve_neg)))

                    I = np.argsort(peakinds_curve)
                    peakinds_curve = peakinds_curve[I]
                    peaks_curve = peaks_curve[I]

                    #plt.plot(curve_smooth[:,fish])
                    #plt.plot(peakinds_curve, peaks_curve, 'x')

                    # find the first peak the crosses the curvature threshold

                    if stim_given[trial]==2:
                        curve_thresh = CBendThresh
                    else:
                        curve_thresh = OBendThresh

                    obend_peaks = np.where(peaks_curve > curve_thresh)[0]

                    # max curvature exibited during movie
                    max_curve = np.max(abs(curve[:, fish]))


                    # now get the kinematic aspects of the response
                    if len(obend_peaks) > 0:
                        obend_happened[fish] = 1
                        obend_peak = obend_peaks[0]
                        obend_peak_ind = peakinds_curve[obend_peak]
                        obend_peak_val = curve[obend_peak_ind, fish]

                        # find the start of the movement as the local minima before the peak of the obend
                        start_o = find_peaks(-abs(curve_smooth[:obend_peak_ind, fish]))[0]

                        # if we cant find the start, the fish is moving at the start of the movie, ignore this fish
                        if len(start_o) > 0:
                            start_o = start_o[-1]

                            obend_start[fish] = start_o*1000/FrameRate


                            # get the angular velocity of the C1 movement in radians per msec
                            # not sure this is right, copying from Matlab code...
                            obend_ang_vel[fish] = obend_peak_val/(1000/FrameRate*(obend_peak_ind - start_o))

                            # use when the speed and curvature returns to near 0 as the end of the movement, beginning at l

                            end_o = obend_peak_ind + np.where((speed[obend_peak_ind:,fish]<0.1) & (abs(curve_smooth[obend_peak_ind:,fish]) < 0.1))[0]

                            # if we cant find the end, the movie cut of the end of the movement. can do this downstream analysis
                            if len(end_o) > 0:
                                end_o = end_o[0]

                                obend_dur[fish] = (end_o - start_o)*1000/FrameRate
                                obend_disp[fish] = np.sqrt(np.square(x_coors[start_o, fish] - x_coors[end_o, fish]) + np.square(y_coors[start_o, fish] - y_coors[end_o, fish]))
                                obend_dorient[fish] = subtract_angles(orient[end_o, fish], orient[start_o, fish])
                                obend_max_curve[fish] = np.max(abs(curve[start_o:end_o, fish]))

                                if obend_disp[fish] < 5: # fish needs to move at least 5 pixels, or else assume its a tracking error
                                    fish_not_tracked[fish] = 1

                                # determine if this is a "multibend o bend" based on if the local minima after the C1 peak is below 0 (normal o-bend) or above 0 (multibend obend)
                                peak_curve = curve[peakinds_curve[obend_peak], fish]
                                if len(peakinds_curve) > (obend_peak + 1):
                                    trough_curve = curve[peakinds_curve[obend_peak+1], fish]
                                    obend_multibend[fish] = np.sign(peak_curve) == np.sign(trough_curve)
                                    # use the difference between peak and trough as c1 length
                                    obend_c1len[fish] = (peakinds_curve[obend_peak+1] - peakinds_curve[obend_peak])*1000/FrameRate

                                # now look for a second O-bend
                                if max(peakinds_curve[obend_peaks]) > end_o:
                                    obend_second_counter[fish] = 1
                                else:
                                    obend_second_counter[fish] = 0

                        else:
                            fish_not_tracked[fish] = 1
                    else:
                        obend_happened[fish] = 0


                    if plot_tracking_results and not fish_not_tracked[fish]:
                        plt.title(fish)
                        plt.plot(curve_smooth[:,fish])
                        plt.plot(speed[:,fish])
                        if obend_happened[fish] == 1:
                            plt.plot(start_o, curve_smooth[start_o, fish], 'o', label='start')
                            plt.plot(obend_peak_ind, obend_peak_val, 'o', label='peak')
                            plt.plot(end_o, curve_smooth[end_o, fish], 'o', label='end')
                        plt.legend()
                        plt.show()
                        print("multibend =")
                        print(obend_multibend[fish])
                        print("second o bend =")
                        print(obend_second_counter[fish])
                        print("max curve = ")
                        print(obend_max_curve[fish])
                        print("dorient =")
                        print(obend_dorient[fish])
                        print("disp =")
                        print(obend_disp[fish])
                        print("dur = ")
                        print(obend_dur[fish])
                        print("c1 length =")
                        print(obend_c1len[fish])
                        print("ang vel = ")
                        print(obend_ang_vel[fish])

                    # nan out non-tracked fish and save arrays

                obend_happened[fish_not_tracked] = np.nan
                track_data["OBendEvents"][trial, :] = obend_happened
                obend_start[fish_not_tracked] = np.nan
                track_data["OBendLatencies"][trial, :] = obend_start
                obend_second_counter[fish_not_tracked] = np.nan
                track_data["DidASecondOBend"][trial, :] = obend_second_counter
                obend_dur[fish_not_tracked] = np.nan
                track_data["OBendDurations"][trial, :] = obend_dur
                obend_disp[fish_not_tracked] = np.nan
                track_data["DispPerOBend"][trial,:] = obend_disp
                obend_max_curve[fish_not_tracked] = np.nan
                track_data["MaxCurvatureOBendEvents"][trial,:] = obend_max_curve
                obend_dorient[fish_not_tracked] = np.nan
                track_data["DeltaOrientPerOBend"][trial,:] = obend_dorient
                obend_multibend[fish_not_tracked] = np.nan
                track_data["DidAMultiBendOBend"][trial,:] = obend_multibend
                obend_c1len[fish_not_tracked] = np.nan
                track_data["C1LengthOBendEvents"][trial,:] = obend_c1len
                obend_ang_vel[fish_not_tracked] = np.nan
                track_data["C1AngVelOBendEvents"][trial,:] = obend_ang_vel


            #gb = Glasbey(base_palette=[(0, 0, 0)])
            gb = Glasbey()
            p = gb.generate_palette(size=nTreat+2)
            col_vec = gb.convert_palette_to_rgb(p)
            col_vec = np.array(col_vec[1:], dtype=float)/255

            #col_vec = np.array(((0,0,0),(0,0,1), (1,0,0), (0,1,0), (0,1,1),(1,1,0)))
            #%
            if names[0] == '':
                names[0] = '0.1% DMSO'

            # use dummy stim times if we dont have all of the stim times recorded -- for example if we didnt manage to track all of the experiments. 
            if len(track_data["TiffTimeInds"]) < 420:
                with open(root_dir + 'dummy_data.pkl', 'rb') as f:
                            dummy_data = pickle.load(f)
                track_data["TiffTimeInds"] = dummy_data["TiffTimeInds"]

            stim_times = []
            for i in range(len(track_data["TiffTimeInds"])):
                stim_times.append((track_data["TiffTimeInds"][i] - track_data["TiffTimeInds"][0]).total_seconds()/60/60)
            stim_times = np.array(stim_times)

            if plot_burst:
                # probability
                FishTrack.plot_burst_data(track_data["OBendEvents"], 'probability of response', rois, stim_times, names, stim_given, 15, col_vec, save_name = save_name)

                #% latency
                FishTrack.plot_burst_data(track_data["OBendLatencies"], 'latency of response', rois, stim_times, names, stim_given, 15, col_vec, save_name = save_name)

                # displacement
                FishTrack.plot_burst_data(track_data["DispPerOBend"], 'displacement (px)', rois, stim_times, names, stim_given, 15, col_vec, save_name = save_name)

                # duration
                FishTrack.plot_burst_data(track_data["OBendDurations"], 'duration (msec)', rois, stim_times, names, stim_given, 15, col_vec, save_name = save_name)

                # curvature
                FishTrack.plot_burst_data(abs(track_data["MaxCurvatureOBendEvents"]), 'bend amplitude (rad)', rois, stim_times, names, stim_given, 15, col_vec,save_name = save_name)

                # multibend
                FishTrack.plot_burst_data(track_data["DidAMultiBendOBend"], 'proportion multibend', rois, stim_times, names, stim_given, 15, col_vec, save_name = save_name)

                # second obend
                FishTrack.plot_burst_data(track_data["DidASecondOBend"], 'did second o-bend', rois, stim_times, names, stim_given, 15, col_vec, save_name = save_name)

                # c1 length
                FishTrack.plot_burst_data(track_data["C1LengthOBendEvents"], 'c1 length', rois, stim_times, names, stim_given, 15, col_vec, save_name = save_name)

                # ang vel
                FishTrack.plot_burst_data(abs(track_data["C1AngVelOBendEvents"]), 'ang velocity', rois, stim_times, names, stim_given, 15, col_vec, save_name = save_name)



            #% now we will analyze the Multitracker .track file

            track_file = glob.glob("*"+str(plate)+".track")
            
            if len(track_file) > 1:
                raise ValueError('check track files from multitracker... more than one found')
            
            track_name = ' '.join(map(str, track_file))
            #%

            def load_multitracker_data(track_name, n_fish):
                print('loading ' + track_name + ' ...will take a few minutes')
                multitrack_data = pd.read_csv(track_name, delimiter='\t')
                frame_inds = multitrack_data.iloc[:, 0].to_numpy()
                t_stmp = multitrack_data.iloc[:, -1].to_numpy()
                ind_vec = np.arange(1, n_fish*3, 3)

                x_coors_multi = multitrack_data.iloc[:, ind_vec].to_numpy()
                y_coors_multi = multitrack_data.iloc[:, ind_vec+1].to_numpy()
                orient_multi = multitrack_data.iloc[:, ind_vec+2].to_numpy()

                print('\a')
                print('loaded : ' + track_name)

                return frame_inds, t_stmp, x_coors_multi, y_coors_multi, orient_multi

            frame_inds, t_stmp, x_coors_multi, y_coors_multi, orient_multi = load_multitracker_data(track_name, n_fish)

            #% load stim index data

            stim_name = track_name.replace("P_0.track", '0.tap')
            stim_name = stim_name.replace("P_1.track", '1.tap')
            print(stim_name)
            #%
            stim_inds = np.loadtxt(stim_name, delimiter='\t')

            # if the experiment go cut off for the stimuli, use the dummy data
            if len(stim_inds) < 420:
                
                stim_inds = np.loadtxt(root_dir + 'dummy_data_' + str(plate) + '.tap', delimiter='\t')
            #%

            n_frames = len(frame_inds)
            st = int(n_frames/2)
            end = st+10000
            # for i in range(n_fish):
            #     plt.plot(x_coors_multi[st:end, i],y_coors_multi[st:end, i] )
            # plt.show()

            #% determine the multitracker online tracking frame rate based on frame indexes

            base_framerate = int(FrameRate/ np.median(np.diff(frame_inds[1:1000])))
            

            #% analyze OMR data

            OMRStart = np.where(frame_inds>=stim_inds[OMRStimStart])[0][0]
            OMREnd = np.where(frame_inds>=stim_inds[OMRStimEnd])[0][0]
            # if len(OMRStart) == 0:
            #     warnings.warn('did not find OMR data properly...')
            #%

            #%


            # % Extract the orientation of the fish during the OMR phase, and
            # % we use asind(sind()) to extract only the left/right version
            # % of the fish's orentation and we
            # % use a fairly harsh 1s median filter. This is done to
            # % minimize noise, like full 180 flips. Since the OMR phase
            # % change is at 30s intervals, this should be OK.
            orient_omr = np.radians(orient_multi[OMRStart:OMREnd+1, :])
            from scipy.ndimage import median_filter
            
            orient_lr = np.rad2deg(np.arcsin(np.sin(orient_omr)))
            orient_lr = medfilt(orient_lr, kernel_size=(base_framerate+1,1))
            # plt.plot(np.nanmean(orient_lr, axis=1))
            # plt.show()
            #%
            # % now we loop through each phase of the transition, and create
            # % a single averaged trace of the orientation that will have a positive
            # % slope if the fish is performing OMR (ie reorientating right
            # % when the motion is to the right, left when the motion is to
            # % the left
            nfr_per_flip = base_framerate*30
            nfr_per_cyc = nfr_per_flip*2
            n_fr_OMR = orient_lr.shape[0]
            n_cyc_OMR = int(n_fr_OMR/(nfr_per_cyc))
            acc_LR = np.zeros((nfr_per_flip, n_fish)) # array to sum up the left/right angles
            n_obs = np.zeros((nfr_per_flip, n_fish)) # keep track of if the fish was actually tracked or not

            for i in range(n_cyc_OMR):
                trace_start = i*nfr_per_cyc
                trace_mid = trace_start+nfr_per_flip
                trace_end = trace_mid+nfr_per_flip
                acc_LR = np.nansum(np.stack((acc_LR, orient_lr[trace_start:trace_mid]), axis=2), axis=2)
                acc_LR = np.nansum(np.stack((acc_LR, orient_lr[trace_end:trace_mid:-1]), axis=2), axis=2)
                n_obs = np.nansum(np.stack((n_obs, ~np.isnan(orient_lr[trace_start:trace_mid])), axis=2), axis=2)
                n_obs = np.nansum(np.stack((n_obs, ~np.isnan(orient_lr[trace_end:trace_mid:-1])), axis=2), axis=2)

            acc_LR = acc_LR/n_obs


            #plt.plot(np.nanmean(acc_LR, axis=1))



            # remove fish that arent tracked in at least half of the stimlulus fiips
            not_omrtracked = np.mean(n_obs, axis=0) < (n_cyc_OMR/2)
            acc_LR[:, not_omrtracked] = np.nan

            # remove first and last observation, since flipping can create artifacts here. 
            acc_LR = acc_LR[1:-2]

            #%
            from scipy.stats import linregress

            omr_slopes = np.zeros((n_fish, 1))
            omr_slopes[:] = np.nan

            for i in range(n_fish):
                y = acc_LR[:,i]
                x = np.arange(acc_LR.shape[0])
                slope, intercept, r_value, p_value, std_err  = linregress(x, y)
                omr_slopes[i] = slope

            omr_slopes = omr_slopes*base_framerate # change units to degrees per second

            #% now analyze the swimming data

            #% extract speed trace
            sav_sz_multi = 9
            sav_ord_multi = 2
            # diff_x_multi = np.diff(savgol_filter(ffill_cols(x_coors_multi), sav_sz_multi, sav_ord_multi, axis=0), axis=0)
            # diff_y_multi = np.diff(savgol_filter(ffill_cols(y_coors_multi), sav_sz_multi, sav_ord_multi, axis=0), axis=0)
            # speed_multi = np.sqrt(np.square(diff_x_multi) + np.square(diff_y_multi))

            diff_x_multi = np.diff(x_coors_multi, axis=0)
            diff_y_multi = np.diff(y_coors_multi, axis=0)

            diff_x_multi[np.isnan(diff_x_multi)] = 0
            diff_y_multi[np.isnan(diff_y_multi)] = 0
            speed_multi = np.sqrt(np.square(diff_x_multi) + np.square(diff_y_multi))
            #remove any suriously large jumps that are caused by tracking errors and filter
            speed_multi[speed_multi > 15] = 0
            speed_multi = savgol_filter(speed_multi, sav_sz_multi, sav_ord_multi, axis=0)

            #% 
            bouts_multi = (speed_multi > SpeedThreshMulti).astype(int)
            bouts_starts = (np.diff(bouts_multi, axis=0) == 1).astype(int)
            bouts_ends = (np.diff(bouts_multi, axis=0) == -1).astype(int)
            disp_multi = np.copy(speed_multi)
            disp_multi[disp_multi < SpeedThreshMulti] = 0 


            indStart = np.where(frame_inds == stim_inds[0])[0][0]
            free_start = np.where(frame_inds == stim_inds[242])[0][0]
            free_end = np.where(frame_inds == stim_inds[269])[0][0]


            #%
            #%


            # loop through each fish and analyze its bouts and turns:
            #for fish in range(n_fish):
            #%

            # arrays to accumulate across fish
            disp_free = np.full(n_fish, np.nan)
            turn_bias_free = np.full(n_fish, np.nan)
            median_turn_free = np.full(n_fish, np.nan)
            disp_up_omr = np.full(n_fish, np.nan)
            disp_down_omr = np.full(n_fish, np.nan)
            turn_bias_omr = np.full(n_fish, np.nan)
            median_turn_omr = np.full(n_fish, np.nan)
            disp_blks = np.full((n_fish,2), np.nan)


            for fish in range(n_fish):
                bout_starts_fish = np.where(bouts_starts[:, fish] == 1)[0]
                bout_ends_fish = np.where(bouts_ends[:, fish] == 1)[0]

                if len(bout_starts_fish) > 0:
                    # make sure there are no problems at the start or the end, and all bouts have a beginning and and ending
                    if bout_ends_fish[0] < bout_starts_fish[0]:
                        bout_starts_fish = bout_starts_fish[1:]
                    if bout_starts_fish[-1] > bout_ends_fish[-1]:
                        bout_starts_fish = bout_starts_fish[:-1]
                    if bout_starts_fish[0] < 100:
                        bout_starts_fish = bout_starts_fish[1:]
                        bout_ends_fish = bout_ends_fish[1:]
                    if n_frames - bout_ends_fish[-1] < 100:
                        bout_starts_fish = bout_starts_fish[:-1]
                        bout_ends_fish = bout_ends_fish[:-1]

                    # remove bouts that are not at least 3 frames long, as these are likely noise threshold crossings. 

                    bout_len_fish = bout_ends_fish - bout_starts_fish
                    bout_ends_fish = bout_ends_fish[bout_len_fish >= 3]
                    bout_starts_fish = bout_starts_fish[bout_len_fish >= 3]

                    if not len(bout_starts_fish) == len(bout_ends_fish):
                        raise ValueError('bouts did not match up in multitracker data for fish ' + str(fish))

                    # get the difference in orientation by subtracing the orientation three frames before the bout from 3 frames after the bout

                    d_orient_fish = orient_multi[bout_ends_fish, fish] - orient_multi[bout_starts_fish, fish]

                    # assume the smalles angular difference is correct. large jumps are caused by 0-360 degree problems
                    d_orient_fish[d_orient_fish > 180] = 360 - d_orient_fish[d_orient_fish > 180]
                    d_orient_fish[d_orient_fish < -180] = 360 + d_orient_fish[d_orient_fish < -180]

                    # categorize as a swim vs a turn based on a threshold of 10, based on visual inspection of the histograms
                    # plt.hist(d_orient_fish, 70)
                    # plt.vlines([-10,10], 0, 1000)
                    # plt.show()

                    turn_swim = abs(d_orient_fish) > 10 # 1 = turn, 0 = swim
                    turns_fish = np.copy(d_orient_fish)
                    turns_fish[turn_swim == 0] = np.nan

                    # now analyze different epocs of the experiment. we will measure "displacement" as the average speed above threshold, multiplied by the number of frames per minute, give us the average displacement per minute in that period

                    # free swimming period
                    bout_st_inds = np.where((bout_starts_fish > free_start) & (bout_starts_fish < free_end))
                    disp_free[fish] = np.mean(disp_multi[free_start:free_end, fish])*base_framerate*60
                    #turn_bias_free[fish] = (np.sum(turn_swim[bout_st_inds]==1) - np.sum(turn_swim[bout_st_inds]==0))/(np.sum(turn_swim[bout_st_inds]==1) + np.sum(turn_swim[bout_st_inds]==0))
                    turn_bias_free[fish] = np.mean(turn_swim[bout_st_inds])
                    median_turn_free[fish] = np.nanmedian(abs(turns_fish[bout_st_inds]))

                    # print('disp free')
                    # print(disp_free[fish])
                    # print('turn bias free')
                    # print(turn_bias_free[fish])
                    # print('median turn free')
                    # print(median_turn_free[fish])

                    # OMR period

                    bout_st_inds = np.where((bout_starts_fish > OMRStart) & (bout_starts_fish < OMREnd))
                    turn_bias_omr[fish] = np.mean(turn_swim[bout_st_inds])
                    median_turn_omr[fish] = np.nanmedian(abs(turns_fish[bout_st_inds]))
                    # print('turn bias omr')
                    # print(turn_bias_omr[fish])
                    # print('median turn omr')
                    # print(median_turn_omr[fish])

                    # % for  the OMR period, we break this into two bins. The fish
                    # % are hyperactive during the first ~4 minutes after the OMR the
                    # % fish are hyperactive, likely due to the effective 1/8 light
                    # % levels from before. Then the rate of the fish dips below
                    # % baseline for the remaining time.
                    omr_up_inds = np.arange(OMRStart, OMRStart+base_framerate*60*4)
                    omr_down_inds = np.arange(OMRStart+base_framerate*60*4, OMREnd)
                    disp_up_omr[fish] = np.mean(disp_multi[omr_up_inds, fish])*base_framerate*60/disp_free[fish]
                    disp_down_omr[fish] = np.mean(disp_multi[omr_down_inds, fish])*base_framerate*60/disp_free[fish]
                    # print('disp up omr')
                    # print(disp_up_omr[fish])
                    # print('disp down omr')
                    # print(disp_down_omr[fish])

                    #% now analyze movement rate during the dark flash training blocks, but outside of the stimuli when the lights are on 

                    # dark flash training blocks, stim 0:239
                    inds_out_flash = np.array([], dtype=int)
                    for ind in stim_inds[0:240]:
                        flash_start = np.where(frame_inds == ind)[0][0]
                        out_start = flash_start + base_framerate*21
                        inds_out_flash = np.hstack((inds_out_flash, np.arange(out_start, out_start+base_framerate*39)))
                    disp_blks[fish, 0] = np.mean(disp_multi[inds_out_flash, fish])*base_framerate*60

                    # retest block, stim 360:419
                    inds_out_flash = np.array([], dtype=int)
                    for ind in stim_inds[360:419]:
                        flash_start = np.where(frame_inds == ind)[0][0]
                        out_start = flash_start + base_framerate*21
                        inds_out_flash = np.hstack((inds_out_flash, np.arange(out_start, out_start+base_framerate*39)))

                    disp_blks[fish, 1] = np.mean(disp_multi[inds_out_flash, fish])*base_framerate*60


            #%
            # get the SSMD per group

            def get_ssmds(dataset, fish_rois):
                # return the strinclty standardized mean difference between the treatment groups and the control. first entry in the roi list is the control group
                n_treat = len(fish_rois)-1
                SSMDs = np.zeros((n_treat))
                fish_data = np.copy(dataset)
                fish_data[~np.isfinite(fish_data)] = np.nan
                cont_data = fish_data[rois[0]]
                mean_cont = np.nanmean(cont_data)
                std_cont = np.nanstd(cont_data)
                for k, fish_ids in enumerate(rois[1:]):
                    mean_gr = np.nanmean(fish_data[fish_ids])
                    std_gr = np.nanstd(fish_data[fish_ids])
                    SSMDs[k] = (mean_gr-mean_cont)/(np.sqrt(np.square(std_cont) + np.square(std_gr)))
                
                return SSMDs

            ssmds = {}
            ssmds['disp_blkTrain'] = get_ssmds(disp_blks[:,0], rois)
            ssmds['disp_blkRet'] = get_ssmds(disp_blks[:,1], rois)
            ssmds['disp_free'] = get_ssmds(disp_free, rois)
            ssmds['turn_bias_free'] = get_ssmds(turn_bias_free, rois)
            ssmds['median_turn_free'] = get_ssmds(median_turn_free, rois)
            ssmds['disp_up_omr'] = get_ssmds(disp_up_omr, rois)
            ssmds['disp_down_omr'] = get_ssmds(disp_down_omr, rois)
            ssmds['turn_bias_omr'] = get_ssmds(turn_bias_omr, rois)
            ssmds['median_turn_omr'] = get_ssmds(median_turn_omr, rois)
            ssmds['omr_slopes'] = get_ssmds(omr_slopes, rois)

            # save the fish data for output

            track_data['disp_blks'] = disp_blks
            track_data['turn_bias_free'] = turn_bias_free
            track_data['median_turn_free'] = median_turn_free
            track_data['disp_up_omr'] = disp_up_omr
            track_data['disp_down_omr'] = disp_down_omr
            track_data['turn_bias_omr'] = turn_bias_omr
            track_data['median_turn_omr'] = median_turn_omr
            track_data['omr_slopes'] = omr_slopes
            track_data['acc_lr'] = acc_LR

            #% now do groupwise analysis of burst track data


            def get_ssmds_burst(stim_data, fish_rois):
                # calculate the striclty standardized mean difference, first averaging each fish's response per block, then using the average and std per group for the SSMD caclulation. Group 1 is treated as the control group
                stim_data[~np.isfinite(stim_data)] = np.nan
                fish_rois = rois
                n_treat = len(fish_rois)-1
                SSMDs = np.zeros((n_treat))

                # parse stimuli
                training_flash = np.where(stim_given == 1)[0]
                test_tap = np.where(stim_given == 2)[0]
                retest_flash = np.where(stim_given == 3)[0]

                # for dark flashes we will analyze the first n flashes for "naieve" response, then the mean of the remaining flashes for the habituation response
                n_init = 5
                
                # hard coded into blocks of 60 for training flashes
                block_inds = np.array((training_flash[:n_init],training_flash[n_init:], retest_flash, test_tap))



                # we will have the naieve, habitating, retest and and tap response for each group as the SSMD
                blk_results = np.full((n_treat, 4), np.nan)

                # get the controls
                cont_data = stim_data[:, fish_rois[0]]
                cont_means =  np.full(6, np.nan)
                cont_stds = np.full((6), np.nan)

                # summary stats for controls
                for i in range(4):
                    mean_fish = np.nanmean(cont_data[block_inds[i], :], axis=0)
                    cont_means[i] = np.nanmean(mean_fish)
                    cont_stds[i] = np.nanstd(mean_fish)

                # loop through treatment groups and blocks
                for treat in range(n_treat):
                    treat_data = stim_data[:, fish_rois[treat+1]]
                    for i in range(4):    
                        mean_fish = np.nanmean(treat_data[block_inds[i], :], axis=0)
                        treat_mean = np.nanmean(mean_fish)
                        treat_std = np.nanstd(mean_fish)
                        blk_results[treat, i] = (treat_mean - cont_means[i])/(np.sqrt(np.square(treat_std) + np.square(cont_stds[i])))
                
                return blk_results

            ssmds['OBendEvents'] = get_ssmds_burst(track_data['OBendEvents'], rois)
            ssmds['OBendLatencies'] = get_ssmds_burst(1000-track_data['OBendLatencies'], rois)  # For latencies subtract from max value (1000), because latencies increase during habituation
            ssmds['DidASecondOBend'] = get_ssmds_burst(track_data['DidASecondOBend'], rois)
            ssmds['DeltaOrientPerOBend'] = get_ssmds_burst(abs(track_data['DeltaOrientPerOBend']), rois)
            ssmds['DispPerOBend'] = get_ssmds_burst(track_data['DispPerOBend'], rois)
            ssmds['OBendDurations'] = get_ssmds_burst(track_data['OBendDurations'], rois)
            ssmds['MaxCurvatureOBendEvents'] = get_ssmds_burst(track_data['MaxCurvatureOBendEvents'], rois)
            ssmds['DidAMultiBendOBend'] = get_ssmds_burst(1-track_data['DidAMultiBendOBend'], rois) # subtract from 1 for proportion of simple o-bends
            ssmds['C1LengthOBendEvents'] = get_ssmds_burst(1000-track_data['C1LengthOBendEvents'], rois)  # subtract from max value (1000), because c1 length increases during habituation
            ssmds['C1AngVelOBendEvents'] = get_ssmds_burst(track_data['C1AngVelOBendEvents'], rois)



            #% convert the ssmds into the same format as the original fingerprints from the matlab-based analysis 

            fingerprint = np.vstack((
                # start with dark flash responses, Naieve, train and test blocks
                np.transpose(ssmds['OBendEvents'][:,:3]), 
                np.transpose(ssmds['DidASecondOBend'][:,:3]),
                np.transpose(ssmds['OBendLatencies'][:,:3]),
                np.transpose(ssmds['DispPerOBend'][:,:3]),
                np.transpose(ssmds['OBendDurations'][:,:3]),
                np.transpose(ssmds['MaxCurvatureOBendEvents'][:,:3]),
                np.transpose(ssmds['DeltaOrientPerOBend'][:,:3]),
                np.transpose(ssmds['DidAMultiBendOBend'][:,:3]),
                np.transpose(ssmds['C1LengthOBendEvents'][:,:3]),
                np.transpose(ssmds['C1AngVelOBendEvents'][:,:3]),
                # now tap blocks
                np.transpose(ssmds['OBendEvents'][:,3]), 
                np.transpose(ssmds['DidASecondOBend'][:,3]),
                np.transpose(ssmds['OBendLatencies'][:,3]),
                np.transpose(ssmds['DispPerOBend'][:,3]),
                np.transpose(ssmds['OBendDurations'][:,3]),
                np.transpose(ssmds['MaxCurvatureOBendEvents'][:,3]),
                np.transpose(ssmds['DeltaOrientPerOBend'][:,3]),
                ssmds['disp_blkTrain'],
                ssmds['disp_blkRet'],
                ssmds['disp_free'],
                ssmds['turn_bias_free'],
                ssmds['median_turn_free'],
                ssmds['disp_up_omr'],
                ssmds['disp_down_omr'],
                ssmds['turn_bias_omr'],
                ssmds['median_turn_omr'],
                ssmds['omr_slopes']
                ))

            fingerprint_order = ['Prob-Naieve', 'Prob-Train', 'Prob-Test',  
                        'TwoMvmt-Naieve', 'TwoMvmt-Train', 'TwoMvmt-Test',
                        'Lat-Naieve', 'Lat-Train', 'Lat-Test',
                        'Disp-Naieve', 'Disp-Train', 'Disp-Test',
                        'Dur-Naieve', 'Dur-Train', 'Dur-Test',
                        'Curve-Naieve', 'Curve-Train', 'Curve-Test',
                        'dOrient-Naieve', 'dOrient-Train', 'dOrient-Test',
                        'SimpleOBend-Naieve', 'SimpleOBend-Train', 'SimpleOBend-Test',
                        'C1Length-Naieve', 'C1Length-Train', 'C1Length-Test',
                        'C1AngVel-Naieve', 'C1AngVel-Train', 'C1AngVel-Test',
                        'Prob-Tap', 'TwoMvmt-Tap','Lat-Tap', 'Disp-Tap', 'Dur-Tap', 'Curve-Tap', 'dOrient-Tap', 
                        'SpntDisp-Train', 'SpntDisp-Test', 'SpntDisp-Free', 'TurnBias-Free', 'MedianTurnAng-Free',
                        'OMR-SpeedUp','OMR-SpeedDown','OMR-TurnBias', 'OMR-MedianTurnAng', 'OMR-Perf']


            plt.imshow(fingerprint, vmin=-2, vmax=2)

            # collect data and save

            ssmds['fingerprint'] = fingerprint
            ssmds['fingerprint_order'] = fingerprint_order
            ssmds['names'] = names


            save_name = Trials[0][:Trials[0].find('plate_')+7] + '_ssmddata_twoMeasures.pkl'
            with open(save_name,"wb") as f:
                pickle.dump(ssmds,f)

            #% save the burst data files

            save_name = Trials[0][:Trials[0].find('plate_')+7] + '_trackdata_twoMeasures.pkl'
            with open(save_name,"wb") as f:
                pickle.dump(track_data,f)
    except:
        print('failure in ' + exp_dir)



# %%
