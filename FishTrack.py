# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 09:19:20 2019

@author: owen.randlett
"""
import cv2, warnings, time, math, sys, glob, serial

import numpy as np
import matplotlib.pyplot as plt
import PySimpleGUI as sg

from skimage.measure import label, regionprops
from scipy.signal import savgol_filter
from numba import jit


def make_rois(height, width, start_coords,end_coords,  n_cols=20, n_rows=15, well_spacing = 2):
    # return a interger label image defining square ROIs based starting at start coorinates (tuple, x and y coordinate) and ending coordiantes

    im_rois = np.zeros((height, width), dtype=int)
    
    well_width = (end_coords[0] - start_coords[0])/n_cols - 1
    well_height = (end_coords[1] - start_coords[1])/n_rows - 1
    
    k = 1
    for i in range(n_rows):
        h_start = int(well_height*i + well_spacing*i) + start_coords[1]
        h_end = int(h_start + well_height)

        for j in range(n_cols):
            w_start = int(well_width*j + well_spacing*j) + start_coords[0]
            w_end = int(w_start + well_width)
            im_rois[h_start:h_end, w_start:w_end] = k
            k+=1
    return im_rois

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def plot_burst_data_all(track_data, treat_ids, cont_id, col_vec, save_str, nStimInBlocks = 60, smooth_window=15, plot_taps = True, plot_retest=True, stim_times = None):
    import warnings
    from scipy.signal import savgol_filter
    
    if np.sum(stim_times == None) > 0:
        stim_times = []
        for i in range(len(track_data["TiffTimeInds"])):
            stim_times.append((track_data["TiffTimeInds"][i] - track_data["TiffTimeInds"][0]).total_seconds()/60/60)
        stim_times = np.array(stim_times)
    
    ids = np.hstack((cont_id, treat_ids)).flatten().astype(int)

    names = track_data['names']
    fish_names = []
    for i in ids:
        fish_names.append(names[i])
    time_inds = stim_times
    stim_given = track_data['stim_given']
    fish_ids = np.array(track_data['rois'])[ids]
    y_text = ['Probability of Response', 
    'Double Responses', 
    'Latency', 
    'Compound O-Bend',
    'Movement Duration', 
    'Displacement', 
    'Reorientation', 
    'Bend Amplitude', 
    'C1 Duration', 
    'C1 Ang. Vel.']
    for d, data_type in enumerate(['OBendEvents', 'DidASecondOBend', 'OBendLatencies', 'DidAMultiBendOBend', 'OBendDurations', 'DispPerOBend', 'DeltaOrientPerOBend', 'MaxCurvatureOBendEvents', 'C1LengthOBendEvents', 'C1AngVelOBendEvents']):

        data = abs(track_data[data_type])
        plt.figure(figsize=(10,7))
        plt.xlabel('time (hr)')
        plt.ylabel(y_text[d])

        n_gr = np.shape(fish_ids)[0]

        for i in range(n_gr): # plot the raw dark flash stimuli
            inds_stim = np.ix_((stim_given==1) | (stim_given==3))[0]
            inds_fish  = fish_ids[i]
            inds_both = np.ix_(inds_stim, inds_fish)

            plt.plot(time_inds[inds_stim], np.nanmean(data[inds_both], axis=1), '.', markersize=3, color= col_vec[i], label=fish_names[i]+' , n='+str(len(inds_fish)) )
           
        lgnd = plt.legend(fontsize = 15, markerscale=3, loc="lower right")  


        for i in range(n_gr): # plot the smoothed data off of the frist 4 blocks, and retest block
            inds_fish  = fish_ids[i]
            #inds_stim = 
            for k in range(5):
                inds_block = np.ix_((stim_given==1) | (stim_given==3))[0][k*nStimInBlocks:k*nStimInBlocks+nStimInBlocks]
                inds_both_block =  np.ix_(inds_block, inds_fish)

                y = np.nanmean(data[inds_both_block], axis=1) 
                x = time_inds[inds_block]
                # remove NaNs
                x = x[~np.isnan(y)]
                y = y[~np.isnan(y)]
                try:
                    y = savgol_filter(y, smooth_window, 2)
                    plt.plot(x,y, '-', color= col_vec[i], linewidth=5, alpha=0.8)
                except:
                    warnings.warn('savgol did not converge')


        # plot taps
        if plot_taps:
            for i in range(n_gr):
                inds_fish  = fish_ids[i]
                inds_block = stim_given == 2
                inds_both_block =  np.ix_(inds_block, inds_fish)
                y1 = np.nanmean(data[inds_both_block], axis=1)
                plt.plot(time_inds[inds_block], y1, 'x', markersize=4, color= col_vec[i])

                try:
                    y2 = savgol_filter(y1, smooth_window, 2)
                    plt.plot(time_inds[inds_block], y2, '-', color= col_vec[i], linewidth=3, alpha=0.8)

                except:
                        warnings.warn('savgol did not converge')


        #plt.rc('font', size=18)
        #plt.rc('legend', fontsize=10)
        #
        invalid = '<>:"/\|?* '

        for char in invalid:
            save_str = save_str.replace(char, '')
        
        if not plot_retest:
            plt.xlim((-0.1,8.1))
        simpleaxis(plt.gca())
        plt.savefig((save_str +'_' +data_type+ '.svg').replace(' ', ''), bbox_inches='tight', transparent=True)
        plt.savefig((save_str +'_' +data_type+ '.png').replace(' ', ''), bbox_inches='tight', transparent=True, dpi=100)

        plt.show()

def plot_cum_diff(data, treat_id, cont_id, save_name, n_norm = 5, ylim=0.8):
    ### calculate cumulative difference relative to controls, as in Randlett et al., Current Biology, 2019
    # n_norm will give the number of inital responses to normalize to
    from scipy import stats
    
    plt.fill_between(np.arange(240), np.ones(240)*-0.05, np.ones(240)*0.05, color=[0.5, 0.5, 0.5], alpha=0.4)
    plt.vlines(60, -1, 1, colors='k', linestyles='dashed')
    plt.vlines(120, -1, 1, colors='k', linestyles='dashed')
    plt.vlines(180, -1, 1, colors='k', linestyles='dashed')
    plt.hlines(0, 0, 240, colors='k', linestyles='dashed')
    stim_given = data['stim_given']
    rois = data['rois']
    names = data['names']
    col_vec = [[0,0,1],  [1,0,0], [0,1,0], [0,0,0], [1,0,1], [0.9, 0.9, 0], [0, 0.4, 0], [0,0.8,0.8]]
    legend_entries = ['probability', 
        'double responses', 
        'latency',
        'simple o-bends', 
        'movement duration', 
        'displacement', 
        'reorientation', 
        'bend amplitude']
    
    for col_id, data_type in enumerate(['OBendEvents', 'DidASecondOBend', 'OBendLatencies', 'DidAMultiBendOBend', 'OBendDurations', 'DispPerOBend', 'DeltaOrientPerOBend', 'MaxCurvatureOBendEvents']):
        if data_type == 'OBendLatencies': # invert so that habituation changes match direction
            data_to_plot = 1000 - abs(data[data_type][stim_given==1, :]) 
        elif data_type == 'DidAMultiBendOBend' : # invert so that habituation changes match direction
            data_to_plot = 1 - abs(data[data_type][stim_given==1, :]) 
        else:
            data_to_plot = abs(data[data_type][stim_given==1, :])
    
        treat_ids = rois[treat_id]
        treat_data = data_to_plot[:,treat_ids]
        n_treat = len(treat_ids)
        cont_ids = rois[cont_id]
        cont_data = data_to_plot[:,cont_ids]
        n_cont = len(cont_ids)

        n_boots = 2000
        cum_diff_dist = np.zeros((240, n_boots))

        for i in range(n_boots):
            mean_treat = np.nanmean(treat_data[:, np.random.randint(0, n_treat, n_treat)], axis=1)
            mean_cont = np.nanmean(cont_data[:, np.random.randint(0, n_cont, n_cont)], axis=1)
            nan_IDs = (np.isnan(mean_treat) | np.isnan(mean_cont))
            norm_vec = np.arange(1,241)
            k = 1
            for el, val in enumerate(norm_vec):
                if nan_IDs[el] == True:
                    norm_vec[el:] = norm_vec[el:]-1
            
            cum_diff_dist[:, i] = np.nancumsum(mean_cont/np.nanmean(mean_cont[:n_norm]) - mean_treat/np.nanmean(mean_treat[0:n_norm]))/norm_vec


        cum_diff_dist[~np.isfinite(cum_diff_dist)] = 0
        mu = np.nanmean(cum_diff_dist, axis=1)
        sigma = np.nanstd(cum_diff_dist, axis=1)
        CI = stats.norm.interval(0.95, loc=mu, scale=sigma/np.sqrt(n_treat))
        CI[0][np.isnan(CI[0])] = mu[np.isnan(CI[0])]
        CI[1][np.isnan(CI[1])] = mu[np.isnan(CI[1])]
        plt.plot(np.arange(240), mu, color=col_vec[col_id], label=legend_entries[col_id])
        plt.fill_between(np.arange(240), CI[0], CI[1], alpha=0.3, color=col_vec[col_id], label='_nolegend_', interpolate=True)
    plt.ylabel('Cum. Mean Diff\nvs. Control')
    plt.xlabel('Stimuli')
    plt.legend(bbox_to_anchor=(1.05, 1.0, 0.3, 0.2), loc='upper left')
    # plt.legend(['probability', 
    #     'double responses', 
    #     'latency',
    #     'simple o-bends', 
    #     'movement duration', 
    #     'displacement', 
    #     'reorientation', 
    #     'bend amplitude'], 
    #     bbox_to_anchor=(1.05, 1.0, 0.3, 0.2), loc='upper left')
    plt.title(names[treat_id] + ', n = ' + str(n_treat) + '\nvs ' + names[cont_id] + ', n = ' + str(n_cont))
    plt.xticks((0, 60, 120, 180, 240))
    plt.ylim((-ylim, ylim))
    plt.xlim((0,240))
   
    invalid = '<>:"/\|?* '

    for char in invalid:
        save_name = save_name.replace(char, '')
    plt.savefig(save_name+'.png', dpi=100, bbox_inches='tight')
    plt.savefig(save_name+'.svg', dpi=100, bbox_inches='tight')
    plt.show()
    
def plot_burst_data(data, ytext, fish_ids, time_inds, fish_names, stim_given, smooth_window, col_vec, save_name, nStimInBlocks = 60, plot_taps = False, plot_retest=True):
    plt.figure(figsize=(10,7))

    plt.xlabel('time (hr)')
    plt.ylabel(ytext)
    
    n_gr = np.shape(fish_ids)[0]
    x = data
    for i in range(n_gr): # plot the raw dark flash stimuli
        inds_stim = np.ix_((stim_given==1) | (stim_given==3))[0]
        inds_fish  = fish_ids[i]
        inds_both = np.ix_(inds_stim, inds_fish)

        plt.plot(time_inds[inds_stim], np.nanmean(data[inds_both], axis=1), '.', markersize=3, color= col_vec[i], label=fish_names[i]+' , n='+str(len(inds_fish)) )
        plt.legend()

    for i in range(n_gr): # plot the smoothed data off of the frist 4 blocks, and retest block
        inds_fish  = fish_ids[i]
        #inds_stim = 
        for k in range(5):
            inds_block = np.ix_((stim_given==1) | (stim_given==3))[0][k*nStimInBlocks:k*nStimInBlocks+nStimInBlocks]
            inds_both_block =  np.ix_(inds_block, inds_fish)

            y = np.nanmean(data[inds_both_block], axis=1) 
            x = time_inds[inds_block]
            # remove NaNs
            x = x[~np.isnan(y)]
            y = y[~np.isnan(y)]
            try:
                y = savgol_filter(y, smooth_window, 2)
                plt.plot(x,y, '-', color= col_vec[i], linewidth=5, alpha=0.8)
            except:
                warnings.warn('savgol did not converge')


    # plot taps
    if plot_taps:
        for i in range(n_gr):
            inds_fish  = fish_ids[i]
            inds_block = stim_given == 2
            inds_both_block =  np.ix_(inds_block, inds_fish)
            y1 = np.nanmean(data[inds_both_block], axis=1)
            plt.plot(time_inds[inds_block], y1, 'x', markersize=4, color= col_vec[i])

            try:
                y2 = savgol_filter(y1, smooth_window, 2)
                plt.plot(time_inds[inds_block], y2, '-', color= col_vec[i], linewidth=3, alpha=0.8)

            except:
                    warnings.warn('savgol did not converge')


    if not plot_retest:
        plt.xlim((-0.1,8))

    plt.rc('font', size=18)
    plt.rc('legend', fontsize=10)
    plt.savefig(save_name.replace('_trackdata.npy', ytext+'.svg').replace(' ', ''))
    plt.savefig(save_name.replace('_trackdata.npy', ytext+'.png').replace(' ', ''), dpi=100)
        
    
    plt.show()


def savfilt_trace(trace_data, sav_sz=11, sav_ord=3):
    # forward fill NaN values and filter with savgol
    trace_filt = savgol_filter(ffill_cols(trace_data), sav_sz, sav_ord, axis=0)
    return trace_filt


def ffill_cols(a, startfillval=0):
    ### fill NaN values with previous value
    mask = np.isnan(a)
    tmp = a[0].copy()
    a[0][mask[0]] = startfillval
    mask[0] = False
    idx = np.where(~mask,np.arange(mask.shape[0])[:,None],0)
    out = np.take_along_axis(a,np.maximum.accumulate(idx,axis=0),axis=0)
    a[0] = tmp
    return out

def get_speeds(centroid_trace, sav_sz=11, sav_ord=3, speed_thresh=15):
    diff_x = np.diff(savgol_filter(ffill_cols(centroid_trace[:,0,:]), sav_sz, sav_ord, axis=0), axis=0)
    diff_y = np.diff(savgol_filter(ffill_cols(centroid_trace[:,1,:]), sav_sz, sav_ord, axis=0), axis=0)
    speeds = np.sqrt(np.square(diff_x) + np.square(diff_y))
    speeds[speeds > speed_thresh] = 0
    speeds = savgol_filter(speeds, sav_sz, sav_ord, axis=0)
    return speeds

def convert_roi_str(roi_str):
    ### convert from a string that has a matlab stype indexing, using a mix of commans and colons, into a vector of indexes
    ### note that ROIs in the spreadsheet should be 1 indexed to match with old maltab formatting
    roi_str = roi_str.strip('[').strip(']').split(',')
    #print(roi_str)
    for k, roi_part in enumerate(roi_str):
        roi_part = roi_part.strip(' ')
        if roi_part.find(':') == -1: # if there is no colon, just a single number
            vec = int(roi_part)-1
        else:
            roi_subparts = roi_part.split(':')
            if len(roi_subparts) == 2:      # only 1 semicolon, count by ones                
                vec = np.arange(int(roi_subparts[0])-1, int(roi_subparts[1])) # subtract 1 from first index to make 0 indexed
            elif len(roi_subparts) == 3:    # 2 semicolons in matlab index style, jump by middle number
                vec = np.arange(int(roi_subparts[0])-1, int(roi_subparts[2]), int(roi_subparts[1]))
            else:
                raise ValueError('problem with ROI parsing for roi string' + str(roi_str))
        if k ==0:
            roi_vec = vec
        else:
            roi_vec = np.hstack((roi_vec, vec))
        
    return roi_vec

def get_serial_ports():
    """ Lists serial port names

        :raises EnvironmentError:
            On unsupported or unknown platforms
        :returns:
            A list of the serial ports available on the system
    """
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result

    
    



def CreateBkg (cam, bkgSec, FrameRate, FramesPerSec, window):
    nFrames_bkg = bkgSec * FramesPerSec
    print('computing background with ', nFrames_bkg , ' frames over ', bkgSec, ' seconds')
    image_result = cam.GetNextImage()
    frame = image_result.GetNDArray()
    height, width = np.shape(frame)
    data_bkg_arr = np.zeros((height, width, nFrames_bkg), dtype=float)
    k = 0
    imageNum = 0
    while k < nFrames_bkg:
        image_result = cam.GetNextImage()
        if imageNum % round(FrameRate/FramesPerSec) == 0:
            frame = image_result.GetNDArray()
            data_bkg_arr[:,:,k] = np.copy(frame)
            k+=1
        frame = cv2.putText(frame, 'Re-calculating background', (250,250), cv2.FONT_HERSHEY_COMPLEX, 1, 255)
        frame = image_resize(frame, width = 512)
        #cv2.imshow(displayWindow, frame)
        #cv2.waitKey(1)
        
        event, values = window.Read(timeout=1, timeout_key='timeout')      # get events for the window with 20ms max wait                   
        window['image'].Update(data=cv2.imencode('.png', frame)[1].tobytes()) # Update image in window
        
        image_result.Release()
        imageNum += 1
    img_bkg = np.percentile(data_bkg_arr, 75, axis = 2)
    
    print ("width: " + str(width) + ", height: " + str(height))
    plt.figure(figsize=(10,10))
    plt.imshow(img_bkg, cmap='gray', vmin=0, vmax = 255)
    plt.axis('off')
    plt.title('background image')
    plt.show()
    
    return img_bkg.astype('uint8')

def ProcessFrame (cam):
    image_result = cam.GetNextImage()
    imageID = image_result.GetFrameID()
    
        
    #image_converted = image_result.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
    frame = image_result.GetNDArray()
    image_result.Release()
    
    return frame, imageID

def TrackFrame(frame, img_bkg, thresh):
    
    diff = cv2.subtract(img_bkg, frame) 
    
    ret, binimage = cv2.threshold(diff,thresh,255,cv2.THRESH_BINARY)
    #binimage = cv2.morphologyEx(binimage, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binimage)
    
    
    if len(stats) > 1: 
        max_area_ind = np.argmax(stats[1:, cv2.CC_STAT_AREA])+1
        boundingrect = (stats[max_area_ind, cv2.CC_STAT_LEFT],
                        stats[max_area_ind, cv2.CC_STAT_TOP], 
                        stats[max_area_ind, cv2.CC_STAT_WIDTH], 
                        stats[max_area_ind, cv2.CC_STAT_HEIGHT])
        area = stats[max_area_ind, cv2.CC_STAT_AREA]
        centroid = centroids[max_area_ind]
        props = regionprops((labels == max_area_ind).astype('uint8'))
        props = props[0]
        orientation = props.orientation
        major_axis_length = props.major_axis_length
    else:
        boundingrect = (0,0,0,0)
        area = np.nan
        centroid = (np.nan, np.nan)
        orientation = np.nan
        major_axis_length = np.nan
    


    return diff, binimage, area, boundingrect, centroid, orientation, major_axis_length

def TrackFrame_wDots(frame, img_bkg, fishLen, thresh,  nSeg, tailThresh, SearchArc = np.pi/4):
    # takes an image and a background image for subtraction, then finds the fish using provided parameters and tried to reconstruct the tail segments
   
    kerSize = int(np.ceil(fishLen) // 2 * 2 + 1)
    sigma = int(np.ceil(fishLen/20))
    padding= int(fishLen)
    segLen = int(np.floor(fishLen/nSeg))
    AreaMin = fishLen/4
    AreaMax = fishLen*10
   
    # preallocate arrays
    pts_x = np.zeros(nSeg)
    pts_y = np.zeros(nSeg)
    pts_x[:] = np.nan
    pts_y[:] = np.nan
    
    # difference from background
    diff = cv2.subtract(img_bkg, frame) 

    # threshold
    ret, binimage = cv2.threshold(diff,thresh,255,cv2.THRESH_BINARY)
    #binimage = cv2.morphologyEx(binimage, cv2.MORPH_OPEN, np.ones((3,3)))
    
    # fine all the connected components
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binimage)
    
    # if we find blobs, assume the largest one is the fish for the first round or if we didnt find anything in the previous frame. Otherwise take the one that is closes to the previous centroid. 
    # process that blob
    if len(stats) > 1: 
        
        blob_ind = np.argmax(stats[1:, cv2.CC_STAT_AREA])+1
         # get area and centroid
        area = stats[blob_ind, cv2.CC_STAT_AREA]

#        # removed part for keeping closest centroid distance for now...
#        if np.isnan(lastCentroid[0]):
#            # use the biggest blob
#            blob_ind = np.argmax(stats[1:, cv2.CC_STAT_AREA])+1
#        else:
#            # use the closest to the previous centroid
#            centDists = np.sqrt(np.sum((centroids - lastCentroid)**2, axis=1))
#            blob_ind = np.argmin(centDists[1:])+1
#        
        
        
        
        if area >= AreaMin and area <= AreaMax:
            centroid = centroids[blob_ind]
            
                    
                
            # get bbox
            boundingrect = np.array((stats[blob_ind, cv2.CC_STAT_LEFT],
                            stats[blob_ind, cv2.CC_STAT_TOP], 
                            stats[blob_ind, cv2.CC_STAT_WIDTH], 
                            stats[blob_ind, cv2.CC_STAT_HEIGHT]))
            
            
            
            # pad the crop so that we can get pieces of the tail that might not survive the harsher thresholding
            bboxExpand = np.copy(boundingrect)
            bboxExpand[0] = boundingrect[0] - padding
            bboxExpand[1] = boundingrect[1] - padding
            
            # elements 2 and 3 will be the last index of the box, different from the cv2 bounding box convention
            bboxExpand[2] = bboxExpand[0] + boundingrect[2] + padding*2
            bboxExpand[3] = bboxExpand[1] + boundingrect[3] + padding*2
            
            # make sure we stay within image size
            im_height, im_width = np.shape(frame)
            bboxExpand[0] = max(bboxExpand[0], 0)
            bboxExpand[1] = max(bboxExpand[1], 0)
            bboxExpand[2] = min(bboxExpand[2], im_width-1)
            bboxExpand[3] = min(bboxExpand[3], im_height-1)
             
            # crop the image
            crop = diff[bboxExpand[1]:bboxExpand[3], bboxExpand[0]:bboxExpand[2]]
    #        plt.imshow(crop)
    #        plt.show()
            
            #crop(binimage[bboxExpand[1]:bboxExpand[3], bboxExpand[0]:bboxExpand[2]]) = 0
           
            # convolve the image
            
            #conv = cv2.filter2D(crop, cv2.CV_32F,  np.ones((kerSize,kerSize))/(kerSize*kerSize))
            conv = cv2.GaussianBlur(crop,ksize=(kerSize,kerSize), sigmaX=sigma)
            
            # brightest pixel should be between the dark eyes
            headCoor = np.unravel_index(np.argmax(conv, axis=None), conv.shape)
            
            # use the center of the bbox to define the initial search direction
            crop_height, crop_width = np.shape(conv)
            cent_bbox = (crop_height/2, crop_width/2)
    
            #define the angle, along which to search for the desired points
            
            search_angle = np.arctan2(cent_bbox[0]-headCoor[0], cent_bbox[1]-headCoor[1])
            search_angles = np.zeros(nSeg-1)
            search_angles[:] = np.nan
            
            
            #Assign to the first coordinates in x,y coord arrays, the head coordinates
            pts_x[0] = headCoor[1];
            pts_y[0] = headCoor[0];
            
            
    
        # serach in arcs for tail
        #%
            for i in range(nSeg-1):
                        
                # update search direction for subsequent points
                if i > 0:
                    search_angles[i-1] = np.copy(search_angle)
                    search_angle = np.arctan2(pts_y[i]-pts_y[i-1], pts_x[i]-pts_x[i-1])
                    
                    
                # calculate search coordinates    
                ang = np.arange(search_angle-SearchArc,search_angle+SearchArc,np.pi/20)
                ArcCoors =pts_y[i] + np.sin(ang)*segLen, pts_x[i] + np.cos(ang)*segLen
                ArcCoors = np.asarray(ArcCoors, dtype=int)
                
                
                indRem = np.argwhere(np.logical_or.reduce(( ArcCoors[0] <=0, ArcCoors[0] >= crop_height, ArcCoors[1] <=0, ArcCoors[1] >= crop_width))).flatten()
                ArcCoors = np.delete(ArcCoors, indRem, 1)

                
                if np.sum(ArcCoors) == 0: # if we found no valid points break the loop
                    break
                
                # find brightest pixel in smoothed image along arc
                pt_max = np.max(conv[ArcCoors[0], ArcCoors[1]])
                pt_ind = np.argmax(conv[ArcCoors[0], ArcCoors[1]])
                pt_y = ArcCoors[0][pt_ind]
                pt_x = ArcCoors[1][pt_ind]
                
          
                # assign this as the center of the tail if the pixel is above a threshold of intesnity (this is to avoid assigning points to space not actually occupied by the fish, like if it is short, or we have gotten off track)
                if pt_max > tailThresh:
                    pts_y[i+1] = pt_y
                    pts_x[i+1] = pt_x
                else: # break if we didnt find a good point on the tail
                    break
            
            # determine the angles between segments for curvature and orientation calculations
            
            angles = np.unwrap(np.arctan2(np.diff(pts_y), np.diff(pts_x)))
            
            #angles[angles == 0] = np.nan # I dont know why this is necessary, but some 0s were sneaking in that shouldnt be there. 
             
            orient = np.nanmean(angles); 
            diffangles = np.diff(angles)
           # diffangles[abs(diffangles) > 1] = 0 # max of 1 radian per segment
            bendAmp = np.nansum(diffangles)
            
            orient = (np.rad2deg(orient) + 360) % 360 # number between 0 and 360 for orientation
            
            pts_x = pts_x + bboxExpand[0]
            pts_y = pts_y + bboxExpand[1]
            
            
        else:
            #print('wrong area')
            bboxExpand = (0,0,0,0)
            boundingrect = (0,0,0,0)
            centroid = (np.nan, np.nan)
            bendAmp = np.nan
            orient = np.nan
    else:
        #print('no blobs')
        bboxExpand = (0,0,0,0)
        boundingrect = (0,0,0,0)
        area = np.nan
        centroid = (np.nan, np.nan)
        bendAmp = np.nan
        orient = np.nan
        
    

    return diff, binimage, area, boundingrect, bboxExpand, centroid, pts_x, pts_y, bendAmp, orient



def UpdateBkg (img_bkg, frame, turnProp, boundingrect):
    img_bkg_new = img_bkg.astype(float)*(1-turnProp) + np.copy(frame).astype(float) * turnProp
    img_bkg_new = np.round(img_bkg_new).astype('uint8')
    
    # dont update the bbox of the found object
    for i in range(boundingrect.shape[1]):
        img_bkg_new[boundingrect[1, i]:boundingrect[1, i]+boundingrect[3, i], boundingrect[0, i]:boundingrect[0, i]+boundingrect[2, i]] = img_bkg[boundingrect[1, i]:boundingrect[1, i]+boundingrect[3, i], boundingrect[0, i]:boundingrect[0, i]+boundingrect[2, i]]
    
        
    return img_bkg_new

def UpdateDisplay_cv2 (frame, boundingrect, binimage):
    disp = np.hstack((cv2.rectangle(frame, boundingrect, 255 ,2), binimage))

    cv2.imshow('ImageWindow', disp)
    z = cv2.waitKey(1)
    if z == ord('q'):#press q to quit
        return False
        cv2.destroyAllWindows()    
    elif cv2.getWindowProperty('ImageWindow',cv2.WND_PROP_VISIBLE) < 1:        
        return False
        cv2.destroyAllWindows()
    else:
        return True

def UpdateDisplay_sg(frame, boundingrect, binimage, window, centroid, orientation, major_axis_length):
    if not np.isnan(orientation): # if we found something, draw the bounding box
        x0, y0= centroid
        

        x1 = int(x0 + math.cos(orientation) * 0.5 * major_axis_length)
        y1 = int(y0 - math.sin(orientation) * 0.5 * major_axis_length)
    
        frame = cv2.arrowedLine(frame, (int(x0),int(y0)), (x1,y1), 255)
    # display the image
    disp = np.hstack((cv2.rectangle(frame, boundingrect, 255 ,2), binimage)) 
    event, values = window.Read(timeout=1, timeout_key='timeout')      # get events for the window with 20ms max wait                   
    window.FindElement('image').Update(data=cv2.imencode('.png', disp)[1].tobytes()) # Update image in window
    
    if event is None:
        return False
    else:
        return True
    

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def UpdateLights(frames_till_df, isi_frames, lightVal, len_df_frames, frames_till_df_over, bendAmpPow, threshVal, nFlashesDone, nFlashes):
    lightValChange = False
    if lightVal == 0: # lights are off, we are in a dark flash, 
        
        frames_till_df -= 1 # decrement the timer to keep the ISI constant
        
        if frames_till_df_over == 0: # if dark flash timer runs out
            lightVal = 1
            lightValChange = True
            frames_till_df_over = len_df_frames
            nFlashesDone += 1
        elif abs(bendAmpPow) > threshVal and nFlashesDone < nFlashes: # if this fish crosses threshold, and we are still in closed loop, turn lights back on
            lightVal = 1
            lightValChange = True
            frames_till_df_over = len_df_frames
            nFlashesDone += 1            
        else: # decrement the dark flash timer
            frames_till_df_over -= 1
            
        
    elif frames_till_df == 0: # lights are on, check if it is time to dark flash
        # turn lights off and reset counters if it is time
        lightVal = 0
        lightValChange = True
        frames_till_df = isi_frames-1
    
    else: # decrement the dark flash ISI timer
        frames_till_df -= 1
    
    return lightVal, frames_till_df, frames_till_df_over, lightValChange, nFlashesDone

def UpdateLights_Orientation(frames_till_df, isi_frames, lightVal, len_df_frames, frames_till_df_over, orient, orientThresh, orientAtOff, nFlashesDone, nFlashes, len_df_frames_open):
    lightValChange = False
    if lightVal == 0: # lights are off, we are in a dark flash, 
        
        frames_till_df -= 1 # decrement the timer to keep the ISI constant
        deltaOrient = np.min((abs(orientAtOff - orient), 360-abs(orientAtOff - orient) ))
        
        if frames_till_df_over == 0: # if dark flash timer runs out, turn lights back on
            lightVal = 1
            lightValChange = True
            nFlashesDone += 1
            
            # reset the timer for the next dark flash. if we will be in open loop test flashes, use the open loop dark flash length
            if nFlashesDone < nFlashes:
                frames_till_df_over = len_df_frames
            else:
                frames_till_df_over = len_df_frames_open
            
            
            
        elif deltaOrient > orientThresh and nFlashesDone < nFlashes: # if we cross threshold, and we are still doing closed loop flashes turn the lights back on 
            lightVal = 1
            lightValChange = True
            nFlashesDone += 1
            
            # reset the timer for the next dark flash. if we will be in open loop test flashes, use the open loop dark flash length
            if nFlashesDone < nFlashes:
                frames_till_df_over = len_df_frames
            else:
                frames_till_df_over = len_df_frames_open
            
            
        else: # decrement the dark flash timer
            frames_till_df_over -= 1
            
    
    # lights are on
    elif frames_till_df == 0: # check if it is time to dark flash
        # turn lights off and reset counters if it is time, record the current orientation
        lightVal = 0
        lightValChange = True
        frames_till_df = isi_frames-1
        orientAtOff = np.copy(orient)
    
    else: # decrement the dark flash ISI timer
        frames_till_df -= 1
    
    return lightVal, frames_till_df, frames_till_df_over, lightValChange, orientAtOff, nFlashesDone



def add_arrows_and_text(im, headCoors_frame, orientations_frame, text_vec=[None], segLen=30, brightness_arrow = 25, brightness_text = 25):

    im_sl = np.copy(im)
    angles = orientations_frame[:] - np.pi
    angles[np.isnan(angles)] = 0
    head_y = headCoors_frame[1,:]
    head_x = headCoors_frame[0,:]
    head_y = head_y[~np.isnan(head_x)].astype(np.int64)
    angles = angles[~np.isnan(head_x)]
    head_x = head_x[~np.isnan(head_x)].astype(np.int64)

    for j in range(len(head_y)):
        try:
            tip_y, tip_x =  head_y[j] + np.sin(angles[j])*segLen, head_x[j] + np.cos(angles[j])*segLen
            im_sl = cv2.arrowedLine(im_sl, (head_x[j], head_y[j]), (int(tip_x), int(tip_y)),(brightness_arrow,brightness_arrow,brightness_arrow), 2, tipLength=0.3)
            if not text_vec[0] == None:
                disp_val = text_vec[j]
                try:
                    disp_val = str(int(disp_val))
                except:
                    disp_val = 'nan'
                im_sl = cv2.putText(im_sl, disp_val, (head_x[j], head_y[j]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (brightness_text,brightness_text,brightness_text), 2, cv2.LINE_AA)
        except:
            print('error drawing lines???')

    return im_sl

#%


def find_blobs(im_track, bkg_im, thresh = 2, fish_len=40):
    # track the potition of each fish within the ROI, based on background subtraction. Determine heading angle based on angle between coordinate of the centroid and the dimmist pixel in the thresholded blob after convlolution -- generally the poitn between the two eyes
    bkg_im = bkg_im.astype('uint8')
    kerSize = int(np.ceil(fish_len/5) // 2 * 2 + 1)
    sigma = int(np.ceil(fish_len/25))
    open_kerSize = int(np.floor(fish_len/20))

    # backgroudn subtract, threshold, open, find connected components
    diff = cv2.subtract(bkg_im,im_track)
    ret, binimage = cv2.threshold(diff,thresh,255,cv2.THRESH_BINARY)
    binimage = cv2.morphologyEx(binimage, cv2.MORPH_OPEN, np.ones((open_kerSize,open_kerSize)))

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binimage, connectivity=4)
    stats = stats[1:] # first component is backgorund, remove it
    centroids = centroids[1:]
    # convlove background subtracted image, to be used for finding head coordinate
    conv = cv2.GaussianBlur(diff,ksize=(kerSize,kerSize), sigmaX=sigma)

    return binimage, stats, centroids, conv
#%
@jit(nopython=True, cache=True)
def get_heading_and_head(binimage, stats, centroids, conv, rois, fish_len=40, padding = 2):
    # preallocate arrays
    height, width = binimage.shape
    n_rois = np.max(rois)
    min_area = fish_len
    max_area = fish_len * 10
    centroids = centroids.astype(np.int64)
    centroids_final = np.empty((n_rois, 2))
    centroids_final[:] = np.nan
    headCoor_final = np.empty((n_rois, 2))
    headCoor_final[:] = np.nan
    heading_dir_final = np.empty(n_rois)
    heading_dir_final[:] = np.nan
    stats_final = np.empty((n_rois, stats.shape[1]))
    stats_final[:] = np.nan
    
    # determine the ROI where each centroid is. If more than one centroid detected in an ROI, assume the one that the fish in is the Oth one. 
    
    n_cents = len(centroids[:,1])
    roi_label = np.empty(n_cents, dtype='int')
    for ind_val in range(n_cents):
        roi_label[ind_val] = rois[centroids[ind_val,1], centroids[ind_val,0]]


    # loop and select the largest ROI as the fish, if more than one are found
    for i in range(n_rois):
        hits = np.where(roi_label == i+1)[0]
        if len(hits) > 0 and np.max(stats[hits, cv2.CC_STAT_AREA]) > min_area: # only accept if area is larger than area min to avoid tracking noise when fish is not visible. 
            if len(hits) == 1:
                hit = hits[0]
            else:
                hit =  int(hits[np.argmax(stats[hits, cv2.CC_STAT_AREA])])
        
            if stats[hit, cv2.CC_STAT_AREA] < max_area: # check if IDd blob is within max bounds

                stats_hit = (stats[hit, 0], stats[hit, 1], stats[hit, 2], stats[hit, 3], stats[hit, 4]) 
                stats_final[i] = stats_hit
                
                # bounding box of blob
                bbox =  (
                    max(stats_hit[cv2.CC_STAT_TOP] - padding, 0),
                    min(stats_hit[cv2.CC_STAT_TOP] + stats_hit[cv2.CC_STAT_HEIGHT] + padding, height),
                    max(stats_hit[cv2.CC_STAT_LEFT] - padding, 0),
                    min(stats_hit[cv2.CC_STAT_LEFT] + stats_hit[cv2.CC_STAT_WIDTH] + padding, width),
                    )

                
                fish_crop = conv[int(bbox[0]):int(bbox[1]), int(bbox[2]): int(bbox[3])]
                
                # plt.imshow(fish_crop)
                # plt.show()
                centroids_final[i,:] = centroids[hit].flatten()
                cent_crop = (centroids_final[i,0] - bbox[2], centroids_final[i,1] -bbox[0])
                
                # use the brigthes point on the convolved image as the head, usually between the two eyes

                #headCoor = np.unravel_index(np.argmax(fish_crop, axis=None), fish_crop.shape)
                max_val = np.max(fish_crop)
                headCoor_vecs = np.where(fish_crop==max_val)
                headCoor = int(np.mean(headCoor_vecs[0])), int(np.mean(headCoor_vecs[1]))
                # head_id = np.argmax(fish_crop)
                # crop_h, crop_w = fish_crop.shape
                # headCoor = (int(np.floor(head_id/crop_w)), head_id%crop_w)
                
                headCoor_final[i,:] = (int(headCoor[1]) + bbox[2], int(headCoor[0]) + bbox[0])

                # angle between centroid and head point taken as fish heading orientation
                heading_dir_final[i] =  np.arctan2(cent_crop[1]-headCoor[0], cent_crop[0]-headCoor[1])

    return centroids_final, headCoor_final, heading_dir_final, stats_final


@jit(nopython=True, cache=True)
def get_pts_from_heading_and_head(conv_im, headCoors, heading_dirs, fish_len = 40,n_points = 7,thresh = 3, search_degrees = np.pi/2.5):

    # uses the pre-computed convolved image, head coordinates and orientation to serach in arcs are reconstruct points along the tail defined by the brightest spot in the convolved image
    height, width = conv_im.shape
    n_rois = headCoors.shape[0]
    seg_len = fish_len/(n_points-1)
    pts_x = np.empty((n_rois, n_points))
    pts_x[:] = np.nan
    pts_y = np.empty((n_rois, n_points))
    pts_y[:] = np.nan
    bend_amps = np.empty(n_rois)
    bend_amps[:] = np.nan
    orients = np.empty(n_rois)
    orients[:] = np.nan

    search_angles = np.zeros(n_points)
    

    for fish in range(n_rois):
        headCoor = headCoors[fish,:].astype(np.int64)
        search_angle = heading_dirs[fish]


        if not np.isnan(search_angle):
            pts_x[fish, 0] = headCoor[0]
            pts_y[fish, 0] = headCoor[1]

            for i in range(n_points-1):
                                        
                                # update search direction for subsequent points
                if i > 0:
                    search_angles[i-1] = search_angle
                    search_angle = np.arctan2(
                        pts_y[fish,i]-pts_y[fish,i-1], 
                        pts_x[fish,i]-pts_x[fish,i-1]
                        )

                ang = np.arange(search_angle-search_degrees,search_angle+search_degrees,np.pi/30)
                ArcCoors =(pts_y[fish, i] + np.sin(ang)*seg_len).astype(np.int64), (pts_x[fish, i] + np.cos(ang)*seg_len).astype(np.int64)
                
                
                indKeep = np.logical_and(
                        np.logical_and(ArcCoors[0] >=0 ,  ArcCoors[0] <= height) ,
                        np.logical_and(ArcCoors[1] >=0 ,  ArcCoors[1] <= width) 
                        )
                    
                
                ArcCoors = (ArcCoors[0][indKeep], ArcCoors[1][indKeep])
                n_good = np.sum(indKeep)
                if n_good == 0: # if we found no valid points break the loop
                    break
                # find brightest pixel in smoothed image along arc
                #arc_vals = conv_im[ArcCoors[0], ArcCoors[1]]

                #%% loop for numba
                arc_vals = np.empty(n_good, dtype='uint8')
                for ind_val in range(n_good):
                    arc_vals[ind_val] = conv_im[ArcCoors[0][ind_val], ArcCoors[1][ind_val]]

                
                pt_max = np.max(arc_vals)
                #pt_ind = np.argmax(arc_vals)
                pt_ind = int(np.mean(np.where(arc_vals == pt_max)[0]))
                pt_y = ArcCoors[0][pt_ind]
                pt_x = ArcCoors[1][pt_ind]
                
                # assign this as the center of the tail if the pixel is above a threshold of intesnity (this is to avoid assigning points to space not actually occupied by the fish, like if it is short, or we have gotten off track)
                if pt_max > thresh:
                    pts_y[fish, i+1] = pt_y
                    pts_x[fish, i+1] = pt_x
                else: # break if we didnt find a good point on the tail
                    break
                        
            # determine the angles between segments for curvature and orientation calculations
        tail_coords = (pts_x, pts_y)

    return tail_coords

@jit(nopython=True, cache=True)
def get_bendamps(tail_coords):
    n_rois = len(tail_coords[0]) 
    orients = np.empty((n_rois))
    orients[:] = np.nan

    bend_amps = np.empty((n_rois))
    bend_amps[:] = np.nan

    for fish in range(n_rois):
            x = tail_coords[0][fish, :]
            y = tail_coords[1][fish, :]
            x[x==0] = np.nan
            y[y==0] = np.nan

            angles = np.arctan2(np.diff(y), np.diff(x))
            
            angles = angles[~np.isnan(angles)]

            n_ang = len(angles)
            if n_ang > 1:
                orients[fish] = np.mean(angles)
                diff_angles = np.diff(angles)

                while np.max(np.abs(diff_angles)) > np.pi:
                    angles[np.where(diff_angles > np.pi)[0] + 1] = angles[np.where(diff_angles > np.pi)[0] + 1] - 2*np.pi
                    angles[np.where(diff_angles <-np.pi)[0] + 1] = angles[np.where(diff_angles <- np.pi)[0] + 1] + 2*np.pi
                    diff_angles = np.diff(angles)
                        
                
                bend_amps[fish] = np.sum(diff_angles)

    return orients, bend_amps

@jit(nopython=True, cache=True)
def get_head_and_tail(binimage, stats, centroids, conv, rois, fish_len=40, padding = 2, n_points=7, tail_thresh=3, search_degrees = np.pi/2.5):
    
    centroids_final, headCoors, heading_dir, stats_fish = get_heading_and_head(binimage, stats, centroids, conv, rois, fish_len=fish_len, padding = padding)
    tail_coords = get_pts_from_heading_and_head(conv, headCoors, heading_dir, fish_len = fish_len,n_points = n_points,thresh = tail_thresh, search_degrees = search_degrees)
    orients, bend_amps = get_bendamps(tail_coords)
    
    return tail_coords, orients, heading_dir, bend_amps, stats_fish

def make_bkg_mask(stats, height, width, padding=7):
    include_image = np.zeros((height, width), dtype=bool)
    include_image[:] = True
    for i in range(len(stats)):
        stats_hit = stats[i,:]
        if not np.isnan(stats_hit[0]):
            bbox =  (
                max(stats_hit[cv2.CC_STAT_TOP] - padding, 0),
                min(stats_hit[cv2.CC_STAT_TOP] + stats_hit[cv2.CC_STAT_HEIGHT] + padding, height),
                max(stats_hit[cv2.CC_STAT_LEFT] - padding, 0),
                min(stats_hit[cv2.CC_STAT_LEFT] + stats_hit[cv2.CC_STAT_WIDTH] + padding, width),
                )
            include_image[int(bbox[0]):int(bbox[1]), int(bbox[2]): int(bbox[3])] = False
    return include_image
