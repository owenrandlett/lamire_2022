#%% Script for aligning the imaging plane data to the anatomy stack, and then registering the coordinates for the ROIs to the Z-Brain
# 
#  make sure we are running in 'imaging' and not 'suite2p' environment, or else we do not have the right skimage packages. 

import h5py, os, tqdm, nrrd, subprocess, glob, napari

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from skimage.registration import phase_cross_correlation
from natsort import natsorted
from scipy.ndimage import zoom, shift
import tifffile
saveRoot = os.path.realpath(r'/mnt/md0/suite2p_output/')

ref_brain = os.path.realpath(r'/media/BigBoy/ciqle/ref_brains/HuC-H2BRFP_ZBrain.nrrd')
images_fld = os.path.realpath(r'/mnt/md0/registration/images')
folders = natsorted(glob.glob(saveRoot + '/*033*_*fish*'))[-1::-1]
folders = folders+ natsorted(glob.glob(saveRoot + '/*_*fish*'))[-1::-1]
print(folders)

#%
def determine_z_planes(IM_anat, f_h5, folder, zoom_factor = 0.25, n_blocks = 5):
    n_planes_anat = IM_anat.T.shape[0]
    pre_reg_zoom = zoom(IM_anat.T, [1, zoom_factor, zoom_factor])
    dset = f_h5['data']
    n_frames, n_planes, height, width = dset.shape
    n_blocks = 5
    window_size = np.floor(n_frames/n_blocks).astype(int)
    skip_size = np.floor(window_size/50).astype(int)
    shifts = np.zeros((n_planes, n_blocks, 3))
    
    for plane in tqdm.trange(n_planes):
        for block in range(n_blocks):

            IM = dset[block*window_size:block*window_size+window_size:skip_size, plane, :, :]
            IM = np.sum(IM, axis=0)
            IM = zoom(IM, zoom_factor)
            
            errors_blocks = np.zeros(n_planes_anat)
            shifts_blocks = np.zeros((n_planes_anat, 2))
            for slice in range(n_planes_anat):
                shifts_blocks[slice, :], errors_blocks[slice], phasediff = phase_cross_correlation(pre_reg_zoom[slice,:,:],IM, normalization=None)
            best_arg = np.argmin(errors_blocks)
            shifts[plane, block, 0] = best_arg
            shifts[plane, block, 1:] = shifts_blocks[best_arg,:]
    fig, ax = plt.subplots(2,1, figsize=(10,20))
    x = window_size/2+np.arange(n_blocks)*window_size
    ax[0].set_title(folder)
    ax[0].plot(x, shifts[:, :,0].T)

    ax[0].set_ylabel('z coordiante',fontsize=16)
    ax[0].set_xlabel('frames',fontsize=16)

    
    ax[1].plot(x, np.median(shifts[:,:,0]-np.median(shifts[:,0,0]), axis=0))
    ax[1].set_ylabel('delta median \nz coordinate',fontsize=16)
    ax[1].set_ylim(-14,14)
    ax[1].set_xlabel('frames',fontsize=16)
    ax[1].hlines([-3.5,0, 3.5], 0,x[-1], colors = ['r', 'k', 'r'], linestyle='dashed')
    plt.savefig(os.path.join(folder, 'Z-position.svg'))
    plt.show()


#%%
#% analyze z shifts first for functional stack relative to anatomy stack, and output a plot of the z-plane, and prepare the anatomy stack for cmtk registration 
for folder in folders:
    anat_stack = os.path.join(folder, 'AnatStack.nrrd')
    f_h5 = h5py.File(folder+r'/func_data.h5', 'r')
    print(folder)

    IM_anat, meta = nrrd.read(anat_stack)
    determine_z_planes(IM_anat, f_h5, folder, zoom_factor = 0.5, n_blocks = 5)
    f_h5.close()

    ## rearrange the anatomy stack to fit with z-brain dimension order for cmtk registration

    IM_anat_rot = IM_anat + 1 # add 1 so all imaged pixels are > 0 
    IM_anat_rot[:,:,0:2] = 0 # remove data from first slices because of moving Z artifact
    IM_anat_rot = IM_anat_rot[:,:,-1::-1] # reverse Z axis
    IM_anat_rot = np.moveaxis(IM_anat_rot,[0,1], [1,0]) # flip x and y

    # write nrrd of anatomy stack for CMTK registration, in correct dimension order
    anat_name = os.path.join(images_fld, os.path.split(folder)[1]+'_01.nrrd')
    nrrd.write(anat_name, IM_anat_rot, meta)


#%% run CMTK on the images folder direcotry, registering everything to Z-brain coordinates 
root_dir = os.path.split(images_fld)[0]


cmd = 'cd ' + root_dir + ' && /home/lab/cmtk/build/bin/munger -v -awr 0102 -X 52 -C 8 -G 80 -R 3 -A "--accuracy 0.4" -W "--accuracy 1.6" -s '+ ref_brain + ' "images"'
print(cmd)

pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
text =pipe.communicate()[0]
print(text)


#% run streamxform on coordinates from using from suite2p planes, output 'plane_data.npy' file for each registered plane for downstream analysis 

# physical size of voxels
xy_rez_mv = meta['spacings'][0]
z_rez_mv = meta['spacings'][2]

xy_rez_zb = 0.798
z_rez_zb = 2

im_ref_ori, meta_ref = nrrd.read(ref_brain)
#


for folder in tqdm.tqdm(folders):
    folder = os.path.realpath(folder)
    anat_name = os.path.join(images_fld, os.path.split(folder)[1] + '_01.nrrd')
    IM_anat_rot, meta = nrrd.read(anat_name)
    IM_anat_rot = np.moveaxis(IM_anat_rot, [0, 1], [1, 0])

    im_ref = im_ref_ori.copy().T / 2
    planes = natsorted(glob.glob(folder+'/suite2p/plane*'))
    fish_name = os.path.split(folder)[1]
    if len(planes)> 0:

        for plane in planes:

            # load plane info
            try:
                ops = np.load(os.path.join(plane, 'ops.npy'), allow_pickle=True).item()
                F = np.load(os.path.join(plane, 'F.npy'), allow_pickle=True)
                iscell = np.load(os.path.join(plane, 'iscell.npy'), allow_pickle=True)
                roi_stats = np.load(os.path.join(plane, 'stat.npy'), allow_pickle=True)
                n_rois = len(roi_stats)
            except:
                print('!!!!! suite 2p files missing for ' + plane)
                continue
            
            IM = ops['refImg'].T

            #%
            # determine imaging plane
            n_slices = IM_anat_rot.shape[2]

            shifts = np.zeros((n_slices, 2)) # order will be y, x, z
            errors = np.zeros(n_slices)
            phasediff = np.zeros(n_slices)
            anat_offset = np.zeros(3)
            for slice in range(n_slices):
                shifts[slice,:], errors[slice], phasediff[slice] = phase_cross_correlation(IM_anat_rot[:,:,slice],IM,  normalization=None)

            #%
            z_coord = np.nanargmin(errors)
            anat_offset[:2] = shifts[z_coord, :]
            anat_offset[2] = z_coord
            print(plane + '  ... offsets ...')
            print(anat_offset)
            ops['anat_stack_offsets'] = anat_offset
            # shifts[plane, block, 0] = best_arg
            # shifts[plane, block, 1:] = shifts_blocks[best_arg,:]


            height, width = IM.shape 
            roi_centroids = np.zeros(n_rois)

            px_before = np.zeros((n_rois, 3), dtype='float')
            im_cents = np.copy(IM_anat_rot[:,:, z_coord])
            for i in range(n_rois):
                # shift x/y appropriately
                x_coord = roi_stats[i]['med'][0] + anat_offset[1] # x and y dimension from suite2p are swapped to account for z-brain coordinate ordering (L/R may be inverted... not sure!)
                y_coord = roi_stats[i]['med'][1] + anat_offset[0]
                if x_coord < width and y_coord < height:
                    im_cents[int(y_coord), int(x_coord)] = 65535
                px_before[i, 0] =  x_coord * xy_rez_mv
                px_before[i, 1] =  y_coord * xy_rez_mv
                px_before[i, 2] = z_coord * z_rez_mv

            # plt.figure(figsize = [50,15])
            # plt.imshow(im_cents)
            # plt.show()

            pxMapName = os.path.join(plane, 'cents_orig.txt')
            pxMapName_out = pxMapName.replace('cents_orig.txt', 'cents_inzbrain.txt')
            np.savetxt(pxMapName, px_before, fmt='%1.3f', delimiter=' ')
            # %
            streamxform = '/home/lab/cmtk/build/bin/streamxform '
            reg_base_dir = os.path.join(os.path.split(images_fld)[0], 'Registration', 'warp')
            reg_files = glob.glob(reg_base_dir + '/*' + os.path.split(folder)[1] + '*.list')
            if len(reg_files) == 0:
                print('didnt find registered file, streamxform will fail')
            elif len(reg_files) > 1:
                print('found more than one registration file, using first one found')

            reg_file = reg_files[0]
            cmd = streamxform + ' -- --inverse ' + reg_file + ' < ' + pxMapName + ' > ' + pxMapName_out

            pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            text =pipe.communicate()[0]
            print(text)
            #%
            cents_zbrain = np.genfromtxt(pxMapName_out, comments='FAILED') # I need to re-do the registrations on a bigger template to make sure that there are not out of boudns pixels (AKA failed)
            cents_zbrain_px = cents_zbrain/(xy_rez_zb, xy_rez_zb, z_rez_zb)
            cents_zbrain_px = cents_zbrain_px.astype(int)



            #%
            for roi in range(n_rois):
                roi_stats[roi]['centroid_zbrain'] = cents_zbrain_px[roi,:]
                roi_stats[roi]['ypix_zbrain'] = roi_stats[roi]['ypix'] - np.mean(roi_stats[roi]['ypix']) + cents_zbrain_px[roi,1]
                roi_stats[roi]['xpix_zbrain'] = roi_stats[roi]['xpix'] - np.mean(roi_stats[roi]['xpix']) + cents_zbrain_px[roi,0]
                
                #im_zbrain_space[cents_zbrain_px[roi, 2],cents_zbrain_px[roi,1], cents_zbrain_px[roi, 0]] = 65535

                try:
                    im_ref[
                        roi_stats[roi]['centroid_zbrain'][2].astype(int), 
                        roi_stats[roi]['ypix_zbrain'].astype(int),
                        roi_stats[roi]['xpix_zbrain'].astype(int)
                        ] = 65535
                except:
                    continue
            plane_data = {
                'name':fish_name,
                'plane':os.path.split(plane)[1],
                'ops':ops,
                'roi_stats':roi_stats,
                'F':F,
                'iscell':iscell
                }

            np.save(os.path.join(folder, os.path.split(folder)[1]+'_'+os.path.split(plane)[1]+'_data.npy'), plane_data)
        #%
        plt.figure(figsize = [50,15])
        plt.imshow(np.mean(im_ref, axis=0))
        plt.show()

        plt.figure(figsize = [50,15])
        plt.imshow(np.rot90(np.mean(im_ref, axis=2)))
        plt.show()
    else:
        print('didnt find any planes for folder: ' + folder)

#%%%%%%%%%%%%%%%%%%%%%%%%%
# below are some extra analyses for looking at the data


#% look at raw data with napari
import napari
folder = folders[0]
print(folder)
f_h5 = h5py.File(folder+r'/func_data.h5', 'r')
dset = f_h5['data']
viewer = napari.view_image(dset)


#%% make a delta F/F movie for dark flashes

folder = folders[-3]
print(folder)

#%
os.chdir(folder)

stim_df = np.load('stimvec_mean.npy')

# find the start of the dark flashes as rises in the kernel
df_start_inds = np.where(np.diff(stim_df)>0.1)[0]
df_start_inds = np.delete(df_start_inds, np.where(np.diff(df_start_inds) == 1)[0]+1)

f_h5 = h5py.File('func_data.h5', 'r')

data = f_h5['data']


before = 20
after = 80
Fresp = data[df_start_inds[0]-before:df_start_inds[0]+after, :, :].astype('double')

for i in tqdm.tqdm(range(len(df_start_inds)-1)):
    Fresp = Fresp + data[df_start_inds[i+1]-before:df_start_inds[i+1]+after, :, :].astype('double')

Fresp = Fresp/i

#%%
dF = np.zeros(Fresp.shape)
F = np.mean(Fresp[5:15,:,:,:], axis=0)
for i in tqdm.tqdm(range(dF.shape[0])):
    dF[i,:,:,:] = (Fresp[i,:,:,:] - F) + 1 / (F + 1)
dFpos = np.copy(dF)
dFpos[dFpos < 0] = 0

dFneg= -np.copy(dF)
dFneg[dFneg < 0] = 0
#%

viewer = napari.Viewer()

viewer.add_image(
    F,
    scale = [10, 0.5799, 0.5799],
    name = 'F'
    )
viewer.add_image(
    dFpos,
    name = 'dF/F-Pos',
    scale = [1, 10, 0.5799, 0.5799]
    )
viewer.add_image(
    dFneg,
    name = 'dF/F-Neg',
    scale = [1, 10, 0.5799, 0.5799]
    )


