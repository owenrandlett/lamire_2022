#%%

# script to process anatomical stack taken before the Ca2+ imaging run, and then to run suite2p on the functional stack.


from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np

import suite2p
import xml.etree.ElementTree as ET

import glob
import os
import pandas as pd
import nrrd
from tifffile import imread, imsave
from PIL import Image
import tqdm
from natsort import natsorted
import h5py

from scipy.ndimage import zoom
#%

x_rez = 0.5799 # in microns
y_rez = 0.5799

z_rez_anat = 1
z_rez_func = 10
imaging_dir = Path.cwd()

#%
dataRoots = [
    os.path.realpath(r'/media/BigBoy/ciqle/2p/20220316*'), 
    #os.path.realpath(r'/media/BigBoy/ciqle/2p/20220309*')
    ]

dataDirs = natsorted(glob.glob(dataRoots[0]+'/*fish*/'))

#%
if len(dataRoots) > 1:
    for root in range(len(dataRoots)-1):
        dataDirs = dataDirs + natsorted(glob.glob(dataRoots[root+1]+'/*fish*/'))
#%
dataDirs = dataDirs[-1::-1]
print(dataDirs)
saveRoot = os.path.realpath(r'/mnt/md0/suite2p_output/')




for dataDir in dataDirs:

    os.chdir(dataDir)
    saveDir = os.path.join(saveRoot, os.path.basename(os.path.normpath(dataDir)))
    Path(saveDir).mkdir(parents=True, exist_ok=True)
    tiff_files = natsorted(glob.glob('*_Ch2_*'))

    #%
    first_stack = [s for s in tiff_files if "_Cycle00001_Ch2_" in s]

    n_planes_anat = len(first_stack)

    # get the stimulus file to save timestamps and voltage recording
    xml = glob.glob('*.xml')[0] # first one seems to be the relevant one

    stim_file_name = glob.glob('*VoltageRecording*.csv')[0]
    print(stim_file_name)
    n_cyc_anat = 128
    
    #%
    sl = imread(first_stack[-1])
    y_size, x_size = sl.shape

    anat_stack = os.path.join(saveDir,'AnatStack.nrrd')

    if Path(anat_stack).is_file():
        print('skipping anat stack writing__already exists')
        IM_anat, meta = nrrd.read(anat_stack)
        IM_anat = IM_anat.T
    else:
    
        IM = np.zeros((n_planes_anat, y_size, x_size), dtype='float')
        k = 0
        for i in tqdm.tqdm(range(n_cyc_anat)):
            for j in range(n_planes_anat):
                sl_name = tiff_files[k]
                IM[j,:,:] = IM[j,:,:] + Image.open(sl_name)
                k+=1
            # plt.imshow(IM[50,:,:])
            # plt.show()


        IM_anat = IM/np.max(IM)*65535
        IM_anat = IM_anat.astype('uint16')
        #%
        # write anatomy stack as NRRD for CMTK registration, with spacing information
        header = {'kinds': ['domain', 'domain', 'domain'], 'units': ['micron', 'micron', 'micron'], 'spacings': [x_rez, y_rez, z_rez_anat]} 
        nrrd.write(anat_stack, np.moveaxis(IM_anat, [0,1,2], [2,1,0]), header)
        imsave(anat_stack.replace('.nrrd', '.tif'), data=IM_anat)

    func_stat_frame = int(stim_file_name[stim_file_name.find('Cycle')+5:stim_file_name.find('Cycle')+10])
    cycle_start_func = 'Cycle'+(stim_file_name[stim_file_name.find('Cycle')+5:stim_file_name.find('Cycle')+10])
    first_func_ind = [i for i, e in enumerate(tiff_files) if cycle_start_func in e][0]
    
    fnames = tiff_files[first_func_ind:]
    print(fnames[0])
    #
    single_cycle = glob.glob('*Cycle00500*')
    n_planes = len(single_cycle)
    n_frames = np.floor(len(fnames)/n_planes).astype(int)


    tree = ET.parse(xml)
    root = tree.getroot()
#%
    t_stamps = np.zeros((n_planes, n_frames), dtype=float)
    for frame in range(func_stat_frame, n_frames+func_stat_frame):
        if frame == 0: # the first frame has two extra entries for the voltage recordings
            adder = 3 
        else:
            adder = 1

        for sl in range(n_planes):
            t_stamps[sl, frame-func_stat_frame] = root[frame+1][sl+adder].attrib['relativeTime']

    # convert t_stamps to msec and truncate

    t_stamps = np.array(t_stamps * 1000, dtype=int)

    #% use t_stamps to get frame rate
    frame_period = np.mean(np.diff(t_stamps[0,:])) 
    frame_rate = 1000/frame_period
    print("frame rate = " + str(np.round(frame_rate, decimals=2)) + ' fps')
    #print(t_stamps)
    #% now load in the voltage_recording

    stim_file = pd.read_csv(stim_file_name)

    stim_volts = stim_file.loc[:,' Input 3'].values
    stim_tstamps = stim_file.loc[:, 'Time(ms)'].values

    # create a 2d matrix to save the relevant voltage for each slice and each frame
    volt_stamps = np.zeros((n_planes, n_frames), dtype=float)

    #assume 5V if we didnt get a recording there
    volt_stamps[:] = 5

    print('sorting out stimulus trace')
    for i in tqdm.trange(n_frames):
        for j in range(n_planes):
            t_stamp = t_stamps[j,i]
            ind = np.where(stim_tstamps <= t_stamp)[0][-1]
            volt_stamps[j,i] = stim_volts[ind]


    stim_vec = (5-np.mean(volt_stamps, axis=0).flatten())/5

    plt.plot(stim_vec)
    plt.show()
    print(np.shape(stim_vec))
    np.save(os.path.join(saveDir,'stimvec_mean.npy'), stim_vec)
    np.save(os.path.join(saveDir,'volt_stamps.npy'), volt_stamps)
    np.save(os.path.join(saveDir,'time_stamps.npy'), t_stamps)



    # make temporary h5py file
    h5_file = os.path.join(saveDir, 'func_data.h5')

    if Path(h5_file).is_file():
        print('skipping writing hdf5 file, already exists')
    else:
        f_h5 = h5py.File(h5_file, 'w')

        k = 0
        IM = np.zeros((1,n_planes, y_size, x_size), dtype=np.uint16)
        for i in range(n_planes):
            IM[:,i,:,:] = Image.open(fnames[k])
            k+=1 

        dset = f_h5.create_dataset('data', data=IM, chunks=True, dtype=np.uint16, maxshape = (None, n_planes, y_size, x_size))
        print('creating hdf5 file')
        for j in tqdm.trange(n_frames-1):
            for i in range(n_planes):
                IM[:,i,:,:] = Image.open(fnames[k])
                k+=1 
            n_frames_already = dset.shape[0]
            dset.resize(n_frames_already+1, axis=0)
            dset[-1,:,:,:] = IM
        f_h5.close()

    f_h5 = h5py.File(h5_file, 'r')


    ops = suite2p.default_ops()

    ops['h5py'] =  ['func_data.h5']
    ops['tau'] = 1.5
    ops['fs'] = frame_rate
    ops['diameter'] = 8
    ops['sparse_mode'] = True
    ops['spatial_scale'] = 1
    ops['nplanes'] = n_planes
    ops['functional_chan'] = 2
    ops['high_pass'] = 300
    ops['nbinned'] = 7000
    ops['data_path'] = [saveDir + r'/']
    ops['save_path0'] = saveDir + r'/'
    ops['batch_size'] = 500
    ops['nimg_init'] = 1000
    ops['look_one_level_down'] = 1
    ops['max_iterations'] = 50
    ops['denoise'] = True
    ops['do_registration'] = 1
    ops['nonrigid'] = True
    ops['keep_movie_raw'] = False
    ops['neuropil_extract'] = True
    ops['delete_bin'] = True
    ops['classifier_path'] = os.path.join(imaging_dir, 'classifiers', 'HuC-H2BGCaMP7.npy')

    output_ops = suite2p.run_s2p(ops=ops)

