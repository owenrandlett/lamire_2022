#%%  run suite2p on folders of images collected on MartinPhoton
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np

import suite2p

import glob
import os
#from funs import OpenStack
from PIL import Image
import tifffile
from tqdm.notebook import tqdm
import shutil
from PIL import Image
imaging_dir = Path.cwd()
def OpenStack(filename):
    """
    Load image stack from tiff-file
    """
    im = Image.open(filename)
    stack = np.empty((im.n_frames, im.size[1], im.size[0]), dtype=np.float32)
    # loop over frames and assign
    for i in range(im.n_frames):
        im.seek(i)
        stack[i, :, :] = np.array(im)
    im.close()
    return stack
    

frame_rate = 1.048576


tmp_dir = os.path.realpath(r'/mnt/md0/tmp')
save_dir_root = os.path.realpath(r'/mnt/md0/suite2p_output_martinphoton')

im_files = []
parent_dirs = [
    os.path.realpath(r'/media/BigBoy/ciqle/MartinPhotonData/20180319-20180323_DarkFlashHabituation'),
    os.path.realpath(r'/media/BigBoy/ciqle/MartinPhotonData/20190513-17_DarkFalshHabituation')
]
for parent_dir in parent_dirs:
    im_files += glob.glob(parent_dir + r'/*/*ad1*_Z_0_0.tif')

print(im_files)

for im_file in tqdm(im_files):

    IM = OpenStack(im_file)

    # resave the tiffs as individual frames to play nice with suite2p
    IM = IM.astype('uint8') + 1

    fnames=[]
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    for f in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, f))

    for frame in range(IM.shape[0]):
        frame_name = tmp_dir + '/' + str(frame) + '.tif'
        tifffile.imsave(frame_name, IM[frame, :, :])
        fnames.append(frame_name)

    fish_name = os.path.split(os.path.split(im_file)[0])[1] + '_' + os.path.split(im_file)[1].replace('_Z_0_0.tif', '')
    save_dir = os.path.join(save_dir_root, fish_name)

    #%
    ops = suite2p.default_ops()

    ops['fs'] = frame_rate
    ops['tau'] = 1.5
    ops['diameter'] = 6
    ops['sparse_mode'] = True
    ops['spatial_scale'] = 1
    ops['nplanes'] = 1
    ops['functional_chan'] = 1
    ops['high_pass'] = 100
    ops['nbinned'] = 7000
    ops['data_path'] = tmp_dir
    ops['save_path0'] = save_dir
    ops['tiff_list'] = fnames
    ops['look_one_level_down'] = 1
    ops['delete_bin'] = True
    ops['denoise'] = True
    ops['neuropil_extract'] = True
    ops['do_registration'] = 1
    ops['nonrigid'] = True
    ops['keep_movie_raw'] = False
    ops['neuropil_extract'] = True
    ops['delete_bin'] = False
    ops['save_path'] = save_dir
    ops['classifier_path'] = os.path.join(imaging_dir, 'classifiers', 'HuC-H2BGCaMP7.npy')
    #%
    output_ops = suite2p.run_s2p(ops=ops)


    #%
    print('donezoes')
    def save_out_im(im_string):
        im_out = output_ops[im_string]
        im_out = (65535*im_out/np.max(im_out)).astype('uint16')
        tifffile.imsave(output_ops['save_path'] + '\\' + im_string + '.tif', im_out)
        plt.imshow(im_out)
        plt.title(im_string)
        plt.show()

    save_out_im('max_proj')
    save_out_im('meanImg')

