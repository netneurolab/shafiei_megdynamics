# construct 3D matrices as nodextsfeaturesxsubj
import h5py
import numpy as np
import time
import glob

datapath = '/path/to/hctsaMEGoutputs/';

# MEG files
tsID = np.loadtxt(datapath +
                  'HCP_MEG_outputs/Schaefer100/sharedIDall_segments_pca_Schaefer100_fullSet_80.txt')
tsID = np.array(tsID).astype(int)

fileNames = sorted(glob.glob(datapath +
                             'HCP_MEG_outputs/Schaefer100/dataSegments/fullSet_80/*_N.mat'))

sharedTS = np.zeros((len(tsID), 100, len(fileNames)))

for n, inMat in enumerate(fileNames):
    start = time.time()
    with h5py.File(inMat, 'r') as src:
        refs = src.get('Operations/ID')[()]
        refID = np.hstack([src[ref][()].squeeze() for ref in refs[0]])
        ts = np.array(src['TS_DataMat'])
    availidx = np.intersect1d(refID, tsID, return_indices=True)
    myTS = ts[availidx[1], :]
    sharedTS[:, :, n] = myTS
    end = time.time()
    print('\nFile', n, 'of', len(fileNames), 'done!',
          '\nRunning time = ', end-start, 'seconds!')

np.save(datapath + 'HCP_MEG_outputs/Schaefer100/sharedTS_segments_pca_Schaefer100_fullSet_80.npy',
        sharedTS)
