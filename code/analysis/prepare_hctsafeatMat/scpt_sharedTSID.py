
import h5py
import numpy as np
import time
import glob

datapath = '/path/to/hctsaMEGoutputs/';

# MEG files
fileNames = sorted(glob.glob(datapath +
                             'HCP_MEG_outputs/Schaefer100/dataSegments/fullSet_80/*_N.mat'))

IDList = []
for n, inMat in enumerate(fileNames):
    start = time.time()
    with h5py.File(inMat, 'r') as src:
        refs = src.get('Operations/ID')[()]
        refID = np.hstack([src[ref][()].squeeze() for ref in refs[0]])
        # refs = src.get('Operations/ID').value[0]
        # refID = np.hstack([src[ref].value.squeeze() for ref in refs])
    IDList.append(set(refID))
    end = time.time()
    print('\nFile', n, 'of', len(fileNames), 'done!',
          '\nRunning time = ', end-start, 'seconds!')

sharedID = np.array(list(set.intersection(*IDList))).astype(int)
np.savetxt(datapath +
           'HCP_MEG_outputs/Schaefer100/sharedIDall_segments_pca_Schaefer100_fullSet_80.txt',
           sharedID)

