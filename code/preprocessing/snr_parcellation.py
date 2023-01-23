import os
import time
import nibabel as nib
import numpy as np
import scipy.io as sio


meg_dir = '/path/to/megdata/and/results/'
parcellationDir = '/path/to/SchaeferParcellation/fslr4k/'

# load subj list
subjList = sio.loadmat(os.path.join(meg_dir, 'myMEGList.mat'))['myMEG']
subjList = [subjList[i][0][0][4:] for i in range(len(subjList))]

# load parcellation data
lhlabel = nib.load(parcellationDir + 'Schaefer100_L.4k.label.gii')
rhlabel = nib.load(parcellationDir + 'Schaefer100_R.4k.label.gii')

parcels = np.concatenate((lhlabel.darrays[0].data, rhlabel.darrays[0].data))

parcelIDs = np.unique(parcels)
parcelIDs = np.delete(parcelIDs, 0)

dataTypes = ['snrData_lcmv']
outpath = os.path.join(meg_dir, 'brainstormResults/snrData_lcmv/')

for dataType in dataTypes:
    snr_parcel = []
    for s, subj in enumerate(subjList):
        startTime = time.time()
        path_to_file = os.path.join(meg_dir, 'brainstormResults', dataType,
                                    subj, (subj + '_snrData.mat'))
        subjFile = sio.loadmat(path_to_file)
        snrData = subjFile['snr']

        # source amplitude of a dipole is assumed to be ~10nAm
        # however, brainstorm says "assumed RMS of the source is 1 uAm"
        # in the original calculation, I used "10" for a without any units
        # now I first solve for scaling, then recalculate snr with a = 10 * (10**-9)
        a = 10 * (10**-9)
        scaling = (10**(snrData/10))/100
        snrData_new = 10 * np.log10((a)**2 * scaling)

        parcellatedData = np.zeros((len(parcelIDs), 1))
        for IDnum in parcelIDs:
            idx = np.where(parcels == IDnum)[0]
            parcellatedData[IDnum-1, :] = np.nanmean(snrData_new[idx])

        snr_parcel.append(np.squeeze(parcellatedData))

        endTime = time.time()

        print('\nSubj %s, dataType %s' % (s, dataType),
                '\n Run time: %s' % (endTime - startTime))


    path_to_output = os.path.join(meg_dir, 'brainstormResults',
                                  dataType)

    outputName_parcel = os.path.join(path_to_output, ('group_' + dataType +
                                                      '_Schaefer100.npy'))
    np.save(outputName_parcel, np.array(snr_parcel))
