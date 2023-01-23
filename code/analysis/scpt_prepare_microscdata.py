import scipy
import abagen
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn import decomposition
from neuromaps import datasets,resampling
from neuromaps.parcellate import Parcellater
from nibabel.freesurfer.io import read_annot

gitrepo_dir = '/Users/gshafiei/gitrepos/shafiei_megdynamics/'
parcellationDir = gitrepo_dir + 'data/SchaeferParcellation/'

# load data
lhlabels = (parcellationDir + 'fslr32k/' +
            'Schaefer2018_100Parcels_7Networks_order_lh.label.gii')
rhlabels = (parcellationDir + 'fslr32k/' +
            'Schaefer2018_100Parcels_7Networks_order_rh.label.gii')
labelinfo = np.loadtxt(parcellationDir + 'fslr32k/' +
                        'Schaefer2018_100Parcels_7Networks_order_info.txt',
                        dtype='str', delimiter='\t')

####################################
# use neuromaps to compile data
####################################
# surface data
ANNOTATIONS = [
    ('abagen', 'genepc1', 'fsaverage', '10k'),
    ('hcps1200', 'myelinmap', 'fsLR', '32k'),
    ('hcps1200', 'thickness', 'fsLR', '32k'),
    ('hill2010', 'devexp', 'fsLR', '164k'),
    ('hill2010', 'evoexp', 'fsLR', '164k'),
    ('raichle', 'cbf', 'fsLR', '164k'),
    ('raichle', 'cbv', 'fsLR', '164k'),
    ('raichle', 'cmr02', 'fsLR', '164k'),
    ('raichle', 'cmruglu', 'fsLR', '164k'),
    ('reardon2018', 'scalingnih', 'civet', '41k'),
    ('reardon2018', 'scalingpnc', 'civet', '41k'),
]


# prepare maps
neuromaps_microsc = []

for (src, desc, space, den) in ANNOTATIONS:
    target = datasets.fetch_annotation(source=src, desc=desc, space=space, den=den)
    if src=='hill2010':
        hemi = 'R'
    else:
        hemi = None

    # this won't be used later, just so that we have both src and trg inputs
    tempmap = datasets.fetch_annotation(desc='myelinmap')

    if not(space=='fsLR' and den=='32k'):
        tempmap_rs, target_rs = resampling.resample_images(src=tempmap, trg=target,
                                                           src_space='fsLR',
                                                           trg_space=space,
                                                           hemi=hemi,
                                                           method='linear',
                                                           resampling='transform_to_alt',
                                                           alt_spec=('fslr', '32k'))
    else:
        tempmap_rs = tempmap
        target_rs = target

    if hemi=='R':
        # mirror if only one hemisphere
        parcellation = [lhlabels, rhlabels]
        parc = Parcellater(parcellation, 'fslr')
        parcellated_target = parc.fit_transform((target_rs[0], target_rs[0]),
                                                 'fslr')
        parcellated_target = scipy.stats.zscore(parcellated_target)

    else:
        parcellation = [lhlabels, rhlabels]
        parc = Parcellater(parcellation, 'fslr')
        parcellated_target = parc.fit_transform(target_rs, 'fslr')
        parcellated_target = scipy.stats.zscore(parcellated_target)

    neuromaps_microsc.append(parcellated_target)

    print('\nMap: %s done!' % desc)

neuromaps_microsc = np.array(neuromaps_microsc).T
microsclabels = [item[1] for item in ANNOTATIONS]

####################################
# load other parcellated data
####################################
# receptor
receptor_data = pd.read_csv(gitrepo_dir + 'data/schaefer100/receptor_data_scale100.csv',
                            header=None)
receptor_names = np.load(gitrepo_dir + 'data/schaefer100/receptor_names_pet.npy')
receptor_data = np.array(receptor_data)

# get receptor PC1
dataMat = scipy.stats.zscore(receptor_data)
pca = decomposition.PCA(n_components=10, svd_solver='full')
pca.fit(dataMat)
node_score = pca.transform(dataMat)
receptorPC1 = -node_score[:, 0]

# load synapcis density and glycolytic index
ucbj_data = np.genfromtxt(gitrepo_dir + 'data/schaefer100/avg_ucbj_bp_n76_reg_parcellated_schaefer100.csv')
glycoidx = np.load(gitrepo_dir + 'data/schaefer100/glycolytic_index_schaefer100.npy')

# load other data
microsclabels.extend(['pc1receptor', 'synapse density', 'glycolytic index'])
neuromaps_microsc = np.hstack((neuromaps_microsc,
                               -node_score[:, 0][:, np.newaxis],
                               ucbj_data[:, np.newaxis],
                               glycoidx[:, np.newaxis]))

# start a list of labels with receptor names
all_labels = list(receptor_names)

# # Here we loaded the parcellated maps.
# # Example code to directly load and parcellate with neuromaps:
# from neuromaps.parcellate import Parcellater
# from nilearn.datasets import fetch_atlas_schaefer_2018
# mni_atlas = fetch_atlas_schaefer_2018(n_rois=100)['maps']
# parcellater = Parcellater(mni_atlas, 'MNI152')
# orig_map = datasets.fetch_annotation(desc='ucbj')
# parcellated = parcellater.fit_transform(orig_map, 'MNI152', True)

####################################
# use abagen to get gene expression for cell types
####################################
# gene expression with abagen
ahba_data = '/home/gshafiei/data1/Projects/packages/abagen'
schaefer_mni = (parcellationDir + 'MNI/Schaefer2018_100Parcels' +
                '_7Networks_order_FSLMNI152_1mm.nii.gz')
expression = abagen.get_expression_data(schaefer_mni, missing='interpolate',
                                        data_dir=ahba_data,
                                        lr_mirror='bidirectional')

celldata = pd.read_csv(gitrepo_dir + 'data/celltypes_PSP.csv')

celltypes = celldata['class'].unique()
celltype_exp = []

for ctype in celltypes:
    geneid = celldata.loc[celldata['class']==ctype, 'gene']
    ind = expression.columns.isin(geneid)
    genedata = np.array(expression.iloc[:, [i for i, x in enumerate(ind) if x]])
    celltype_exp.append(np.mean(genedata, axis=1))
celltype_exp = np.array(celltype_exp).T

all_labels.extend(list(celltypes))

####################################
# cortical layer data from BigBrainWarp
####################################
# cortical layers
bigbrainwarppath = '/home/gshafiei/data1/Projects/packages/BigBrainWarp/spaces/'
schaefer_fsaverage_lh = read_annot(parcellationDir + 'FreeSurfer5.3/' +
                                   'fsaverage/label/' +
                                   'lh.Schaefer2018_100Parcels_7Networks' +
                                   '_order.annot')
schaefer_fsaverage_rh = read_annot(parcellationDir + 'FreeSurfer5.3/' +
                                   'fsaverage/label/' +
                                   'rh.Schaefer2018_100Parcels_7Networks' +
                                   '_order.annot')

parcellatedData_all = np.zeros((6, 100))
for i in range(6):
    lh_file = (bigbrainwarppath + 'tpl-fsaverage/' +
            'tpl-fsaverage_hemi-L_den-164k_desc-layer%s_thickness.shape.gii' % str(i+1))
    rh_file = (bigbrainwarppath + 'tpl-fsaverage/' +
            'tpl-fsaverage_hemi-R_den-164k_desc-layer%s_thickness.shape.gii' % str(i+1))

    data_lh = nib.load(lh_file).darrays[0].data
    data_rh = nib.load(rh_file).darrays[0].data

    parcelIDs_lh = np.unique(schaefer_fsaverage_lh[0])
    parcelIDs_lh = np.delete(parcelIDs_lh, 0)

    parcellatedData_lh = np.zeros((1, len(parcelIDs_lh)))

    for IDnum in parcelIDs_lh:
        idx = np.where(schaefer_fsaverage_lh[0] == IDnum)[0]
        parcellatedData_lh[:, IDnum-1] = np.nanmean(data_lh[idx])


    parcelIDs_rh = np.unique(schaefer_fsaverage_rh[0])
    parcelIDs_rh = np.delete(parcelIDs_rh, 0)

    parcellatedData_rh = np.zeros((1, len(parcelIDs_rh)))

    for IDnum in parcelIDs_rh:
        idx = np.where(schaefer_fsaverage_rh[0] == IDnum)[0]
        parcellatedData_rh[:, IDnum-1] = np.nanmean(data_rh[idx])

    parcellatedData = np.hstack((parcellatedData_lh, parcellatedData_rh))

    parcellatedData_all[i, :] = parcellatedData

layer_data = parcellatedData_all.T
all_labels.extend(list(np.arange(6)+1))
all_data = np.hstack((receptor_data, celltype_exp, layer_data))

####################################
# stack all data and save for PLS
####################################
all_labels.extend(microsclabels)
all_data = np.hstack((all_data, neuromaps_microsc))

# save for PLS
np.save(gitrepo_dir + 'data/schaefer100/all_microsc_schaefer100.npy', all_data)
np.save(gitrepo_dir + 'data/schaefer100/all_microsc_labels.npy', all_labels)
