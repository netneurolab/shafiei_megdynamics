import pyls
import scipy
import mayavi
import numpy as np
import pandas as pd
import seaborn as sns
import fcn_megdynamics
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from mapalign.align import iterative_alignment
from netneurotools import stats as netneurostats
from scipy.spatial.distance import squareform, pdist


gitrepo_dir = '/Users/gshafiei/gitrepos/shafiei_megdynamics/'
parcellationDir = gitrepo_dir + 'data/SchaeferParcellation/'
datapath = gitrepo_dir + 'data/schaefer100/'

outpath = gitrepo_dir + 'Figures_MEG/'

# load atlas info
lhlabels = (parcellationDir + 'fslr32k/' +
            'Schaefer2018_100Parcels_7Networks_order_lh.label.gii')
rhlabels = (parcellationDir + 'fslr32k/' +
            'Schaefer2018_100Parcels_7Networks_order_rh.label.gii')
labelinfo = np.loadtxt(parcellationDir + 'fslr32k/' +
                       'Schaefer2018_100Parcels_7Networks_order_info.txt',
                       dtype='str', delimiter='\t')
rsnlabels = []
for row in range(0, len(labelinfo), 2):
    rsnlabels.append(labelinfo[row].split('_')[2])

# load coordinates and estimate distance
coor = np.loadtxt(parcellationDir + 'Schaefer_100_centres.txt', dtype=str)
coor = coor[:, 1:].astype(float)

# get custom colormaps
cmap_seq, cmap_seq_r, megcmap, megcmap2, categ_cmap, cmap_OrYel = fcn_megdynamics.make_colormaps()

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.sans-serif'] = ['Myriad Pro']
plt.rcParams['font.size'] = 18.0

# load data for PLS
all_data = np.load(datapath + 'all_microsc_schaefer100.npy')
all_labels = list(np.load(datapath + 'all_microsc_labels.npy'))

# load hctsa features
avg_sharedTS_pca = np.load(datapath + 'avg_hctsafeat_schaefer100.npy')
sharedID_pca = np.loadtxt(datapath + 'hctsafeatID_schaefer100.txt')
sharedID_pca = (np.array(sharedID_pca)-1).astype(int)
megops = pd.read_csv(gitrepo_dir + 'data/ops_modified.txt', header=None)
validmegops = megops.iloc[sharedID_pca]

####################################
# pls analysis
####################################

def spinCV(iSpin, orig_x_wei, orig_y_wei, spins):
    coords = coor
    trainpct=0.75
    nnodes = len(coords)
    P = squareform(pdist(coords, metric="euclidean"))
    testnull = np.zeros((spins.shape[1], ))
    t = np.zeros((nnodes, ))
    Y_null = Y[spins[:, iSpin], :]
    lv = 0

    for node in range(nnodes):
        # distance from a node to all others
        distances = P[node, :]
        idx = np.argsort(distances)

        train_idx = idx[:int(np.floor(trainpct * nnodes))]
        test_idx = idx[int(np.floor(trainpct * nnodes)):]

        Xtrain = X[train_idx, :]
        Xtest = X[test_idx, :]
        Ytrain = Y_null[train_idx, :]
        Ytest = Y_null[test_idx, :]

        # pls analysis
        train_result = pyls.behavioral_pls(Xtrain, Ytrain, n_boot=0,
                                            n_perm=0, test_split=0)
        null_x_wei = train_result['x_weights']
        null_y_wei = train_result['y_weights']

        temp = [orig_x_wei[node], null_x_wei]
        realigned_null_x_wei, _ = iterative_alignment(temp, n_iters=1)

        temp = [orig_y_wei[node], null_y_wei]
        realigned_null_y_wei, _ = iterative_alignment(temp, n_iters=1)

        t[node], _ = scipy.stats.spearmanr(Xtest @ realigned_null_x_wei[1][:, lv],
                        Ytest @ realigned_null_y_wei[1][:, lv])
    testnull[iSpin]  = np.mean(t)

    return testnull


def pls_cv_distance_dependent_par(X, Y, coords, trainpct=0.75, lv=0,
                              testnull=False, spins=None, nspins=1000):
    """
    Distance-dependent cross validation.
    Parameters
    ----------
    X : (n, p1) array_like
        Input data matrix. `n` is the number of brain regions.
    Y : (n, p2) array_like
        Input data matrix. `n` is the number of brain regions.
    coords : (n, 3) array_like
        Region (x,y,z) coordinates. `n` is the number of brain regions.
    trainpct : float
        Percent observations in train set. 0 < trainpct < 1.
        Default = 0.75.
    lv : int
        Index of latent variable to cross-validate. Default = 0.
    testnull : Boolean
        Whether to calculate and return null mean test-set correlation.
    spins : (n, nspins) array_like
        Spin-test permutations. Required if testnull=True.
    nspins : int
        Number of spin-test permutations. Only used if testnull=True
    Returns
    -------
    train : (nplit, ) array
        Training set correlation between X and Y scores.
    test : (nsplit, ) array
        Test set correlation between X and Y scores.
    """

    nnodes = len(coords)
    train = np.zeros((nnodes, ))
    test = np.zeros((nnodes, ))

    orig_x_wei = []
    orig_y_wei = []

    P = squareform(pdist(coords, metric='euclidean'))

    for k in range(nnodes):

        # distance from a node to all others
        distances = P[k, :]
        idx = np.argsort(distances)

        train_idx = idx[:int(np.floor(trainpct * nnodes))]
        test_idx = idx[int(np.floor(trainpct * nnodes)):]

        Xtrain = X[train_idx, :]
        Xtest = X[test_idx, :]
        Ytrain = Y[train_idx, :]
        Ytest = Y[test_idx, :]

        # pls analysis
        train_result = pyls.behavioral_pls(Xtrain, Ytrain, n_boot=0, n_perm=0, test_split=0)

        train[k], _ = scipy.stats.spearmanr(train_result['x_scores'][:, lv],
                               train_result['y_scores'][:, lv])
        # project weights, correlate predicted scores in the test set
        test[k], _ = scipy.stats.spearmanr(Xtest @ train_result['x_weights'][:, lv],
                              Ytest @ train_result['y_weights'][:, lv])
        orig_x_wei.append(train_result['x_weights'])
        orig_y_wei.append(train_result['y_weights'])

    # if testnull=True, get distribution of mean null test-set correlations.
    if testnull:
        print('Running null test-set correlations, will take time')
        results = Parallel(n_jobs=42)(delayed(spinCV)(i, orig_x_wei, orig_y_wei, spins) for i in range(nspins))
        testnull = np.array([results[i][i] for i in range(nspins)])
    else:
        testnull = None

    if testnull:
        return train, test, testnull
    else:
        return train, test


# spin test
nspinsall=10000
surf_path = (gitrepo_dir + 'data/surfaces/')

surfaces = [surf_path + 'L.sphere.32k_fs_LR.surf.gii',
            surf_path + 'R.sphere.32k_fs_LR.surf.gii']

centroids, hemiid = fcn_megdynamics.get_gifti_centroids(surfaces,
                                                        lhlabels,
                                                        rhlabels)

allspins = netneurostats.gen_spinsamples(centroids, hemiid,
                                         n_rotate=nspinsall, seed=272)

X = scipy.stats.zscore(avg_sharedTS_pca.T)
Y = scipy.stats.zscore(all_data)

results = pyls.behavioral_pls(X, Y, n_boot=nspinsall, n_perm=nspinsall,
                              permsamples=allspins, test_split=0)
pyls.save_results(outpath + 'pls/plsresults_schaefer100.hdf5', results)

train, test = pls_cv_distance_dependent_par(X, Y, coords=coor)
np.save(outpath + 'pls/pls_train_schaefer100.npy', train)
np.save(outpath + 'pls/pls_test_schaefer100.npy', test)

# plot train and test score correlation distribution
fig, ax = plt.subplots(figsize=(3, 4))
sns.boxplot(data=[train, test], ax=ax)
ax.set_xticklabels(['train', 'test'])
ax.set_ylabel('score correlation')
plt.savefig(outpath + 'pls/scorecorr_pls_schaefer100.svg',
            bbox_inches='tight', dpi=300,
            transparent=True)

# visualize pls results
# covariance explained
lv = 0  # latent variable
cv = results['singvals']**2 / np.sum(results['singvals']**2)
null_singvals = results['permres']['permsingvals']
cv_spins = null_singvals**2 / sum(null_singvals**2)

myplot = sns.scatterplot(np.arange(len(cv[:10])), cv[:10]*100,
                         facecolors='darkslategrey', s=70)
plt.boxplot(cv_spins[:10, :].T * 100, positions=range(len(cv[:10])),
            showcaps=False, showfliers=False, flierprops=dict(marker='.'))
sns.despine(ax=myplot, trim=False)
myplot.axes.set_title('PLS')
myplot.axes.set_xlabel('latent variable')
myplot.axes.set_ylabel('covariance explained (%)')
myplot.figure.set_figwidth(5)
myplot.figure.set_figheight(4)
plt.savefig(outpath + 'pls/covarexp_wnull_pls_schaefer400.svg',
            bbox_inches='tight', dpi=300,
            transparent=True)

# correlate and plot scores
x = -results['x_scores'][:, lv]
y = -results['y_scores'][:, lv]

corr = scipy.stats.spearmanr(x, y)
pvalspin = fcn_megdynamics.get_spinp(x, y, corrval=corr[0], nspin=10000,
                                     corrtype='spearman',
                                     lhannot=lhlabels, rhannot=rhlabels)

title = ('LV%s: PLS p(spin) = %1.4f \n Score corr: spearman r = %1.3f - p(spin) = %1.4f'
         % (lv+1, results.permres['pvals'][lv], corr[0], pvalspin))
xlab = 'node score (meg)'
ylab = 'node score (microsc)'
fcn_megdynamics.scatterregplot(x, y, title, xlab, ylab, 50)
plt.tight_layout()
plt.savefig(outpath + 'pls/scores_lv1_schaefer400.svg',
            bbox_inches='tight', dpi=300,
            transparent=True)

# plot on brain
plotlabel = 'meg'
toplot = -results['x_scores'][:, lv]
brains = fcn_megdynamics.plot_conte69(toplot, lhlabels, rhlabels,
                                      vmin=np.percentile(toplot, 2.5),
                                      vmax=np.percentile(toplot, 97.5),
                                      colormap='Greys', customcmap=megcmap,
                                      colorbartitle=plotlabel,
                                      surf='inflated')

mayavi.mlab.figure(brains[0]).scene.parallel_projection = True
mayavi.mlab.figure(brains[1]).scene.parallel_projection = True
mayavi.mlab.figure(brains[0]).scene.background = (1, 1, 1)
mayavi.mlab.figure(brains[1]).scene.background = (1, 1, 1)

mayavi.mlab.savefig(outpath + 'pls/%s_score_lh_schaefer100.png' % plotlabel,
                    figure=brains[0])
mayavi.mlab.savefig(outpath + 'pls/%s_score_rh_schaefer100.png' % plotlabel,
                    figure=brains[1])

# medial view
mayavi.mlab.view(azimuth=0, elevation=90, distance=450, figure=brains[0])
mayavi.mlab.view(azimuth=0, elevation=-90, distance=450, figure=brains[1])

mayavi.mlab.savefig(outpath + 'pls/%s_score_lh_med_schaefer100.png' % plotlabel,
                    figure=brains[0])
mayavi.mlab.savefig(outpath + 'pls/%s_score_rh_med_schaefer100.png' % plotlabel,
                    figure=brains[1])
mayavi.mlab.close(all=True)

plotlabel = 'microsc'
toplot = -results['y_scores'][:, lv]
brains = fcn_megdynamics.plot_conte69(toplot, lhlabels, rhlabels,
                                      vmin=np.percentile(toplot, 2.5),
                                      vmax=np.percentile(toplot, 97.5),
                                      colormap='Greys', customcmap=megcmap,
                                      colorbartitle=plotlabel,
                                      surf='inflated')

mayavi.mlab.figure(brains[0]).scene.parallel_projection = True
mayavi.mlab.figure(brains[1]).scene.parallel_projection = True
mayavi.mlab.figure(brains[0]).scene.background = (1, 1, 1)
mayavi.mlab.figure(brains[1]).scene.background = (1, 1, 1)

mayavi.mlab.savefig(outpath + 'pls/%s_score_lh_schaefer100.png' % plotlabel,
                    figure=brains[0])
mayavi.mlab.savefig(outpath + 'pls/%s_score_rh_schaefer100.png' % plotlabel,
                    figure=brains[1])

# medial view
mayavi.mlab.view(azimuth=0, elevation=90, distance=450, figure=brains[0])
mayavi.mlab.view(azimuth=0, elevation=-90, distance=450, figure=brains[1])

mayavi.mlab.savefig(outpath + 'pls/%s_score_lh_med_schaefer100.png' % plotlabel,
                    figure=brains[0])
mayavi.mlab.savefig(outpath + 'pls/%s_score_rh_med_schaefer100.png' % plotlabel,
                    figure=brains[1])
mayavi.mlab.close(all=True)

# plot loadings for microsc
err = (results.bootres['y_loadings_ci'][:, lv, 1]
      - results.bootres['y_loadings_ci'][:, lv, 0]) / 2
sorted_idx = np.argsort(-results['y_loadings'][:, lv])
plt.figure(figsize=(14, 7))
plt.ion()
plt.bar(np.arange(len(all_labels)), -results['y_loadings'][sorted_idx, lv],
        yerr=err[sorted_idx])
plt.xticks(np.arange(len(all_labels)), labels=np.array(all_labels)[sorted_idx],
           rotation='vertical')
plt.ylabel('microsc loadings')
plt.tight_layout()

plt.savefig(outpath + 'pls/microscloading_lv1_schaefer400.svg',
            bbox_inches='tight', dpi=300,
            transparent=True)

# get meg loadings by swaping X and Y matrices in PLS
xload = pyls.behavioral_pls(Y, X, n_boot=0, n_perm=0, test_split=0)

# save meg loadings in a csv file
validmegops.reset_index(inplace=True, drop=True)
univarCorr = pd.DataFrame(data=validmegops.values, columns=['CodeString'])
univarCorr['loadings'] = -xload['y_loadings'][:, lv]
univarCorr['absoluteLoad'] = np.abs(-xload['y_loadings'][:, lv])
univarCorr.sort_values('loadings', ascending=False, inplace=True)

# # could also sort based on absolute value of loadings instead of loadings
# univarCorr.sort_values('absoluteLoad', ascending=False, inplace=True)

# save
univarCorr.to_csv(outpath +
                    'pls/megweight_lv1_schaefer100.csv',
                    index=False)
