import scipy
import mayavi
import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
import sklearn.decomposition
import matplotlib.pyplot as plt
from statsmodels.stats import multitest
from netneurotools import stats as netneurostats
from mapalign.align import iterative_alignment
import fcn_megdynamics


gitrepo_dir = '/Users/gshafiei/gitrepos/shafiei_megdynamics/'
parcellationDir = gitrepo_dir + 'data/SchaeferParcellation/'
datapath = gitrepo_dir + 'data/schaefer100/'

outpath = gitrepo_dir + 'Figures_MEG/'

# load data
avg_sharedTS_pca = np.load(datapath + 'avg_hctsafeat_schaefer100.npy')
sharedID_pca = np.loadtxt(datapath + 'hctsafeatID_schaefer100.txt')
sharedID_pca = (np.array(sharedID_pca)-1).astype(int)
megops = pd.read_csv(gitrepo_dir + 'data/ops_modified.txt', header=None)
validmegops = megops.iloc[sharedID_pca]

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
distance = sklearn.metrics.pairwise_distances(coor)

# get custom colormaps
cmap_seq, cmap_seq_r, megcmap, megcmap2, categ_cmap, cmap_OrYel = fcn_megdynamics.make_colormaps()

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.sans-serif'] = ['Myriad Pro']
plt.rcParams['font.size'] = 18.0

####################################
# PCA analysis
####################################
featMat = avg_sharedTS_pca.T

dataMat = scipy.stats.zscore(featMat)
pca = sklearn.decomposition.PCA(n_components=10, svd_solver='full')
pca.fit(dataMat)
node_score = pca.transform(dataMat)
pc_wei = pca.components_

# var explained in PCA
varexpall = pca.explained_variance_ratio_
myplot = sns.scatterplot(x=np.arange(len(varexpall)), y=varexpall,
                         facecolors='darkslategrey')
sns.despine(ax=myplot, trim=False)
myplot.axes.set_title('PCA - hctsa feat')
myplot.axes.set_xlabel('components')
myplot.axes.set_ylabel('variance explained (%)')
myplot.figure.set_figwidth(5)
myplot.figure.set_figheight(5)
plt.savefig(outpath + 'pca/varexp_PCA_schaefer100.svg',
            bbox_inches='tight', dpi=300,
            transparent=True)

ncomp = 0
toplot = node_score[:, ncomp]
brains = fcn_megdynamics.plot_conte69(toplot, lhlabels, rhlabels,
                                      vmin=np.percentile(toplot, 2.5),
                                      vmax=np.percentile(toplot, 97.5),
                                      colormap='viridis', customcmap=megcmap,
                                      colorbartitle=('node score - PC %s - ' +
                                                     'VarExp = %0.3f')
                                                     % (ncomp+1, varexpall[ncomp]),
                                      surf='inflated')

mayavi.mlab.savefig(outpath + 'pca/pc%s_lh_schaefer100.png' % (ncomp+1), figure=brains[0])
mayavi.mlab.savefig(outpath + 'pca/pc%s_rh_schaefer100.png' % (ncomp+1), figure=brains[1])

# PC1 in intrinsic networks
pc1score = pd.DataFrame(scipy.stats.zscore(node_score[:, 0]), columns=['score'])
pc1score['rsn'] = rsnlabels
meanrsnScore = pc1score.groupby(['rsn']).median()
idx = np.argsort(-meanrsnScore.squeeze())
plot_order = [meanrsnScore.index[k] for k in idx]
sns.set(style='ticks', palette='pastel')
plt.figure()
ax = sns.boxplot(x='rsn', y='score', data=pc1score,
                 width=.45, fliersize=3, showcaps=False,
                 order=plot_order, showfliers=False)

sns.despine(ax=ax, offset=5, trim=True)
ax.axes.set_title('intrinsic networks')

plt.setp(ax.artists, edgecolor = 'k', facecolor='w')
plt.setp(ax.lines, color='k')

ax.figure.set_figwidth(7)
ax.figure.set_figheight(6)
plt.tight_layout()

plt.savefig(outpath  + 'pca/intrinsicnetwork_schaefer100.svg',
            bbox_inches='tight', dpi=300,
            transparent=True)

# univariate analysis of PCA weis
X = avg_sharedTS_pca.T
Y = node_score[:, :2]
rho = np.zeros((X.shape[1], Y.shape[1]))
for i in range(X.shape[1]):
    for j in range(Y.shape[1]):
        tmpcorr = scipy.stats.pearsonr(X[:, i], Y[:, j])
        rho[i, j] = tmpcorr[0]

surf_path = (gitrepo_dir + '/data/surfaces/')

surfaces = [surf_path + 'L.sphere.32k_fs_LR.surf.gii',
            surf_path + 'R.sphere.32k_fs_LR.surf.gii']

centroids, hemiid = fcn_megdynamics.get_gifti_centroids(surfaces,
                                                        lhlabels,
                                                        rhlabels)
spins = netneurostats.gen_spinsamples(centroids, hemiid,
                                      n_rotate=10000, seed=272)

n_spins = spins.shape[1]
rhoPerm = np.zeros((X.shape[1], Y.shape[1], n_spins))
for spin in range(n_spins):
    for i in range(X.shape[1]):
        for j in range(Y.shape[1]):
            rhoPerm[i, j, spin] = scipy.stats.pearsonr(
                                    X[spins[:, spin], i],
                                    Y[:, j])[0]
    print('\nspin %i' % spin)

pvalPerm = np.zeros((X.shape[1], Y.shape[1]))
corrected_pval = np.zeros((X.shape[1], Y.shape[1]))
sigidx = []
nonsigidx = []
for comp in range(Y.shape[1]):
    for feat in range(X.shape[1]):
        permmean = np.mean(rhoPerm[feat, comp, :])
        pvalPerm[feat, comp] = (len(np.where(abs(rhoPerm[feat, comp, :]
                                                 - permmean) >=
                                             abs(rho[feat, comp]
                                                 - permmean))[0])+1)/(n_spins +
                                                                      1)
    multicomp = multitest.multipletests(pvalPerm[:, comp], alpha=0.05,
                                        method='fdr_bh')
    corrected_pval[:, comp] = multicomp[1]
    sigidx.append(np.where(multicomp[1] < 0.05)[0])
    nonsigidx.append(np.where(multicomp[1] >= 0.05)[0])


rhoCopy = rho.copy()
rhoCopy[nonsigidx[0], 0] = 0
rhoCopy[nonsigidx[1], 1] = 0

var = 0
sortedCorr = np.sort(rho[:, var])
sortedrhoCopy = np.sort(rhoCopy[:, var])
sigthresh1 = np.max(np.where(sortedrhoCopy < 0)[0])
sigthresh2 = np.min(np.where(sortedrhoCopy > 0)[0])

myfig = plt.figure()
plt.plot(sortedCorr)
plt.vlines([sigthresh1, sigthresh2], ymin=-1, ymax=1)
plt.xlabel('# of features')
plt.ylabel('pearson r')
plt.title('PC%s, sigthresh1 = %s, sigthresh2 = %s' % (str(var+1),
                                                      str(sigthresh1),
                                                      str(sigthresh2)))
myfig.set_figwidth(7)
myfig.set_figheight(7)
plt.savefig(outpath + 'pca/PC%s_pearsonr_correctPval_fdr05_schaefer100.svg' % str(var+1),
            bbox_inches='tight', dpi=300,
            transparent=True)

# save to csv
validmegops.reset_index(inplace=True, drop=True)
univarCorr = pd.DataFrame(data=validmegops.values, columns=['CodeString'])

var = 0
univarCorr['Loading'] = rho[:, var]
univarCorr['absoluteLoading'] = np.abs(rho[:, var])
univarCorr['spinPvalue'] = pvalPerm[:, var]
univarCorr['spinPvalue_fdrCorrected'] = corrected_pval[:, var]

univarCorr.sort_values('absoluteLoading', ascending=False, inplace=True)

univarCorr.to_csv(outpath +
                  'pca/csvFiles/pc%s_allpearson_fdr05_schaefer100.csv'
                  % str(var+1), index=False)

####################################
# SNR
####################################
avg_snr = np.load(datapath + 'group_snrData_lcmv_Schaefer100.npy')

# univariate analysis of weis
X = avg_sharedTS_pca.T
Y = avg_snr
rho = np.zeros((X.shape[1], 1))
for i in range(X.shape[1]):
    tmpcorr = scipy.stats.pearsonr(X[:, i], Y)  # or pearsonr
    rho[i, 0] = tmpcorr[0]

surf_path = (gitrepo_dir + 'data/surfaces/')

surfaces = [surf_path + 'L.sphere.32k_fs_LR.surf.gii',
            surf_path + 'R.sphere.32k_fs_LR.surf.gii']

centroids, hemiid = fcn_megdynamics.get_gifti_centroids(surfaces,
                                                        lhlabels,
                                                        rhlabels)
spins = netneurostats.gen_spinsamples(centroids, hemiid,
                                      n_rotate=10000, seed=272)

n_spins = spins.shape[1]
rhoPerm = np.zeros((X.shape[1], n_spins))
for spin in range(n_spins):
    for i in range(X.shape[1]):
        rhoPerm[i, spin] = scipy.stats.pearsonr(X[spins[:, spin], i], Y)[0]
    print('\nspin %i' % spin)

pvalPerm = np.zeros((X.shape[1], 1))
corrected_pval = np.zeros((X.shape[1], 1))
sigidx = []
nonsigidx = []
for feat in range(X.shape[1]):
    permmean = np.mean(rhoPerm[feat, :])
    pvalPerm[feat, 0] = (len(np.where(abs(rhoPerm[feat, :] - permmean) >=
                                      abs(rho[feat, 0] - permmean))[0]) +
                         1)/(n_spins + 1)
# # we want to bre more conservative so that we remove more features
# multicomp = multitest.multipletests(pvalPerm[:, 0], alpha=0.05,
#                                     method='fdr_bh')
# corrected_pval[:, comp] = multicomp[1]
# sigidx.append(np.where(multicomp[1] < 0.05)[0])
# nonsigidx.append(np.where(multicomp[1] >= 0.05)[0])

sigidx.append(np.where(pvalPerm[:, 0] < 0.05)[0])
nonsigidx.append(np.where(pvalPerm[:, 0] >= 0.05)[0])


# save to csv
validmegops.reset_index(inplace=True, drop=True)
univarCorr = pd.DataFrame(data=validmegops.values, columns=['CodeString'])

var = 0
univarCorr['PearsonR'] = rho[:, var]
univarCorr['absolutePearsonR'] = np.abs(rho[:, var])
univarCorr['spinPvalue'] = pvalPerm[:, var]
univarCorr['spinPvalue_fdrCorrected'] = corrected_pval[:, var]

univarCorr.sort_values('absolutePearsonR', ascending=False, inplace=True)

univarCorr.to_csv(outpath +
                  'pca/csvFiles/snr_allpearson_schaefer100.csv', index=False)

# PCA with non-sig features
featMat_snr_rm = featMat[:, nonsigidx[0]]
dataMat = scipy.stats.zscore(featMat_snr_rm)
pca = sklearn.decomposition.PCA(n_components=10, svd_solver='full')
pca.fit(dataMat)
node_score_snr = pca.transform(dataMat)
pc_wei_snr = pca.components_

# var explained in PCA
varexpall = pca.explained_variance_ratio_
myplot = sns.scatterplot(np.arange(len(varexpall)), varexpall,
                         facecolors='darkslategrey')
sns.despine(ax=myplot, trim=False)
myplot.axes.set_title('PCA - hctsa feat')
myplot.axes.set_xlabel('components')
myplot.axes.set_ylabel('variance explained (%)')
myplot.figure.set_figwidth(5)
myplot.figure.set_figheight(5)

plt.savefig(outpath + 'pca/varexp_PCA_schaefer100_snr_rm.svg',
            bbox_inches='tight', dpi=300,
            transparent=True)

toplot = node_score_snr[:, 0]
brains = fcn_megdynamics.plot_conte69(toplot, lhlabels, rhlabels,
                                      vmin=np.percentile(toplot, 0),
                                      vmax=np.percentile(toplot, 100),
                                      colormap='viridis', customcmap=megcmap,
                                      colorbartitle=('node score - PC %s - ' +
                                                     'VarExp = %0.3f')
                                                     % (ncomp+1, varexpall[ncomp]),
                                      surf='inflated')

mayavi.mlab.savefig(outpath + 'pca/pc1_snr_rm_lh_schaefer100.png', figure=brains[0])
mayavi.mlab.savefig(outpath + 'pca/pc1_snr_rm_rh_schaefer100.png', figure=brains[1])

toplot = avg_snr
brains = fcn_megdynamics.plot_conte69(toplot, lhlabels, rhlabels,
                                      vmin=np.percentile(toplot, 0),
                                      vmax=np.percentile(toplot, 100),
                                      colormap='viridis', customcmap=megcmap,
                                      colorbartitle='SNR (dB)',
                                      surf='inflated')

mayavi.mlab.savefig(outpath + 'pca/snr_lh_schaefer100.png', figure=brains[0])
mayavi.mlab.savefig(outpath + 'pca/snr_rh_schaefer100.png', figure=brains[1])

# correlate and plot
x = node_score[:, 0]
y = node_score_snr[:, 0]

corr = scipy.stats.spearmanr(x, y)
pvalspin = fcn_megdynamics.get_spinp(x, y, corrval=corr[0], nspin=10000,
                                     corrtype='spearman',
                                     lhannot=lhlabels, rhannot=rhlabels)

title = 'spearman r = %1.3f - p (spin) = %1.4f' % (corr[0], pvalspin)
xlab = 'PC1 - original'
ylab = 'PC1 - snrFeat removed'
plt.figure()
fcn_megdynamics.scatterregplot(x, y, title, xlab, ylab, 60)

plt.savefig(outpath + 'pca/pc1_feat_snr_rm.svg',
            bbox_inches='tight', dpi=300,
            transparent=True)


# univariate analysis of weis for snr removed features
featMat_snr_rm = featMat[:, nonsigidx[0]]
validmegops_snr_rm = validmegops.iloc[nonsigidx[0], :]
X = featMat_snr_rm
Y = node_score_snr[:, :1]
rho = np.zeros((X.shape[1], Y.shape[1]))
for i in range(X.shape[1]):
    for j in range(Y.shape[1]):
        tmpcorr = scipy.stats.pearsonr(X[:, i], Y[:, j])  # or pearsonr
        rho[i, j] = tmpcorr[0]

surf_path = (gitrepo_dir + 'data/surfaces/')

surfaces = [surf_path + 'L.sphere.32k_fs_LR.surf.gii',
            surf_path + 'R.sphere.32k_fs_LR.surf.gii']

centroids, hemiid = fcn_megdynamics.get_gifti_centroids(surfaces,
                                                        lhlabels,
                                                        rhlabels)
spins = netneurostats.gen_spinsamples(centroids, hemiid,
                                      n_rotate=10000, seed=272)

n_spins = spins.shape[1]
rhoPerm = np.zeros((X.shape[1], Y.shape[1], n_spins))
for spin in range(n_spins):
    for i in range(X.shape[1]):
        for j in range(Y.shape[1]):
            rhoPerm[i, j, spin] = scipy.stats.pearsonr(
                                    X[spins[:, spin], i],
                                    Y[:, j])[0]
    print('\nspin %i' % spin)

pvalPerm = np.zeros((X.shape[1], Y.shape[1]))
corrected_pval = np.zeros((X.shape[1], Y.shape[1]))
sigidx = []
nonsigidx = []
for comp in range(Y.shape[1]):
    for feat in range(X.shape[1]):
        permmean = np.mean(rhoPerm[feat, comp, :])
        pvalPerm[feat, comp] = (len(np.where(abs(rhoPerm[feat, comp, :]
                                                 - permmean) >=
                                             abs(rho[feat, comp]
                                                 - permmean))[0])+1)/(n_spins +
                                                                      1)
    multicomp = multitest.multipletests(pvalPerm[:, comp], alpha=0.05,
                                        method='fdr_bh')
    corrected_pval[:, comp] = multicomp[1]
    sigidx.append(np.where(multicomp[1] < 0.05)[0])
    nonsigidx.append(np.where(multicomp[1] >= 0.05)[0])


rhoCopy = rho.copy()
rhoCopy[nonsigidx[0], 0] = 0

var = 0
sortedCorr = np.sort(rho[:, var])
sortedrhoCopy = np.sort(rhoCopy[:, var])
sigthresh1 = np.max(np.where(sortedrhoCopy < 0)[0])
sigthresh2 = np.min(np.where(sortedrhoCopy > 0)[0])

myfig = plt.figure()
plt.plot(sortedCorr)
plt.vlines([sigthresh1, sigthresh2], ymin=-1, ymax=1)
plt.xlabel('# of features')
plt.ylabel('pearson r')
plt.title('PC%s, sigthresh1 = %s, sigthresh2 = %s' % (str(var+1),
                                                      str(sigthresh1),
                                                      str(sigthresh2)))
myfig.set_figwidth(7)
myfig.set_figheight(7)
plt.savefig(outpath + 'pca/PC%s_snr_rm_pearsonr_correctPval_fdr05_schaefer100.svg' % str(var+1),
            bbox_inches='tight', dpi=300,
            transparent=True)


# save to csv
validmegops_snr_rm.reset_index(inplace=True, drop=True)
univarCorr = pd.DataFrame(data=validmegops_snr_rm.values, columns=['CodeString'])

var = 0
univarCorr['Loading'] = rho[:, var]
univarCorr['absoluteLoading'] = np.abs(rho[:, var])
univarCorr['spinPvalue'] = pvalPerm[:, var]
univarCorr['spinPvalue_fdrCorrected'] = corrected_pval[:, var]

univarCorr.sort_values('absoluteLoading', ascending=False, inplace=True)

univarCorr.to_csv(outpath +
                  'pca/csvFiles/pc%s_snr_rm_allpearson_fdr05_schaefer100.csv'
                  % str(var+1), index=False)



## regress out snr
avg_snr = np.load(datapath + 'group_snrData_lcmv_Schaefer100.npy')

# correlate and plot
x = avg_snr
y = node_score[:, 0]

corr = scipy.stats.spearmanr(x, y)
pvalspin = fcn_megdynamics.get_spinp(x, y, corrval=corr[0], nspin=10000,
                                     corrtype='spearman',
                                     lhannot=lhlabels, rhannot=rhlabels)

title = 'spearman r = %1.3f - p (spin) = %1.4f' % (corr[0], pvalspin)
xlab = 'SNR (dB)'
ylab = 'PC1'
plt.figure()
fcn_megdynamics.scatterregplot(x, y, title, xlab, ylab, 60)

plt.savefig(outpath  + 'pca/pc1_snr.svg',
            bbox_inches='tight', dpi=300,
            transparent=True)

# regress snr
featmat_res = np.zeros_like(featMat)
for feat in range(featmat_res.shape[1]):
    x = scipy.stats.zscore(avg_snr[:, np.newaxis])
    y = scipy.stats.zscore(featMat[:, feat][:, np.newaxis])
    betas = np.linalg.lstsq(x, y, rcond=None)[0]

    # residuals = y-yhat = y-x@betas (betas: coef and intercept)
    featresid = y - (x @ betas)

    featmat_res[:, feat] = featresid.flatten()

dataMat = scipy.stats.zscore(featmat_res)
pca = sklearn.decomposition.PCA(n_components=10, svd_solver='full')
pca.fit(dataMat)
node_score_snrreg = pca.transform(dataMat)
pc_wei_snrreg = pca.components_

# var explained in PCA
varexpall = pca.explained_variance_ratio_
myplot = sns.scatterplot(np.arange(len(varexpall)), varexpall,
                         facecolors='darkslategrey')
sns.despine(ax=myplot, trim=False)
myplot.axes.set_title('PCA - hctsa feat')
myplot.axes.set_xlabel('components')
myplot.axes.set_ylabel('variance explained (%)')
myplot.figure.set_figwidth(5)
myplot.figure.set_figheight(5)

plt.savefig(outpath + 'pca/varexp_PCA_schaefer100_snrreg.svg',
            bbox_inches='tight', dpi=300,
            transparent=True)

toplot = node_score_snrreg[:, 0]
brains = fcn_megdynamics.plot_conte69(toplot, lhlabels, rhlabels,
                                      vmin=np.percentile(toplot, 0),
                                      vmax=np.percentile(toplot, 100),
                                      colormap='viridis', customcmap=megcmap,
                                      colorbartitle=('node score - PC %s - ' +
                                                     'VarExp = %0.3f')
                                                     % (ncomp+1, varexpall[ncomp]),
                                      surf='inflated')

mayavi.mlab.savefig(outpath + 'pca/pc1_snrreg_lh_schaefer100.png', figure=brains[0])
mayavi.mlab.savefig(outpath + 'pca/pc1_snrreg_rh_schaefer100.png', figure=brains[1])

# correlate and plot
x = node_score[:, 0]
y = node_score_snrreg[:, 0]

corr = scipy.stats.spearmanr(x, y)
pvalspin = fcn_megdynamics.get_spinp(x, y, corrval=corr[0], nspin=10000,
                                     corrtype='spearman',
                                     lhannot=lhlabels, rhannot=rhlabels)

title = 'spearman r = %1.3f - p (spin) = %1.4f' % (corr[0], pvalspin)
xlab = 'PC1 original'
ylab = 'PC1 snr regressed'
plt.figure()
fcn_megdynamics.scatterregplot(x, y, title, xlab, ylab, 60)

plt.savefig(outpath  + 'pca/pc1orig_pc1_snrregressed.svg',
            bbox_inches='tight', dpi=300,
            transparent=True)

####################################
# SNR empty room
####################################
# load data
avg_sharedTS_pca_noise = np.load(datapath + 'avg_hctsafeat_schaefer100_noise.npy')
sharedID_pca_noise = np.loadtxt(datapath + 'hctsafeatID_schaefer100_noise.txt')
sharedID_pca_noise = (np.array(sharedID_pca_noise)-1).astype(int)
megops = pd.read_csv(gitrepo_dir + 'data/ops_modified.txt', header=None)

xy, x_ind, y_ind = np.intersect1d(sharedID_pca, sharedID_pca_noise,
                                  return_indices=True)

validmegops_sharedwnoise = megops.iloc[xy]
sharedTS_orig = avg_sharedTS_pca[x_ind, :]
sharedTS_noise = avg_sharedTS_pca_noise[y_ind, :]

# PCA
featMat_orig = sharedTS_orig.T
featMat_noise = sharedTS_noise.T

dataMat = scipy.stats.zscore(featMat_orig)
pca = sklearn.decomposition.PCA(n_components=10, svd_solver='full')
pca.fit(dataMat)
node_score_orig = pca.transform(dataMat)
pc_wei_orig = pca.components_
scaledWei_orig = pca.components_.T @ np.diag(pca.singular_values_)


dataMat = scipy.stats.zscore(featMat_noise)
pca = sklearn.decomposition.PCA(n_components=10, svd_solver='full')
pca.fit(dataMat)
node_score_noise = pca.transform(dataMat)
pc_wei_noise = pca.components_
scaledWei_noise = pca.components_.T @ np.diag(pca.singular_values_)

# var explained in PCA
varexpall = pca.explained_variance_ratio_
myplot = sns.scatterplot(np.arange(len(varexpall)), varexpall,
                         facecolors='darkslategrey')
sns.despine(ax=myplot, trim=False)
myplot.axes.set_title('PCA - hctsa feat')
myplot.axes.set_xlabel('components')
myplot.axes.set_ylabel('variance explained (%)')
myplot.figure.set_figwidth(5)
myplot.figure.set_figheight(5)
plt.savefig(outpath + 'pca/emptyRoom/varexp_PCA_schaefer100_noise.svg',
            bbox_inches='tight', dpi=300,
            transparent=True)

# align weights
temp = [pc_wei_orig.T, pc_wei_noise.T]
realigned_wei, _ = iterative_alignment(temp, n_iters=1)

node_score_noise_realigned = featMat_noise @ realigned_wei[1][:, 0]

toplot = node_score_noise_realigned
brains = fcn_megdynamics.plot_conte69(toplot, lhlabels, rhlabels,
                                      vmin=np.percentile(toplot, 2.5),
                                      vmax=np.percentile(toplot, 97.5),
                                      colormap='viridis', customcmap=megcmap,
                                      colorbartitle=('node score - PC %s - ' +
                                                     'VarExp = %0.3f')
                                                     % (ncomp+1, varexpall[ncomp]),
                                      surf='inflated')

mayavi.mlab.savefig(outpath + 'pca/emptyRoom/pc%s_lh_schaefer100.png' % (ncomp+1), figure=brains[0])
mayavi.mlab.savefig(outpath + 'pca/emptyRoom/pc%s_rh_schaefer100.png' % (ncomp+1), figure=brains[1])

# univariate analysis of weis
X = featMat_noise
Y = node_score_noise_realigned[:, np.newaxis]
rho = np.zeros((X.shape[1], Y.shape[1]))
for i in range(X.shape[1]):
    for j in range(Y.shape[1]):
        tmpcorr = scipy.stats.pearsonr(X[:, i], Y[:, j])  # or pearsonr
        rho[i, j] = tmpcorr[0]

# save to csv
validmegops_sharedwnoise.reset_index(inplace=True, drop=True)
univarCorr = pd.DataFrame(data=validmegops_sharedwnoise.values, columns=['CodeString'])

var = 0
univarCorr['Loading'] = rho[:, var]
univarCorr['absoluteLoading'] = np.abs(rho[:, var])

univarCorr.sort_values('absoluteLoading', ascending=False, inplace=True)

univarCorr.to_csv(outpath +
                  'pca/emptyRoom/pc%s_allpearson_noise_realign.csv'
                  % str(var+1), index=False)


# correlate and plot
x = node_score_orig[:, 0]
y = node_score_noise_realigned

corr = scipy.stats.spearmanr(x, y)
pvalspin = fcn_megdynamics.get_spinp(x, y, corrval=corr[0], nspin=10000,
                                     corrtype='spearman',
                                     lhannot=lhlabels, rhannot=rhlabels)

title = 'spearman r = %1.3f - p (spin) = %1.4f' % (corr[0], pvalspin)
xlab = 'PC1 original'
ylab = 'PC1 noise realigned'
plt.figure()
fcn_megdynamics.scatterregplot(x, y, title, xlab, ylab, 60)

plt.savefig(outpath  + 'pca/emptyRoom/pc1orig_pc1emptyRoom_realign.svg',
            bbox_inches='tight', dpi=300,
            transparent=True)
