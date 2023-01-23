import scipy
import sklearn
import numpy as np
import nibabel as nib
import seaborn as sns
import sklearn.metrics
import palettable as pal
from scipy.spatial.distance import cdist
from netneurotools import stats as netneurostats
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap


def get_gifti_centroids(surfaces, lhannot, rhannot):
    lhsurface, rhsurface = [nib.load(s) for s in surfaces]

    centroids, hemiid = [], []
    for n, (annot, surf) in enumerate(zip([lhannot, rhannot],
                                          [lhsurface, rhsurface])):
        vert, face = [d.data for d in surf.darrays]
        labels = np.squeeze(nib.load(annot).darrays[0].data)

        for lab in np.unique(labels):
            if lab == 0:
                continue
            coords = np.atleast_2d(vert[labels == lab].mean(axis=0))
            roi = vert[np.argmin(cdist(vert, coords), axis=0)[0]]
            centroids.append(roi)
            hemiid.append(n)

    centroids = np.row_stack(centroids)
    hemiid = np.asarray(hemiid)

    return centroids, hemiid


def get_spinp(x, y, corrval, nspin, lhannot, rhannot, corrtype):
    surf_path = ('/Users/gshafiei/gitrepos/shafiei_megdynamics/data/surfaces/')
    surfaces = [surf_path + 'L.sphere.32k_fs_LR.surf.gii',
                surf_path + 'R.sphere.32k_fs_LR.surf.gii']
    lhannot = lhannot
    rhannot = rhannot

    centroids, hemiid = get_gifti_centroids(surfaces, lhannot,
                                                         rhannot)
    spins = netneurostats.gen_spinsamples(centroids, hemiid,
                                          n_rotate=nspin, seed=272)

    permuted_r = np.zeros((nspin, 1))
    for spin in range(nspin):
        if corrtype == 'spearman':
            permuted_r[spin] = scipy.stats.spearmanr(x[spins[:, spin]], y)[0]
        elif corrtype == 'pearson':
            permuted_r[spin] = scipy.stats.pearsonr(x[spins[:, spin]], y)[0]

    permmean = np.mean(permuted_r)
    pvalspin = (len(np.where(abs(permuted_r - permmean) >=
                             abs(corrval - permmean))[0])+1)/(nspin+1)
    return pvalspin


def get_spinidx(nspin, lhannot, rhannot):
    surf_path = ('/Users/gshafiei/gitrepos/shafiei_megdynamics/data/surfaces/')
    surfaces = [surf_path + 'L.sphere.32k_fs_LR.surf.gii',
                surf_path + 'R.sphere.32k_fs_LR.surf.gii']
    lhannot = lhannot
    rhannot = rhannot

    centroids, hemiid = get_gifti_centroids(surfaces, lhannot,
                                                         rhannot)
    spins = netneurostats.gen_spinsamples(centroids, hemiid,
                                          n_rotate=nspin, seed=272)
    return spins


def scatterregplot(x, y, title, xlab, ylab, pointsize):
    myplot = sns.scatterplot(x, y,
                             facecolor=np.array([128/255, 128/255, 128/255]),
                             legend=False, rasterized=True)
    sns.regplot(x, y, scatter=False, ax=myplot,
                line_kws=dict(color='k'))
    sns.despine(ax=myplot, trim=False)
    myplot.axes.set_title(title)
    myplot.axes.set_xlabel(xlab)
    myplot.axes.set_ylabel(ylab)
    myplot.figure.set_figwidth(5)
    myplot.figure.set_figheight(5)
    return myplot


def plot_conte69(data, lhlabel, rhlabel, surf='midthickness',
                 vmin=None, vmax=None, colormap='viridis', customcmap=None,
                 colorbar=True, num_labels=4, orientation='horizontal',
                 colorbartitle=None, backgroundcolor=(1, 1, 1),
                 foregroundcolor=(0, 0, 0), **kwargs):

    """
    Plots surface `data` on Conte69 Atlas

    (This is a modified version of plotting.plot_conte69 from netneurotools.
     This version will be merged with the one on netneurotools in future.)

    Parameters
    ----------
    data : (N,) array_like
        Surface data for N parcels
    lhlabel : str
        Path to .gii file (generic GIFTI file) containing labels to N/2 parcels
        on the left hemisphere
    rhlabel : str
        Path to .gii file (generic GIFTI file) containing labels to N/2 parcels
        on the right hemisphere
    surf : {'midthickness', 'inflated', 'vinflated'}, optional
        Type of brain surface. Default: 'midthickness'
    vmin : float, optional
        Minimum value to scale the colormap. If None, the min of the data will
        be used. Default: None
    vmax : float, optional
        Maximum value to scale the colormap. If None, the max of the data will
        be used. Default: None
    colormap : str, optional
        Any colormap from matplotlib. Default: 'viridis'
    colorbar : bool, optional
        Wheter to display a colorbar. Default: True
    num_labels : int, optional
        The number of labels to display on the colorbar.
        Available only if colorbar=True. Default: 4
    orientation : str, optional
        Defines the orientation of colorbar. Can be 'horizontal' or 'vertical'.
        Available only if colorbar=True. Default: 'horizontal'
    colorbartitle : str, optional
        The title of colorbar. Available only if colorbar=True. Default: None
    backgroundcolor : tuple of float values with RGB code in [0, 1], optional
        Defines the background color. Default: (1, 1, 1)
    foregroundcolor : tuple of float values with RGB code in [0, 1], optional
        Defines the foreground color (e.g., colorbartitle color).
        Default: (0, 0, 0)
    kwargs : key-value mapping
        Keyword arguments for `mayavi.mlab.triangular_mesh()`

    Returns
    -------
    scene : mayavi.Scene
        Scene object containing plot
    """

    from netneurotools.datasets import fetch_conte69
    try:
        from mayavi import mlab
    except ImportError:
        raise ImportError('Cannot use plot_conte69() if mayavi is not '
                          'installed. Please install mayavi and try again.')

    opts = dict()
    opts.update(**kwargs)

    try:
        surface = fetch_conte69()[surf]
    except KeyError:
        raise ValueError('Provided surf "{}" is not valid. Must be one of '
                         '[\'midthickness\', \'inflated\', \'vinflated\']'
                         .format(surf))
    lhsurface, rhsurface = [nib.load(s) for s in surface]

    lhlabels = nib.load(lhlabel).darrays[0].data
    rhlabels = nib.load(rhlabel).darrays[0].data
    lhvert, lhface = [d.data for d in lhsurface.darrays]
    rhvert, rhface = [d.data for d in rhsurface.darrays]

    # add NaNs for subcortex
    data = np.append(np.nan, data)

    # get lh and rh data
    lhdata = np.squeeze(data[lhlabels.astype(int)])
    rhdata = np.squeeze(data[rhlabels.astype(int)])

    # plot
    lhplot = mlab.figure()
    rhplot = mlab.figure()
    lhmesh = mlab.triangular_mesh(lhvert[:, 0], lhvert[:, 1], lhvert[:, 2],
                                  lhface, figure=lhplot, colormap=colormap,
                                  mask=np.isnan(lhdata), scalars=lhdata,
                                  vmin=vmin, vmax=vmax, **opts)
    lhmesh.module_manager.scalar_lut_manager.lut.nan_color = [0.863, 0.863,
                                                              0.863, 1]
    lhmesh.update_pipeline()
    if type(customcmap) != str:
        lut = lhmesh.module_manager.scalar_lut_manager.lut.table.to_array()
        lut[:, :3] = customcmap.colors * 255
        lhmesh.module_manager.scalar_lut_manager.lut.table = lut
        mlab.draw()
    if colorbar is True:
        mlab.colorbar(title=colorbartitle, nb_labels=num_labels,
                      orientation=orientation)
    rhmesh = mlab.triangular_mesh(rhvert[:, 0], rhvert[:, 1], rhvert[:, 2],
                                  rhface, figure=rhplot, colormap=colormap,
                                  mask=np.isnan(rhdata), scalars=rhdata,
                                  vmin=vmin, vmax=vmax, **opts)
    rhmesh.module_manager.scalar_lut_manager.lut.nan_color = [0.863, 0.863,
                                                              0.863, 1]
    rhmesh.update_pipeline()
    if type(customcmap) != str:
        lut = rhmesh.module_manager.scalar_lut_manager.lut.table.to_array()
        lut[:, :3] = customcmap.colors * 255
        rhmesh.module_manager.scalar_lut_manager.lut.table = lut
        mlab.draw()
    if colorbar is True:
        mlab.colorbar(title=colorbartitle, nb_labels=num_labels,
                      orientation=orientation)
    mlab.view(azimuth=180, elevation=90, distance=450, figure=lhplot)
    mlab.view(azimuth=180, elevation=-90, distance=450, figure=rhplot)

    mlab.figure(bgcolor=backgroundcolor, fgcolor=foregroundcolor,
                figure=lhplot)
    mlab.figure(bgcolor=backgroundcolor, fgcolor=foregroundcolor,
                figure=rhplot)

    return lhplot, rhplot


def make_colormaps():
    cmap_seq = LinearSegmentedColormap.from_list('mycmap', list(reversed(
            np.array(pal.cmocean.sequential.Matter_20.mpl_colors[:-1]))))

    cmap_seq_r = LinearSegmentedColormap.from_list('mycmap', list(
                   np.array(pal.cmocean.sequential.Matter_20.mpl_colors[:-1])))

    cmap_seq_v2 = LinearSegmentedColormap.from_list('mycmap', list(reversed(
                np.array(pal.cartocolors.sequential.SunsetDark_7.mpl_colors))))

    cmap_seq_v2_disc = ListedColormap(list(reversed(
                np.array(pal.cartocolors.sequential.SunsetDark_6.mpl_colors))))

    cmap_OrYel = LinearSegmentedColormap.from_list('mycmap', list(
                np.array(pal.cartocolors.sequential.OrYel_7.mpl_colors)))
    # PinkYl_7

    # test = np.abs(np.random.rand(5, 12))
    # plt.figure()
    # plt.imshow(test, interpolation='nearest', cmap=cmap_seq)
    # plt.colorbar()

    colors = np.vstack([np.array(cmap_seq(i)[:3]) for i in range(256)])
    megcmap = ListedColormap(colors)

    colors = np.vstack([np.array(cmap_seq_v2(i)[:3]) for i in range(256)])
    megcmap2 = ListedColormap(colors)

    colors = np.vstack([np.array(cmap_seq_v2_disc(i)[:3]) for i in range(6)])
    categ_cmap = ListedColormap(np.vstack((np.ones((43, 3)) * colors[0],
                                           np.ones((43, 3)) * colors[1],
                                           np.ones((43, 3)) * colors[2],
                                           np.ones((43, 3)) * colors[3],
                                           np.ones((42, 3)) * colors[4],
                                           np.ones((42, 3)) * colors[5])))

    return cmap_seq, cmap_seq_r, megcmap, megcmap2, categ_cmap, cmap_OrYel
