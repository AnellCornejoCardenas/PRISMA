#Imports
from matplotlib import __version__
import matplotlib.cbook as cbook
from matplotlib import colors as mcol
from matplotlib import collections as collec
import matplotlib.transforms as mtransforms
import matplotlib.path as mpltPath
import numpy as np
import matplotlib.pyplot as plt

class hex_grid(object):
    """
    """     
    def __init__(self, x, y, cellsize=0, gridsize=100, extent=None, xscale='linear', yscale='linear') -> None:
        super().__init__()

        x, y = cbook.delete_masked_points(x, y)
        self.x = np.array(x, float)
        self.y = np.array(y, float)

        self.xscale = xscale
        self.yscale = yscale
        self._logscale()

        self.extent = extent
        self._extent()
        
        self.cellsize = int(cellsize)
        self.gridsize = int(gridsize)
        self._gridsize()
        self._offsets = None

        self.grid()


    def _logscale(self):
        if self.xscale == 'log':
            if np.any(self.x <= 0.0): raise ValueError("x contains non-positive values, so can not" " be log-scaled")
            self.x = np.log10(self.x)
        if self.yscale == 'log':
            if np.any(self.y <= 0.0): raise ValueError("y contains non-positive values, so can not" " be log-scaled")
            self.y = np.log10(self.y)
    

    def _extent(self):
        if self.extent is not None:
            if len(self.extent) == 4: self.xmin, self.xmax, self.ymin, self.ymax = self.extent
            elif len(self.extent) == 2:
                self.xmin, self.xmax = self.extent
                self.ymin, self.ymax = self.extent
            else: raise ValueError("extent lenght must to be 2 or 4")
        else:
            self.xmin, self.xmax = (np.min(self.x), np.max(self.x)) if len(self.x) else (0, 1)
            self.ymin, self.ymax = (np.min(self.y), np.max(self.y)) if len(self.y) else (0, 1)

            # to avoid issues with singular data, expand the min/max pairs
            self.xmin, self.xmax = mtransforms.nonsingular(self.xmin, self.xmax, expander=0.1)
            self.ymin, self.ymax = mtransforms.nonsingular(self.ymin, self.ymax, expander=0.1)

            # In the x-direction, the hexagons exactly cover the region from
            # xmin to xmax. Need some padding to avoid roundoff errors.
            padding = 1.e-9 * (self.xmax - self.xmin)
            self.xmin -= padding
            self.xmax += padding


    def _gridsize(self):
        # Set the size of the hexagon grid
        if self.cellsize == 0:
            self.nx = self.gridsize
            self.sx = (self.xmax - self.xmin) / self.nx
            self.cellsize = self.sx
        else:
            self.sx = self.cellsize
            self.nx = np.ceil((self.xmax - self.xmin) / self.sx).astype(int)
            self.gridsize = self.nx

            padding = np.ceil((self.xmax - self.xmin)/self.nx)
            padding = padding*self.nx - (self.xmax - self.xmin)
            self.xmin -= padding/2
            self.xmax += padding/2

        #self.ny = np.ceil((self.ymax - self.ymin) / self.sx / np.sqrt(3)).astype(int)
        #self.sy = (self.ymax - self.ymin) / self.ny
        self.sy = self.sx * np.sqrt(3)
        self.ny = np.ceil((self.ymax - self.ymin) / self.sy).astype(int)

        self._polygon = [self.sx, self.sy/3] * np.array([[.5, -.5], [.5, .5], [0., 1.], [-.5, .5], [-.5, -.5], [0., -1.]])


    def grid(self):
        x = (self.x - self.xmin) / self.sx
        y = (self.y - self.ymin) / self.sy

        ix1 = np.round(x).astype(int)
        iy1 = np.round(y).astype(int)
        ix2 = np.floor(x).astype(int)
        iy2 = np.floor(y).astype(int)

        self._nx1 = self.nx + 1
        self._ny1 = self.ny + 1
        self._nx2 = self.nx
        self._ny2 = self.ny
        self._n = self._nx1 * self._ny1 + self._nx2 * self._ny2

        d1 = (x - ix1) ** 2 + 3.0 * (y - iy1) ** 2
        d2 = (x - ix2 - 0.5) ** 2 + 3.0 * (y - iy2 - 0.5) ** 2
        self._bdist = (d1 < d2)

        # create accumulation arrays
        """
        lattice1 = np.empty((self._nx1*self._ny1), dtype=object)
        d1 = (iy1+self._ny1*ix1)[self._bdist]
        c1 = np.unique(d1)      
        ind = np.nonzero(self._bdist)[0]
        lattice1[c1] = [ind[(d1==i)] for i in c1]
        lattice2 = np.empty((self._nx2*self._ny2), dtype=object)
        d2 = (iy2+self._ny2*ix2)[~self._bdist]
        c2 = np.unique(d2)      
        ind = np.nonzero(~self._bdist)[0]
        lattice2[c2] = [ind[(d2==i)] for i in c2]
        self.lattice = np.concatenate([lattice1, lattice2])
        self.n_accum = np.array([len(i) if i is not None else 0 for i in self.lattice])
        """ 

        lattice1 = np.empty((self._nx1, self._ny1), dtype=object)
        for i in range(self._nx1):
            for j in range(self._ny1): lattice1[i, j] = []
        
        lattice2 = np.empty((self._nx2, self._ny2), dtype=object)
        for i in range(self._nx2):
            for j in range(self._ny2): lattice2[i, j] = []
        
        if self.extent is None:
            for i in range(len(x)):
                if self._bdist[i]: lattice1[ix1[i], iy1[i]].append(i)
                else: lattice2[ix2[i], iy2[i]].append(i)
        else:
            for i in range(len(x)):
                if self._bdist[i]:
                    if 0 <= ix1[i] < self._nx1 and 0 <= iy1[i] < self._ny1: lattice1[ix1[i], iy1[i]].append(i)
                else:
                    if 0 <= ix2[i] < self._nx2 and 0 <= iy2[i] < self._ny2: lattice2[ix2[i], iy2[i]].append(i)
        
        self.lattice = np.concatenate([lattice1.ravel(), lattice2.ravel()])
        self.n_accum = np.array([len(i) for i in self.lattice])
        self.good_idxs = self.n_accum > 0
   
        return self.lattice, self.n_accum


    def _set_offsets(self):
        self._offsets = np.zeros((self._n, 2), float)
        self._offsets[:self._nx1 * self._ny1, 0] = np.repeat(np.arange(self._nx1), self._ny1)
        self._offsets[:self._nx1 * self._ny1, 1] = np.tile(np.arange(self._ny1), self._nx1)
        self._offsets[self._nx1 * self._ny1:, 0] = np.repeat(np.arange(self._nx2) + 0.5, self._ny2)
        self._offsets[self._nx1 * self._ny1:, 1] = np.tile(np.arange(self._ny2), self._nx2) + 0.5
        self._offsets[:, 0] *= self.sx
        self._offsets[:, 1] *= self.sy
        self._offsets[:, 0] += self.xmin
        self._offsets[:, 1] += self.ymin


    def _make_grid(self, ax, idxs=slice(None)):
        # remove accumulation bins with no data
        if self._offsets is None: self._set_offsets()
        offsets = self._offsets[idxs, :]

        if self.xscale == 'log' or self.yscale == 'log':
            polygons = np.expand_dims(self._polygon, 0) + np.expand_dims(offsets, 1)
            if self.xscale == 'log':
                polygons[:, :, 0] = 10.0 ** polygons[:, :, 0]
                self.xmin = 10.0 ** self.xmin
                self.xmax = 10.0 ** self.xmax
            if self.yscale == 'log':
                polygons[:, :, 1] = 10.0 ** polygons[:, :, 1]
                self.ymin = 10.0 ** self.ymin
                self.ymax = 10.0 ** self.ymax
            grid = collec.PolyCollection(polygons)
        else:
            grid = collec.PolyCollection([self._polygon],offsets=offsets, transOffset=mtransforms.AffineDeltaTransform(ax.transData) if __version__ > '3.3' else mtransforms.IdentityTransform())
            
            #grid = collec.PolyCollection([self._polygon],offsets=offsets,
            #    transOffset=mtransforms.AffineDeltaTransform(ax.transData) if __version__ > '3.3' else mtransforms.IdentityTransform(), offset_position="data")  #original
        return grid


    def get_grid(self, ax , idxs=slice(None)):
        grid = self._make_grid(ax , idxs=idxs)
        if self.xscale == 'log' or self.yscale == 'log': grid = grid.get_paths()
        else: grid = (grid.get_offsets().reshape(-1,1,2) + grid.get_paths()[0].vertices.reshape(-1,7,2))
        return grid


    def eval_C(self, C, mincnt=0, reduce_C_function=np.mean):
        if reduce_C_function == "pesado":
            AA = C[0]
            MM = C[1]
            self.c_accum = np.array([np.sum(AA[l]*MM[l])/np.sum(MM[l]) if self.n_accum[i] > 0 else np.nan for i,l in enumerate(self.lattice)])
        else:
            self.c_accum = np.array([reduce_C_function(C[l]) if self.n_accum[i] > 0 else np.nan for i,l in enumerate(self.lattice)])
        self.c_accum[np.isnan(self.c_accum)] = np.nanmin(self.c_accum)


    def hexbin(self, C=None, mincnt=1, bins=None, reduce_C_function=np.mean, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, edgecolors='face',marginals=False, ax=plt.axes(), **kwargs):

        if C is None: accum = self.n_accum
        else: 
            self.eval_C(C, mincnt=mincnt, reduce_C_function=reduce_C_function)
            accum = self.c_accum
           
        # remove accumulation bins with no data
        good_idxs = slice(None) if not mincnt else self.good_idxs & (self.n_accum >= mincnt)
        self._collection = self._make_grid(ax , good_idxs)
        accum = accum[good_idxs]

        # Set normalizer if bins is 'log'
        if bins == 'log':
            if norm is not None: api.warn_external("Only one of 'bins' and 'norm' arguments can be supplied, ignoring bins={bins}")
            else:
                norm = mcol.LogNorm(vmin=vmin, vmax=vmax)
                vmin = vmax = None
            bins = None

        # autoscale the norm with current accum values if it hasn't been set
        if norm is not None:
            if norm.vmin is None and norm.vmax is None: norm.autoscale(accum)

        if bins is not None:
            if not np.iterable(bins):
                minimum, maximum = min(accum), max(accum)
                bins -= 1  # one less edge than bins
                bins = minimum + (maximum - minimum) * np.arange(bins) / bins
            bins = np.sort(bins)
            accum = bins.searchsorted(accum)

        if linewidths is None: linewidths = [1.0]

        self._collection.set_array(accum)
        self._collection.set_cmap(cmap)
        self._collection.set_norm(norm)
        self._collection.set_alpha(alpha)
        self._collection.set_linewidths(linewidths)
        self._collection.set_edgecolors(edgecolors)
        self._collection.update(kwargs)

        if __version__ > '3.3': self._collection._scale_norm(norm, vmin, vmax)
        else:
            if vmin is not None or vmax is not None: self._collection.set_clim(vmin, vmax)
            else: self._collection.autoscale_None()

        if self.xscale == 'log': ax.set_xscale(self.xscale)
        if self.yscale == 'log': ax.set_yscale(self.yscale)

        corners = ((self.xmin, self.ymin), (self.xmax, self.ymax))
        ax.update_datalim(corners)
        ax._request_autoscale_view(tight=True)

        # add the collection last
        ax.add_collection(self._collection, autolim=False)        
        return self._collection


    def marginals(self, C=None, mincnt=1, bins=None, reduce_C_function=np.mean,
               cmap=None, norm=None, vmin=None, vmax=None,
               alpha=None, linewidths=None, edgecolors='face',
               marginals=False, ax=plt.axes, **kwargs):
      
        if C is None: C = np.ones(self.x.size)
        xorig = self.x.copy()
        yorig = self.y.copy()

        def coarse_bin(x, y, bin_edges):
            """
            Sort x-values into bins defined by *bin_edges*, then for all the
            corresponding y-values in each bin use *reduce_c_function* to
            compute the bin value.
            """
            nbins = len(bin_edges) - 1
            # Sort x-values into bins
            bin_idxs = np.searchsorted(bin_edges, self.x) - 1
            mus = np.zeros(nbins) * np.nan
            for i in range(nbins):
                # Get y-values for each bin
                yi = y[bin_idxs == i]
                if len(yi) > 0: mus[i] = reduce_C_function(yi)
            return mus

        if self.xscale == 'log': bin_edges = np.geomspace(self.xmin, self.xmax, self.nx + 1)
        else: bin_edges = np.linspace(self.xmin, self.xmax, self.nx + 1)
        xcoarse = coarse_bin(xorig, C, bin_edges)

        verts, values = [], []
        for bin_left, bin_right, val in zip(bin_edges[:-1], bin_edges[1:], xcoarse):
            if np.isnan(val): continue
            verts.append([(bin_left, 0),
                          (bin_left, 0.05),
                          (bin_right, 0.05),
                          (bin_right, 0)])
            values.append(val)
        values = np.array(values)

        trans = ax.get_xaxis_transform(which='grid')

        hbar = collec.PolyCollection(verts, transform=trans, edgecolors='face')
        hbar.set_array(values)
        hbar.set_cmap(cmap)
        hbar.set_norm(norm)
        hbar.set_alpha(alpha)
        hbar.update(kwargs)
        ax.add_collection(hbar, autolim=False)

        if self.yscale == 'log':
            bin_edges = np.geomspace(self.ymin, self.ymax, 2 * self.ny + 1)
        else:
            bin_edges = np.linspace(self.ymin, self.ymax, 2 * self.ny + 1)
        ycoarse = coarse_bin(yorig, C, bin_edges)

        verts, values = [], []
        for bin_bottom, bin_top, val in zip(bin_edges[:-1], bin_edges[1:], ycoarse):
            if np.isnan(val): continue
            verts.append([(0, bin_bottom),
                          (0, bin_top),
                          (0.05, bin_top),
                          (0.05, bin_bottom)])
            values.append(val)
        values = np.array(values)

        trans = ax.get_yaxis_transform(which='grid')

        vbar = collec.PolyCollection(verts, transform=trans, edgecolors='face')
        vbar.set_array(values)
        vbar.set_cmap(cmap)
        vbar.set_norm(norm)
        vbar.set_alpha(alpha)
        vbar.update(kwargs)
        ax.add_collection(vbar, autolim=False)
        
        self._collection.hbar = hbar
        self._collection.vbar = vbar
        
        def on_changed(collection):
            hbar.set_cmap(collection.get_cmap())
            hbar.set_clim(collection.get_clim())
            vbar.set_cmap(collection.get_cmap())
            vbar.set_clim(collection.get_clim())

        if __version__ > '3.3': self._collection.callbacks.connect('changed', on_changed)
        else: self._collection.callbacksSM.connect('changed', on_changed)

        return self._collection


    def get_FOV(self, size=0):
        if size==0: size = self.xmax - self.xmin
        fov = self._polygon * np.sqrt(3)/2 * size
        fov = np.vstack([fov, fov[0]])

        paths = mpltPath.Path(fov, closed=True)
        infov = paths.contains_points(self._offsets)

        return fov, infov