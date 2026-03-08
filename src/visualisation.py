"""
visualisation.py — Generalised topology visualisation for 2D and 3D cases.

Provides 2D grid plots for planar topologies and 3D voxel plots for
volumetric topologies. Supports both single-material (density) and
multi-material (coloured phase) representations.

Set HEADLESS = True before importing to use the Agg backend (saves to file
instead of displaying). Or call set_headless() before any plotting.
"""

import os
import numpy as np

# Allow headless mode via environment variable
HEADLESS = os.environ.get("MPLBACKEND", "").lower() == "agg"


def set_headless(headless=True):
    """Switch to Agg backend for saving plots to file."""
    global HEADLESS
    HEADLESS = headless
    if headless:
        import matplotlib
        matplotlib.use('Agg')


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)


def _setup_figure(ax, figsize=(10, 5), projection=None):
    """Create figure/axes if needed."""
    if ax is None:
        if not HEADLESS:
            plt.ion()
        if projection == '3d':
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def _finish_plot(fig, pause=0.01, save_path=None):
    """Draw, pause, optionally save."""
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if not HEADLESS:
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(pause)


def plot_topology_2d(x, nelx, nely, title="2D Topology", ax=None,
                     pause=0.01, save_path=None):
    """
    Visualise a 2D single-material density field on a grid.

    Parameters
    ----------
    x : ndarray (nelx*nely,)
        Element densities.
    nelx, nely : int
        Mesh dimensions.
    title : str
    ax : matplotlib Axes, optional
    pause : float
    save_path : str, optional
        If provided, save figure to this path.
    """
    fig, ax = _setup_figure(ax, figsize=(12, 4))
    ax.clear()
    density = x.reshape((nelx, nely)).T
    ax.imshow(
        1.0 - density,
        cmap='gray',
        interpolation='none',
        origin='upper',
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title(title)
    ax.set_xlabel('x elements')
    ax.set_ylabel('y elements')
    ax.set_aspect('equal')
    _finish_plot(fig, pause, save_path)
    return fig, ax


def plot_topology_2d_multi(alpha, nelx, nely, colors,
                           title="2D Multi-Material Topology",
                           ax=None, pause=0.01, save_path=None):
    """
    Visualise a 2D multi-material topology as an RGB bitmap.

    Parameters
    ----------
    alpha : ndarray (nelx*nely, p)
    nelx, nely : int
    colors : ndarray (p, 3)
    title : str
    ax : matplotlib Axes, optional
    pause : float
    save_path : str, optional
    """
    fig, ax = _setup_figure(ax, figsize=(12, 4))
    ax.clear()
    nel = nelx * nely
    p = alpha.shape[1]
    I = np.zeros((nel, 3))
    for j in range(p):
        I += alpha[:, j:j+1] * colors[j:j+1, :]

    I = np.clip(I, 0, 1)
    I = I.reshape((nelx, nely, 3)).transpose(1, 0, 2)
    ax.imshow(I, interpolation='none', origin='upper')
    ax.set_title(title)
    ax.set_xlabel('x elements')
    ax.set_ylabel('y elements')
    ax.set_aspect('equal')
    _finish_plot(fig, pause, save_path)
    return fig, ax


def plot_topology_3d(x, nelx, nely, nelz, threshold=0.5,
                     title="3D Topology", ax=None, pause=0.5,
                     save_path=None):
    """
    Visualise a 3D single-material topology using voxels.

    Only elements with density above `threshold` are displayed.

    Parameters
    ----------
    x : ndarray (nelx*nely*nelz,)
    nelx, nely, nelz : int
    threshold : float
    title : str
    ax : matplotlib Axes3D, optional
    pause : float
    save_path : str, optional
    """
    fig, ax = _setup_figure(ax, figsize=(12, 8), projection='3d')
    ax.clear()

    # Reshape: element indexing is (iz * nelx * nely + ix * nely + iy)
    density = x.reshape((nelz, nelx, nely))
    voxels = np.transpose(density, (1, 2, 0))  # (nelx, nely, nelz)
    filled = voxels > threshold

    # Colour by density
    colors_arr = np.zeros(filled.shape + (4,))
    for ix in range(nelx):
        for iy in range(nely):
            for iz in range(nelz):
                if filled[ix, iy, iz]:
                    d = voxels[ix, iy, iz]
                    colors_arr[ix, iy, iz] = [0.2, 0.4, 0.8, min(d, 1.0)]

    ax.voxels(filled, facecolors=colors_arr, edgecolor='k', linewidth=0.1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    max_range = max(nelx, nely, nelz)
    ax.set_xlim(0, max_range)
    ax.set_ylim(0, max_range)
    ax.set_zlim(0, max_range)

    _finish_plot(fig, pause, save_path)
    return fig, ax


def plot_topology_3d_multi(alpha, nelx, nely, nelz, colors, threshold=0.3,
                           title="3D Multi-Material Topology", ax=None,
                           pause=0.5, save_path=None):
    """
    Visualise a 3D multi-material topology using coloured voxels.

    Parameters
    ----------
    alpha : ndarray (nel, p)
    nelx, nely, nelz : int
    colors : ndarray (p, 3)
    threshold : float
    title : str
    ax : matplotlib Axes3D, optional
    pause : float
    save_path : str, optional
    """
    fig, ax = _setup_figure(ax, figsize=(12, 8), projection='3d')
    ax.clear()

    nel = nelx * nely * nelz
    p = alpha.shape[1]

    # Compute RGB per element
    rgb = np.zeros((nel, 3))
    for j in range(p):
        rgb += alpha[:, j:j+1] * colors[j:j+1, :]
    rgb = np.clip(rgb, 0, 1)

    # Void is the last phase
    void_frac = alpha[:, -1] if p > 1 else np.zeros(nel)
    solid_frac = 1.0 - void_frac

    solid_3d = solid_frac.reshape((nelz, nelx, nely))
    rgb_3d = rgb.reshape((nelz, nelx, nely, 3))

    solid_3d = np.transpose(solid_3d, (1, 2, 0))
    rgb_3d = np.transpose(rgb_3d, (1, 2, 0, 3))

    filled = solid_3d > threshold
    colors_arr = np.zeros(filled.shape + (4,))
    for ix in range(nelx):
        for iy in range(nely):
            for iz in range(nelz):
                if filled[ix, iy, iz]:
                    colors_arr[ix, iy, iz, :3] = rgb_3d[ix, iy, iz, :]
                    colors_arr[ix, iy, iz, 3] = min(solid_3d[ix, iy, iz], 1.0)

    ax.voxels(filled, facecolors=colors_arr, edgecolor='k', linewidth=0.1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    max_range = max(nelx, nely, nelz)
    ax.set_xlim(0, max_range)
    ax.set_ylim(0, max_range)
    ax.set_zlim(0, max_range)

    _finish_plot(fig, pause, save_path)
    return fig, ax


def show_final():
    """Call at end of optimisation to keep plot windows open."""
    if not HEADLESS:
        plt.ioff()
        plt.show()
