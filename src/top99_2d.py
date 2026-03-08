"""
top99_2d.py — 2D Single-Material Topology Optimisation (Sigmund 99-line code).

A faithful Python implementation of Sigmund's classic SIMP-based topology
optimisation code for minimum compliance of 2D structures.

Reference:
    Sigmund, O. (2001). "A 99 line topology optimization code written in Matlab."
    Structural and Multidisciplinary Optimization, 21(2), 120-127.

Usage:
    python top99_2d.py                          # Default cantilever 60x20
    python top99_2d.py --nelx 80 --nely 40      # Custom mesh
    python top99_2d.py --save                   # Save final plot to file (headless)

Boundary condition: Cantilever beam
    - Left edge fully fixed
    - Point load downward at bottom-right corner
"""

import argparse
import sys
import os

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

# Add parent path for module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def top99_2d(nelx=60, nely=20, volfrac=0.5, penal=3.0, rmin=1.5,
             max_iter=200, headless=False):
    """
    2D topology optimisation using SIMP method with OC update.

    Parameters
    ----------
    nelx : int
        Number of elements in x.
    nely : int
        Number of elements in y.
    volfrac : float
        Target volume fraction (0 < volfrac < 1).
    penal : float
        Penalisation power for SIMP (typically 3).
    rmin : float
        Filter radius (typically ~1.5).
    max_iter : int
        Maximum number of iterations.
    headless : bool
        If True, use Agg backend and save final plot to file.
    """
    # Set headless mode BEFORE importing matplotlib-dependent modules
    if headless:
        import matplotlib
        matplotlib.use('Agg')
        from visualisation import set_headless, plot_topology_2d, show_final
        set_headless(True)
    else:
        from visualisation import plot_topology_2d, show_final

    from FEM_models import element_stiffness_2d, prepare_fe_2d, make_filter_2d
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("2D Single-Material Topology Optimisation (Sigmund 99-line)")
    print(f"  Mesh: {nelx} x {nely} = {nelx*nely} elements")
    print(f"  Volume fraction: {volfrac}")
    print(f"  Penalisation: {penal}, Filter radius: {rmin}")
    print("=" * 60)

    nel = nelx * nely
    Emin = 1e-9   # Minimum stiffness (void)
    E0 = 1.0      # Solid stiffness

    # === ELEMENT STIFFNESS MATRIX ===
    KE = element_stiffness_2d(nu=0.3)

    # === PREPARE FE ASSEMBLY ===
    iK, jK, edofMat, freedofs, fixeddofs, F, ndof = prepare_fe_2d(nelx, nely)

    # === BUILD FILTER ===
    H, Hs = make_filter_2d(nelx, nely, rmin)

    # === INITIALISE DESIGN VARIABLES ===
    x = np.full(nel, volfrac)          # Current density
    xPhys = x.copy()                   # Physical (filtered) density
    U = np.zeros(ndof)

    # === OPTIMISATION LOOP ===
    change = 1.0
    iteration = 0

    if not headless:
        plt.ion()
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    while change > 0.01 and iteration < max_iter:
        iteration += 1

        # --- FE Analysis ---
        # Element Young's moduli (SIMP)
        E = Emin + xPhys ** penal * (E0 - Emin)

        # Assemble global stiffness matrix
        sK = (KE.flatten(order='F')[np.newaxis]).T * E[np.newaxis]
        sK = sK.flatten(order='F')
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        K = (K + K.T) / 2.0

        # Solve KU = F
        U[freedofs] = spsolve(K[freedofs, :][:, freedofs], F[freedofs].flatten())

        # --- Objective & Sensitivity ---
        Ue = U[edofMat]  # (nel, 8)
        ce = np.sum((Ue @ KE) * Ue, axis=1)  # (nel,)
        obj = np.sum(E * ce)
        dc = -penal * xPhys ** (penal - 1) * (E0 - Emin) * ce
        dv = np.ones(nel)

        # --- Apply Filter ---
        dc = np.array(H * (xPhys * dc / Hs)[:, np.newaxis]).flatten()
        dv = np.array(H * (dv / Hs)[:, np.newaxis]).flatten()

        # --- OC Update ---
        x_old = x.copy()
        x = oc_update(nel, x, volfrac, dc, dv, move=0.2)
        xPhys = np.array(H * x[:, np.newaxis] / Hs[:, np.newaxis]).flatten()

        change = np.max(np.abs(x - x_old))

        # --- Print & Visualise ---
        print(f"  Iter: {iteration:4d}  Obj: {obj:11.4f}  Vol: {np.mean(xPhys):6.3f}  Change: {change:8.4f}")

        if not headless:
            plot_topology_2d(xPhys, nelx, nely,
                            title=f"Iter {iteration}, Obj={obj:.2f}, Change={change:.4f}",
                            ax=ax, pause=0.01)

    # Save final result
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'top99_2d_{nelx}x{nely}.png')
    plot_topology_2d(xPhys, nelx, nely,
                    title=f"Final: Iter {iteration}, Obj={obj:.2f}",
                    ax=ax, save_path=save_path)
    print(f"\n  Final topology saved to: {save_path}")

    print("=" * 60)
    print(f"Converged after {iteration} iterations. Final objective: {obj:.4f}")
    print("=" * 60)

    if not headless:
        show_final()

    return xPhys, obj


def oc_update(nel, x, volfrac, dc, dv, move=0.2):
    """
    Optimality Criteria (OC) update of design variables.

    Parameters
    ----------
    nel : int
        Number of elements.
    x : ndarray (nel,)
        Current densities.
    volfrac : float
        Target volume fraction.
    dc : ndarray (nel,)
        Compliance sensitivities.
    dv : ndarray (nel,)
        Volume sensitivities.
    move : float
        Move limit.

    Returns
    -------
    xnew : ndarray (nel,)
        Updated densities.
    """
    l1, l2 = 0.0, 1e9
    while (l2 - l1) / (l1 + l2) > 1e-3:
        lmid = 0.5 * (l2 + l1)
        xnew = np.maximum(
            0.001,
            np.maximum(
                x - move,
                np.minimum(
                    1.0,
                    np.minimum(
                        x + move,
                        x * np.sqrt(-dc / dv / lmid)
                    )
                )
            )
        )
        if np.sum(xnew) > volfrac * nel:
            l1 = lmid
        else:
            l2 = lmid
    return xnew


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="2D Single-Material Topology Optimisation (Sigmund 99-line)"
    )
    parser.add_argument("--nelx", type=int, default=60, help="Elements in x (default: 60)")
    parser.add_argument("--nely", type=int, default=20, help="Elements in y (default: 20)")
    parser.add_argument("--volfrac", type=float, default=0.5, help="Volume fraction (default: 0.5)")
    parser.add_argument("--penal", type=float, default=3.0, help="SIMP penalty (default: 3.0)")
    parser.add_argument("--rmin", type=float, default=1.5, help="Filter radius (default: 1.5)")
    parser.add_argument("--maxiter", type=int, default=200, help="Max iterations (default: 200)")
    parser.add_argument("--save", action="store_true", help="Headless mode: save plot to file")
    args = parser.parse_args()

    top99_2d(
        nelx=args.nelx,
        nely=args.nely,
        volfrac=args.volfrac,
        penal=args.penal,
        rmin=args.rmin,
        max_iter=args.maxiter,
        headless=args.save,
    )
