"""
multitop_2d.py — 2D Multi-Material Topology Optimisation.

Faithful Python implementation of the alternating active-phase algorithm
from Tavakoli & Mohseni, matching the multitop3.m MATLAB code.

The multi-material problem is decomposed into pairwise binary sub-problems
(bi_top), each solved with the OC method.

Reference:
    Tavakoli & Mohseni (2014). "Alternating active-phase algorithm for
    multimaterial topology optimization problems: a 115-line MATLAB
    implementation."

Usage:
    python multitop_2d.py                    # Default: 40x40, 4 phases
    python multitop_2d.py --nx 60 --ny 30    # Custom mesh
    python multitop_2d.py --save             # Headless mode

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def multitop_2d(nx=40, ny=40, tol_out=0.001, tol_f=0.05,
                iter_max_in=2, iter_max_out=500,
                p=4, q=3,
                e=None, v=None, rf=3.0,
                colors=None, headless=False):
    """
    2D multi-material topology optimisation.

    Parameters
    ----------
    nx, ny : int
        Mesh dimensions.
    tol_out : float
        Outer loop convergence tolerance.
    tol_f : float
        Filter adaptation tolerance.
    iter_max_in : int
        Inner iterations per binary sub-problem.
    iter_max_out : int
        Maximum outer iterations.
    p : int
        Number of material phases (including void).
    q : float
        SIMP penalty exponent.
    e : array-like (p,)
        Young's moduli per phase (last should be ~0 for void).
    v : array-like (p,)
        Volume fractions per phase (must sum to 1).
    rf : float
        Filter radius.
    colors : array-like (p, 3)
        RGB colours per phase.
    headless : bool
        If True, save output to file instead of displaying.
    """
    if headless:
        import matplotlib
        matplotlib.use('Agg')
        from visualisation import set_headless, plot_topology_2d_multi, show_final
        set_headless(True)
    else:
        from visualisation import plot_topology_2d_multi, show_final

    from FEM_models import element_stiffness_2d, prepare_fe_2d, make_filter_2d
    import matplotlib.pyplot as plt

    # Default material properties: 3 materials + void
    if e is None:
        e = np.array([3.0, 2.0, 1.0, 1e-9])
    else:
        e = np.array(e, dtype=float)
    if v is None:
        v = np.array([0.2, 0.2, 0.2, 0.4])
    else:
        v = np.array(v, dtype=float)
    if colors is None:
        colors = np.array([
            [1.0, 0.0, 0.0],   # Red   (material 1, stiffest)
            [0.0, 0.0, 0.45],  # Blue  (material 2)
            [0.0, 1.0, 0.0],   # Green (material 3)
            [1.0, 1.0, 1.0],   # White (void)
        ])
    else:
        colors = np.array(colors, dtype=float)

    p = len(e)

    print("=" * 60)
    print("2D Multi-Material Topology Optimisation")
    print(f"  Mesh: {nx} x {ny} = {nx*ny} elements")
    print(f"  Phases: {p} (including void)")
    print(f"  Young's moduli: {e}")
    print(f"  Volume fractions: {v}")
    print(f"  SIMP penalty q={q}, Filter radius rf={rf}")
    print("=" * 60)

    nel = nx * ny

    # === ELEMENT STIFFNESS MATRIX ===
    KE = element_stiffness_2d(nu=0.3)

    # === PREPARE FE ASSEMBLY ===
    iK, jK, edofMat, freedofs, fixeddofs, F, ndof = prepare_fe_2d(nx, ny)

    # === BUILD FILTER ===
    H, Hs = make_filter_2d(nx, ny, rf)

    # === INITIALISE DESIGN VARIABLES ===
    # alpha(i, j) = volume fraction of phase j in element i
    alpha = np.zeros((nel, p))
    for i in range(p):
        alpha[:, i] = v[i]

    # === OUTER ITERATION LOOP ===
    change_out = 2 * tol_out
    iter_out = 0

    if not headless:
        plt.ion()
    fig_ax = [None, None]

    while (iter_out < iter_max_out) and (change_out > tol_out):
        alpha_old = alpha.copy()

        # Alternating active-phase: loop over all material pairs (a, b)
        for a in range(p):
            for b in range(a + 1, p):
                obj, alpha = bi_top(
                    a, b, nx, ny, p, v, e, q, alpha,
                    H, Hs, iter_max_in,
                    KE, iK, jK, edofMat, freedofs, F, ndof
                )

        iter_out += 1
        change_out = np.max(np.abs(alpha.flatten() - alpha_old.flatten()))

        print(f"  Iter: {iter_out:5d}  Obj: {obj:11.4f}  Change: {change_out:10.8f}")

        # Update filter if convergence is slow
        if (change_out < tol_f) and (rf > 3):
            tol_f = 0.99 * tol_f
            rf = 0.99 * rf
            H, Hs = make_filter_2d(nx, ny, rf)

        # Visualise
        if not headless and iter_out % 5 == 0:
            fig, ax = plot_topology_2d_multi(
                alpha, nx, ny, colors,
                title=f"Iter {iter_out}, Obj={obj:.2f}, Change={change_out:.6f}",
            )
            fig_ax = [fig, ax]

    # Save final result
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'multitop_2d_{nx}x{ny}.png')

    fig, ax = plot_topology_2d_multi(
        alpha, nx, ny, colors,
        title=f"Final: Iter {iter_out}, Obj={obj:.2f}",
        save_path=save_path,
    )
    print(f"\n  Final topology saved to: {save_path}")
    print("=" * 60)
    print(f"Converged after {iter_out} iterations. Final objective: {obj:.4f}")
    print("=" * 60)

    if not headless:
        show_final()

    return alpha, obj


def bi_top(a, b, nx, ny, p, v, e, q, alpha_old,
           H, Hs, iter_max_in,
           KE, iK, jK, edofMat, freedofs, F, ndof):
    """
    Binary sub-problem solver: optimise between phases a and b.

    This is the core of the alternating active-phase algorithm.
    Matches the bi_top function in multitop3.m exactly.

    Parameters
    ----------
    a, b : int
        Indices of the two phases being optimised.
    nx, ny : int
        Mesh dimensions.
    p : int
        Total number of phases.
    v : ndarray (p,)
        Target volume fractions.
    e : ndarray (p,)
        Young's moduli.
    q : float
        SIMP penalty.
    alpha_old : ndarray (nel, p)
        Current phase fractions.
    H, Hs : sparse matrix, ndarray
        Filter matrix and row sums.
    iter_max_in : int
        Number of inner iterations.
    KE : ndarray (8, 8)
        Element stiffness matrix.
    iK, jK : ndarray
        Sparse index arrays.
    edofMat : ndarray (nel, 8)
        Element DOF connectivity.
    freedofs : ndarray
        Free DOFs.
    F : ndarray
        Load vector.
    ndof : int
        Total DOFs.

    Returns
    -------
    obj : float
        Objective function value.
    alpha : ndarray (nel, p)
        Updated phase fractions.
    """
    nel = nx * ny
    alpha = alpha_old.copy()
    U = np.zeros(ndof)

    for iter_in in range(iter_max_in):
        # --- FE Analysis ---
        # Effective Young's modulus: E = sum_i e_i * alpha_i^q
        E = np.zeros(nel)
        for phase in range(p):
            E += e[phase] * alpha[:, phase] ** q

        # Assemble stiffness
        sK = (KE.flatten(order='F')[np.newaxis]).T * E[np.newaxis]
        sK = sK.flatten(order='F')
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        K = (K + K.T) / 2.0

        # Solve
        U[freedofs] = spsolve(K[freedofs, :][:, freedofs], F[freedofs].flatten())

        # --- Objective & Sensitivity ---
        Ue = U[edofMat]
        ce = np.sum((Ue @ KE) * Ue, axis=1)
        obj = np.sum(E * ce)

        # Sensitivity w.r.t. alpha_a for the (a,b) pair
        dc = -(q * (e[a] - e[b]) * alpha[:, a] ** (q - 1)) * ce

        # --- Filter sensitivities ---
        dc = np.array(
            H * (alpha[:, a] * dc)[:, np.newaxis]
        ).flatten() / Hs / np.maximum(1e-3, alpha[:, a])
        dc = np.minimum(dc, 0.0)

        # --- Compute bounds ---
        move = 0.2
        # r = available fraction after removing all other phases except a and b
        r = np.ones(nel)
        for k in range(p):
            if (k != a) and (k != b):
                r = r - alpha[:, k]

        lower = np.maximum(0.0, alpha[:, a] - move)
        upper = np.minimum(r, alpha[:, a] + move)

        # --- OC Update (bisection on Lagrange multiplier) ---
        l1, l2 = 0.0, 1e9
        while (l2 - l1) / (l1 + l2) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            alpha_a = np.maximum(
                lower,
                np.minimum(
                    upper,
                    alpha[:, a] * np.sqrt(-dc / lmid)
                )
            )
            if np.sum(alpha_a) > nel * v[a]:
                l1 = lmid
            else:
                l2 = lmid

        alpha[:, a] = alpha_a
        alpha[:, b] = r - alpha_a

    return obj, alpha


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="2D Multi-Material Topology Optimisation"
    )
    parser.add_argument("--nx", type=int, default=40, help="Elements in x (default: 40)")
    parser.add_argument("--ny", type=int, default=40, help="Elements in y (default: 40)")
    parser.add_argument("--maxiter", type=int, default=500, help="Max outer iters (default: 500)")
    parser.add_argument("--save", action="store_true", help="Headless mode: save to file")
    args = parser.parse_args()

    multitop_2d(
        nx=args.nx,
        ny=args.ny,
        iter_max_out=args.maxiter,
        headless=args.save,
    )
