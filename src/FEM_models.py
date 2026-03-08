"""
FEM_models.py — Shared Finite Element Method code for topology optimisation.

Provides element stiffness matrices, FE assembly preparation, and density filters
for both 2D (Q4 bilinear quad) and 3D (8-node hexahedral) elements.

References:
    - Sigmund, O. (2001). "A 99 line topology optimization code written in Matlab."
    - Tavakoli & Mohseni (2014). "Alternating active-phase algorithm for
      multimaterial topology optimization problems."
"""

import numpy as np
from scipy.sparse import coo_matrix


# =============================================================================
# 2D Finite Element Code
# =============================================================================

def element_stiffness_2d(nu=0.3):
    """
    Compute the 8x8 element stiffness matrix for a 2D bilinear Q4 element
    under plane stress, unit Young's modulus, unit element size.

    This is the exact formulation from Sigmund's 99-line code and the
    multitop3.m MATLAB code.

    Parameters
    ----------
    nu : float
        Poisson's ratio (default 0.3).

    Returns
    -------
    KE : ndarray (8, 8)
        Element stiffness matrix.
    """
    A11 = np.array([
        [12,  3, -6, -3],
        [ 3, 12,  3,  0],
        [-6,  3, 12, -3],
        [-3,  0, -3, 12]
    ], dtype=float)
    A12 = np.array([
        [-6, -3,  0,  3],
        [-3, -6, -3, -6],
        [ 0, -3, -6,  3],
        [ 3, -6,  3, -6]
    ], dtype=float)
    B11 = np.array([
        [-4,  3, -2,  9],
        [ 3, -4, -9,  4],
        [-2, -9, -4, -3],
        [ 9,  4, -3, -4]
    ], dtype=float)
    B12 = np.array([
        [ 2, -3,  4, -9],
        [-3,  2,  9, -2],
        [ 4,  9,  2,  3],
        [-9, -2,  3,  2]
    ], dtype=float)

    KE = (1.0 / (1.0 - nu**2) / 24.0) * (
        np.block([[A11, A12], [A12.T, A11]])
        + nu * np.block([[B11, B12], [B12.T, B11]])
    )
    return KE


def prepare_fe_2d(nelx, nely):
    """
    Prepare index arrays for efficient 2D FE assembly.

    Matches the vectorised assembly from Sigmund's code (88-line style).

    Parameters
    ----------
    nelx : int
        Number of elements in x-direction.
    nely : int
        Number of elements in y-direction.

    Returns
    -------
    iK : ndarray (64*nelx*nely,)
        Row indices for sparse stiffness matrix.
    jK : ndarray (64*nelx*nely,)
        Column indices for sparse stiffness matrix.
    edofMat : ndarray (nelx*nely, 8)
        Element DOF connectivity matrix.
    freedofs : ndarray
        Free DOF indices.
    fixeddofs : ndarray
        Fixed DOF indices.
    F : ndarray
        Load vector.
    ndof : int
        Total number of DOFs.
    """
    ndof = 2 * (nelx + 1) * (nely + 1)

    # Node numbering (column-major, matching MATLAB reshape order)
    nodenrs = np.arange(1, (1 + nelx) * (1 + nely) + 1).reshape(
        (1 + nely), (1 + nelx), order='F'
    )

    # Element DOF vector — top-left node of each element, first DOF
    edofVec = (2 * nodenrs[0:-1, 0:-1] + 1).reshape(-1, order='F')

    # Full element DOF matrix (8 DOF per element)
    # DOF order: [n1x, n1y, n2x, n2y, n3x, n3y, n4x, n4y]
    # Nodes: bottom-left, bottom-right, top-right, top-left
    offsets = np.array([0, 1, 2*nely + 2, 2*nely + 3, 2*nely, 2*nely + 1, -2, -1])
    edofMat = np.tile(edofVec.reshape(-1, 1), (1, 8)) + np.tile(offsets, (nelx * nely, 1))

    # Index arrays for sparse assembly
    iK = np.kron(edofMat, np.ones((1, 8), dtype=int)).flatten()
    jK = np.kron(edofMat, np.ones((8, 1), dtype=int)).flatten()

    # --- Cantilever Boundary Condition ---
    # Fixed: all DOFs on left edge (x=0)
    # Load: downward point load at bottom-right corner
    fixeddofs = np.arange(0, 2 * (nely + 1))  # 0-indexed
    F = np.zeros((ndof, 1))
    F[2 * (nelx + 1) * (nely + 1) - 1, 0] = -1.0  # Bottom-right corner, y-direction

    alldofs = np.arange(ndof)
    freedofs = np.setdiff1d(alldofs, fixeddofs)

    # Convert to 0-indexed
    edofMat = edofMat - 1

    return iK - 1, jK - 1, edofMat, freedofs, fixeddofs, F, ndof


def make_filter_2d(nelx, nely, rmin):
    """
    Build the density/sensitivity filter matrix for 2D problems.

    Parameters
    ----------
    nelx : int
        Number of elements in x.
    nely : int
        Number of elements in y.
    rmin : float
        Filter radius.

    Returns
    -------
    H : sparse matrix (nel, nel)
        Filter weight matrix.
    Hs : ndarray (nel,)
        Row sums of H.
    """
    nfilter = nelx * nely * ((2 * (int(np.ceil(rmin)) - 1) + 1) ** 2)
    iH = np.zeros(nfilter, dtype=int)
    jH = np.zeros(nfilter, dtype=int)
    sH = np.zeros(nfilter)
    k = 0

    for i1 in range(nelx):
        for j1 in range(nely):
            e1 = i1 * nely + j1
            ir = int(np.ceil(rmin)) - 1
            for i2 in range(max(i1 - ir, 0), min(i1 + ir + 1, nelx)):
                for j2 in range(max(j1 - ir, 0), min(j1 + ir + 1, nely)):
                    e2 = i2 * nely + j2
                    iH[k] = e1
                    jH[k] = e2
                    sH[k] = max(0.0, rmin - np.sqrt((i1 - i2)**2 + (j1 - j2)**2))
                    k += 1

    H = coo_matrix((sH[:k], (iH[:k], jH[:k])),
                    shape=(nelx * nely, nelx * nely)).tocsc()
    Hs = np.array(H.sum(axis=1)).flatten()
    return H, Hs


# =============================================================================
# 3D Finite Element Code
# =============================================================================

def element_stiffness_3d(nu=0.3):
    """
    Compute the 24x24 element stiffness matrix for an 8-node hexahedral
    element with unit Young's modulus and unit element size.

    Uses the analytical integration for a regular hexahedron under
    isotropic linear elasticity.

    Parameters
    ----------
    nu : float
        Poisson's ratio (default 0.3).

    Returns
    -------
    KE : ndarray (24, 24)
        Element stiffness matrix.
    """
    # Constitutive matrix (3D isotropic)
    E0 = 1.0
    C = E0 / ((1.0 + nu) * (1.0 - 2.0 * nu)) * np.array([
        [1-nu, nu,   nu,   0,            0,            0           ],
        [nu,   1-nu, nu,   0,            0,            0           ],
        [nu,   nu,   1-nu, 0,            0,            0           ],
        [0,    0,    0,    (1-2*nu)/2.0, 0,            0           ],
        [0,    0,    0,    0,            (1-2*nu)/2.0, 0           ],
        [0,    0,    0,    0,            0,            (1-2*nu)/2.0],
    ])

    # 2x2x2 Gauss quadrature
    gp = 1.0 / np.sqrt(3.0)
    gauss_pts = np.array([-gp, gp])
    weights = np.array([1.0, 1.0])

    KE = np.zeros((24, 24))

    for xi_w, xi in zip(weights, gauss_pts):
        for eta_w, eta in zip(weights, gauss_pts):
            for zeta_w, zeta in zip(weights, gauss_pts):
                # Shape function derivatives (natural coords) for 8-node hex
                # Node ordering: (0,0,0), (1,0,0), (1,1,0), (0,1,0),
                #                (0,0,1), (1,0,1), (1,1,1), (0,1,1)
                dN_dxi = np.array([
                    [-(1-eta)*(1-zeta), (1-eta)*(1-zeta), (1+eta)*(1-zeta),
                     -(1+eta)*(1-zeta), -(1-eta)*(1+zeta), (1-eta)*(1+zeta),
                     (1+eta)*(1+zeta), -(1+eta)*(1+zeta)],
                    [-(1-xi)*(1-zeta), -(1+xi)*(1-zeta), (1+xi)*(1-zeta),
                     (1-xi)*(1-zeta), -(1-xi)*(1+zeta), -(1+xi)*(1+zeta),
                     (1+xi)*(1+zeta), (1-xi)*(1+zeta)],
                    [-(1-xi)*(1-eta), -(1+xi)*(1-eta), -(1+xi)*(1+eta),
                     -(1-xi)*(1+eta), (1-xi)*(1-eta), (1+xi)*(1-eta),
                     (1+xi)*(1+eta), (1-xi)*(1+eta)]
                ]) / 8.0

                # Jacobian for unit cube [0,1]^3 → physical
                # For a unit cube, J = 0.5 * I (element size = 1)
                J = 0.5 * np.eye(3)
                detJ = np.linalg.det(J)
                invJ = np.linalg.inv(J)

                # Derivatives in physical coords
                dN_dx = invJ @ dN_dxi

                # Strain-displacement matrix B (6x24)
                B = np.zeros((6, 24))
                for n in range(8):
                    B[0, 3*n]   = dN_dx[0, n]  # εxx
                    B[1, 3*n+1] = dN_dx[1, n]  # εyy
                    B[2, 3*n+2] = dN_dx[2, n]  # εzz
                    B[3, 3*n]   = dN_dx[1, n]  # γxy
                    B[3, 3*n+1] = dN_dx[0, n]
                    B[4, 3*n+1] = dN_dx[2, n]  # γyz
                    B[4, 3*n+2] = dN_dx[1, n]
                    B[5, 3*n]   = dN_dx[2, n]  # γxz
                    B[5, 3*n+2] = dN_dx[0, n]

                KE += xi_w * eta_w * zeta_w * (B.T @ C @ B) * detJ

    return KE


def prepare_fe_3d(nelx, nely, nelz):
    """
    Prepare index arrays for efficient 3D FE assembly.

    Parameters
    ----------
    nelx : int
        Number of elements in x.
    nely : int
        Number of elements in y.
    nelz : int
        Number of elements in z.

    Returns
    -------
    iK : ndarray
        Row indices for sparse stiffness matrix.
    jK : ndarray
        Column indices for sparse stiffness matrix.
    edofMat : ndarray (nel, 24)
        Element DOF connectivity.
    freedofs : ndarray
        Free DOF indices.
    fixeddofs : ndarray
        Fixed DOF indices.
    F : ndarray
        Load vector.
    ndof : int
        Total DOFs.
    """
    nel = nelx * nely * nelz
    nnode = (nelx + 1) * (nely + 1) * (nelz + 1)
    ndof = 3 * nnode

    # Node numbering: nodes indexed as (iy, ix, iz) with y fastest, then x, then z
    # This matches a column-major like ordering
    def node_id(ix, iy, iz):
        return iz * (nelx + 1) * (nely + 1) + ix * (nely + 1) + iy

    # Build element DOF matrix
    edofMat = np.zeros((nel, 24), dtype=int)
    elem = 0
    for iz in range(nelz):
        for ix in range(nelx):
            for iy in range(nely):
                # 8 nodes of hex element
                # Bottom face (z=iz): n0(ix,iy), n1(ix+1,iy), n2(ix+1,iy+1), n3(ix,iy+1)
                # Top face (z=iz+1): n4(ix,iy), n5(ix+1,iy), n6(ix+1,iy+1), n7(ix,iy+1)
                n = [
                    node_id(ix, iy, iz),
                    node_id(ix + 1, iy, iz),
                    node_id(ix + 1, iy + 1, iz),
                    node_id(ix, iy + 1, iz),
                    node_id(ix, iy, iz + 1),
                    node_id(ix + 1, iy, iz + 1),
                    node_id(ix + 1, iy + 1, iz + 1),
                    node_id(ix, iy + 1, iz + 1),
                ]
                dofs = []
                for ni in n:
                    dofs.extend([3 * ni, 3 * ni + 1, 3 * ni + 2])
                edofMat[elem, :] = dofs
                elem += 1

    # Index arrays for sparse assembly
    iK = np.kron(edofMat, np.ones((1, 24), dtype=int)).flatten()
    jK = np.kron(edofMat, np.ones((24, 1), dtype=int)).flatten()

    # --- Cantilever Boundary Condition ---
    # Fixed: all DOFs on face x=0
    # Load: uniform downward load on face x=nelx (distributed over all nodes on that face)
    fixed_nodes = []
    for iz in range(nelz + 1):
        for iy in range(nely + 1):
            fixed_nodes.append(node_id(0, iy, iz))
    fixeddofs = np.array([], dtype=int)
    for n in fixed_nodes:
        fixeddofs = np.append(fixeddofs, [3*n, 3*n+1, 3*n+2])

    # Uniform load on face x=nelx, direction -y
    F = np.zeros((ndof, 1))
    load_nodes = []
    for iz in range(nelz + 1):
        for iy in range(nely + 1):
            load_nodes.append(node_id(nelx, iy, iz))
    n_load = len(load_nodes)
    for n in load_nodes:
        F[3*n + 1, 0] = -1.0 / n_load  # Distribute load evenly, y-direction

    alldofs = np.arange(ndof)
    freedofs = np.setdiff1d(alldofs, fixeddofs)

    return iK, jK, edofMat, freedofs, fixeddofs, F, ndof


def make_filter_3d(nelx, nely, nelz, rmin):
    """
    Build the density/sensitivity filter matrix for 3D problems.

    Parameters
    ----------
    nelx : int
        Elements in x.
    nely : int
        Elements in y.
    nelz : int
        Elements in z.
    rmin : float
        Filter radius.

    Returns
    -------
    H : sparse matrix (nel, nel)
        Filter weight matrix.
    Hs : ndarray (nel,)
        Row sums of H.
    """
    nel = nelx * nely * nelz
    ir = int(np.ceil(rmin)) - 1
    nfilter = nel * ((2 * ir + 1) ** 3)
    iH = np.zeros(nfilter, dtype=int)
    jH = np.zeros(nfilter, dtype=int)
    sH = np.zeros(nfilter)
    k = 0

    for i1 in range(nelx):
        for j1 in range(nely):
            for k1 in range(nelz):
                e1 = k1 * nelx * nely + i1 * nely + j1
                for i2 in range(max(i1 - ir, 0), min(i1 + ir + 1, nelx)):
                    for j2 in range(max(j1 - ir, 0), min(j1 + ir + 1, nely)):
                        for k2 in range(max(k1 - ir, 0), min(k1 + ir + 1, nelz)):
                            e2 = k2 * nelx * nely + i2 * nely + j2
                            dist = np.sqrt(
                                (i1 - i2)**2 + (j1 - j2)**2 + (k1 - k2)**2
                            )
                            if k < nfilter:
                                iH[k] = e1
                                jH[k] = e2
                                sH[k] = max(0.0, rmin - dist)
                                k += 1

    H = coo_matrix((sH[:k], (iH[:k], jH[:k])),
                    shape=(nel, nel)).tocsc()
    Hs = np.array(H.sum(axis=1)).flatten()
    return H, Hs
