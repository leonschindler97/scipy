"""Multipoint boundary value problem solver."""
from warnings import warn

import numpy as np

from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu
from scipy.integrate._bvp import (
    EPS,
    TERMINATION_MESSAGES,
    stacked_matmul,
    print_iteration_header,
    print_iteration_progress,
    modify_mesh,
    estimate_rms_residuals,
    collocation_fun,
    create_spline,
    BVPResult)

def estimate_fun_jac(fun, X, Y, F0=None):
    n = Y[0].shape[0]
    N = len(X)
    M = np.sum([x_i.size for x_i in X])

    dtype = Y[0].dtype

    df_dy = np.empty((n, n, M), dtype=dtype)

    l = 0 # Node counter
    for i in range(N):
        y_i = Y[i]

        h = EPS**0.5 * (1 + np.abs(Y[i]))
        m = X[i].size
        if F0 is None:
            f0 = fun(X[i], Y[i], region=i)
        else:
            f0 = F0[i]
        for j in range(n):
            y_i_new = y_i.copy()
            y_i_new[j] += h[j]
            hi = y_i_new[j] - y_i[j]
            f_new = fun(X[i], y_i_new, region=i)
            df_dy[:, j, l:l+m] = (f_new - f0) / hi
        l += m

    return df_dy


def estimate_bc_jac(bc, Ya, Yb, bc0=None):
    N = Ya.shape[0] # Number of regions
    n = Ya.shape[1] 

    if bc0 is None:
        bc0 = bc(Ya, Yb)

    dtype = Ya.dtype

    dbc_dya = np.empty((N * n, N * n), dtype=dtype)
    h = EPS**0.5 * (1 + np.abs(Ya))
    for i in range(N):
        for j in range(n):
            Ya_new = Ya.copy()
            Ya_new[i, j] += h[i, j]
            bc_new = bc(Ya_new, Yb)
            dbc_dya[:, i*n + j] = (bc_new - bc0) / h[i, j]

    h = EPS**0.5 * (1 + np.abs(Yb))
    dbc_dyb = np.empty((N * n, N * n), dtype=dtype)
    for i in range(N):
        for j in range(n):
            Yb_new = Yb.copy()
            Yb_new[i, j] += h[i, j]
            bc_new = bc(Ya, Yb_new)
            dbc_dyb[:, i*n + j] = (bc_new - bc0) / h[i, j]

    return dbc_dya, dbc_dyb


def compute_jac_indices(n, X):
    N = len(X) # Number of regions
    M = sum([x_i.size for x_i in X]) # Total number of nodes
    K = sum([x_i.size - 1 for x_i in X]) # Total number of regions

    i_col = np.repeat(np.arange(K * n), n)
    j_cols = []

    l = 0 # Node counter
    for x_i in X:
        m = x_i.size; b = m - 1
        j_col = (np.tile(np.arange(n), n * b) + # subindex
                    np.repeat(np.arange(b) * n + l * n, n**2)) # node index
        j_cols.append(j_col)
        l += m
    j_col = np.hstack(j_cols)

    i_bc = np.tile(np.arange(K * n, M * n), n * N)
    j_bcs_a = []; j_bcs_b = []

    l = 0 # Node counter
    for x_i in X:
        m = x_i.size; b = m - 1
        j_bc_a = np.repeat(np.arange(n), N * n) + l * n
        j_bc_b = np.repeat(np.arange(n), N * n) + (l + b) * n

        j_bcs_a.append(j_bc_a)
        j_bcs_b.append(j_bc_b)
        l += m
    j_bc_a = np.hstack(j_bcs_a)
    j_bc_b = np.hstack(j_bcs_b)

    i = np.hstack((i_col, i_col,     i_bc, i_bc))
    j = np.hstack((j_col, j_col + n, j_bc_a, j_bc_b))

    return i, j


def construct_global_jac(n, X,
                         i_jac, j_jac, 
                         df_dy, df_dy_middle, 
                         dbc_dya, dbc_dyb):
    # Total number of regions
    K = sum([x_i.size - 1 for x_i in X])

    df_dy = np.transpose(df_dy, (2, 0, 1))
    df_dy_middle = np.transpose(df_dy_middle, (2, 0, 1))

    dtype = df_dy.dtype

    # Computing diagonal n x n blocks.
    dPhi_dy_0 = np.empty((K, n, n), dtype=dtype)
    dPhi_dy_0[:] = -np.identity(n)
    l = 0 # Node counter
    k = 0 # Segment counter
    for x_i in X:
        m = x_i.size; b = m - 1

        # Segment length 
        h_i = np.diff(x_i)
        h_i = h_i[:, np.newaxis, np.newaxis]

        dPhi_dy_0[k:k+b] -= h_i / 6 * (df_dy[l:l+m-1] + 2 * df_dy_middle[k:k+b])
        T = stacked_matmul(df_dy_middle[k:k+b], df_dy[l:l+m-1])
        dPhi_dy_0[k:k+b]  -= h_i**2 / 12 * T

        l += m
        k += b

    # Computing off-diagonal n x n blocks.
    dPhi_dy_1 = np.empty((K, n, n), dtype=dtype)
    dPhi_dy_1[:] = np.identity(n)
    l = 0 # Node counter
    k = 0 # Segment counter
    for x_i in X:
        m = x_i.size; b = m - 1

        # Segment length 
        h_i = np.diff(x_i)
        h_i = h_i[:, np.newaxis, np.newaxis]

        dPhi_dy_1[k:k+b] -= h_i / 6 * (df_dy[l+1:l+m] + 2 * df_dy_middle[k:k+b])
        T = stacked_matmul(df_dy_middle[k:k+b], df_dy[l+1:l+m])
        dPhi_dy_1[k:k+b]  += h_i**2 / 12 * T

        l += m
        k += b

    values = np.hstack((
        dPhi_dy_0.ravel(), 
        dPhi_dy_1.ravel(), 
        dbc_dya.ravel(order='F'),
        dbc_dyb.ravel(order='F')))
    J = coo_matrix((values, (i_jac, j_jac)))
    return csc_matrix(J)


def assemble_boundary_values(Y):
    N = len(Y)

    Ya = np.vstack([Y[i][:, 0] for i in range(N)])
    Yb = np.vstack([Y[i][:, -1] for i in range(N)])

    return Ya, Yb


def solve_newton(n, h, col_fun, bc, jac, Y, bvp_tol, bc_tol):
    N = len(Y) # Number of regions
    M = sum([y_i.shape[1] for y_i in Y]) # Total number of nodes
    K = sum([y_i.shape[1] - 1 for y_i in Y]) # Total number of segments

    # We know that the solution residuals at the middle points of the mesh
    # are connected with collocation residuals  r_middle = 1.5 * col_res / h.
    # As our BVP solver tries to decrease relative residuals below a certain
    # tolerance, it seems reasonable to terminated Newton iterations by
    # comparison of r_middle / (1 + np.abs(f_middle)) with a certain threshold,
    # which we choose to be 1.5 orders lower than the BVP tolerance. We rewrite
    # the condition as col_res < tol_r * (1 + np.abs(f_middle)), then tol_r
    # should be computed as follows:
    tol_r = 2/3 * h * 5e-2 * bvp_tol

    # Maximum allowed number of Jacobian evaluation and factorization, in
    # other words, the maximum number of full Newton iterations. A small value
    # is recommended in the literature.
    max_njev = 4

    # Maximum number of iterations, considering that some of them can be
    # performed with the fixed Jacobian. In theory, such iterations are cheap,
    # but it's not that simple in Python.
    max_iter = 8

    # Minimum relative improvement of the criterion function to accept the
    # step (Armijo constant).
    sigma = 0.2

    # Step size decrease factor for backtracking.
    tau = 0.5

    # Maximum number of backtracking steps, the minimum step is then
    # tau ** n_trial.
    n_trial = 4

    # Derive datatype
    dtype = Y[0].dtype

    # Collocation
    r_col, Y_middle, F, F_middle = col_fun(Y)
    # Boundary
    Ya, Yb = assemble_boundary_values(Y)
    bc_res = bc(Ya, Yb)  
    # Residual
    res = np.hstack((r_col.ravel(order='F'), bc_res))

    def ravel(Z):
        # Concatenate all regions together and ravel all components
        return np.hstack(Z).ravel(order='F')
    
    def unravel(u):
        # Divide the vector into the individual regions
        Z = [np.empty_like(y_i) for y_i in Y]
        l = 0 # Counter into total nodes
        for z_i in Z:
            m = z_i.shape[1] # Number of nodes
            z_i[:] = u[l*n:(l+m)*n].reshape((n, -1), order='F')
            l += m
        return Z

    # Vector of unknowns
    u = ravel(Y)

    njev = 0
    singular = False
    recompute_jac = True
    for p in range(max_iter):
        if recompute_jac:
            J = jac(Y, Y_middle, F, F_middle, bc_res)

            njev += 1
            try:
                LU = splu(J)
            except RuntimeError:
                singular = True
                break

            u_step = LU.solve(res)
            cost = np.dot(u_step, u_step)

        alpha = 1
        for trial in range(n_trial + 1):
            # Step
            u_new = u - alpha * u_step
            Y_new = unravel(u_new)

            # Collocation
            r_col, Y_middle, F, F_middle = col_fun(Y_new)

            # Boundary
            Ya, Yb = assemble_boundary_values(Y_new)
            bc_res = bc(Ya, Yb)  
            
            # Residual
            res = np.hstack((r_col.ravel(order='F'), bc_res))

            u_step_new = LU.solve(res)
            cost_new = np.dot(u_step_new, u_step_new)
            if cost_new < (1 - 2 * alpha * sigma) * cost:
                break

            if trial < n_trial:
                alpha *= tau

        # Update the raveled and unraveled values
        u = u_new
        Y = Y_new

        if njev == max_njev:
            break

        # Unravel f_middle
        f_middle = np.empty((n, K), dtype=dtype)
        k = 0 # Total segment counter
        for f_middle_i in F_middle:
            b = f_middle_i.shape[1] # Number of segments in this region
            f_middle[:, k:k+b] = f_middle_i
            k += b

        # Check for convergence
        if (np.all(np.abs(r_col) < tol_r * (1 + np.abs(f_middle))) and
                np.all(np.abs(bc_res) < bc_tol)):
            break

        if alpha == 1:
            u_step = u_step_new
            cost = cost_new
            recompute_jac = False
        else:
            recompute_jac = True

    return Y, singular


def prepare_sys(n, fun, bc, X):
    """Create the function and the Jacobian for the collocation system."""
    N = len(X) # Number of regions
    M = sum([x_i.size for x_i in X]) # Total number of nodes
    K = sum([x_i.size - 1 for x_i in X]) # Total number of regions

    i_jac, j_jac = compute_jac_indices(n, X)

    def col_fun(Y):
        dtype = Y[0].dtype

        Y_middle = []; F = []; F_middle = [];
        r_col = np.empty((n, K), dtype=dtype)

        l = 0 # Node counter
        k = 0 # Segment counter
        for i, (x_i, y_i) in enumerate(zip(X, Y)):
            m = x_i.size # Number of nodes in this region
            b =  m - 1 # Number of segments in this region

            # Wrap the fun
            def fun_wrap(x, y, _):
                return fun(x, y, region=i)

            h_i = np.diff(x_i)
            r_col_i, y_middle, f, f_middle = collocation_fun(
                fun_wrap, y_i, None, x_i, h_i)

            Y_middle.append(y_middle)
            F.append(f)
            F_middle.append(f_middle)

            r_col[:, k:k+b] = r_col_i

            # Offset total node counter for next region
            l += m
            k += b
        return r_col, Y_middle, F, F_middle

    def sys_jac(Y, Y_middle, F, F_middle, bc0):
        X_middle = []
        for x_i in X:
            h_i = np.diff(x_i)
            X_middle.append(x_i[:-1] + 0.5 * h_i)

        # Compute derivates at every position
        df_dy = estimate_fun_jac(fun, X, Y)
        df_dy_middle = estimate_fun_jac(fun, X_middle, Y_middle, F_middle)
        
        # Values at boundaries
        Ya, Yb = assemble_boundary_values(Y)
        dbc_dya, dbc_dyb = estimate_bc_jac(bc, Ya, Yb, bc0)

        return construct_global_jac(n, X, 
                                    i_jac, j_jac, 
                                    df_dy, df_dy_middle, 
                                    dbc_dya, dbc_dyb)

    return col_fun, sys_jac


def solve_mpbvp(fun, bc, X, Y, p=None, S=None, fun_jac=None, bc_jac=None,
              tol=1e-3, max_nodes=1000, verbose=0, bc_tol=None):
    # Keep the same function signature as solve_bvp, but check for 
    # parameters which are currently not implemented.
    if p is not None:
        raise RuntimeError("Parameters are currently not supported.")
    if S is not None:
        raise RuntimeError("Singular terms are currently not supported.")
    if fun_jac is not None or bc_jac is not None:
        raise RuntimeError("Derivate functions are currently not supported.")
    
    # Check the arguments
    X = [np.asarray(x, dtype=float) for x in X]
    if any(x.ndim != 1 for x in X):
        raise ValueError("`X` must be a list of 1 dimensional arrays.")
    if any(np.any(np.diff(x) <= 0) for x in X):
        raise ValueError("`X` must be a list of strictly increasing arrays.")

    # Make sure that complex is used if any of the input is complex
    Y = [np.asarray(y) for y in Y]
    if any(np.issubdtype(y.dtype, np.complexfloating) for y in Y):
         dtype = complex
    else:
         dtype = float 
    Y = [y.astype(dtype, copy=False) for y in Y]

    if any(y.ndim != 2 for y in Y):
        raise ValueError("`Y` must be a list of 2 dimensional arrays.")
    if len(X) != len(Y):
        raise ValueError("`X` and `Y` must use the same number of regions.")
    if any(y.shape[1] != x.shape[0] for x, y in zip(X, Y)):
        raise ValueError(f"Regions must be of the same length in `X` and `Y`.")
   
    if tol < 100 * EPS:
        warn(f"`tol` is too low, setting to {100 * EPS:.2e}", stacklevel=2)
        tol = 100 * EPS

    if verbose not in [0, 1, 2]:
        raise ValueError("`verbose` must be in [0, 1, 2].")

    if bc_tol is None:
        bc_tol = tol

    # Number of regions
    N = len(X)
    # Dimension
    n = Y[0].shape[0]

    # Verify the function and boundary
    for i, (x, y) in enumerate(zip(X, Y)):
        f = fun(x, y, region=i)
        if f.shape != y.shape:
            raise ValueError(f"`fun` return is expected to have shape {y.shape}, "
                         f"but actually has {f.shape} for region {i}.")
    Ya, Yb = assemble_boundary_values(Y)
    bc_res = bc(Ya, Yb)  
    if bc_res.shape != (N * n,):
        raise ValueError(f"`bc` return is expected to have shape {(N * n,)}, "
                    f"but actually has {bc_res.shape}.")
    
    # Maximum number of iterations
    max_iteration = 10

    status = 0
    iteration = 0
    if verbose == 2:
        print_iteration_header()

    while True:
        M = sum([x_i.size for x_i in X]) # Total number of nodes
        K = sum([x_i.size - 1 for x_i in X]) # Total number of segments

        h = np.empty((K,), dtype=dtype)
        k = 0 # Total segment counter
        for x_i in X:
            b = x_i.size - 1 # Number of segments in this region
            h[k:k+b] = np.diff(x_i)
            k += b

        col_fun, jac_sys = prepare_sys(n, fun, bc, X)
        Y, singular = solve_newton(n, h, col_fun, bc, jac_sys,
                                      Y, tol, bc_tol)
        iteration += 1

        # Collocation
        r_col, Y_middle, F, F_middle = col_fun(Y)
        # Boundary
        Ya, Yb = assemble_boundary_values(Y)
        bc_res = bc(Ya, Yb)
        max_bc_res = np.max(abs(bc_res))

        # This relation is not trivial, but can be verified.z
        r_middle = 1.5 * r_col / h


        rms_res = []
        S = []

        # 
        nodes_added = 0
        Insert_1 = []; Insert_2 = []

        k = 0 # Segment count
        max_rms_res = 0.0
        for i, (x_i, y_i, f_i, f_middle_i) in enumerate(zip(X, Y, F, F_middle)):          
            # Compute number of nodes and segments in this region  
            m = x_i.size
            b = m - 1
            # Segment lengths and residual at middle points
            h_i = np.diff(x_i)
            r_middle_i = r_middle[:, k:k+b]
            # Compute spline for this region
            s_i = create_spline(y_i, f_i, x_i, h_i)
            S.append(s_i)
            # Estimate residuals
            rms_res_i = estimate_rms_residuals(
                lambda x, y, _: fun(x, y, region=i),
                s_i, x_i, h_i, None, r_middle_i, f_middle_i)
            rms_res.append(rms_res_i)
            # Update highest residual value
            max_rms_res = max(max_rms_res, np.max(rms_res_i))

            # Compute number of nodes to be added (one or two nodes per 
            # segment)
            insert_1, = np.nonzero((rms_res_i > tol) & (rms_res_i < 100 * tol))
            insert_2, = np.nonzero(rms_res_i >= 100 * tol)

            Insert_1.append(insert_1)
            Insert_2.append(insert_2)

            nodes_added_i = insert_1.shape[0] + 2 * insert_2.shape[0]
            nodes_added += nodes_added_i

            k += b
        # Convert residuals to array
        rms_res = np.hstack(rms_res)

        # Add nodes
        for i, (insert_1, insert_2) in enumerate(zip(Insert_1, Insert_2)):
            if insert_1.shape[0] == 0 and insert_2.shape[0] == 0:
                continue
            x_i = modify_mesh(X[i], insert_1, insert_2)
            y_i = s_i(x_i)
            X[i] = x_i
            Y[i] = y_i

        if singular:
            status = 2
            break

        if m + nodes_added > max_nodes:
            status = 1
            if verbose == 2:
                nodes_added = f"({nodes_added})"
                print_iteration_progress(iteration, max_rms_res, max_bc_res,
                                        M, nodes_added)
            break

        if verbose == 2:
            print_iteration_progress(iteration, max_rms_res, max_bc_res, M,
                                     nodes_added)

        if max_bc_res <= bc_tol:
            status = 0
            break
        elif iteration >= max_iteration:
            status = 3
            break

    if verbose > 0:
        if status == 0:
            print(f"Solved in {iteration} iterations, number of nodes {M}. \n"
                  f"Maximum relative residual: {max_rms_res:.2e} \n"
                  f"Maximum boundary residual: {max_bc_res:.2e}")
        elif status == 1:
            print(f"Number of nodes is exceeded after iteration {iteration}. \n"
                  f"Maximum relative residual: {max_rms_res:.2e} \n"
                  f"Maximum boundary residual: {max_bc_res:.2e}")
        elif status == 2:
            print("Singular Jacobian encountered when solving the collocation "
                  f"system on iteration {iteration}. \n"
                  f"Maximum relative residual: {max_rms_res:.2e} \n"
                  f"Maximum boundary residual: {max_bc_res:.2e}")
        elif status == 3:
            print("The solver was unable to satisfy boundary conditions "
                  f"tolerance on iteration {iteration}. \n"
                  f"Maximum relative residual: {max_rms_res:.2e} \n"
                  f"Maximum boundary residual: {max_bc_res:.2e}")

    return BVPResult(sol=S, x=X, y=Y, yp=F, rms_residuals=rms_res,
                     niter=iteration, status=status,
                     message=TERMINATION_MESSAGES[status], success=status == 0)