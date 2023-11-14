import numpy as np
from scipy import linalg
from opt_einsum import contract

def compute_V_Woodberry(X, pi, checks=False):
    n, p = X.shape
    K = pi.shape[1]-1
    # Constructing the diagonal blocks and the matrix to inverse when applying Woodberry
    Un = np.einsum('ik, ij -> kij', pi, X)
    Up = np.einsum('k,  ij -> kij', np.ones(K+1), np.eye(p))
    U = np.concatenate([Un, Up], axis=1)
    assert U.shape == (K+1, n+p, p)
    Ablocks = ((pi.T[:, np.newaxis, :] * X.T[np.newaxis, ...]) @ X) 
    # equivalent to the above but much slower:
    # Ablocks = np.einsum('ij,iJ, ik -> kjJ', X, X, pi, optimize='optimal')
    assert Ablocks.shape == (K+1, p, p)
    A_inv_times_U = np.linalg.solve(Ablocks, np.swapaxes(U, 1, 2))
    matrix_to_inverse = -np.eye(n+p) + np.tensordot(U, A_inv_times_U, axes=[(0, 2), (0, 1)] )
    # equivalent to     -np.eye(n+p) + np.einsum('kuj, kjv -> uv', U, A_inv_times_U)

    # Hessians
    H = np.stack([(np.diag(pi[i]) - np.outer(pi[i], pi[i].T)) for i in range(n)])
    # A_k^{-1} X for each diagonal block
    Ablocks_inverse_XT = np.linalg.solve(Ablocks, X.T[np.newaxis, ...])
    # the RHS to be solved in the linear system with matrix matrix_to_inverse
    b = np.transpose(U @ Ablocks_inverse_XT, [2, 1, 0]) @ H
    solve_RHS = np.transpose(b, [1, 2, 0])
    # equivalent but faster than the following einsum
    # solve_RHS_old = np.einsum('kuj, kji, ikK -> uKi',
    #    U, Ablocks_inverse_XT, H)
    # assert np.isclose(solve_RHS, solve_RHS_old).all()

    # solving the linear system and reshaping
    old_shape = solve_RHS.shape
    solved = np.linalg.solve(matrix_to_inverse, solve_RHS.reshape((n+p, (K+1)*n))).reshape(old_shape)
    # the correction term in Woodberry's formula
    V2 = np.tensordot(solved, solve_RHS, axes=[(0, 2), (0, 2)])
    # equivalent but faster than the following einsum
    # V2_old = np.einsum('uki, uKi -> kK', solved, solve_RHS)
    # assert np.isclose(V2, V2_old).all()

    # the main term in Woodberry's formula
    V1 = np.einsum('ij, kji, ikK, ikL -> KL',
            X, Ablocks_inverse_XT, H, H, optimize=True)
    V1 - V2
    return np.einsum('ikK -> kK', H) - (V1 - V2)


def compute_V_solve(X, pi, checks=False):
    n, p = X.shape
    K = pi.shape[1]-1
    Un = np.einsum('ik, ij -> kji', pi, X)
    UU = Un.reshape((p*(K+1), n))
    Ablocks = ((pi.T[:, np.newaxis, :] * X.T[np.newaxis, ...]) @ X) 
    # equivalent to the above but much slower:
    # Ablocks = np.einsum('ij,iJ, ik -> kjJ', X, X, pi, optimize='optimal')
    S_fast = linalg.block_diag(*Ablocks)- UU @ UU.T

    if checks:
        S = np.zeros((p*(K+1), p*(K+1)))
        for i in range(n):
            Hi = np.diag(pi[i]) - np.outer(pi[i], pi[i].T)
            S += np.kron(Hi, np.outer(X[i], X[i].T))
        assert np.isclose(S_fast, S).all()

    Hstacks = np.hstack([
                    np.kron(np.diag(pi[i]) - np.outer(pi[i], pi[i].T),X[i][:, np.newaxis])
                    for i in range(n)
                ])
    solved = np.linalg.solve(S_fast, Hstacks)
    Htensor = Hstacks.reshape((p*(K+1), n, K+1))
    solved_tensor = solved.reshape((p*(K+1), n, K+1))
    V = np.einsum('uik, uiK -> kK', Htensor, solved_tensor)

    if checks:
        for i in range(n):
            assert np.isclose(
                Htensor[:, i, :],
                Hstacks[:, i*(K+1):(i+1)*(K+1)]
            ).all()
            assert np.isclose(
                solved_tensor[:, i, :],
                solved[:, i*(K+1):(i+1)*(K+1)]
            ).all()

    if checks:
        print("Start inversing to compute M")
        M = np.linalg.pinv(S)
        assert M.shape == (p*(K+1), p*(K+1))
        print("start iterating")
        V_slow = np.zeros((K+1, K+1))
        for i in range(n):
            Hi = np.diag(pi[i]) - np.outer(pi[i], pi[i].T)
            Hi_otimes_xi = np.kron(Hi, X[i][:, np.newaxis]) # make sure X[i] is a column vector
            V_slow += Hi_otimes_xi.T @ M @ Hi_otimes_xi
        assert np.isclose(V_slow, V).all()

    if checks:
        V_fast = np.zeros((K+1, K+1))
        for i in range(n):
            Hi = np.diag(pi[i]) - np.outer(pi[i], pi[i].T)
            Hi_otimes_xi = np.kron(Hi, X[i][:, np.newaxis]) # make sure X[i] is a column vector
            V_fast += Hi_otimes_xi.T @ np.linalg.solve(S, Hi_otimes_xi)

        assert np.isclose(V_fast, V).all()

    H = np.stack([(np.diag(pi[i]) - np.outer(pi[i], pi[i].T)) for i in range(n)])
    return np.einsum('ikK -> kK', H) - V





