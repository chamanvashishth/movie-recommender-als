import numpy as np


class ALSOptimizer:

    def __init__(
        self,
        factors=20,
        reg=0.01
    ):
        self.factors = factors
        self.reg = reg

    def update_users(
        self,
        R,
        U,
        V
    ):

        k = self.factors

        eye = np.eye(k)

        VtV = V.T @ V

        for u in range(R.shape[0]):

            idx = R[u].indices

            if len(idx) == 0:
                continue

            V_i = V[idx]

            ratings = R[u, idx].toarray().ravel()

            A = (
                V_i.T @ V_i
                + self.reg * eye
            )

            b = V_i.T @ ratings

            U[u] = np.linalg.solve(A, b)

        return U

    def update_items(
        self,
        R,
        U,
        V
    ):

        k = self.factors

        eye = np.eye(k)

        Rt = R.T.tocsr()

        for i in range(Rt.shape[0]):

            idx = Rt[i].indices

            if len(idx) == 0:
                continue

            U_u = U[idx]

            ratings = Rt[i, idx].toarray().ravel()

            A = (
                U_u.T @ U_u
                + self.reg * eye
            )

            b = U_u.T @ ratings

            V[i] = np.linalg.solve(A, b)

        return V
