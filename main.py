"""
Power method for computing the dominant eigenvalue and eigenvector
of a large sparse nonnegative matrix.

Course: NI-MPI – Numerická matematika
"""

from __future__ import annotations

import numpy as np
import scipy.io
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def is_irreducible(matrix: csr_matrix) -> bool:
    """
    Test irreducibility of a nonnegative matrix using graph theory.

    A matrix is irreducible iff its directed graph
    is strongly connected.
    """
    n_components, _ = connected_components(
        matrix,
        directed=True,
        connection="strong",
    )

    return n_components == 1


def l1_norm(x: np.ndarray) -> np.single:
    """
    Compute the L1 norm in single precision.

    Using explicit float32 accumulation ensures consistent precision
    throughout the algorithm.
    """
    return np.sum(np.abs(x), dtype=np.single)


def power_method(
    matrix: csr_matrix,
    tol: float = 1e-7,
    max_iter: int = 1000,
    criterion: int = 1,
) -> tuple[np.ndarray, float, int] | None:
    """
    Compute the dominant eigenpair using the power method.

    Parameters
    ----------
    matrix : csr_matrix
        Sparse square matrix A in CSR format (float32).
        CSR stores the matrix row by row and only keeps nonzero entries,
        which allows efficient matrix–vector multiplication.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum number of iterations.
    criterion : int
        Convergence criterion:
        1 -> ||x_k - x_{k-1}||_1 < tol
        2 -> ||A x_k - lambda_k x_k||_1 < tol

    Returns
    -------
    (x, lambda_, k) or None
        Normalized eigenvector, dominant eigenvalue, and iteration count.
        Returns None if convergence was not achieved.
    """
    n = matrix.shape[0]

    # Initial vector: constant vector with entries 1 / n (float32)
    x = np.full(n, np.single(1.0 / n), dtype=np.single)

    for k in range(1, max_iter + 1):
        x_prev = x.copy()

        # Power iteration step: y = A x
        y = matrix.dot(x)

        # L1 normalization
        norm_y = l1_norm(y)
        if norm_y == np.single(0.0):
            # Degenerate case: power method breaks down
            raise ValueError("Zero vector encountered during iteration.")

        x = y / norm_y

        lambda_ = norm_y

        # Convergence criteria
        if criterion == 1:
            # Difference of successive iterates
            diff = l1_norm(x - x_prev)
            if diff < tol:
                return x, float(lambda_), k

        elif criterion == 2:
            # Residual norm ||Ax - lambda x||_1
            ax = matrix.dot(x)
            residual = l1_norm(ax - lambda_ * x)
            print(f"residual: {residual}")
            print(f"tol: {tol}")
            if residual < tol:
                return x, float(lambda_), k

        else:
            raise ValueError("Criterion must be 1 or 2.")

    # Convergence not achieved within max_iter
    return None


def main() -> None:
    """Load matrix, run power method, and print results."""

    # Load the matrix from a Matrix Market file, convert it to CSR format,
    # and store all values in single precision (float32).
    # CSR format enables efficient row-wise dot products.
    matrix = csr_matrix(
        scipy.io.mmread("higgs-twitter.mtx"),
        dtype=np.single,
    )

    print(f"Is matrix irreducible: {is_irreducible(matrix)}")

    result = power_method(
        matrix=matrix,
        tol=1e-7,
        max_iter=1000,
        criterion=1,
    )

    if result is None:
        print("Convergence criterion was not met.")
        return

    eigenvector, eigenvalue, iterations = result

    # Ensure L1 normalization of the resulting eigenvector
    eigenvector /= l1_norm(eigenvector)

    # Five largest components of the eigenvector
    top_indices = np.argsort(eigenvector)[-5:][::-1]

    print("Five largest components of the eigenvector:")
    for idx in top_indices:
        print(f"({idx}, {eigenvector[idx]:.5f})")

    print(f"Dominant eigenvalue: {eigenvalue:.5f}")
    print(f"Iterations: {iterations}")


if __name__ == "__main__":
    main()
