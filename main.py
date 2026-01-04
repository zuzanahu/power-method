from scipy.io import mmread


def print_sparse_adjacency_matrix(path: str) -> None:
    """
    Load an MTX file and print its sparse adjacency matrix
    as (row, column, value) entries.
    """
    matrix = mmread(path).tocoo()

    for row, col, value in zip(matrix.row, matrix.col, matrix.data):
        # +1 because Matrix Market uses 1-based indexing
        print(f"A[{row + 1}][{col + 1}] = {value}")


if __name__ == "__main__":
    print_sparse_adjacency_matrix("higgs-twitter.mtx")
