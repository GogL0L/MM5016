## Description
"""
The answer to task 2 of lab 6 is the function qr_decompose together with
the main function.
"""


from math import sqrt


def matrix_multiply(matrix1, matrix2):
    m = len(matrix1)
    n = len(matrix2)
    p = len(matrix2[0])
    element = lambda i,j: sum([matrix1[i][k] * matrix2[k][j]
                               for k in range(n)])
    row = lambda i: [element(i,j) for j in range(p)]
    return [row(i) for i in range(m)]


def transpose(matrix):
    """ Transposes a square matrix. """
    n = len(matrix)
    return [[matrix[i][j] for i in range(n)] for j in range(n)]


def dot_product(v,u):
    """ Takes the dot product of the vectors v and u. I.e. the inner product
    given that the coordinates are in a orthonormal basis.
    """
    n = len(v)
    return sum([v[k]*u[k] for k in range(n)])


def vector_add(v,u):
    """ Add vectors componentwise. """
    n = len(v)
    return [v[k] + u[k] for k in range(n)]


def vector_sub(v,u):
    """ Add vectors componentwise. """
    n = len(v)
    if n == 0:
        return []
    return [v[k] - u[k] for k in range(n)]


def vector_sum(list_of_vectors, size=1):
    """ Returns the sum of the list. """
    n = size
    zero = [0 for i in range(n)]
    if list_of_vectors == []:
        return zero
    else:
        n = len(list_of_vectors[0])
        zero = [0 for i in range(n)]
        return vector_add(list_of_vectors[0],
                          vector_sum(list_of_vectors[1:], size=n))


def scalar_product(m,v):
    """ Takes the scalar product of the scalar m and the vector v. """
    n = len(v)
    return [m * v[k] for k in range(n)]


def norm(v):
    """ Returns the euclidean norm given the basis is orthonormal. """
    return sqrt(dot_product(v,v))


def normalise(v):
    """ Normalises the vector 'v' using the euclidean norm. """
    return scalar_product(1 / norm(v), v)


v = [0,1,1]
print(normalise(v))


def projection(u,v):
    """ Takes the projection of v on u, given that the coordinates
    are in a orthnormal basis.
    """
    scalar = dot_product(u,v) / dot_product(u,u)
    return scalar_product(scalar, u)


def gram_schmidt(basis):
    """ Returns an orthonormal basis. """
    n = len(basis)
    if n <= 1:
        return list(map(normalise,basis))
    else:
        tail_orthogonal_basis = gram_schmidt(basis[:-1])
        head_original_basis = basis[-1]
        project_head_on = lambda u: projection(u, head_original_basis)
        non_orthogonal_components = list(map(project_head_on,
                                             tail_orthogonal_basis))
        non_orthogonal_component = vector_sum(non_orthogonal_components)
        head_orthogonal_basis = vector_sub(head_original_basis,
                                           non_orthogonal_component)
        return tail_orthogonal_basis + [normalise(head_orthogonal_basis)]


def qr_decompose(A):
    """ Returns a unitary matrix Q and an upper triangular matrix R
    such that A = QR.
    """
    n = len(A)
    basis_A = transpose(A)
    orthogonal_basis = gram_schmidt(basis_A)
    Q = transpose(orthogonal_basis)

    R = [[dot_product(orthogonal_basis[i], basis_A[j]) for j in range(n)]
         for i in range(n)]
    return Q, R


def main():
    """ Solution for task 2 of lab assignment 6. """
    A = [[0,1,1], [1,1,2], [0,0,3]]
    Q, R = qr_decompose(A)
    Qt = transpose(Q)
    print("Original matrix A:", matrix_multiply(Q, R))
    print("Result of Q:", Q)
    print("Result of R:", R)
    print("Result of QQ^t:", matrix_multiply(Q,Qt))
    print("Result of QR:", matrix_multiply(Q, R))


if __name__ == "__main__":
    main()
