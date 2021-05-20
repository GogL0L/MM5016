## Description
"""
The column vector 'X' will be symbolised as a list of size 1 lists, that is
X = [[x1], [x2] ,[x3], ... , [xn]]. An m by n matrix 'A' will be symbolised by a list of size
m consisting of lists of size n. The solution to task 2 in the lab will be the function
'jacobi_method'.
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
    """ Transposes an n by m matrix. """
    n = len(matrix)
    m = len(matrix[0])
    return [[matrix[i][j] for i in range(n)] for j in range(m)]


def dot_product(v,u):
    """ Takes the dot product of the vectors v and u. I.e. the inner product
    given that the coordinates are in a orthonormal basis.
    """
    n = len(v)
    vt = transpose(v)
    return matrix_multiply(vt,u)[0][0]


def matrix_add(A,B):
    """ Add matrices componentwise. """
    n = len(A)
    m = len(A[0])
    return [[A[i][j] + B[i][j] for j in range(m)] for i in range(n)]


def scale(l,A):
    """ Scales the matrix 'A' with the scalar m. """
    n = len(A)
    m = len(A[0])
    return [[l * A[i][j] for j in range(m)] for i in range(n)]


def matrix_sub(A,B):
    """ Add vectors componentwise. """
    negative_B = scale(-1, B)
    return matrix_add(A, negative_B)


def norm(A):
    """ Returns the euclidean norm given the basis is orthonormal. """
    return sqrt(dot_product(A,A))


def normalise(v):
    """ Normalises the vector 'v' using the euclidean norm. """
    return scale(1 / norm(v), v)


v = [[ 0 ],[ 1 ],[ 1 ]]
print(normalise(v))


def projection(u,v):
    """ Takes the projection of v on u, given that the coordinates
    are in a orthnormal basis.
    """
    scalar = dot_product(u,v) / dot_product(u,u)
    return scale(scalar, u)


def additive_decomposition(A):
    """ Decomposes an n by n matrix 'A' into an lower triangular
    matrix 'L', diagonal matrix 'U' and an upper triangular
    matrix 'U' such that 'A = D + L + U'. This function
    returns the 3 tuple '(D, L, U)'.
    """
    n = len(A)
    D = [[A[i][j] if i==j else 0 for j in range(n)] for i in range(n)]
    L = [[A[i][j] if i<j else 0 for j in range(n)] for i in range(n)]
    U = [[A[i][j] if i>j else 0 for j in range(n)] for i in range(n)]
    return D,L,U


def diagonal_inverse(D):
    """ Given a diagonal n by n matrix 'D' with non-zero diagonals this function 
    returns the inverse matrix.
    """
    n = len(D)
    return [[1 / D[i][j] if i==j else 0 for j in range(n)] for i in range(n)]


def euclidean_distance(v,u):
    """ Given the n by 1 vectors 'v' and 'u', this function will return
    the euclidean distance between them.
    """
    uv = matrix_sub(v,u)
    return norm(uv)


def zero_matrix(A):
    """ Returns the zero matrix of the same size. """
    n = len(A)
    m = len(A[0])
    return [[0 for j in range(m)] for i in range(n)]


def jacobi_next_value(X, D, L, U, b):
    """ This returns the the next vector approximation based on the formula: 
    X_k = D^(-1) (L+U)X^(k-1) + D^(-1)b, from the jacobi method.
    X_new = D^(-1)(b - (L+U)X)
    """
    #Making functions a bit shorter
    add = lambda a,b: matrix_add(a,b)
    sub = lambda a,b: matrix_sub(a,b)
    mult = lambda a,b: matrix_multiply(a,b)

    second_factor = sub(b, mult(add(L,U),X))
    return mult(diagonal_inverse(D), second_factor)


def jacobi_method(A, b, approximate_solution,
                  tolerance=0.01, iteration=0, max_iterations=100,
                  debug = False):
    """ Given an n by n matrix A and an n by 1 vector b, this function 
    returns the solution x to the equation Ax = b, with the accuracy
    specified by the tolerance. If the tolerance is not met within
    'max_iterations' ammount of iterations, an exception will be raised.
    """
    if debug:
        print("Approximate solution:", approximate_solution)
    if iteration > max_iterations:
        raise RecursionError("<jacobi_method: max iterations exceded!>")

    b_approximate = matrix_multiply(A, approximate_solution)
    distance = euclidean_distance(b, b_approximate)
    if debug:
        print("Distance:", distance)
    if distance <= tolerance:
        return approximate_solution
    else:
        X = approximate_solution
        D, L, U = additive_decomposition(A)
        X_new = jacobi_next_value(X, D, L, U, b)
        return jacobi_method(A, b,
                             X_new,
                             tolerance, iteration + 1, max_iterations,
                             debug)


def problem3():
    """ The solution to task 2. """
    A = [[5, -2, 3], [-3, 9, 1], [2, -1, -7]]
    b = [[-1], [2], [3]]
    result = jacobi_method(A,b, zero_matrix(b), debug= False)
    print("A:", A)
    print("x from jacobi_method:", result)
    x_from_gaussian_elimination = [[ 59/317 ], [ 105/317 ], [ -134/317 ]]
    print("x from gaussian elimination:", x_from_gaussian_elimination)
    print("b from approximation:", matrix_multiply(A,result))
    print("b:", b)


def gradient_descent(function, gradient_function, vector_approximation,
                     tolerance = 0.01, learning_step_size = 1,
                     iteration=0, max_iterations=600,
                     debug=False):
    """Finds a local minimum of "function" using the gradient descent method."""
    gradient = gradient_function(vector_approximation)
    if debug:
        print(f"Iteration {iteration} vector: {vector_approximation}")
        print(f"Iteration {iteration} value: {function(vector_approximation)}")
        print(f"Iteration {iteration} norm of gradient: {norm(gradient)}")
    if iteration >= max_iterations:
        raise RecursionError("<gradient_descent: max_iterations exceeded!>")
    if norm(gradient) <= tolerance:
        return function(vector_approximation), vector_approximation
    else:
        scaled_gradient = scale(learning_step_size, gradient)
        new_approximation = matrix_sub(vector_approximation, scaled_gradient)
        return gradient_descent(function, gradient_function, new_approximation,
                                tolerance, learning_step_size, 
                                iteration+1, max_iterations,
                                debug)


def problem4():
    """ Solution to problem 3. """
    a,b = 1,5
    f_tupled = lambda x1, x2: (a-x1) ** 2 + b * (x2 - x1 ** 2) **2
    gradient_tupled = lambda x1, x2: [[-4 * b * x1 * (x2 - x1 ** 2) - 2 * (a-x1)],
                                       [2 * b * (x2 - x1 ** 2) ]]
    f = lambda X: f_tupled(X[0][0], X[1][0])
    gradient = lambda X: gradient_tupled(X[0][0], X[1][0])
    start_vector = [[ -1.4 ], [ 2 ]]

    value, vector = gradient_descent(f, gradient, start_vector,
                              learning_step_size = 0.02,
                              debug=False)
    print("Minimum vector found by gradient descent:", vector)
    print("Real minimum vector:", [[1], [1]])
    print("Minimum value found by gradient descent:", value)
    print("Real minimum value:", f([[1], [1]]))


def main():
    """ Be asked to present the solution to problem 3 or 4. """
    print("At any point type 'exit' to exit.")
    while True:
        problem = input("To which problem do you want to display a solution? ")
        if problem == "exit":
            break
        elif problem == "3":
            problem3()
        elif problem == "4":
            problem4()
        else:
            print("Invalid input, try again.")

if __name__ == "__main__":
    main()
