import matplotlib.pyplot as plt
import numpy as np


## Description
"""
In order to
see the solutions to all the tasks run this as a python 3 file
in a terminal.
"""


## Linear algebra code from previous labs


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

## Euler method from previous lab


def vector_euler_method(derivative, initial_value, stepsize):
    """ Given that 'derivative' is a vector valued function of (x,y) 
    where x is a normal varaible and y is vector valued and that
    'initial_value' is a tuple of the form (x0, y(x0)), where y(x0) is
    vector valued. This method returns a function that
    approximates vector valued y in the equation dy/dx = derivative(x,y).
    """
    add = lambda x,y: [x[i] + y[i] for i in range(len(x))]
    scale = lambda s,x: [s * x[i] for i in range(len(x))]
    minus = lambda x,y : add(x, scale(-1,y))

    step_to_goal = lambda x, goal: x+stepsize if x <= goal else x - stepsize
    y_next = lambda x, y, goal: (add(y,
                                     scale(stepsize, derivative(x,y))
                                     if x <= goal
                                 else minus(y, scale(stepsize, derivative(x,y)))))



    def y(x):
        x0, y0 = initial_value
        xk, yk = x0, y0
        while x0 <= xk < x or x0 >= xk > x:
            yk = y_next(xk, yk, x)
            xk = step_to_goal(xk, x)
        return yk

    
    return y


## Functions related to secant method from previous lab


## Iterate function
def iterate(function, root_approximates, tolerance, next_values_function,
            iteration=0, max_iterations=100, debug=False):
    """ Approximates a root x for the equation function(x) = 0,
    by applying the next_values_function on the list root_approximates 
    until tolerance is met on the last element of the list, in that
    case that last element is returned. If debug is True, then a tuple of
    that mentioned value and the iteration is returned. An error is raised 
    if iteration exceeds max_iterations.
    """
    # The root_approximates list will usually be (always, in the context of
    # this asignment) updated by calculating a new value based on all
    # the values in the root_approximates list, and then appending that new value,
    # and deleting the first element in root_approximates.
    # So the last element in root_approximates will be the most recent estimate of
    # the root. In the case of Newton's method, the root_approximates list
    # will just be a single value surrounded by a list.
    newest_root_approximate = root_approximates[-1]
    satisfies_tolerance = abs(function(newest_root_approximate)) <= tolerance
    if iteration > max_iterations:
        raise RecursionError("<iterate: maximum ammount of iterations reached>")
    elif satisfies_tolerance and debug:
        return newest_root_approximate, iteration
    elif satisfies_tolerance and not debug:
        return newest_root_approximate
    else:
        next_values = next_values_function(root_approximates)
        return iterate(function, next_values, tolerance, next_values_function,
                       iteration+1, max_iterations, debug)


def secant_next_values_function(function):
    """ The next_values_function for the Secant method. """
    f = function
    next_value_function = lambda x0, x : x - f(x) * (x0 - x) / ( f(x0) - f(x) ) 
    # As mentioned in the comments for iterate, this updates function
    # updates root_approximates by calculating the new approximate root,
    # appending it, and deleting the first element of the list.
    return lambda root_approximates: [root_approximates[1],
                                      next_value_function(root_approximates[0],
                                                          root_approximates[1])]


def find_root_secant(function, root_approximates,
                     tolerance=0.00001, debug=False):
    """ Solve the root x for the equation function(x) = 0 using
    the secant method, where root_approximates is a list of 2 approximates
    of the root.
    """
    next_values_function = secant_next_values_function(function)
    return iterate(function, root_approximates, tolerance, next_values_function,
                   debug=debug)


## Methods related to this lab


def tridiagonal_elimination(A,y):
    """ Returns the solution x to the equation Ax = y,
    where A is an n by n tridiagonal matrix and
    y is an n dimensional column vector. Algorithm copied
    from wikipedia.
    """
    n = len(y)
    a = lambda i: A[i][i-1]
    b = lambda i: A[i][i]
    c = lambda i: A[i][i+1]
    d = lambda i: y[i][0]


    def c_prim(i):
        if i == 0:
            return c(i) / b(i)
        else:
            return c(i) / (b(i) - a(i) * c_prim(i-1))


    def d_prim(i):
        if i == 0:
            return d(i) / b(i)
        else:
            return (d(i) - a(i) * d_prim(i-1)) / (b(i) - a(i) * c_prim(i-1))


    def x(i):
        if i == n-1:
            return d_prim(i)
        else:
            return d_prim(i) - c_prim(i) * x(i+1)


    x_vector = [[x(i)] for i in range(n)]
    return x_vector


def finite_difference_method(p, q, r, ya, yb, interval, partitions):
    """ Numerically solves the function y(x) from the equation 
    y''(x) = p(x) y' + q(x) y + r(x) with boundary values y(a) = ya
    and y(b) = yb, with the specified stepsize, in the interval given
    as a tuple (left, right) symbolising that x ranges from
    left <= x <= right. This function then returns a list of x values
    partitioned according to the interval and stepsize aswell as a corresponding 
    list of y values.
    """
    n = partitions
    left, right = interval
    h = (right - left) / n
    x = np.linspace(left, right, n+1)
    
    first_row_diagonal = [-2 - h ** 2 * q(x[1]), 1 - (h/2) * p(x[1])]
    last_row_diagonal = [1 + (h/2) * p(x[n-1]), -2 - h ** 2 * q(x[n-1])]
    row_i_diagonal = lambda i: [1 + (h/2) * p(x[i]),
                                -2 - h ** 2 * q(x[i]),
                                1 - (h/2) * p(x[i])]
    f = [[h ** 2 * r(x[1]) - (1 + (h/2) * p(x[1])) * ya if i == 1
          else h ** 2 * r(x[n-1]) - (1 - (h/2) * p(x[n-1])) * yb if i == n-1
          else h ** 2 * r(x[i])] for i in range(1,n)]

    A = [first_row_diagonal + (n-1-2) * [0] if i==1
         else (n-1-2) * [0] + last_row_diagonal if i == n-1
         else (i-2) * [0] + row_i_diagonal(i) + (n-1-1-i) * [0]
         for i in range(1,n)]
    u = tridiagonal_elimination(A,f)
    return x, transpose([[ya]] + u + [[yb]])[0]


def shooting_method(f, ya, yb, interval, partitions):
    """ Numerically solves the function y(x) from the equation 
    y''(x) = f(x,y,y') with boundary values y(a) = ya
    and y(b) = yb, with the specified stepsize, in the interval given
    as a tuple (left, right) symbolising that x ranges from
    left <= x <= right. It is calculated by using the secant method
    on the euler step method with the initial value y'(a) being the
    variable solved for in the secant method.
    This function then returns a list of x values
    partitioned according to the interval and stepsize aswell as a corresponding 
    list of y values.
    """
    a,b = interval
    y_prim = lambda x,y: [y[1], f(x, y[0], y[1])]

    y_z = lambda z: vector_euler_method(y_prim, (a, [ya, z]), 0.0001)
    phi = lambda z: y_z(z)(b)[0] - yb

    z = find_root_secant(phi, [1,4])

    x = list(np.linspace(a, b, partitions))
    #y = list(map(y_z(z), x))
    y = [y_z(z)(xi)[0] for xi in x]
    return x,y


def main():
    p = lambda t: 0
    q = lambda t: 3/2
    r = lambda t: 0
    interval = (0,1)
    ta, tb = 4,1
    f = lambda t, x, x_prim: q(t) * x
    
    t1, x1 = finite_difference_method(p, q, r, ta, tb, interval, 10)
    t2, x2 = finite_difference_method(p, q, r, ta, tb, interval, 100)

    t3, x3 = shooting_method(f, ta, tb, interval, 10)
    t4, x4 = shooting_method(f, ta, tb, interval, 100)
    
    plt.plot(t1, x1, label="Finite difference dx = 0.1")
    plt.plot(t2, x2, label="Finite difference dx = 0.01")
    plt.plot(t3, x3, label="Shooting method using simple euler method dx = 0.1")
    plt.plot(t4, x4, label="Shooting method using simple euler method dx = 0.01")

    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Approximations of the solution to x''(t) = 3x/2")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
