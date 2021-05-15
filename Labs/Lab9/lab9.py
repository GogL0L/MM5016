

import matplotlib.pyplot as plt
import numpy as np


def plot_function(function, interval):
    """ Plots the function oven the given interval, where the interval
    is represented by a tuple.
    """
    x_min, x_max = interval
    x = np.linspace(x_min , x_max ,100)

    y = np.array(map(function, x))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.plot(x,y, 'r')
    plt.show()


def linear_interpolation(data_set):
    """ Takes a list of data points in form of tuple coordinates, that is:
    [(x0, y0), (x1, y1) ...] and returns an apporixmated function given by
    linear interpolation.
    """
    if len(data_set) <= 1:
        raise ValueError("<linear_interpolation: not enough data points>")
        
    data_set_sorted = sorted(data_set)
    if len(data_set_sorted) == 2:
        (x0, y0), (x1, y1) = data_set_sorted
        k = (y1 - y0) / (x1 - x0)
        m = y0
        interpolated_function = lambda x: k * (x - x0) + m
        return interpolated_function
    else:
        head = data_set_sorted[:2]
        head_interval_end = head[1][0] 
        tail = data_set_sorted[1:]
        head_function = linear_interpolation(head)
        tail_function = linear_interpolation(tail)
        complete_function = lambda x:( head_function(x) if x <= head_interval_end
                                       else tail_function(x))
        return complete_function


data = [(0,0), (1,1), (2,4)]
fun = linear_interpolation(data)
print(fun(1.3))

plot_function(fun, (0,5))


def product(factors):
    """ Returns factor[0] * factor[1] ..."""
    if len(factors) == 0:
        return 1
    else:
        return factors[0] * product(factors[1:])


print(product([1,2,3]))


def langrange_polynomial_coefficients(data_set):
    """ Takes a list of data points in form of tuple coordinates, that is:
    [(x0, y0), (x1, y1) ...] and returns the list [a0, a1, a2...], so that
    the lagrange polynomial is a0 + a1x + a2x^2 ...
    """
    l = lambda j, x: product([


def lagrange_interpolation(data_set):
    """ Takes a list of data points in form of tuple coordinates, that is:
    [(x0, y0), (x1, y1) ...] and returns an apporixmated function given by
    lagrange interpolation.
    """
    data_set_sorted = sorted(data_set)
    x_points = [P[0] for P in data_set_sorted]
    y_points = [P[1] for P in data_set_sorted]
    l = lambda j, x: product([ (x-xm) / (x_points[j] - xm)
                               for xm in x_points if xm != x_points[j]])

    k = len(data_set)
    lagrange_function = lambda x : sum([y_points[j] * l(j,x) for j in range(k+1)])
    return lagrange_function


print(lagrange_interpolation([(1,1), (2,2)])(5)
