import matplotlib.pyplot as plt
import numpy as np


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


def linear_interpolation_vector(data_set, query_points):
    """ Transforms the the query_points list from the format of 
    [x1,x2,x3...] to [f(x1), f(x2), f(x3) ...],
    where f is the linearly interpolated function from data_set.
    """
    f = linear_interpolation(data_set)
    return list(map(f,query_points))


def product(factors):
    """ Returns factor[0] * factor[1] ..."""
    if len(factors) == 0:
        return 1
    else:
        return factors[0] * product(factors[1:])


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
    lagrange_function = lambda x : sum([y_points[j] * l(j,x) for j in range(k)])
    return lagrange_function


def lagrange_interpolation_vector(data_set, query_points):
    """ Transforms the the query_points list from the format of 
    [x1,x2,x3...] to [f(x1), f(x2), f(x3) ...],
    where f is the lagrange interpolated function from data_set.
    """
    f = lagrange_interpolation(data_set)
    return list(map(f,query_points))


def problem1a():
    data = [(1,2), (2,2.5), (3,7), (4,10.5), (6,12.75), (8,13), (10,13)]
    x = np.linspace(1 , 10 , 18)
    y = linear_interpolation_vector(data, x)
    plt.plot(x,y)
    plt.show()


def problem1b():
    data = [(1,2), (2,2.5), (3,7), (4,10.5), (6,12.75), (8,13), (10,13)]
    x = np.linspace(1 , 10 , 18)
    y = lagrange_interpolation_vector(data, x)
    plt.plot(x,y)
    plt.show()


def euler_method(derivative, initial_value, stepsize):
    """ Given that 'derivative' is a function of (x,y)
    and that the 'initial_value' is a tuple of the form
    (x0, y(x0)), this method returns a function that
    approximates y in the equation dy/dx = derivative(x,y).
    """
    step_to_goal = lambda x, goal: x+stepsize if x < goal else x - stepsize
    y_next = lambda x, y, goal: (y + stepsize * derivative(x,y) if x < goal
                                 else y - stepsize * derivative(x,y) )


    def y(x):
        x0, y0 = initial_value
        xk, yk = x0, y0
        while x0 <= xk < x or x0 >= xk > x:
            xk = step_to_goal(xk, x)
            yk = y_next(xk, yk, x)
        return yk

    
    return y


def problem2():
    derivative = lambda x,y: y-x
    initial_value = (0, 0.5)
    y_analytic = lambda x: x + 1 - 0.5 * np.exp(x)
    y_euler1 = euler_method(derivative, initial_value, 0.1)
    y_euler2 = euler_method(derivative, initial_value, 0.05)
    y_euler3 = euler_method(derivative, initial_value, 0.01)

    x = np.linspace(0,10, 100)
    y_exact = list(map(y_analytic, x))
    y1 = list(map(y_euler1, x))
    y2 = list(map(y_euler2, x))
    y3 = list(map(y_euler3, x))

    plt.plot(x, y_exact, label='Exact')
    plt.plot(x, y1, label='0.1 step')
    plt.plot(x, y2, label='0.05 step')
    plt.plot(x, y3, label='0.01 step')
    plt.legend()
    plt.show()


def main():
    while True:
        print("At any point type 'exit' to exit.")
        user_input = input("What problem do you want to display? (1a, 1b or 2) ")
        if user_input == "1a":
            problem1a()
        elif user_input == "1b":
            problem1b()
        elif user_input == "2":
            problem2()
        elif user_input == "exit":
            break
        else:
            print("Not valid input")


if __name__ == "__main__":
    main()
