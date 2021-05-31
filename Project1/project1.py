from math import sqrt, sin,  cos, pi, exp
import numpy as np
import matplotlib.pyplot as plt


## Description
"""
In order to display the graphs of the approximated solutions to
the ODE and statistics regarding the error, run this as a python 3 
file in a terminal.
"""


def average(data_set):
    """ Given a list of data [a1, a2, a3, ..., an], this returns
    the average given by (a1 + a2 + a3 + ... + an) / n.
    """
    n = len(data_set)
    return sum(data_set) / n


def standard_deviation(estimated_data, correct_data):
    """ Given estimated_data = [e1, e2, e3, ..., en] and
    correct_data = [c1, c2, c3, ..., cn], this returns the
    squareroot of the variance, i.e. 
    (((e1 - c1)^2 + (e2 - c2)^2 + ... + (en - cn)^2) / n)^0.5.
    """
    n = len(estimated_data)
    differences = [estimated_data[i] - correct_data[i] for i in range(n)]
    differences_squared = list(map(lambda x: x ** 2, differences))
    return sqrt(average(differences_squared))


def biggest_error(estimated_data, correct_data):
    """ Returns the biggest error difference between an estimated
    data point and its correct value.
    """
    n = len(estimated_data)
    differences = [estimated_data[i] - correct_data[i] for i in range(n)]
    differences_abs = [abs(differences[i]) for i in range(n)]
    return max(differences_abs)


def runge_kutta_2_method(derivative, orbit, stepsize):
    """ Given that 'derivative' is a function of (x,y)
    and that the 'initial_value' is a tuple of the form
    (x0, y(x0)), this method returns a function that
    approximates y using runge kutta method in the equation 
    dy/dx = derivative(x,y).
    """

    initial_value = orbit[-1]


    def y(x):
        x0, y0 = initial_value
        xk, yk = x0, y0
        while x0 <= xk < x:
            k1 = derivative(xk,yk)
            k2 = derivative(xk + 2 * stepsize / 3, yk + 2 * stepsize * k1 / 3)
            x_next = xk + stepsize 
            y_next = yk + stepsize * (k1 / 4 + 3 * k2 / 4)
            xk, yk = x_next, y_next
        return yk

    
    return y


def runge_kutta_4_method(derivative, orbit, stepsize):
    """ Given that 'derivative' is a function of (x,y)
    and that the 'initial_value' is a tuple of the form
    (x0, y(x0)), this method returns a function that
    approximates y using runge kutta method in the equation 
    dy/dx = derivative(x,y).
    """
    initial_value = orbit[-1]

    def y(x):
        x0, y0 = initial_value
        xk, yk = x0, y0
        while x0 <= xk < x:
            k1 = derivative(xk,yk)
            k2 = derivative(xk + stepsize / 2, yk + stepsize * k1 / 2)
            k3 = derivative(xk + stepsize / 2, yk + stepsize * k2 / 2)
            k4 = derivative(xk + stepsize, yk + stepsize * k3)
            x_next = xk + stepsize 
            y_next = yk + (1/6) * stepsize * (k1 + 2 * k2 + 2 * k3 + k4)
            xk, yk = x_next, y_next
        return yk

    
    return y


def euler_method(derivative, orbit, stepsize):
    """ Given that 'derivative' is a function of (x,y)
    and that the 'initial_value' is a tuple of the form
    (x0, y(x0)), this method returns a function that
    approximates y in the equation dy/dx = derivative(x,y).
    """
    initial_value = orbit[-1]
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


def bashforth_method(derivative, initial_value_orbit, stepsize):
    """ Given that 'derivative' is a function of (x,y) and that the 'initial_value' is a tuple of the form
    (x0, y(x0)), this method returns a function that
    approximates y using the Adams-Bashforth method in the equation 
    dy/dx = derivative(x,y).
    """


    def y(x):
        orbit = initial_value_orbit.copy()
        while 0 <= orbit[-1][0] < x:
            (x0,y0), (x1,y1) = orbit[-2:]
            x_next = x1 + stepsize 
            y_next = (y1 + (3/2) * stepsize * derivative(x1,y1)
                      - (1/2) * stepsize * derivative(x0,y0))
            orbit.append((x_next, y_next))
            
        return orbit[-1][1]

    
    return y


def display_chart_comparison(title, y1_data, y1_label, y2_data, y2_label,
                             x_data_labels,
                             x_label, y_label):
    # data to plot
    n_groups = len(y1_data)
    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, y1_data, bar_width,
    alpha=opacity,
    color='b',
    label=(y1_label))

    rects2 = plt.bar(index + bar_width, y2_data, bar_width,
    alpha=opacity,
    color='g',
    label=(y2_label))

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(index + bar_width, x_data_labels)
    plt.legend()

    plt.tight_layout()
    plt.show()


def display_graph(x_data_list, y_data_list, y_data_labels,
                  x_axis_label, y_axis_label, title):
    """ Displays multiple functions in one graph. """
    n = len(x_data_list)
    for i in range(n):
        plt.plot(x_data_list[i], y_data_list[i], label=y_data_labels[i])
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(title)

    plt.legend()
    plt.show()


def display_all(ODE_function,
                ODE_problem_string_representation,
                ODE_solver,
                ODE_solver_string_representation,
                initial_value_orbit,
                data_points_list,
                interval,
                analytic_function):
    """ First it displays a graph of the all the calculated solutions linearly
    interpolated. When the window is closed it proceeds to display statistics
    regarding the standard deviation and biggest error.
    """
    (start, end) = interval
    interval_size = end - start
    step_size = lambda N: interval_size / N
    x_intervals = list(map(lambda N: np.linspace(start, end, N),
                           data_points_list))
    approximated_functions = list(map(
        lambda N: ODE_solver(ODE_function, initial_value_orbit, step_size(N)),
        data_points_list))
    y_intervals = [list(map(approximated_functions[i], x_intervals[i]))
                   for i in range(len(x_intervals))]

    correct_y_intervals = [list(map(analytic_function, x_intervals[i]))
                           for i in range(len(x_intervals))]
   
    x_labels = list(map(str, data_points_list))
    deviations = [standard_deviation(y_intervals[i], correct_y_intervals[i])
                  for i in range(len(y_intervals))]
    max_errors = [biggest_error(y_intervals[i], correct_y_intervals[i])
                  for i in range(len(y_intervals))]
    title = ('Error for approximating the solution to ' +
              ODE_problem_string_representation +
              ' using ' +
              ODE_solver_string_representation)

    display_graph_string = ("Graph of approximated solutions to " +
                            ODE_problem_string_representation +
                            " calculated using " +
                            ODE_solver_string_representation +
                            " and then linearly interpolated ")
    x_data_largest = np.linspace(start, end, 640)
    analytic_y_data = list(map(analytic_function, x_data_largest))
    display_graph(x_intervals + [x_data_largest],
                  y_intervals + [analytic_y_data],
                  x_labels + ["Analytic function"],
                  "t", "u(t)",
                  display_graph_string)
                              
                  

    display_chart_comparison(title, deviations, "standard deviation",
                             max_errors, "Biggest absolute value error.",
                             x_labels,
                             "Number of points", "Error")


def F(t,u):
    """ The function symbolising the second the derivative from
    the ODE for Project 1. I.e du/dt = cos(pi * t) + u(t).
    """
    return cos(pi * t) + u

F_label = "du/dt = cos(pi * t) + u(t)"


analytic_solution = lambda t:( (pi * sin(pi*t))/(pi ** 2  + 1)
                               - (cos(pi*t))/(pi ** 2 + 1)
                               + ( 2 + 1/(pi **2 + 1) ) * exp(t)
                               )


N = [10, 20, 40, 80, 160, 320, 640]
interval = (0,2)
orbit = [(-1, analytic_solution(-1)), (0, analytic_solution(0))]


def task_a():
    """ Prints out the graph for the errors using Euler's method. """
    #display(F, euler_method, N, interval, analytic_function)
    display_all(F,
                F_label,
                euler_method,
                "Euler's method",
                [(0,2)],
                N,
                interval,
                analytic_solution)


def task_b():
    """ Prints out the graph for the errors using Euler's method. """
    #display(F, euler_method, N, interval, analytic_function)
    display_all(F,
                F_label,
                runge_kutta_2_method,
                "Second order Runge-Kutta method",
                orbit,
                N,
                interval,
                analytic_solution)


def task_c():
    """ Prints out the graph for the errors using Euler's method. """
    #display(F, euler_method, N, interval, analytic_function)
    display_all(F,
                F_label,
                runge_kutta_4_method,
                "Fourth order Runge-Kutta method",
                orbit,
                N,
                interval,
                analytic_solution)


def task_d():
    """ Prints out the graph for the errors using Euler's method. """
    #display(F, euler_method, N, interval, analytic_function)
    display_all(F,
                F_label,
                bashforth_method,
                "Adams-Bashforth method",
                orbit,
                N,
                interval,
                analytic_solution)


def main():
    while True:
        print("At any point, type 'exit', to exit")
        message =(
""" 
To display solutions to 'a', 'b', 'c', or 'd'
simply type in the string. Observe that for each solution
there will first be a graph displayed of the functions, and upon
closing that window, a new window with statistics regarding
the error will pop up. After that one is closed, the terminal
program resumes.
""")
        user_input = input(message)
        if user_input == "a":
            task_a()
        elif user_input == "b":
            task_b()
        elif user_input == "c":
            task_c()
        elif user_input == "d":
            task_d()
        elif user_input == "exit":
            print("The program will now exit.")
            break
        else:
            print("Invalid input")


if __name__ == "__main__":
    main()
