## Description

# The solution to task 1 will be the functions find_root_newton and
# find_root_secant.

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
    satisfies_tolerance = function(newest_root_approximate) <= tolerance
    if iteration > max_iterations:
        raise RecursionError("<iterate: maximum ammount of iterations reached>")
    elif satisfies_tolerance and debug:
        return newest_root_approximate, iteration
    elif satisfies_tolerance and not debug:
        return newest_root_approximate
    else:
        next_values = next_values_function(root_approximates)
        return iterate(function, next_values, tolerance, next_values_function,
                       iteration+1, max_iterations)

## Newtons's method
# The next values function for Newton's method just takes a list of size
# one and returns a list of size one.
def newton_next_values_function(function, derivative_of_function):
    """ The next_values function for Newton's method. """
    f, f_prim = function, derivative_of_function
    next_value_function = lambda x: x - (f(x) / f_prim(x))
    return lambda guess_roots: [ next_value_function(guess_roots[0]) ]


def find_root_newton(function, function_derivative, root_approximate, tolerance,
                     debug=False):
    """ Solve the root x for the equation function(x) = 0 using
    Newton's method. 
    """
    root_approximates = [root_approximate]
    next_values_function = newton_next_values_function(function,
                                                       function_derivative)
    return iterate(function, root_approximates, tolerance, next_values_function,
                   debug=debug)

## Secant method
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


def find_root_secant(function, root_approximates, tolerance, debug=False):
    """ Solve the root x for the equation function(x) = 0 using
    the secant method.
    """
    next_values_function = secant_next_values_function(function)
    return iterate(function, root_approximates, tolerance, next_values_function,
                   debug=debug)
