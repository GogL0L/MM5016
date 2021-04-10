

## Description
"""
The functions for integrating using the trapezoidal rule or Simpson's rule
are 'trapezoidal_integration' and 'simpson_integration' respectively. The
solution to task 1 and 2 can be displayed by running "python lab3.py" in a
terminal in the directory of this script.
"""


from math import sqrt


def partition_interval(start, end, number_of_partitions):
    """ Divides the interval start<= x <= end into 
    ammount_of_partitions ammount of partitions, with each partition 
    having the same size. The return value, is a list of the partitioned 
    intervals, symbolysed by tuples.
    """
    # dx symbolises the partition_size
    dx = (end - start) / number_of_partitions
    intervals = [(start + i * dx, start + (i+1) * dx)
                 for i in range(number_of_partitions)]
    return intervals


## Trapezoidal integration
def trapezoidal_integration(function, start, end, number_of_partitions):
    """ Numerically integrates function(x) over the interval 
    start <= x <= end using the trapezoidal rule. number_of_partitions
    symbolises the ammount of partitioned intervals that we will use the
    trapezoidal rule on.
    """
    f = function
    # From result (7) in the reference material, with t = (a,b)
    trapezoidal_rule = lambda t : (t[1]-t[0]) * (f(t[0]) + f(t[1])) / 2

    intervals = partition_interval(start, end, number_of_partitions)
    areas_of_intervals = map(trapezoidal_rule, intervals)
    return sum(areas_of_intervals)


## Simpson integration
def simpson_integration(function, start, end, number_of_partitions):
    """ Numerically integrates function(x) over the interval 
    start <= x <= end using Simpson's rule. number_of_partitions
    symbolises the ammount of partitioned intervals that we will use
    Simpson's rule on.
    """
    
    f = function
    # From result (9) in the reference material
    simpson_rule_untupled = lambda a,b: ((1/6) * (b-a) * (f(a)
                                                           + 4 * f((a+b)/2)
                                                           + f(b)))
    # simpsons_rule_untupled, but takes a tuple (a,b) instead
    simpson_rule = lambda t: simpson_rule_untupled(t[0],t[1])

    intervals = partition_interval(start, end, number_of_partitions)
    areas_of_intervals = map(simpson_rule, intervals)
    return sum(areas_of_intervals)


## Solution to task 1 and 2
def main():
    """ Prints the solution to task 1 and 2 in this lab assignment. """
    result1 = trapezoidal_integration(sqrt, 0, 2, 4)
    print("Integrating sqrt(x) from x=0 to x=2 using the trapezoidal rule over",
          "4 partitions yields:", result1)

    result2 = simpson_integration(sqrt, 0, 2, 4)
    print("Integrating sqrt(x) from x=0 to x=2 using Simpson's rule over",
          "4 partitions yields:", result2)

    analytic_result = (2 / 3) * sqrt(2) ** 3
    print("Integrating sqrt(x) from x=0 to x=2 analytically yields:",
          analytic_result)
        

if __name__ == "__main__":
    main()
