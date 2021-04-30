## Description
"""
A column vector 'X' will be symbolised as a list of size 1 lists, that is
X = [[x1], [x2] ,[x3], ... , [xn]]. An m by n matrix 'A' will be symbolised by a list of size
m consisting of lists of size n. The solution to task 1 in the lab will be the function
'gaussian_elimination'. The solution to task 2 will be given by running this script in
terminal.
"""


def matrix_multiply(matrix1, matrix2):
    m = len(matrix1)
    n = len(matrix2)
    p = len(matrix2[0])
    element = lambda i,j: sum([matrix1[i][k] * matrix2[k][j]
                               for k in range(n)])
    row = lambda i: [element(i,j) for j in range(p)]
    return [row(i) for i in range(m)]


def create_identity(n):
    """ Creates an n by n identity matrix. """
    return [[1 if i==j else 0 for j in range(n)] for i in range(n)]


def switch(row1, row2, n):
    """ Creates an n by n matrix which switches the rows
    row1 and row2 on a matrix if this matrix returned is multiplied
    on the left. Rows are indexed such that the first row is row 0.
    """
    I = create_identity(n)
    return [I[row2] if i==row1
            else I[row1] if i==row2
            else I[i] for i in range(n)]


def add_multiple_of_row_to_other_row(multiple, row, other_row, n):
    """ Creates an 'n' by 'n' matrix which if multiplied with on the
    left of another matrix, then that other matrix will add
    'multiple' times the row 'row' to the row 'other_row'.
    rows are indexed such that the first row is row 0.
    """
    I = create_identity(n)
    new_other_row = [1 if j==other_row and other_row != row
                     else multiple if j==row
                     else 0 for j in range(n)]
    return [new_other_row if i==other_row else I[i] for i in range(n)]


def scale_row(multiple, row, n):
    """ Create an 'n' by 'n' matrix which if multiplied with on the
    left of another matrix, then that other matrix will have the
    row 'row' be mutliplied with 'multiple'.
    """
    return add_multiple_of_row_to_other_row(multiple, row, row, n)


def transpose(matrix):
    """ Transposes a square matrix. """
    n = len(matrix)
    return [[matrix[i][j] for i in range(n)] for j in range(n)]


def normaliser(matrix):
    """ Normalises a non-zero diagonal matrix."""
    n = len(matrix)
    return [[0 if i!=j
             else 1/matrix[i][j] for j in range(n)]
            for i in range(n)]


def eliminate_column(matrix, column_index):
    """ Given a square matrix 'matrix' with non-zero diagonal element in the 
    column 'column_index', this functions returns the matrix to eliminate all 
    the non diagonal elements in the column. Columns are numbered such that the
    first one is 0.
    """
    column = transpose(matrix)[column_index]
    diagonal_index = column_index
    diagonal_element = column[diagonal_index]
    n = len(matrix)
    column_element_is_correct_except_diagonal = [True if i==diagonal_index
                                                 else column[i] == 0
                                                 for i in range(n) ]
    if all(column_element_is_correct_except_diagonal):
        I = create_identity(n)
        return I
    else:
        index_to_fix = column_element_is_correct_except_diagonal.index(False)
        incorrect_element = column[index_to_fix]
        inverse = - incorrect_element / diagonal_element
        eliminate = add_multiple_of_row_to_other_row(inverse,
                                                        diagonal_index,
                                                        index_to_fix,
                                                        n)
        matrix_prim = matrix_multiply(eliminate, matrix)
        rest_of_elimination = eliminate_column(matrix_prim, column_index)
        return matrix_multiply(rest_of_elimination, eliminate)


def get_pivot_index(matrix, column_index):
    """ Returns the element with the greatest absolute value in the under the
    diagonal in the column 'column_index' in the matrix 'matrix'. Indexing starts
    from zero. If the whole column is zero, the index value -1 is returned.
    """
    column = transpose(matrix)[column_index]
    diagonal_index = column_index
    under_diagonal = column[diagonal_index:]
    max_element, min_element = max(under_diagonal), min(under_diagonal)
    if abs(max_element) >= abs(min_element):
        pivot_element = max_element
    else:
        pivot_element = min_element

    if pivot_element == 0:
        return -1
    else:
        return diagonal_index + under_diagonal.index(pivot_element)


def swap_pivot(matrix, column_index):
    """ Given a matrix 'matrix', with a non-zero pivot below the diagonal
    in the column 'column_index', this function returns the matrix
    to be multiplied with on the left in order to put the the pivot on
    the diagonal. 
    """
    n = len(matrix)
    column = transpose(matrix)[column_index]
    diagonal_index = column_index
    pivot_index = get_pivot_index(matrix, column_index)
    return switch(diagonal_index, pivot_index, n)


def check_column(matrix, column_index, debug=False):
    """ Checks the column 'column_index' (indexing starts from zero) in the 
    matrix 'matrix'. An index integer of the pivot and the string saying "swap"
    are returned if there's some element below the diagonal that has a a 
    strictly bigger absolute value 
    than the diagonal element. "eliminate" is returned if the no pivoting 
    is needed, but there are other non-zero elements in the column. "zero" is 
    returned if the whole column is zero. If the diagonal element is zero and
    the rest of the column is zero, "finnished" is returned.
    """
    n = len(matrix)
    columns = transpose(matrix)
    column = columns[column_index]
    diagonal_index = column_index
    pivot_index = get_pivot_index(matrix, column_index)
    element_is_zero_list = [x == 0 for x in column]
    non_diagonal_element_is_zero_list = [True if i == diagonal_index
                                         else element_is_zero_list[i]
                                         for i in range(n)]
    column_is_eliminated = all(non_diagonal_element_is_zero_list)
    if pivot_index == -1:
        status = "zero"
    elif pivot_index != diagonal_index:
        status = "swap"
    elif column_is_eliminated:
        status = "finnished"
    else:
        status = "eliminate"

    if debug:
        print("Status of column", column_index, ":", status)
    return status


def gaussian_elimination(A, b, left_most_column_to_check=0, debug=False):
    """ If 'A' is an n by n matrix and 'b' is an n by 1 vector,
    then this function returns the n by 1 vector 'x' which solves
    the equation 'Ax = b'. Observe that the vector's elements
    must be surrounded by squarebrackets, for example:
    [[1],[2],[3]] is correct and [1,2,3] is not. An exception is
    raised if the determinant of A is zero.
    """
    if debug:
        print(A)

    l = left_most_column_to_check
    n = len(b)
    if l == n:
        M = normaliser(A)
        b_prim = matrix_multiply(M, b)
        if debug:
            print("The solution is x=", b_prim)
        return b_prim

    column_status = check_column(A,l, debug)
    if column_status == "finnished":
        return gaussian_elimination(A, b, l+1, debug)
    elif column_status == "zero":
        raise ValueError("<gaussian_elimination:Can't have a zero determinant>")
    elif column_status == "swap":
        M = swap_pivot(A, l)
        A_prim, b_prim = matrix_multiply(M, A), matrix_multiply(M, b)
        return gaussian_elimination(A_prim, b_prim, l, debug)
    elif column_status == "eliminate":
        M = eliminate_column(A, l)
        A_prim, b_prim = matrix_multiply(M, A), matrix_multiply(M, b)
        return gaussian_elimination(A_prim, b_prim, l+1, debug)
    else:
        return "something went wrong"


def main():
    """ Solution to task 2 in lab assignment 5. """
    A = [[0.143, 0.357, 2.01], [-1.31, 0.911, 1.99], [11.2, -4.3, -0.605]]
    b = [[-5.173], [-5.458], [4.415]]
    x = gaussian_elimination(A,b,debug=True)
    print("The result of multiplying 'A' with our solution from the function:")
    print(matrix_multiply(A,x))
    print("Original b:")
    print(b)


if __name__ == "__main__":
    main()
