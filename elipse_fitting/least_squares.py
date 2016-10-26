import numpy as np


def least_squares_circle(data):
    """Want to find the parameters a, b, c, d such that it minimizes the error for the linear system:
            a*x_i^2 + b*y_i^2 + c*x_i +d*y_i - 1 = 0
            To solve for the unknowns create a matrix:

            | a_11 a_12 a_13 a_14| |a|   |b_1|
            | a_21 a_22 a_23 a_24| |b| = |b_2|
            | a_31 a_32 a_33 a_34| |c|   |b_3|
            | a_41 a_42 a_43 a_44| |d|   |b_4|
            """
    a_11 = a_12 = a_13 = a_14 = a_21 = a_22 = a_23 = a_24 = a_31 = a_32 = a_33 = a_34 = 0
    a_41 = a_42 = a_43 = a_44 = b_1 = b_2 = b_3 = b_4 = 0

    for i in range(0, len(data)):
        a_11 += data[i][0]**4
        a_12 += data[i][0]**2 * data[i][1]**2
        a_13 += data[i][0]**3
        a_14 += data[i][0]**2 * data[i][1]
        a_22 += data[i][1]**4
        a_23 += data[i][0] * data[i][1]**2
        a_24 += data[i][1] ** 3
        a_33 += data[i][0]
        a_34 += data[i][0] * data[i][1]
        a_44 += data[i][1]**2
        b_1 +=  data[i][0]**2
        b_4 += data[i][1]

    a_21 = a_12
    a_31 = a_13
    a_32 = a_23
    a_41 = a_14
    a_42 = a_24
    a_43 = a_34
    b_2 = a_44
    b_3 = a_33

    matrix = np.array([[a_11, a_12, a_13, a_14],
                       [a_21, a_22, a_23, a_24],
                       [a_31, a_32, a_33, a_34],
                       [a_41, a_42, a_43, a_44]])
    answer = np.array([b_1, b_2, b_3, b_4])

    unknowns = np.linalg.solve(matrix, answer)