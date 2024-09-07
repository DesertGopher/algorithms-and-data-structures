from math import ceil, log

import numpy as np
from scipy import linalg
from sympy import Matrix
import tensorflow as tf


def classic_multiplication(A, B):
    """
    Умножение двух матриц A и B.

    Параметры:
    A (list of list of int/float): Первая матрица
    B (list of list of int/float): Вторая матрица

    Возвращает:
    list of list of int/float: Результат умножения матриц
    """
    # Определяем размеры матриц
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])

    # Проверка на согласованность размеров
    if cols_A != rows_B:
        raise ValueError("Число столбцов первой матрицы должно быть равно числу строк второй матрицы.")

    # Инициализация результата с нулями
    result = [[0] * cols_B for _ in range(rows_A)]

    # Умножение матриц
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]

    return result


def numpy_multiplication(a_matrix, b_matrix):
    return np.dot(a_matrix, b_matrix)


def scipy_multiplication(A, B):
    """Умножение матриц с использованием библиотеки SciPy."""
    result = linalg.blas.sgemm(1.0, A, B)
    return result


def sympy_multiplication(A, B):
    """
    Умножение матриц с использованием библиотеки SymPy.
    Символьное умножение
    """
    A_matrix = Matrix(A)
    B_matrix = Matrix(B)
    result = A_matrix * B_matrix
    return result


def tensorflow_multiplication(A, B):
    """
    Умножение матриц с использованием библиотеки TensorFlow.
    Оптимизированное умножение на GPU/CPU
    """
    A_tensor = tf.constant(A)
    B_tensor = tf.constant(B)
    result = tf.matmul(A_tensor, B_tensor)
    return result.numpy()


def pad_matrix(a_matrix):
    (rows, cols) = a_matrix.shape
    new_size = max(rows, cols)

    # Если матрица уже имеет чётные размеры, возвратим её как есть
    if rows % 2 == 0 and cols % 2 == 0:
        return a_matrix

    # Если размер нечётный, увеличиваем до чётного
    if new_size % 2 != 0:
        new_size += 1

    # Создаём новую матрицу с увеличенным размером и заполняем нулями
    padded_matrix = np.zeros((new_size, new_size), dtype=a_matrix.dtype)
    padded_matrix[:rows, :cols] = a_matrix  # Копируем исходную матрицу

    return padded_matrix


def custom_strassen_multiplication(a_matrix, b_matrix):
    # Запоминаем исходные размеры матриц
    original_size = a_matrix.shape

    # Если размер матриц нечётный, дополняем их до чётного размера
    if len(a_matrix) % 2 != 0 or len(a_matrix[0]) % 2 != 0:
        a_matrix = pad_matrix(a_matrix)
        b_matrix = pad_matrix(b_matrix)

    n = len(b_matrix)

    # Базовый случай для матриц 1x1
    if n == 1:
        return a_matrix * b_matrix

    # Если размеры матрицы 2x2 или меньше, переходим к классическому умножению
    if n <= 2:
        return numpy_multiplication(a_matrix, b_matrix)

    mid = n // 2
    A11, A12, A21, A22 = a_matrix[:mid, :mid], a_matrix[:mid, mid:], a_matrix[mid:, :mid], a_matrix[mid:, mid:]
    B11, B12, B21, B22 = b_matrix[:mid, :mid], b_matrix[:mid, mid:], b_matrix[mid:, :mid], b_matrix[mid:, mid:]

    M1 = custom_strassen_multiplication(A11 + A22, B11 + B22)
    M2 = custom_strassen_multiplication(A21 + A22, B11)
    M3 = custom_strassen_multiplication(A11, B12 - B22)
    M4 = custom_strassen_multiplication(A22, B21 - B11)
    M5 = custom_strassen_multiplication(A11 + A12, B22)
    M6 = custom_strassen_multiplication(A21 - A11, B11 + B12)
    M7 = custom_strassen_multiplication(A12 - A22, B21 + B22)

    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6

    # Собираем результат
    C = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))

    # Обрезаем итоговую матрицу до исходного размера
    return C[:original_size[0], :original_size[1]]


def read(filename):
    lines = open(filename).read().splitlines()
    A = []
    B = []
    matrix = A
    for line in lines:
        if line != "":
            matrix.append([int(el) for el in line.split("\t")])
        else:
            matrix = B
    return A, B


def print_matrix(matrix):
    for line in matrix:
        print("\t".join(map(str, line)))


def ikj_matrix_product(A, B):
    n = len(A)
    C = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for k in range(n):
            for j in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C


def add(A, B):
    n = len(A)
    C = [[0 for j in range(0, n)] for i in range(0, n)]
    for i in range(0, n):
        for j in range(0, n):
            C[i][j] = A[i][j] + B[i][j]
    return C


def subtract(A, B):
    n = len(A)
    C = [[0 for j in range(0, n)] for i in range(0, n)]
    for i in range(0, n):
        for j in range(0, n):
            C[i][j] = A[i][j] - B[i][j]
    return C


def strassenR(A, B):
    n = len(A)

    if n <= 2:
        return ikj_matrix_product(A, B)
    else:
        # initializing the new sub-matrices
        new_size = n // 2
        a11 = [[0 for j in range(0, new_size)] for i in range(0, new_size)]
        a12 = [[0 for j in range(0, new_size)] for i in range(0, new_size)]
        a21 = [[0 for j in range(0, new_size)] for i in range(0, new_size)]
        a22 = [[0 for j in range(0, new_size)] for i in range(0, new_size)]

        b11 = [[0 for j in range(0, new_size)] for i in range(0, new_size)]
        b12 = [[0 for j in range(0, new_size)] for i in range(0, new_size)]
        b21 = [[0 for j in range(0, new_size)] for i in range(0, new_size)]
        b22 = [[0 for j in range(0, new_size)] for i in range(0, new_size)]

        aResult = [[0 for j in range(0, new_size)] for i in range(0, new_size)]
        bResult = [[0 for j in range(0, new_size)] for i in range(0, new_size)]

        # dividing the matrices in 4 sub-matrices:
        for i in range(0, new_size):
            for j in range(0, new_size):
                a11[i][j] = A[i][j]  # top left
                a12[i][j] = A[i][j + new_size]  # top right
                a21[i][j] = A[i + new_size][j]  # bottom left
                a22[i][j] = A[i + new_size][j + new_size]  # bottom right

                b11[i][j] = B[i][j]  # top left
                b12[i][j] = B[i][j + new_size]  # top right
                b21[i][j] = B[i + new_size][j]  # bottom left
                b22[i][j] = B[i + new_size][j + new_size]  # bottom right

        # Calculating p1 to p7:
        aResult = add(a11, a22)
        bResult = add(b11, b22)
        p1 = strassenR(aResult, bResult)  # p1 = (a11+a22) * (b11+b22)

        aResult = add(a21, a22)  # a21 + a22
        p2 = strassenR(aResult, b11)  # p2 = (a21+a22) * (b11)

        bResult = subtract(b12, b22)  # b12 - b22
        p3 = strassenR(a11, bResult)  # p3 = (a11) * (b12 - b22)

        bResult = subtract(b21, b11)  # b21 - b11
        p4 = strassenR(a22, bResult)  # p4 = (a22) * (b21 - b11)

        aResult = add(a11, a12)  # a11 + a12
        p5 = strassenR(aResult, b22)  # p5 = (a11+a12) * (b22)

        aResult = subtract(a21, a11)  # a21 - a11
        bResult = add(b11, b12)  # b11 + b12
        p6 = strassenR(aResult, bResult)  # p6 = (a21-a11) * (b11+b12)

        aResult = subtract(a12, a22)  # a12 - a22
        bResult = add(b21, b22)  # b21 + b22
        p7 = strassenR(aResult, bResult)  # p7 = (a12-a22) * (b21+b22)

        # calculating c21, c21, c11 e c22:
        c12 = add(p3, p5)  # c12 = p3 + p5
        c21 = add(p2, p4)  # c21 = p2 + p4

        aResult = add(p1, p4)  # p1 + p4
        bResult = add(aResult, p7)  # p1 + p4 + p7
        c11 = subtract(bResult, p5)  # c11 = p1 + p4 - p5 + p7

        aResult = add(p1, p3)  # p1 + p3
        bResult = add(aResult, p6)  # p1 + p3 + p6
        c22 = subtract(bResult, p2)  # c22 = p1 + p3 - p2 + p6

        # Grouping the results obtained in a single matrix:
        C = [[0 for j in range(0, n)] for i in range(0, n)]
        for i in range(0, new_size):
            for j in range(0, new_size):
                C[i][j] = c11[i][j]
                C[i][j + new_size] = c12[i][j]
                C[i + new_size][j] = c21[i][j]
                C[i + new_size][j + new_size] = c22[i][j]
        return C


def strassen_multiplication(A, B):
    # print("A", A)
    # print("B", B)
    # print(type(A), type(B))
    assert type(A) == list and type(B) == list
    assert len(A) == len(A[0]) == len(B) == len(B[0])

    # Make the matrices bigger so that you can apply the strassen
    # algorithm recursively without having to deal with odd
    # matrix sizes
    nextPowerOfTwo = lambda n: 2 ** int(ceil(log(n, 2)))
    n = len(A)
    m = nextPowerOfTwo(n)
    APrep = [[0 for i in range(m)] for j in range(m)]
    BPrep = [[0 for i in range(m)] for j in range(m)]
    for i in range(n):
        for j in range(n):
            APrep[i][j] = A[i][j]
            BPrep[i][j] = B[i][j]
    CPrep = strassenR(APrep, BPrep)
    C = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            C[i][j] = CPrep[i][j]
    return C
