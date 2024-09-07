import numpy as np


def classic_multiplication(a_matrix, b_matrix):
    return np.dot(a_matrix, b_matrix)


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


def strassen_multiplication(a_matrix, b_matrix):
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
        return classic_multiplication(a_matrix, b_matrix)

    mid = n // 2
    A11, A12, A21, A22 = a_matrix[:mid, :mid], a_matrix[:mid, mid:], a_matrix[mid:, :mid], a_matrix[mid:, mid:]
    B11, B12, B21, B22 = b_matrix[:mid, :mid], b_matrix[:mid, mid:], b_matrix[mid:, :mid], b_matrix[mid:, mid:]

    M1 = strassen_multiplication(A11 + A22, B11 + B22)
    M2 = strassen_multiplication(A21 + A22, B11)
    M3 = strassen_multiplication(A11, B12 - B22)
    M4 = strassen_multiplication(A22, B21 - B11)
    M5 = strassen_multiplication(A11 + A12, B22)
    M6 = strassen_multiplication(A21 - A11, B11 + B12)
    M7 = strassen_multiplication(A12 - A22, B21 + B22)

    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6

    # Собираем результат
    C = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))

    # Обрезаем итоговую матрицу до исходного размера
    return C[:original_size[0], :original_size[1]]
