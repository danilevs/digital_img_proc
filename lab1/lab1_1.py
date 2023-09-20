import numpy as np

# 1 Сравнить две матрицы со случайными значениями на равенство
def compare_matrix(a, b):
    return (a == b).all()

A = np.random.randint(0, 2, 5)
B = np.random.randint(0, 2, 5)
print(A, B)
if compare_matrix(A, B):
    print("Equal")
else:
    print("Not equal")

# Сравнение одинаковых матриц
B1 = A1 = np.array([[1, 2], [3, 4]])
assert compare_matrix(A1, B1), "Ошибка: Одинаковые матрицы не считаются равными"
assert np.array_equal(A1, B1), "Ошибка: Одинаковые матрицы не считаются равными"

# Сравнение неравных матриц
A2 = np.array([[1, 2], [3, 4]])
B2 = np.array([[5, 6], [7, 8]])
assert not compare_matrix(A2, B2), "Ошибка: Неравные матрицы считаются равными"
assert not np.array_equal(A2, B2), "Ошибка: Неравные матрицы считаются равными"
print("Test pass success")