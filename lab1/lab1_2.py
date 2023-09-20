import numpy as np

# 2 Найти одинаковые элементы в двух массивах
def search_common_elements_1(a, b):
    # Получаем массив общими элементами и нулями
    common_indices = np.where(a == b, a, 0)
    # Возвращение не нулевых элементов
    return common_indices[common_indices != 0]

def search_common_elements_2(a, b):
    # Находим индексы одинаковых элементов
    common_indices = np.where(Z1 == Z2)
    # Получаем значения одинаковых элементов и возвращаем
    return Z1[common_indices]

# ============================================================
Z1 = np.random.randint(0, 10, 10)
Z2 = np.random.randint(0, 10, 10)
print(Z1)
print(Z2)
print("Одинаковые элементы:", search_common_elements_1(Z1, Z2))
print("Одинаковые элементы:", search_common_elements_2(Z1, Z2))

# Сравнение одинаковых массивов
Z2 = Z1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
assert np.array_equal(search_common_elements_1(Z1, Z2), Z1), "Ошибка: ожидалось полное совпадение массивов"
assert np.array_equal(search_common_elements_2(Z1, Z2), Z1), "Ошибка: ожидалось полное совпадение массивов"

# Сравнение неравных массивов
Z2 = np.array([9, 8, 7, 6, 4, 5, 3, 2, 1])
assert (len(search_common_elements_1(Z1, Z2)) == 0), "Ошибка: ожидалось полное не совпадение массивов"
assert (len(search_common_elements_2(Z1, Z2)) == 0), "Ошибка: ожидалось полное не совпадение массивов"

print("Test pass success")
