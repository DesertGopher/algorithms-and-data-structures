import random
import numpy as np

rng = np.random.default_rng(seed=random.randint(a=0, b=2**32))
N, M = map(int, input("Введите N и M (от 3 до 10 через пробел): ").split())
A, B = rng.integers(-10, 11, size=(N, M)), rng.integers(-10, 11, size=min(N, M))
print(f"\nИсходная матрица A:\n {A}\n\nМассив B:\n {B}")

[A.__setitem__((i, i), B[i]) for i in range(min(N, M))]
print(f"\nИзмененная матрица A:\n {A}")
