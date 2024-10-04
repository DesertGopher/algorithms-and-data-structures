def classic_large_multiplication(a, b):
    a = str(str(a)[::-1])
    b = str(str(b)[::-1])

    result = [0] * (len(a) + len(b))
    for i in range(len(a)):
        for j in range(len(b)):
            result[i + j] += int(a[i]) * int(b[j])
            if result[i + j] >= 10:
                result[i + j + 1] += result[i + j] // 10
                result[i + j] %= 10
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return int(''.join(map(str, result[::-1])))


def karatsuba(x, y):
    if x < 10 or y < 10:
        return x * y
    m = max(len(str(x)), len(str(y))) // 2
    high1, low1 = divmod(x, 10 ** m)
    high2, low2 = divmod(y, 10 ** m)
    z0 = karatsuba(low1, low2)
    z1 = karatsuba((low1 + high1), (low2 + high2))
    z2 = karatsuba(high1, high2)
    return (z2 * 10 ** (2 * m)) + ((z1 - z2 - z0) * 10 ** m) + z0
