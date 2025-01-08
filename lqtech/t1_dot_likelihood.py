import numpy as np

a = np.array([
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ],
    [
        [10, 20, 30, 40],
        [50, 60, 70, 80],
    ],
    [
        [100, 200, 300, 400],
        [500, 600, 700, 800],
    ],
])
X = np.sum(a, axis=0, keepdims=True)

b = np.array([
    [
        [9]
    ],
    [
        [10]
    ],
    [
        [11]
    ]
])

print(a.shape)
print(b.shape)

c = a * b
print(c)
