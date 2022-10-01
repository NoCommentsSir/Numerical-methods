n = int(input())
class Matrix:
    def __init__(self, length):
        self.matrix = [[int(input()) for _ in range(length)] for _ in range(length)]


A = Matrix(n)
U = [[0 for _ in range(n)] for _ in range(n)]
L = [[0 for _ in range(n)] for _ in range(n)]
Q = [[0 for _ in range(n)] for _ in range(n)]
P = [[0 for _ in range(n)] for _ in range(n)]
maxx = 10**(-10)
def det(a,b):
    det = 1
    for i in range(len(a)):
        det = det * a[i][i]*b[i][i]
    return det
def multi(a,b):
    M = []
    for i in range(len(a)):
        row = []
        for j in range(len(a)):
            sum = 0
            for k in range(len(a)):
                sum += a[i][k]*b[k][j]
            row.append(sum)
        M.append(row)
    return M

for i in range(n):
    for j in range(n):
        L[i][i] = 1
        P[i][i] = 1
        Q[i][i] = 1
for i in range(n):
    for j in range(n):
        if A.matrix[i][j] > maxx:
            maxx = A.matrix[i][j]

for i in range(n):
    for j in range(n):
        if A.matrix[i][j] == maxx:
            temp_str_1 = A[0]
            temp_str_2 = P[0]
            A[0] = A[i]
            P[0] = P[i]
            A[i] = temp_str_1
            P[i] = temp_str_2
for j in range(n):
    if A[0][j] == maxx:
        for i in range(n):
            temp_col_1 = A[i][0]
            temp_col_2 = Q[i][0]
            A[i][0] = A[i][j]
            Q[i][0] = Q[i][j]
            A[i][j] = temp_col_1
            Q[i][j] = temp_col_2


for i in range(n):
    for j in range(n):
        if i <= j:
            sum = 0
            for k in range(i):
                sum = sum + L[i][k]*U[k][j]
            U[i][j] = A[i][j] - sum
        elif i > j:
            sum = 0
            for k in range(j):
                sum = sum + L[i][k] * U[k][j]
            L[i][j] = (A[i][j] - sum)/U[j][j]
print(f'Det A = {det(L,U)}')
print("Multiplication L and U:")
M_1 = multi(L,U)
M_2 = multi(P,A)
M_2 = multi(M_2, Q)
for i in range(n):
    for j in range(n):
        print(M_1[i][j], end = ' ')
    print()
print('\n')
print("Multiplication PA and Q:")
for i in range(n):
    for j in range(n):
        print(M_2[i][j], end = ' ')
    print()
print('\n')
for i in range(n):
    for j in range(n):
        print(U[i][j], end = ' ')
    print()