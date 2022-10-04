from math import *
n = int(input())
class Matrix:

    def __init__(self, length):
        self._length = length

    def get_matrix(self):
        self.matrix = [[int(input()) for _ in range(self._length)] for _ in range(self._length)]

    def det(self, a, b):
        det = 1
        for i in range(len(a)):
            det = det * a[i][i] * b[i][i]
        self.deter = det

    def LU_matrix(self, a, b):
        for i in range(n):
            for j in range(n):
                if i <= j:
                    sum = 0
                    for k in range(i):
                        sum = sum + a.matrix[i][k] * b.matrix[k][j]
                    b.matrix[i][j] = A.matrix[i][j] - sum
                elif i > j:
                    sum = 0
                    for k in range(j):
                        sum = sum + a.matrix[i][k] * b.matrix[k][j]
                    a.matrix[i][j] = (A.matrix[i][j] - sum) / b.matrix[j][j]
        self.L = Help_Matrix(a._length)
        self.L.matrix = a.matrix
        self.U = Help_Matrix(b._length)
        self.U.matrix = b.matrix

    def multi(self, a, b):
        M = []
        for i in range(a._length):
            row = []
            for j in range(a._length):
                sum = 0
                for k in range(a._length):
                    sum += a.matrix[i][k] * b.matrix[k][j]
                row.append(sum)
            M.append(row)
        self.matrix = M

    def swap_row(self, a, b):
        temp = self.matrix[a]
        self.matrix[a] = self.matrix[b]
        self.matrix[b] = temp

    def swap_elem(self, a_1, a_2, b_1, b_2):
        temp = self.matrix[a_1][a_2]
        self.matrix[a_1][a_2] = self.matrix[b_1][b_2]
        self.matrix[b_1][b_2] = temp

    def display(self):
        for i in range(self._length):
            for j in range(self._length):
                print(self.matrix[i][j], end=' ')
            print()

    def norma(self):
        sum = 0
        for i in range(self._length):
            for j in range(self._length):
                sum += self.matrix[i][j]**2
        self.norm = sqrt(sum)


class Help_Matrix(Matrix):
    def __init__(self, length):
        self._length = length
        self.matrix = [[0 for _ in range(self._length)] for _ in range(self._length)]
        for i in range(self._length):
            for j in range(self._length):
                self.matrix[i][i] = 1

class Vector:

    def __init__(self, length):
        self._length = length

    def get_vector(self):
        self.vector = [int(input()) for _ in range(self._length)]

    def display(self):
        for i in range(self._length):
            print(self.vector[i], end=' ')

    def multi_mat_vec(self, a, b):
        M = []
        for i in range(a._length):
            sum = 0
            for j in range(a._length):
                sum += a.matrix[i][j] * b.vector[j]
            M.append(sum)
        self.vector = M

    def swap_vec_lem(self, a, b):
        temp = self.vector[a]
        self.vector[a] = self.vector[b]
        self.vector[b] = temp

class Help_Vector(Vector):
    def __init__(self, length):
        self._length = length
        self.vector = [0 for _ in range(self._length)]

A = Matrix(n)
print("Введите матрицу. Для проверки работы программы (3 2 -1, 2 -1 5, 1 7 -1)")
A.get_matrix()
B = Vector(n)
print("Введите вектор. Для проверки работы программы (4 23 5)")
B.get_vector()
Y = Help_Vector(n)
Y_m = Help_Matrix(n)
Y_m.matrix = [[0 for _ in range(n)] for _ in range(n)]
X = Help_Vector(n)
X_m = Help_Matrix(n)
I = Help_Matrix(n)
X_m.matrix = [[0 for _ in range(n)] for _ in range(n)]
U = Help_Matrix(n)
L = Help_Matrix(n)
Q = Help_Matrix(n)
P = Help_Matrix(n)
maxx = 10**(-10)

Buf = Matrix(n)
buf = [[None for _ in range(n)] for _ in range(n)]
for i in range(n):
    for j in range(n):
        buf[i][j] = A.matrix[i][j]
Buf.matrix = buf
for i in range(n):
    for j in range(n):
        if A.matrix[i][j] > maxx:
            maxx = A.matrix[i][j]

for i in range(n):
    for j in range(n):
        if A.matrix[i][j] == maxx:
            A.swap_row(0,i)
            P.swap_row(0,i)
            B.swap_vec_lem(0,i)

for j in range(n):
    if A.matrix[0][j] == maxx:
        for i in range(n):
            A.swap_elem(i,0,i,j)
            Q.swap_elem(i, 0, i, j)
A.LU_matrix(L, U)

for i in range(n):
    sum = 0
    for j in range(n):
        sum += Y.vector[j]*A.L.matrix[i][j]
    Y.vector[i] = B.vector[i] - sum

for i in reversed(range(n)):
    sum = 0
    for j in range(n):
        sum += X.vector[j]*A.U.matrix[i][j]
    X.vector[i] = (Y.vector[i] - sum)/A.U.matrix[i][i]

for i in range(n):
    for j in range(n):
        sum = 0
        for k in range(n):
            sum += A.L.matrix[i][k] * Y_m.matrix[k][j]
        Y_m.matrix[i][j] = I.matrix[i][j] - sum

for i in reversed(range(n)):
    for j in range(n):
        sum = 0
        for k in range(n):
            sum += A.U.matrix[i][k] * X_m.matrix[k][j]
        X_m.matrix[i][j] = (Y_m.matrix[i][j] - sum)/A.U.matrix[i][i]
A.det(A.L.matrix,A.U.matrix)
A.norma()
print("Ответы к пунктам:")
print(f'a) Det A = {A.deter}')
print('б)')
X.display()
print()
print("в)")
X_m.display()
print("г)")
X_m.norma()
print(f'Число обусловленности = {X_m.norm*A.norm}')
print('\n')
print('=================================')
print('Проверки по пунктам:')
print('а)')
print("Multiplication L and U:")
M_1 = Matrix(n)
M_1.multi(A.L,A.U)
M_1.display()
M_2 = Matrix(n)
M_2.multi(P,Buf)
M_2.multi(M_2, Q)
print("Multiplication PA and Q:")
M_2.display()
print('б)')
Rezult = Help_Vector(n)
Rezult.multi_mat_vec(A, X)
Rezult.display()
print()
print('в)')
Rez_2 = Matrix(n)
Rez_2.multi(A, X_m)
Rez_3 = Matrix(n)
Rez_3.multi(X_m, A)
Rez_2.display()
print()
Rez_3.display()
