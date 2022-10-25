from math import *
from random import randint
n = int(input())
class Matrix:

    def __init__(self, n):
        self.rang = n
        self._length = n
        self.check = False

    def get_matrix(self):
        self.matrix = [[randint(-50, 50) for _ in range(self._length)] for _ in range(self._length)]
        for i in range(self._length):
            self.matrix[2][i] = self.matrix[0][i] + self.matrix[1][i]
            self.matrix[3][i] = self.matrix[0][i] - self.matrix[1][i]

    def det(self, a, b):
        det = 1
        for i in range(len(a)):
            det = det * a[i][i] * b[i][i]
        self.deter = det

    def LU_matrix(self, U, L, P, Q):
        eps = 1e-6
        for k in range(self._length):
            maxx = 0
            max_row = 0
            max_col = 0
            for i in range(k,self._length):
                for j in range(k,self._length):
                    if abs(self.matrix[i][j]) > maxx:
                        maxx = self.matrix[i][j]
                        max_row = i;
                        max_col = j;
            if(max_col != 0 or max_row != 0):
                P.swap_row(max_row, k)
                self.swap_row(max_row, k)
                Q.swap_col(max_col, k)
                self.swap_col(max_col, k)
            else:
                continue
            if abs(self.matrix[k][k]) > eps :
                for i in range(k+1, self._length):
                    self.matrix[i][k] = self.matrix[i][k]/self.matrix[k][k]
                    for j in range(k+1, self._length):
                        self.matrix[i][j] = self.matrix[i][j] - self.matrix[k][j]*self.matrix[i][k]
            else:
                self.rang = k
                break
        for i in range(self._length):
            for j in range(self._length):
                if i > j:
                    L.matrix[i][j] = self.matrix[i][j]
                else:
                    U.matrix[i][j] = self.matrix[i][j]

        self.check = True

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

    def swap_col(self, a, b):
        for i in range(self._length):
            temp = self.matrix[i][a]
            self.matrix[i][a] = self.matrix[i][b]
            self.matrix[i][b] = temp

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

    def kron_kap(self, b):
        U = Help_Matrix(self._length+1)
        L = Help_Matrix(self._length+1)
        Q = Help_Matrix(self._length+1)
        P = Help_Matrix(self._length+1)
        Buff = Matrix(self._length+1)
        buff = [[None for _ in range(Buff._length)] for _ in range(Buff._length)]
        for i in range(Buff._length):
            for j in range(Buff._length):
                if i < Buff._length-1 and j < Buff._length-1:
                    buff[i][j] = self.matrix[i][j]
                elif j == Buff._length - 1 and i < Buff._length-1:
                    buff[i][j] = b.vector[i]
                elif i == Buff._length - 1:
                    buff[i][j] = 0
        Buff.matrix = buff
        Buff.display()
        Buff.LU_matrix(U,L,P,Q)
        print(Buff.rang)
        U.display()
        return Buff.rang


class Help_Matrix(Matrix):
    def __init__(self, n):
        self._length = n
        self.matrix = [[0 for _ in range(self._length)] for _ in range(self._length)]
        for i in range(self._length):
            for j in range(self._length):
                self.matrix[i][i] = 1


class Vector:

    def __init__(self, n):
        self._length = n

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
print("Генерация матрицы")
A.get_matrix()
B = Vector(n)
print("Генерация вектора значений")
B.get_vector()
print()
Y = Help_Vector(n)
X = Help_Vector(n)
Z = Help_Vector(n)
I = Help_Matrix(n)
U = Help_Matrix(n)
L = Help_Matrix(n)
Q = Help_Matrix(n)
P = Help_Matrix(n)

Buf = Matrix(n)
buf = [[None for _ in range(A._length)] for _ in range(A._length)]
for i in range(A._length):
    for j in range(A._length):
        buf[i][j] = A.matrix[i][j]
Buf.matrix = buf
Buf.LU_matrix(U, L, P, Q)
A.display()
print("Ранг матрицы")
print(Buf.rang)
U.display()
A.kron_kap(B)
print("Совместность системы:")
x = True
if True == x:
    print("Система совместна!")
    B.multi_mat_vec(P, B)
    for i in range(n):
        sum = 0
        for j in range(i):
            sum += Y.vector[j]*L.matrix[i][j]
        Y.vector[i] = B.vector[i] - sum

    for i in reversed(range(n)):
        sum = 0
        for j in range(i+1, n):
            sum += Z.vector[j]*U.matrix[i][j]
        if U.matrix[i][i] != 0:
            Z.vector[i] = (Y.vector[i] - sum)/U.matrix[i][i]
        else:
            Z.vector[i] = 0
    X.multi_mat_vec(Q, Z)
    print("Решение уравнения")
    X.display()
    print('\n')
    print('=================================')
    print('Проверки по пунктам:')
    print('а)')
    print("Multiplication L and U:")
    M_1 = Matrix(n)
    M_1.multi(L, U)
    M_1.display()
    M_2 = Matrix(n)
    M_2.multi(P, A)
    M_2.multi(M_2, Q)
    print("Multiplication PA and Q:")
    M_2.display()
    print('б)')
    Rezult = Help_Vector(n)
    Rezult.multi_mat_vec(A, X)
    Rezult.display()



#Нахождение обратной матрицы
# I.multi(P, I)
#
# for i in range(n):
#     for j in range(n):
#         sum = 0
#         for k in range(i):
#             sum += L.matrix[i][k] * Y_m.matrix[k][j]
#         Y_m.matrix[i][j] = I.matrix[i][j] - sum
#
# for i in reversed(range(n)):
#     for j in range(n):
#         sum = 0
#         for k in range(i+1, n):
#             sum += U.matrix[i][k] * Z_m.matrix[k][j]
#         Z_m.matrix[i][j] = (Y_m.matrix[i][j] - sum)/U.matrix[i][i]
#
# X_m.multi(Q, Z_m)
