from math import *
import copy
from random import randint
n = int(input())
class Matrix:

    def __init__(self, n):
        self.rang = n
        self._length = n
        self.check = False

    def get_matrix(self):
        self.matrix = [[randint(-50, 50) for _ in range(self._length)] for _ in range(self._length)]

    def det(self, a, b):
        det = 1
        for i in range(len(a)):
            det = det * a[i][i] * b[i][i]
        self.deter = det

    def LU_matrix(self, U, L, P, Q):
        eps = 1e-10
        for k in range(self._length):
            maxx = 0
            max_row = 0
            max_col = 0
            for i in range(k,self._length):
                for j in range(k,self._length):
                    if abs(self.matrix[i][j]) > maxx:
                        maxx = abs(self.matrix[i][j])
                        max_row = i;
                        max_col = j;
            if(max_col != 0 or max_row != 0):
                P.swap_row(max_row, k)
                self.swap_row(max_row, k)
                Q.swap_col(max_col, k)
                self.swap_col(max_col, k)
            else:
                self.rang = k
            if abs(self.matrix[k][k]) - eps > 0:
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
        Buff.LU_matrix(U,L,P,Q)
        return Buff.rang

    def transpose(self):
        for i in range(self._length):
            for j in range(i+1, self._length):
                x = self.matrix[i][j]
                self.matrix[i][j] = self.matrix[j][i]
                self.matrix[j][i] = x
        return self


    def norm_vec(self, a):
        norm = 0
        for i in range(self._length):
            norm += a[i]*a[i]
        return sqrt(norm)

    def scal(self, a, b):
        multi = 0
        for i in range(self._length):
            multi += a[i]*b[i]
        return multi

    def proj(self, a, b):
        new_a = []
        for i in range(self._length):
            new_a.append((self.scal(a,b)/self.scal(b,b))*b[i])
        return new_a

    def gram_s(self):
        new_matrix = []
        self.transpose()
        for i in range(self._length):
            a = self.matrix[i]
            for j in new_matrix:
                p = self.proj(a, j)
                for k in range(self._length):
                    a[k] = a[k]-p[k]
            norm = self.norm_vec(a)
            for k in range(self._length):
                a[k] = a[k] / norm
            new_matrix.append(a)
        self.matrix = new_matrix
        self.transpose()
        return self



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
        self.vector = [randint(-50, 50) for _ in range(self._length)]

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
print("Генерация матрицы:")
A.get_matrix()
A.display()
print()
print("Генерация вектора значений:")
B = Vector(n)
B.get_vector()
B.display()
print("\n")
print("==================================================================================")
Y = Help_Vector(n)
X = Help_Vector(n)
I = Help_Matrix(n)
U = Help_Matrix(n)
L = Help_Matrix(n)
P = Help_Matrix(n)
Y_m = Help_Matrix(n)
Z_m = Help_Matrix(n)
QLU = Help_Matrix(n)
Buf = Matrix(n)
Q = copy.deepcopy(A)
R = copy.deepcopy(A)
# Buf.LU_matrix(U, L, P, Q)
# A.rang = Buf.rang
Q.gram_s()
Q_1 = Matrix(n)
Q_1 = copy.deepcopy(Q)
Q_1 = Q_1.transpose()
R.multi(Q_1, A)
R_2 = copy.deepcopy(R)
R_2.LU_matrix(U, L, P, QLU)
I.multi(P, I)
for i in range(n):
    for j in range(n):
        sum = 0
        for k in range(i):
            sum += L.matrix[i][k] * Y_m.matrix[k][j]
        Y_m.matrix[i][j] = I.matrix[i][j] - sum

for i in reversed(range(n)):
    for j in range(n):
        sum = 0
        for k in range(i+1, n):
            sum += U.matrix[i][k] * Z_m.matrix[k][j]
        Z_m.matrix[i][j] = (Y_m.matrix[i][j] - sum)/U.matrix[i][i]

Z_m.multi(QLU, Z_m)

Y.multi_mat_vec(Q_1, B)
X.multi_mat_vec(Z_m, Y)
print("Вектор значений:")
X.display()
print()
print("==================================================================================")
print("Проверки по этапам\n")
print("Этап 1 - правильность разложения на Q и R:\n")
Buf.multi(Q, R)
Buf.display()
print()
print("Этап 2 - правильность найденного вектора значений:\n")
M = Vector(n)
M.multi_mat_vec(A,X)
M.display()
#===============================================================================================================================================
#Решение 1 и 2 пункта
# print("Ранг матрицы")
# print(A.rang)
# print("Совместность системы:")
# if A.rang == A.kron_kap(B):
#     print("Система совместна!")
#     B.multi_mat_vec(P, B)
#     for i in range(n):
#         sum = 0
#         for j in range(i):
#             sum += Y.vector[j]*L.matrix[i][j]
#         Y.vector[i] = B.vector[i] - sum
#
#     for i in reversed(range(A.rang)):
#         sum = 0
#         for j in range(i+1, n):
#             sum += Z.vector[j]*U.matrix[i][j]
#         Z.vector[i] = (Y.vector[i] - sum)/U.matrix[i][i]
#     X.multi_mat_vec(Q, Z)
#     print("Решение уравнения")
#     X.display()
#     print('\n')
#     print('=================================')
#     print('Проверки по пунктам:')
#     print('а)')
#     print("Multiplication L and U:")
#     M_1 = Matrix(n)
#     M_1.multi(L, U)
#     M_1.display()
#     print("==================================================================================")
#     M_2 = Matrix(n)
#     M_2.multi(P, A)
#     M_2.multi(M_2, Q)
#     print("Multiplication PA and Q:")
#     M_2.display()
#     print("==================================================================================")
#     print('б)')
#     Rezult = Help_Vector(n)
#     Rezult.multi_mat_vec(A, X)
#     Rezult.display()
# else:
#     print("Система не совместна!")
#     print('Проверка:')
#     print("Multiplication L and U:")
#     M_1 = Matrix(n)
#     M_1.multi(L, U)
#     M_1.display()
#     print("==================================================================================")
#     M_2 = Matrix(n)
#     M_2.multi(P, A)
#     M_2.multi(M_2, Q)
#     print("Multiplication PA and Q:")
#     M_2.display()
