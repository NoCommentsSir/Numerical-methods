from math import *
import copy
from random import randint
n = int(input())
eps = 1e-6
class Matrix:

    def __init__(self, n):
        self.rang = n
        self._length = n
        self.check = False

    def get_matrix(self):
        self.matrix = [[randint(-50,50) for _ in range(self._length)] for _ in range(self._length)]

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

    def sum_matrix(self, A, B):
        for i in range(self._length):
            for j in range(self._length):
                self.matrix[i][j] = A.matrix[i][j]+B.matrix[i][j]

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

    def multi_const(self, k):
        for i in range(self._length):
            for j in range(self._length):
                self.matrix[i][j] *= k

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

    def decompose(self, L, D, R):
        for i in range(self._length):
            for j in range(self._length):
                if i < j:
                    R.matrix[i][j] = self.matrix[i][j]
                elif i == j:
                    D.matrix[i][j] = self.matrix[i][j]
                else:
                    L.matrix[i][j] = self.matrix[i][j]

    def change(self):
        sum_max = 0
        for k in range(n):
            sum_el = 0
            for i in range(n):
                if i != k:
                    sum_el += abs(self.matrix[k][i])
            if sum_max < sum_el:
                sum_max = sum_el
        for k in range(n):
            if sum_max >= abs(self.matrix[k][k]):
                self.matrix[k][k] += sum_max

    def Obr_matrix(self):
        I = Help_Matrix(self._length)
        U = Help_Matrix(self._length)
        L = Help_Matrix(self._length)
        P = Help_Matrix(self._length)
        Y_m = Help_Matrix(self._length)
        Z_m = Help_Matrix(self._length)
        QLU = Help_Matrix(n)
        Buf = Matrix(n)
        Buf = copy.deepcopy(self)
        Buf.LU_matrix(U, L, P, QLU)
        I.multi(P, I)
        for i in range(self._length):
            for j in range(self._length):
                sum = 0
                for k in range(i):
                    sum += L.matrix[i][k] * Y_m.matrix[k][j]
                Y_m.matrix[i][j] = I.matrix[i][j] - sum

        for i in reversed(range(self._length)):
            for j in range(self._length):
                sum = 0
                for k in range(i + 1, self._length):
                    sum += U.matrix[i][k] * Z_m.matrix[k][j]
                Z_m.matrix[i][j] = (Y_m.matrix[i][j] - sum) / U.matrix[i][i]
        Z_m.multi(QLU, Z_m)
        return Z_m

    def Jacobi(self, B, C, q, x, A):
        k = 1
        iterator_prev = Help_Vector(self._length)
        diff = Help_Vector(self._length)
        Rez = Help_Vector(self._length)
        Exam = Help_Vector(self._length)
        iterator_curr = copy.deepcopy(C)
        norm_iterator = C.norm
        while q / (1 - q) * norm_iterator > eps:
            iterator_prev = copy.deepcopy(iterator_curr)
            iterator_curr.multi_mat_vec(B, iterator_prev)
            iterator_curr.sum_vector(iterator_curr, C)
            diff.razn_vector(iterator_curr, iterator_prev)
            diff.norma()
            norm_iterator = diff.norm
            k += 1
        Rez.multi_mat_vec(self, iterator_curr)
        print('Ax = b in Jacobi:')
        Rez.display()
        print()
        print("Проверка точности")
        Exam.multi_mat_vec(A, x)
        Exam.razn_vector(Rez, Exam)
        Exam.display()
        print()
        return k

    def Seidel(self, Bmatrix, C, q, x, A):
        k = 1
        iterator_prev = Help_Vector(self._length)
        Rez = Help_Vector(self._length)
        Exam = Help_Vector(self._length)
        iterator_curr = copy.deepcopy(C)
        norm_iterator = C.norm
        while q / (1 - q) * norm_iterator > eps:
            diff = Help_Vector(self._length)
            iterator_prev = copy.deepcopy(iterator_curr)
            for i in range(self._length):
                k1 = 0
                for j in range(self._length):
                    k1 += Bmatrix.matrix[i][j]*iterator_curr.vector[j]
                iterator_curr.vector[i] = k1 + C.vector[i]
            diff.razn_vector(iterator_curr, iterator_prev)
            diff.norma()
            norm_iterator = diff.norm
            k += 1
        Rez.multi_mat_vec(self, iterator_curr)
        print('Ax = b in Seidel:')
        Rez.display()
        print()
        print("Проверка точности")
        Exam.multi_mat_vec(A, x)
        Exam.razn_vector(Rez, Exam)
        Exam.display()
        print()
        return k

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
        self.vector = []

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

    def norma(self):
        sum = 0
        for i in range(self._length):
            sum += self.vector[i]**2
        self.norm = sqrt(sum)

    def sum_vector(self, A, B):
        for i in range(self._length):
            self.vector[i] = A.vector[i]+B.vector[i]

    def multi_vec(self, A, B):
        sum = 0
        for i in range(self._length):
            sum += A.vector[i]*B.vector[i]
        return sum

    def razn_vector(self, A, B):
        for i in range(self._length):
            self.vector[i] = A.vector[i]-B.vector[i]

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
Buff1 = copy.deepcopy(A)
L1 = Help_Matrix(n)
U1 = Help_Matrix(n)
Q1 = Help_Matrix(n)
P1 = Help_Matrix(n)
y = Help_Vector(n)
x = Help_Vector(n)
z = Help_Vector(n)
b = Help_Vector(n)
Buff1.LU_matrix(U1, L1, P1, Q1)
b.multi_mat_vec(P1, B)
for i in range(n):
    sum = 0
    for j in range(i):
        sum += y.vector[j]*L1.matrix[i][j]
    y.vector[i] = b.vector[i] - sum
for i in reversed(range(n)):
    sum = 0
    for j in range(i+1, n):
        sum += z.vector[j]*U1.matrix[i][j]
    z.vector[i] = (y.vector[i] - sum)/U1.matrix[i][i]
x.multi_mat_vec(Q1, z)
A_T = Matrix(n)
A_T = copy.deepcopy(A)
A_T.transpose()
A_diag = copy.deepcopy(A)
A_diag.change()
A_cv_form = Matrix(n)
A_cv_form.multi(A, A_T)
Buf = Matrix(n)
Buf = copy.deepcopy(A)
Buf.change()
L = Help_Matrix(n)
L.matrix = [[0 for _ in range(n)] for _ in range(n)]
R = Help_Matrix(n)
R = copy.deepcopy(L)
D = Help_Matrix(n)
D = copy.deepcopy(L)
Buf.decompose(L, D, R)
D_inv = D.Obr_matrix()
Bmatrix = Help_Matrix(n)
Bmatrix.sum_matrix(L,R)
Bmatrix.multi(D_inv, Bmatrix)
Bmatrix.multi_const(-1)
C = Help_Vector(n)
C.multi_mat_vec(D_inv, B)
Bmatrix.norma()
C.norma()
if Bmatrix.norm < 1:
    q = Bmatrix.norm
if Bmatrix.norm > 1:
    q = 1e-1
print('q:')
print(q)
k = ceil(log((eps/ C.norm * (1 - q)), q))
print('k:')
print(k)
print('Jacobi:')
c = copy.deepcopy(A_diag)
count_jac = A_diag.Jacobi(Bmatrix, C, q, x, A)
print(count_jac)
print('Seidel:')
count_sei = A_diag.Seidel(Bmatrix, C, q, x, A)
print(count_sei)

# Y = Help_Vector(n)
# X = Help_Vector(n)
# I = Help_Matrix(n)
# U = Help_Matrix(n)
# L = Help_Matrix(n)
# P = Help_Matrix(n)
# Y_m = Help_Matrix(n)
# Z_m = Help_Matrix(n)
# QLU = Help_Matrix(n)
# Buf = Matrix(n)
# Q = copy.deepcopy(A)
# R = copy.deepcopy(A)
# # Buf.LU_matrix(U, L, P, Q)
# # A.rang = Buf.rang
# Q.gram_s()
# Q_1 = Matrix(n)
# Q_1 = copy.deepcopy(Q)
# Q_1 = Q_1.transpose()
# R.multi(Q_1, A)
# R_2 = copy.deepcopy(R)
# R_2.LU_matrix(U, L, P, QLU)
# I.multi(P, I)
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
# Z_m.multi(QLU, Z_m)
#
# Y.multi_mat_vec(Q_1, B)
# X.multi_mat_vec(Z_m, Y)
# print("Вектор значений:")
# X.display()
# print()
# print("==================================================================================")
# print("Проверки по этапам\n")
# print("Этап 1 - правильность разложения на Q и R:\n")
# Buf.multi(Q, R)
# Buf.display()
# print()
# print("Этап 2 - правильность найденного вектора значений:\n")
# M = Vector(n)
# M.multi_mat_vec(A,X)
# M.display()
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
