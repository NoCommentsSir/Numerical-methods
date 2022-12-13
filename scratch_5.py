import math
import copy
from random import randint
from datetime import datetime

import numpy
n = int(input())
eps = 1e-6
delta = 1e-3
class Matrix:

    def __init__(self, n):
        self.rang = n
        self._length = n
        self.check = False
        self.operation = 0

    def get_matrix(self):
        self.matrix = [[randint(-1000,1000) for _ in range(self._length)] for _ in range(self._length)]

    def det(self, a, b):
        det = 1
        for i in range(len(a)):
            det = det * a[i][i] * b[i][i]
        self.deter = det

    def LU_matrix(self, U, L, P, Q):
        eps = 1e-10
        operation = 0
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
                    operation += 1
                    for j in range(k+1, self._length):
                        self.matrix[i][j] = self.matrix[i][j] - self.matrix[k][j]*self.matrix[i][k]
                        operation += 2
            else:
                self.rang = k
                break
        for i in range(self._length):
            for j in range(self._length):
                if i > j:
                    L.matrix[i][j] = self.matrix[i][j]
                else:
                    U.matrix[i][j] = self.matrix[i][j]
        self.operation += operation
        self.check = True

    def sum_matrix(self, A, B):
        for i in range(self._length):
            for j in range(self._length):
                self.matrix[i][j] = A.matrix[i][j]+B.matrix[i][j]

    def razn_matrix(self, A, B):
        for i in range(self._length):
            for j in range(self._length):
                self.matrix[i][j] = A.matrix[i][j]-B.matrix[i][j]

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
        count = 0
        for i in range(self._length):
            for j in range(self._length):
                sum += self.matrix[i][j]**2
                count += 2
        count += 11
        self.norm = math.sqrt(sum)
        self.operation += count

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
        return math.sqrt(norm)

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

    def uravn(self, Func):
        U = Help_Matrix(self._length)
        L = Help_Matrix(self._length)
        P = Help_Matrix(self._length)
        Q = Help_Matrix(self._length)
        Y = Help_Vector(self._length)
        Z = Help_Vector(self._length)
        X1 = Help_Vector(self._length)
        Buf = Matrix(self._length)
        Buf = copy.deepcopy(self)
        Buf.LU_matrix(U, L, P, Q)
        count = Buf.operation
        B = copy.deepcopy(Func)
        B.multi_mat_vec(P, B)
        for i in range(self._length):
            sum = 0
            for j in range(i):
                sum += Y.vector[j]*L.matrix[i][j]
                count += 2
            Y.vector[i] = B.vector[i] - sum
            count += 1

        for i in reversed(range(Buf.rang)):
            sum = 0
            for j in range(i+1, self._length):
                sum += Z.vector[j]*U.matrix[i][j]
                count += 2
            Z.vector[i] = (Y.vector[i] - sum)/U.matrix[i][i]
            count += 2
        X1.multi_mat_vec(Q, Z)
        X1.operation += count
        X1.operation += B.operation
        return X1

    def Jacobi(self, B, C, q, x):
        B.display()
        print()
        C.display()
        print()
        self.display()
        print()
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
        Exam.razn_vector(x, iterator_curr)
        Exam.display()
        Rez.multi_mat_vec(self, iterator_curr)
        print('Ax = b in Jacobi:')
        Rez.display()
        print()
        print("Проверка точности")

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
        self.operation = 0
        self.matrix = [[0 for _ in range(self._length)] for _ in range(self._length)]
        for i in range(self._length):
            for j in range(self._length):
                self.matrix[i][i] = 1


class Vector:

    def __init__(self, n):
        self._length = n
        self.vector = []
        self.operation = 0

    def get_vector(self):
        self.vector = [randint(-1000, 1000) for _ in range(self._length)]

    def display(self):
        for i in range(self._length):
            print(self.vector[i], end=' ')
        print('')

    def multi_mat_vec(self, a, b):
        M = []
        count = 0
        for i in range(a._length):
            sum = 0
            for j in range(a._length):
                sum += a.matrix[i][j] * b.vector[j]
                count += 2
            M.append(sum)
        self.vector = M
        self.operation += count

    def multi_const(self, k):
        for i in range(self._length):
            self.vector[i] *= k

    def swap_vec_lem(self, a, b):
        temp = self.vector[a]
        self.vector[a] = self.vector[b]
        self.vector[b] = temp

    def norma(self):
        sum = 0
        count = 0
        for i in range(self._length):
            sum += self.vector[i]**2
            count += 2
        count += 11
        self.norm = math.sqrt(sum)
        self.operation += count

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
        self.operation = 0

def get_vector_f(X):
    F = Vector(10)
    F.vector = [
        math.cos(X.vector[1] * X.vector[0]) - math.exp(-3 * X.vector[2]) + X.vector[3] * X.vector[4] ** 2 - X.vector[5] - math.sinh(2 * X.vector[7]) * X.vector[8] + 2 * X.vector[9] + 2.000433974165385440,
        math.sin(X.vector[1] * X.vector[0]) + X.vector[2] * X.vector[8] * X.vector[6] - math.exp(-X.vector[9] + X.vector[5]) + 3 * X.vector[4] ** 2 - X.vector[5] * (X.vector[7] + 1) + 10.886272036407019994,
        X.vector[0] - X.vector[1] + X.vector[2] - X.vector[3] + X.vector[4] - X.vector[5] + X.vector[6] - X.vector[7] + X.vector[8] - X.vector[9] - 3.1361904761904761904,
        2 * math.cos(-X.vector[8] + X.vector[3]) + X.vector[4] / (X.vector[2] + X.vector[0]) - math.sin(X.vector[1] ** 2) + math.cos(X.vector[6] * X.vector[9]) ** 2 - X.vector[7] - 0.1707472705022304757,
        math.sin(X.vector[4]) + 2 * X.vector[7] * (X.vector[2] + X.vector[0]) - math.exp(-X.vector[6] * (-X.vector[9] + X.vector[5])) + 2 * math.cos(X.vector[1]) - 1.0 / (-X.vector[8] + X.vector[3]) - 0.3685896273101277862,
        math.exp(X.vector[0] - X.vector[3] - X.vector[8]) + X.vector[4] ** 2 / X.vector[7] + math.cos(3 * X.vector[9] * X.vector[1]) / 2 - X.vector[5] * X.vector[2] + 2.0491086016771875115,
        X.vector[1] ** 3 * X.vector[6] - math.sin(X.vector[9] / X.vector[4] + X.vector[7]) + (X.vector[0] - X.vector[5]) * math.cos(X.vector[3]) + X.vector[2] - 0.7380430076202798014,
        X.vector[4] * (X.vector[0] - 2 * X.vector[5]) ** 2 - 2 * math.sin(-X.vector[8] + X.vector[2]) + 0.15e1 * X.vector[3] - math.exp(X.vector[1] * X.vector[6] + X.vector[9]) + 3.5668321989693809040,
        7 / X.vector[5] + math.exp(X.vector[4] + X.vector[3]) - 2 * X.vector[1] * X.vector[7] * X.vector[9] * X.vector[6] + 3 * X.vector[8] - 3 * X.vector[0] - 8.4394734508383257499,
        X.vector[9] * X.vector[0] + X.vector[8] * X.vector[1] - X.vector[7] * X.vector[2] + math.sin(X.vector[3] + X.vector[4] + X.vector[5]) * X.vector[6] - 0.78238095238095238096
    ]
    return F

def get_Jacobi_matrix(X):
    J = Matrix(10)
    J.matrix = [
        [-X.vector[1] * math.sin(X.vector[1] * X.vector[0]),
         -X.vector[0] * math.sin(X.vector[1] * X.vector[0]),
         3 * math.exp(-3 * X.vector[2]),
         X.vector[4] ** 2, 2 * X.vector[3] * X.vector[4],
        -1, 0, -2 * math.cosh(2 * X.vector[7]) * X.vector[8],
         -math.sinh(2 * X.vector[7]), 2],
        [X.vector[1] * math.cos(X.vector[1] * X.vector[0]), X.vector[0] * math.cos(X.vector[1] * X.vector[0]),
         X.vector[8] * X.vector[6], 0, 6 * X.vector[4], -math.exp(-X.vector[9] + X.vector[5]) - X.vector[7] - 1,
         X.vector[2] * X.vector[8], -X.vector[5], X.vector[2] * X.vector[6], math.exp(-X.vector[9] + X.vector[5])],
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
        [-X.vector[4] / (X.vector[2] + X.vector[0]) ** 2, -2 * X.vector[1] * math.cos(X.vector[1] ** 2),
         -X.vector[4] / (X.vector[2] + X.vector[0]) ** 2, -2 * math.sin(-X.vector[8] + X.vector[3]),
        1.0 / (X.vector[2] + X.vector[0]), 0, -2 * math.cos(X.vector[6] * X.vector[9]) * X.vector[9] * math.sin(X.vector[6] * X.vector[9]), -1,
        2 * math.sin(-X.vector[8] + X.vector[3]), -2 * math.cos(X.vector[6] * X.vector[9]) * X.vector[6] * math.sin(X.vector[6] * X.vector[9])],
        [2 * X.vector[7], -2 * math.sin(X.vector[1]), 2 * X.vector[7], 1.0 / (-X.vector[8] + X.vector[3]) ** 2, math.cos(X.vector[4]),
        X.vector[6] * math.exp(-X.vector[6] * (-X.vector[9] + X.vector[5])), -(X.vector[9] - X.vector[5]) * math.exp(-X.vector[6] * (-X.vector[9] + X.vector[5])),
         2 * X.vector[2] + 2 * X.vector[0], -1.0 / (-X.vector[8] + X.vector[3]) ** 2, -X.vector[6] * math.exp(-X.vector[6] * (-X.vector[9] + X.vector[5]))],
        [math.exp(X.vector[0] - X.vector[3] - X.vector[8]), -1.5 * X.vector[9] * math.sin(3 * X.vector[9] * X.vector[1]),
         -X.vector[5], -math.exp(X.vector[0] - X.vector[3] - X.vector[8]), 2 * X.vector[4] / X.vector[7], -X.vector[2], 0,
         -X.vector[4] ** 2 / X.vector[7] ** 2, -math.exp(X.vector[0] - X.vector[3] - X.vector[8]), -1.5 * X.vector[1] * math.sin(3 * X.vector[9] * X.vector[1])],
        [math.cos(X.vector[3]), 3 * X.vector[1] ** 2 * X.vector[6], 1, -(X.vector[0] - X.vector[5]) * math.sin(X.vector[3]),
        X.vector[9] / X.vector[4] ** 2 * math.cos(X.vector[9] / X.vector[4] + X.vector[7]), -math.cos(X.vector[3]), X.vector[1] ** 3, -math.cos(X.vector[9] / X.vector[4] + X.vector[7]),
         0, -1.0 / X.vector[4] * math.cos(X.vector[9] / X.vector[4] + X.vector[7])],
        [2 * X.vector[4] * (X.vector[0] - 2 * X.vector[5]), -X.vector[6] * math.exp(X.vector[1] * X.vector[6] + X.vector[9]),
         -2 * math.cos(-X.vector[8] + X.vector[2]), 1.5, (X.vector[0] - 2 * X.vector[5]) ** 2, -4 * X.vector[4] * (X.vector[0] - 2 * X.vector[5]),
         -X.vector[1] * math.exp(X.vector[1] * X.vector[6] + X.vector[9]), 0, 2 * math.cos(-X.vector[8] + X.vector[2]), -math.exp(X.vector[1] * X.vector[6] + X.vector[9])],
        [-3, -2 * X.vector[7] * X.vector[9] * X.vector[6], 0, math.exp(X.vector[4] + X.vector[3]), math.exp(X.vector[4] + X.vector[3]),
        -7.0 / X.vector[5] ** 2, -2 * X.vector[1] * X.vector[7] * X.vector[9], -2 * X.vector[1] * X.vector[9] * X.vector[6], 3, -2 * X.vector[1] * X.vector[7] * X.vector[6]],
        [X.vector[9], X.vector[8], -X.vector[7], math.cos(X.vector[3] + X.vector[4] + X.vector[5]) * X.vector[6], math.cos(X.vector[3] + X.vector[4] + X.vector[5]) * X.vector[6],
        math.cos(X.vector[3] + X.vector[4] + X.vector[5]) * X.vector[6], math.sin(X.vector[3] + X.vector[4] + X.vector[5]), -X.vector[2], X.vector[1], X.vector[0]]
    ]
    return J

def Newton(X):
    F = Vector(10)
    J = Matrix(10)
    x_next = Help_Vector(10)
    x_curr = copy.deepcopy(X)
    delta_x = Help_Vector(10)
    count = 0
    operation = 0
    delta_x.razn_vector(x_next, x_curr)
    delta_x.norma()
    q = delta_x.norm
    operation += delta_x.operation
    start_time = datetime.now()
    while(q > eps):
        F = get_vector_f(x_curr)
        F.multi_const(-1)
        operation += 100
        J = get_Jacobi_matrix(x_curr)
        x_next = J.uravn(F)
        x_next.norma()
        operation += x_next.operation
        x_curr.sum_vector(x_next, x_curr)
        operation += 10
        q = x_next.norm
        count += 1
    time = datetime.now() - start_time
    arr = []
    arr.append(count)
    arr.append(operation)
    arr.append(time)
    arr.append(x_curr)
    return arr

def Modified_Newton(X):
    F = Vector(10)
    J = Matrix(10)
    F1 = Vector(10)
    F1 = get_vector_f(X)
    F2 = Help_Vector(10)
    count = 0
    operation = 0
    delta_F = Help_Vector(10)
    delta_F.razn_vector(F2,F1)
    delta_F.norma()
    operation += delta_F.operation
    q1 = delta_F.norm
    x_next = Help_Vector(10)
    x_curr = copy.deepcopy(X)
    J = get_Jacobi_matrix(x_curr)
    J.norma()
    operation += J.operation
    alpha = eps*J.norm
    delta_x = Help_Vector(10)
    delta_x.razn_vector(x_next, x_curr)
    delta_x.norma()
    q = delta_x.norm
    operation += delta_x.operation
    start_time = datetime.now()
    while(q > eps and q1 > alpha):
        F = get_vector_f(x_curr)
        F.multi_const(-1)
        operation += 100
        x_next = J.uravn(F)
        x_next.norma()
        operation += x_next.operation
        F.multi_const(-1)
        operation += 100
        x_curr.sum_vector(x_next, x_curr)
        operation += 10
        F3 = Vector(10)
        F3 = get_vector_f(x_curr)
        F3.razn_vector(F3, F)
        operation += 10
        F3.norma()
        operation += F3.operation
        q1 = F3.norm
        q = x_next.norm
        count += 1
    time = datetime.now() - start_time
    arr = []
    arr.append(count)
    arr.append(operation)
    arr.append(time)
    arr.append(x_curr)
    return arr

def Auto_Newton(X):
    F = Vector(10)
    J = Matrix(10)
    x_next = Help_Vector(10)
    x_curr = copy.deepcopy(X)
    delta_x = Help_Vector(10)
    count = 0
    operation = 0
    temp = 0
    delta_x.razn_vector(x_next, x_curr)
    delta_x.norma()
    q = delta_x.norm
    operation += delta_x.operation
    start_time = datetime.now()
    while (q > eps and temp <= n):
        F = get_vector_f(x_curr)
        F.multi_const(-1)
        operation += 100
        J = get_Jacobi_matrix(x_curr)
        x_next = J.uravn(F)
        x_next.norma()
        operation += x_next.operation
        x_curr.sum_vector(x_next, x_curr)
        operation += 10
        q = x_next.norm
        count += 1
        temp += 1
    Arr = Modified_Newton(x_curr)
    time = datetime.now() - start_time
    count += Arr[0]
    operation += Arr[1]
    arr = []
    arr.append(count)
    arr.append(operation)
    arr.append(time)
    arr.append(Arr[3])
    return arr

def Auto_Newton_vol_2(X):
    F = Vector(10)
    J = Matrix(10)
    J1 = Matrix(10)
    J1 = get_Jacobi_matrix(X)
    J2 = Help_Matrix(10)
    count = 0
    operation = 0
    delta_J = Help_Matrix(10)
    delta_J.razn_matrix(J2, J1)
    operation += 100
    delta_J.norma()
    operation += delta_J.operation
    q1 = delta_J.norm
    x_next = Help_Vector(10)
    x_curr = copy.deepcopy(X)
    delta_x = Help_Vector(10)
    delta_x.razn_vector(x_next, x_curr)
    delta_x.norma()
    q = delta_x.norm
    operation += delta_x.operation
    start_time = datetime.now()
    while (q > eps):
        if(q1 < delta):
            Arr = Modified_Newton(x_curr)
            time = datetime.now() - start_time
            count += Arr[0]
            operation += Arr[1]
            arr = []
            arr.append(count)
            arr.append(operation)
            arr.append(time)
            arr.append(Arr[3])
            return arr
        F = get_vector_f(x_curr)
        F.multi_const(-1)
        operation += 100
        J = get_Jacobi_matrix(x_curr)
        x_next = J.uravn(F)
        x_next.norma()
        operation += x_next.operation
        x_curr.sum_vector(x_next, x_curr)
        J2 = get_Jacobi_matrix(x_curr)
        J2.razn_matrix(J2,J)
        J2.norma()
        operation = operation + 110 + J2.operation
        q1 = J2.norm
        q = x_next.norm
        count += 1
    time = datetime.now() - start_time
    arr = []
    arr.append(count)
    arr.append(operation)
    arr.append(time)
    arr.append(x_curr)
    return arr

def Hybrid_Newton(X):
    F = Vector(10)
    J = Matrix(10)
    F1 = Vector(10)
    F1 = get_vector_f(X)
    F2 = Help_Vector(10)
    X1 = []
    count = 0
    operation = 0
    delta_F = Help_Vector(10)
    delta_F.razn_vector(F2,F1)
    delta_F.norma()
    operation += delta_F.operation
    q1 = delta_F.norm
    x_next = Help_Vector(10)
    x_curr = copy.deepcopy(X)
    alpha = 0
    delta_x = Help_Vector(10)
    delta_x.razn_vector(x_next, x_curr)
    delta_x.norma()
    q = delta_x.norm
    operation += delta_x.operation
    start_time = datetime.now()
    temp = 0
    while(q > eps and q1 > alpha):
        if(temp%n == 0):
            J = get_Jacobi_matrix(x_curr)
            J.norma()
            operation += J.operation
            alpha = eps * J.norm
        F = get_vector_f(x_curr)
        F.multi_const(-1)
        operation += 100
        x_next = J.uravn(F)
        x_next.norma()
        operation += x_next.operation
        F.multi_const(-1)
        operation += 100
        x_curr.sum_vector(x_next, x_curr)
        operation += 10
        F3 = Vector(10)
        F3 = get_vector_f(x_curr)
        F3.razn_vector(F3, F)
        operation += 10
        F3.norma()
        operation += F3.operation
        q1 = F3.norm
        q = x_next.norm
        X1.append(x_curr)
        temp += 1
        count += 1
    time = datetime.now() - start_time
    arr = []
    arr.append(count)
    arr.append(operation)
    arr.append(time)
    arr.append(x_curr)
    arr.append(len(X1))
    return arr

X = Vector(10)
X.vector = [0.5, 0.5, 1.5, -1, -0.2, 1.5, 0.5, -0.5, 1.5, -1.5]
print("==================================================================================")
print("Полный метод Ньютона")
k1 = Newton(X)
print("Решение:")
k1[3].display()
print("Затраченное время:")
print(k1[2])
print("Количество операций:")
print(k1[1])
print(f'Количество итераций: {k1[0]}')
print("==================================================================================")
print("Модифицированный метод Ньютона")
k2 = Modified_Newton(X)
print("Решение:")
k2[3].display()
print("Затраченное время:")
print(k2[2])
print("Количество операций:")
print(k2[1])
print(f'Количество итераций: {k2[0]}')
print("==================================================================================")
print("Автоматический метод Ньютона (пользователь)")
if (k1[0] <= n):
    print("Полный метод Ньютона из первого пункта!")
else:
    k3 = Auto_Newton(X)
    print("Решение:")
    k3[3].display()
    print("Затраченное время:")
    print(k3[2])
    print("Количество операций:")
    print(k3[1])
    print(f'Количество итераций: {k3[0]}')
print("==================================================================================")
print("Автоматический метод Ньютона")
k4 = Auto_Newton_vol_2(X)
print("Решение:")
k4[3].display()
print("Затраченное время:")
print(k4[2])
print("Количество операций:")
print(k4[1])
print(f'Количество итераций: {k4[0]}')
print("==================================================================================")
print("Гибридный метод Ньютона (пользователь)")
k5 = Hybrid_Newton(X)
print("Решение:")
k5[3].display()
print("Затраченное время:")
print(k5[2])
print("Количество операций:")
print(k5[1])
print(f'Количество итераций: {k5[0]}')
print(f'Количество корней: {k5[4]}')


















# Buff1 = copy.deepcopy(A_diag)
# L1 = Help_Matrix(n)
# U1 = Help_Matrix(n)
# Q1 = Help_Matrix(n)
# P1 = Help_Matrix(n)
# y = Help_Vector(n)
# x = Help_Vector(n)
# z = Help_Vector(n)
# b = Help_Vector(n)
# Buff1.LU_matrix(U1, L1, P1, Q1)
# b.multi_mat_vec(P1, B)
# for i in range(n):
#     sum = 0
#     for j in range(i):
#         sum += y.vector[j]*L1.matrix[i][j]
#     y.vector[i] = b.vector[i] - sum
# for i in reversed(range(n)):
#     sum = 0
#     for j in range(i+1, n):
#         sum += z.vector[j]*U1.matrix[i][j]
#     z.vector[i] = (y.vector[i] - sum)/U1.matrix[i][i]
# x.multi_mat_vec(Q1, z)
# # A_T = Matrix(n)
# # A_T = copy.deepcopy(A)
# # A_T.transpose()
# # A_cv_form = Matrix(n)
# # A_cv_form.multi(A, A_T)
# Buf = Matrix(n)
# Buf = copy.deepcopy(A_diag)
# L = Help_Matrix(n)
# L.matrix = [[0 for _ in range(n)] for _ in range(n)]
# R = Help_Matrix(n)
# R = copy.deepcopy(L)
# D = Help_Matrix(n)
# D = copy.deepcopy(L)
# Buf.display()
# A_diag.display()
# Buf.decompose(L, D, R)
#
# D_inv = D.Obr_matrix()
# Bmatrix = Help_Matrix(n)
# Bmatrix.sum_matrix(L,R)
# D.display()
# D_inv.display()
# Bmatrix.display()
# Bmatrix.multi(D_inv, Bmatrix)
# Bmatrix.display()
# Bmatrix.multi_const(-1)
# Bmatrix.display()
# C = Help_Vector(n)
# C.multi_mat_vec(D_inv, B)
# Bmatrix.norma()
# C.norma()
# if Bmatrix.norm < 1:
#     q = Bmatrix.norm
# if Bmatrix.norm > 1:
#     q = 1e-1
# print('q:')
# print(q)
# k = ceil(log((eps/ C.norm * (1 - q)), q))
# print('k:')
# print(k)
# print('Jacobi:')
# A_diag.display()
# c = copy.deepcopy(A_diag)
# count_jac = c.Jacobi(Bmatrix, C, q, x)
# print(count_jac)
# print('Seidel:')
# count_sei = c.Seidel(Bmatrix, C, q, x, A)
# print(count_sei)

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
