from math import *
def fact(n):
    factorial = 1
    while n > 1:
        factorial *= n
        n -= 1
    return factorial
def Geron(eps_u, a):
    n_k = 1 + (a-1)/2 - ((a-1)**2)/8
    while abs(a - n_k*n_k) > eps_u:
       n_k = 0.5*(n_k + a/n_k)
    return abs(n_k)
def v_func(n):
    return (1+n)/(0.6*n)
def ch_Teylor(i, eps_w):
    n = 0
    Sum = 0
    while abs((i**(2*n))/fact(2*n)) > eps_w:
        Sum += (i ** (2 * n)) / fact(2 * n)
        n += 1
    return Sum
def sin_Teylor(i, eps_y):
    n = 0
    Sum = 0
    while abs(((-1)**n)*(i ** (2 * n + 1)) / fact(2 * n + 1)) > eps_y:
        Sum += ((-1) ** n) * (i ** (2 * n + 1)) / fact(2 * n + 1)
        n += 1
    return Sum
s = []
eps = 10**(-6)
arr_U = []
c_u = 0.4
eps_u = eps/3*c_u
arr_V = []
c_v = 0.68
arr_W = []
c_w = 0.79
eps_w = eps/3*c_w
arr_Y = []
c_y = 1.119
eps_y = eps/3*c_y
N = 0.2
arr_Z = []
real_Z = []
while N < 0.31:
    s.append(N)
    N += 0.01
for i in s:
    k = i*i + 0.3
    U = Geron(eps_u, k)/(1+i)
    arr_U.append(U)
for i in s:
    V = v_func(i)
    arr_V.append(V)
for i in arr_U:
    W = ch_Teylor(i, eps_w)
    arr_W.append(W)
for i in arr_V:
    Y = sin_Teylor(i, eps_y)
    arr_Y.append(Y)
for i in range(0, len(s)):
    Z = arr_W[i]*arr_Y[i]
    arr_Z.append(Z)
for x in s:
    zx = cosh(((x**2 + 0.3)**0.5)/(1+x))*sin((1+x)/(0.6*x))
    real_Z.append(zx)
Delta = []
for i in range(0, len(s)):
    delta = abs(real_Z[i]-arr_Z[i])
    Delta.append(delta)
print("x | z(x) | ~z(x) | delta_z")
for i in range(0, len(s)):
    print(f'{s[i]} |  {real_Z[i]} | {arr_Z[i]} | {Delta[i]}')
for i in Delta:
    print(f'{i < eps} + {i > 10**(-10)}')
