import numpy as np
import time

def f_function(x, y):
    return ((-5*(np.pi**2)/4)*np.sin(np.pi*x)*np.cos(np.pi*y/2))

def u_function(x, y):
    return (np.sin(np.pi*x)*np.cos(np.pi*y/2))

def gS_function(x):
    return (np.sin(np.pi*x))

def algorithm_approx(m, tol, itmax):
    a = 0
    b = 1

    h = (b - a)/m
    lbd = 1
    mu = 2*(1 + lbd)
    
    # Como a = c e h = k, x = y.
    x = np.zeros((m + 1))
    for i in range(0, m, 1):
        x[i] = a + i*h

    u = np.zeros((m + 1, m + 1))
    f = np.zeros((m + 1, m + 1))
    for i in range(0, m, 1):
        for j in range(0, m, 1):
            u[i][j] = 0
            f[i][j] = f_function(x[i], x[j])

    for i in range(0, m, 1):
        u[i][0] = gS_function(x[i])
        u[i][m] = 0

    for j in range(1, m - 1, 1):
        u[0][j] = 0

    for j in range(1, m - 1, 1):
        u[m][j] = (4*u[m-1][j] - u[m-2][j])/3

    norm = 100
    z = np.linalg.norm(u)
    l = 0

    while (l <= itmax):
        for i in range(1, m - 1, 1):
            for j in range(1, m - 1, 1):
                u[i][j] = (u[i+1][j] + u[i-1][j] + lbd*(u[i][j+1] + u[i][j-1]) - h*h*f[i][j])/(mu)

        for j in range(1, m - 1, 1):
            u[m][j] = (4*u[m-1][j] - u[m-2][j])/3
        
        z = np.linalg.norm(u)
        
        if(abs(z - norm) <= tol):
            return u, abs(z - norm)
        else:
            norm = z

        l += 1

    return u, "Numero maximo de iteracoes atingido."

def main():
    m = int(input("Digite o numero de intervalos na direcao x e y: "))
    tolerancia = float(input("Digite um valor para a tolerancia: "))
    itmax = int(input("Digite o numero maximo de iteracoes: "))

    inicio = time.time()
    u_approx, erro_norma = algorithm_approx(m, tolerancia, itmax)
    fim = time.time()

    delta_t = fim - inicio

    print("Tempo do metodo aproximado: " + str(delta_t))
    print("Erro da norma da aproximacao: " + str(erro_norma))

if __name__ == "__main__":
    main()