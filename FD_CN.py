import numpy as np
from matplotlib import pyplot as plt

def u_function(x, t):
    return (np.e**(-t)*np.sin(np.pi*x)*np.cos(np.pi*x/2))
    
def f_function(x):
    return  (np.sin(np.pi*x)*np.cos(np.pi*x/2))

def g_function(x, t):
    return ((1/4)*(np.pi**2)*(np.e**(-t))*((4*np.sin(np.pi*x/2)*(np.cos(np.pi*x))) + 5*np.sin(np.pi*x)*(np.cos(np.pi*x/2)) - np.e**(-t)*(np.sin(np.pi*x)*(np.cos(np.pi*x/2)))))

def foward_diff_exact():
    jmax = 1000
    alpha = 1
    m = 100
    k = 1/20100
    T = jmax*k
    h = 1/m
    lbd = ((alpha**2)*k)/(h**2)
    T1 = 0
    T2 = 0

    w = np.zeros([m, jmax])
    
    for i in range(0, m-1):
        w[i][0] = u_function(i, i)
    
    for j in range(0, jmax-1):
        w[0][j] = T1
        w[m-1][j] = T2

        for i in range(1, m-1):
            w[i][j+1] = (1-2*lbd)*w[i][j] + lbd*(w[i-1][j] + w[i+1][j]) + k*g_function(i, i)
            
    return w


def foward_diff_approx():
    jmax = 1000
    alpha = 1
    m = 100
    k = 1/20100
    T = jmax*k
    h = 1/m
    lbd = ((alpha**2)*k)/(h**2)
    T1 = 0
    T2 = 0

    w = np.zeros([m, jmax])

    for i in range(0, m-1):
        w[i][0] = f_function(i)

    for j in range(0, jmax-1):
        w[0][j] = T1
        w[m-1][j] = T2

        for i in range(1, m-1):
            w[i][j+1] = (1-2*lbd)*w[i][j] + lbd*(w[i-1][j] + w[i+1][j]) + k*g_function(i, i)

    return w

def main():
    A = foward_diff_exact()
    B = foward_diff_approx()
    C = A - B

    plt.matshow(A)
    plt.colorbar()
    plt.show()

    plt.matshow(B)
    plt.colorbar()
    plt.show()

    plt.matshow(C)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()
