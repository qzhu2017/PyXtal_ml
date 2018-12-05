import numpy as np
from math import factorial
from optparse import OptionParser


def CG(j1, m1, j2, m2, j3, m3):
    '''
    Calculates the Clebsch-Gordon coefficient
    for coupling (j1,m1) and (j2,m2) to give (j3,m3).

    Args:
    j1: float
        Total angular momentum 1.

    j2: float
        Total angular momentum 2.

    j3: float
        Total angular momentum 3.

    m1: float
        z-component of angular momentum 1.

    m2: float
        z-component of angular momentum 2.

    m3: float
        z-component of angular momentum 3.

    Returns
    cg_coeff : float
        Requested Clebsch-Gordan coefficient.
    '''
    if m3 != m1 + m2:
        return 0
    vmin = int(np.max([-j1 + j2 + m3, -j1 + m1, 0]))
    vmax = int(np.min([j2 + j3 + m1, j3 - j1 + j2, j3 + m3]))

    C = np.sqrt((2.0 * j3 + 1.0) * factorial(j3 + j1 - j2) *
                factorial(j3 - j1 + j2) * factorial(j1 + j2 - j3) *
                factorial(j3 + m3) * factorial(j3 - m3) /
                (factorial(j1 + j2 + j3 + 1) *
                 factorial(j1 - m1) * factorial(j1 + m1) *
                 factorial(j2 - m2) * factorial(j2 + m2)))
    S = 0
    for v in range(vmin, vmax + 1):
        S += (-1.0) ** (v + j2 + m2) / factorial(v) * \
            factorial(j2 + j3 + m1 - v) * factorial(j1 - m1 + v) / \
            factorial(j3 - j1 + j2 - v) / factorial(j3 + m3 - v) / \
            factorial(v + j1 - j2 - m3)
    C = C * S
    return C


def wigner_d(beta, J, M, MP):
    '''
    Small Wigner d function
    Ref:  Quantum theory of angular momentum D.A. Varshalovich 1988
    Args:
        beta: float
              Second euler angle of rotation

        J: int
           Total Angular Momentum

        M: int
            Eigenvalue of angular momentum along axis after rotation

        MP: int
            Eigenvalue of angular momentum along rotated axis

    Returns:
        d the wigner_D matrix element for a defined rotation
        '''
    constant = -1**(J-MP)*np.sqrt(factorial(J+M)*factorial(J-M)
                                  * factorial(J+MP)*factorial(J-MP))
    n_max = int(np.min([J-M, J-MP]))
    d = 0
    for k in range(n_max+1):
        d += (-1**(k) * np.cos(beta/2)**(M+MP+2*k) * np.sin(beta/2)**(2*J-M-MP-2*k) /
              factorial(k) / factorial(J-M-k) / factorial(J-MP-k) / factorial(M+MP+k))

    d *= constant

    return d


def wigner_D(alpha, beta, gamma, J, M, MP):
    '''
    Large Wigner D function
    Ref:  Quantum theory of angular momentum D.A. Varshalovich 1988
    Args:
        alpha: float
               First euler angle of rotation

        beta: float
              Second euler angle of rotation

        gamma: float
               Thirs euler angle of rotation

        J: int
           Total Angular Momentum

        M: int
            Eigenvalue of angular momentum along axis after rotation

        MP: int
            Eigenvalue of angular momentum along rotated axis

    Returns:
        the wigner_D matrix element for a defined rotation
        '''
    return np.exp(-M*alpha*1j)*wigner_d(beta, M, MP, J)*np.exp(-MP*gamma*1j)
