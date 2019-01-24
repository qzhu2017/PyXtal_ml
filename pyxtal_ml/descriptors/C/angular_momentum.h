#ifndef ANGULAR_MOMENTUM_H
#define ANGULAR_MOMENTUM_H

double CG(double j1, double m1, double j2, double m2, double j3, double m3, long long int *factorial);

double complex wigner_D(double alpha, double beta, double gamma, double j, double m, double mp, long long int *factorial);

#endif
