#ifndef SPHERICAL_HARMONICS_H
#define SPHERICAL_HARMONICS_H

double complex U(double j, double m, double m_prime, double psi, double theta, double phi, long long int *factorial);

long double complex c(double j, double m_prime, double m, double *site, int site_size, double *neighbors, int neighbors_size, double *G, int G_size, double Rc, long long int *factorial);

#endif
