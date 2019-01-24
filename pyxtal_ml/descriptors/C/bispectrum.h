#ifndef BISPECTRUM_H
#define BISPECTRUM_H

long double complex Bispectrum(double j1, double j2, int j, double *site, int site_size, double *neighbors, int neighbors_size, double *G, int G_size, double Rc, long long int *factorial);

void site_bispectrum(double *B, int dim_B, int j_max, double *site, double *N, int dim_N, double *G, int dim_G, double cutoff, long long int *Factorial);

int calc_B_length(int j_max);
#endif
