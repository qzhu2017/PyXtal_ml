#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "factorial.h"
#include "bispectrum.h"
#include "array_ops.h"

void Bis(int j_max, double cutoff, int dim_G, double *site, double *N, double *G, double *B)
{

	//define the dynamical arrays F, B, N, G
	int dim_F = 3 * (j_max + 1);
	int dim_B = calc_B_length(j_max);
	int dim_N = 3*dim_G;
	long long int *F = malloc(dim_F * sizeof(long long int));

	factorial_values(F, dim_F);
	site_bispectrum(B, dim_B, j_max, site, N, dim_N, G, dim_G, cutoff, F);
}


int main(int argc, char *argv[])
{
	FILE *fp;
	int j_max = atoi(argv[1]);
	printf("Calculating the site bispectrum coefficients: j_max=%d\n", j_max);
	double cutoff=6.5;
	double site[3];

	//define the dynamical arrays F, B, N, G
	int dim_F = 3 * (j_max + 1);
	int dim_B = calc_B_length(j_max);
	int dim_G = 131;
	int dim_N = 3*dim_G;

	long long int *F = malloc(dim_F * sizeof(long long int));
	double *B = malloc(dim_B * sizeof(double));
	double *N = malloc(dim_N * sizeof(double));
	double *G = malloc(dim_G * sizeof(double));

	factorial_values(F, dim_F);
	//disp_1d_array_int(F, dim_F);

	fp = fopen("../data/site_0.txt", "r");
	for(int i=0; i<3; ++i){
		fscanf(fp, "%lf", &site[i]);
	}
	fclose(fp);

	fp = fopen("../data/neighbors_0.txt", "r");
	for(int i=0; i<dim_N; ++i){
		fscanf(fp, "%lf", &N[i]);
	}
	fclose(fp);

	fp = fopen("../data/basis_0.txt", "r");
	for(int i=0; i<dim_G; ++i){
		fscanf(fp, "%lf", &G[i]);
	}
	fclose(fp);
	/* QZ: for the next step of Python API,
	 * we should just call this library as follows 
	 * j_max, site, neighbors, G, cutoff should be passed from the python code */
	site_bispectrum(B, dim_B, j_max, site, N, dim_N, G, dim_G, cutoff, F);
	printf("Returned: (%d)\n", dim_B);
	disp_1d_array(B, dim_B);
	
	//release the memory
	free(F);
	free(B);
	free(N);
	free(G);
}


