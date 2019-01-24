#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "angular_momentum.h"
#include "spherical_harmonics.h"
#include "array_ops.h"
#include "bispectrum.h"

/****************************************************************
FUNCTION NAME : Bispectrum
RETURN TYPE : complex
PARAMETERS :  double j1: free integer/half-integer parameter
	      double j2: free integer/half-integer parameter
	      int j: free integer parameter
	      double *site: a pointer to the first element of an array of 3 elements
	                    corresponding to the x, y, and z coordinates of the central
			    atom of the bispectrum calculation
	      
	      double *neighbors: a pointer to the first element of an array of 3*n elements
	                         corresponding to the cartesian coordinates of each neighbor
				 to the central atom up to a cutoff radius, this should be formatted
				 as a 1-d array where every 3 elements corresponds to an atom
				 eg: {x1, y1, z1, x2, y2, z2 ,..., xn, yn, zn}

	      double *G: a pointer to the first element of an array of n elements corresponding to
	      		 the values of the radial basis function for each neighbor site

	      double Rc: the cutoff radius of calculation

	      long int *factorial: a pointer to the first element of an array of precomputed factorial values


DESCRIPTION : this function computes the bispectrum coefficeints of an atomic site given its neighbors up to a
	      cutoff distance Rc
****************************************************************/
long double complex Bispectrum(double j1, double j2, int j, double *site, int site_size, double *neighbors, int neighbors_size, double *G, int G_size, double Rc, long long int *factorial){

	/* Input error statements */
	if (site_size != 3){
		printf("Site corresponds x, y, and z coordinates of site");
		exit(0);
	}


	else if ((neighbors_size % 3) != 0){
		printf("Site neighbors length not a multiple of 3");
		exit(0);
	}


	else if((int)(neighbors_size/3) != G_size){
		printf("Radial basis values not the same length as neighbors");
		exit(0);
	}

	else{}


	long double complex B = 0 + 0*I;  // bispectrum coefficient
	long double complex C, C1, C2, CG1, CG2;  // c coefficients and clebsch gordon coefficients
	double m1, m_prime1, m1bound, m_prime1_bound;  // integer/half-integer parameters
	int m, m_prime;  // integer parameters

	// sum from -j : j
	for(m=-j; m<(j+1); ++m){

		/*sum from -j to j */
		
		for(m_prime=-j; m_prime<(j+1); ++m_prime){
			m1bound = Min_ab(j1, ((double) m)+j2);
			m1 = Max_ab(-j1, ((double) m)-j2);

			/* 4-D hyperspherical harmonics of neighbor-site pairs 
			 * multiplied by cutoff function
			 * value and radial basis value */
			C = c((double) j, (double) m_prime, (double) m, site, site_size, neighbors, neighbors_size, G, G_size, Rc, factorial);

			while(m1 < (m1bound + 0.5) ){
	
				m_prime1_bound = Min_ab(j1, ((double) m_prime)+j2); 
				m_prime1 = Max_ab(-j1, ((double) m_prime) - j2);

				while(m_prime1 < (m_prime1_bound + 0.5)){
					// 4-D hyper-spherical harmonics
					C1 = c(j1, (double) m_prime1, (double) m1, site, site_size, neighbors, neighbors_size, G, G_size, Rc, factorial);
					C2 = c(j2, (double) m_prime-m_prime1, (double) m-m1, site, site_size, neighbors, neighbors_size, G, G_size, Rc, factorial);
					// Clebsch-Gordon Coefficients
					CG1 = CG(j1, (double) m1, j2, (double) m-m1, (double) j, (double) m, factorial);
					CG2 = CG(j1, (double) m_prime1, j2, (double) m_prime-m_prime1, (double) j, (double) m_prime, factorial);
					B += CG1 * CG2 * conj(C) * C1 * C2;
					m_prime1 += 1.0;
				}
				m1 += 1.0;

			}
		}
	}

	return B;
}


/****************************************************************
FUNCTION NAME : site_bispectrum
RETURN TYPE : void
PARAMETERS :  
	      double *B: a pointer to the target B coefficents 1*M array

	      int dim_B: the length of B array

	      int j_max: the maximum interger number of j angular momentum (int)

	      double *site: a pointer to a 1*3 array corresponding 
	                    to the x, y, z of central atom 
	      
	      double *N: a pointer to 1*3N array corresponding 
	                    to the neighboring atoms

	      int dim_N: the length of N array

	      double *G: a pointer to the array of elemental index of neighbor atoms
	      		 1*N array

	      double Rc: the cutoff radius of calculation

DESCRIPTION : this function computes the site bispectrum coefficeints (B)
              of an atomic site given its neighbors
****************************************************************/

void site_bispectrum(double *B, int dim_B, int j_max, double *site, double *N, int dim_N, double *G, int dim_G, double cutoff, long long int *Factorial){
	
	double complex out;
	int min_index;
	double j1, j2;

	int count = 0;
	for(int j12=0; j12<=2*j_max; ++j12){
		min_index = Min_ab(j12, (double)j_max);
		j1 = (double) j12/2;
		j2 = (double) j12/2;

		for(int j=0; j<=min_index; ++j){
			out = Bispectrum(j1, j2, j, site, 3, N, dim_N, G, dim_G, cutoff, Factorial);
			B[count] = creal(out);
			count += 1;
		}
	}

}

/****************************************************************
FUNCTION NAME calc_B_length
RETURN TYPE : the dimension of bispectrum coefficient 
PARAMETERS :  int j_max: the maximum interger number of j angular momentum 
******************************************************************/

int calc_B_length(int j_max){

	int min_index;
	int count = 0;
	for(int j12=0; j12<=2*j_max; ++j12){
		min_index = Min_ab(j12, (double)j_max);
		for(int j=0; j<=min_index; ++j){
			count += 1;
		}
	}

	return count;
}
