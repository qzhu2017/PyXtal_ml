#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "angular_momentum.h"
#include "cutoffs.h"
#include "spherical_harmonics.h"

/****************************************************************
FUNCTION NAME : U
RETURN TYPE : complex
PARAMETERS : double j: integer/half-integer parameter (QZ: must be integer?)
	     double m: integer/half-integer parameter
	     double m_prime: integer/half-integer parameter
	     psi, theta, phi: coordinate angles
	     long int *factorial: a pointer to the first element of an array of precomputed factorial values
DESCRIPTION : this function computes the 4-d hyperspherical harmonic
****************************************************************/
double complex U(double j, double m, double m_prime, double psi, double theta, double phi, long long int *factorial){

	double complex sph_harm=0;
	double mp = -j;
	///* sum from -j to j of the product of wignerD values and complex exponentials */
	while (mp <= j) {
		//double complex tmp = -mp*psi*I;
		sph_harm += (wigner_D(phi, theta, -phi, j, m, mp, factorial) * cexp(-mp*psi*I) *
				wigner_D(phi, -theta, -phi, j, mp, m_prime, factorial));
		mp += 1.0;
	}

	return sph_harm;
}

/****************************************************************
FUNCTION NAME : c
RETURN TYPE : complex
PARAMETERS : double j:  free integer/half-integer parameter
	     double m_prime: integer/half-integer parameter
	     double m: integer/half-integer parameter
	     double *site: pointer to first element of array of cartesian coordinates
	     of central atom
	     
	     double *neighbors: pointer to first element of array
	     of neighbors cartesian coordinates length 3*n

	     double *G: pointer to first element of array of radial basis values
	     for each neighbor length n

	     double Rc: cutoff radius

	     long int *factorial: pointer to first element of array of precomputed
	     factorial values

DESCRIPTION : sum of hyperspherical harmonics multiplied by cutoff function value and
radial basis value for each neighbor site pair
****************************************************************/
long double complex c(double j, double m_prime, double m, double *site, int site_size, double *neighbors, int neighbors_size, double *G, int G_size, double Rc, long long int *factorial){
	/* input error handling */
	if (site_size != 3){
		printf("Site corresponds to x, y, and z coordinates");
		exit(0);
	}


	else if ((neighbors_size % 3) != 0){
		printf("Site neighbors length not a multiple of 3");
		exit(0);
	}


	else if ((int)(neighbors_size/3) != G_size){
		printf("Radial basis values not the same length as neighbors");
		exit(0);
	}

	else{}

	/* initialize cartesian values, and the hyperspherical radius and angles along with the radial basis value*/
	double x, y, z, r, Gval, psi, theta, phi;
	/* initialize the result*/
	double dot = 0;
	/* initialize pointers to the neighbor list and radial basis values
	 * we increment the neighbor list by "z values" ie. every third value
	 * until the end of the array (ntimes)
	 * we incriment the g values n times by every value*/

	for (int i=0; i<(G_size); ++i){
			

		// position vector by dereferencing the pointers
		x = neighbors[3*i] - site[0];  // x component is the second address in memory behind i
		y = neighbors[3*i+1] - site[1];  // y component is the first address in memory behind i
		z = neighbors[3*i+2] - site[2];  // z component is the address at i
		// magnitude of position vector
		r = sqrt( pow(x,2) + pow(y,2) + pow(z,2) );
		// dereference g to get the radial basis value
		Gval = G[i];

		// compute angles from position vector and magnitude
		if (r > pow(10.0, -10)){
			// psi computation
	       		psi = asin(r/Rc);


			// theta computation
			if (fabs(z/r - 1.0) < pow(10.0, -8)){
				theta = 0.0;
			}

			else if (fabs(z/r + 1.0) < pow(10.0, -8)){
				theta = M_PI;
			}

			else{
				theta = acos(z/r);
			}



			// phi computation
			if (x < 0.0){
				phi = M_PI + atan(y/x);
			}
		
			else if (0.0 < x && y < 0.0){
				phi = 2.0*M_PI + atan(y/x);
			}

			else if (0.0 < x && 0.0 <= y){
				phi = atan(y/x);
			}

			else if (x == 0.0 && 0.0 < y){
				phi = 0.5 * M_PI;
			}

			else if (x == 0.0 && y < 0.0){
				phi = 1.5 * M_PI;
			}

			else{
				phi = 0.0;
			}

			/*sum the conjugate of the 4-D spherical harmonic multiplied by the radial basis value
			 * and the cutoff function value */
			dot += Gval * conj(U(j, m, m_prime, psi, theta, phi, factorial)) * cosine_cutoff(r, Rc);
		}

		else{
			continue;
		}
	}
	return dot;
}	
