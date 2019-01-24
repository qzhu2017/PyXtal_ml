#include <math.h>
#include <complex.h>
#include <stdio.h>
#include "binomial.h"
#include "array_ops.h"
#include "angular_momentum.h"


/****************************************************************
FUNCTION NAME : CG
RETURN TYPE : double
PARAMETERS : double j1: integer/half integer parameter - total angular momentum 1
	     double j2: integer/half integer parameter - total angular momentum 2
	     double j3: integer/half integer parameter - resultant total angular momentum 
	     double m1: integer/half integer parameter - z angular momentum 1
	     double m2: integer/half integer parameter - z angular momentum 2
	     double m3: integer/half integer parameter - resultant z angular momentum 
	     long int *factorial:  a pointer to the first element of an array of precomputed factorial values
DESCRIPTION : 
	Computes the Clebsch Gordon Coefficient of the coupling of two angular momenta j1,m1 and j2,m2 to
	a coupled state j3, m3.  This coefficient represents the probability amplitude that two angular
	momentum states are coupled into a resultant angular momentum state.
****************************************************************/
double CG(double j1, double m1, double j2, double m2, double j3, double m3, long long int *factorial){

	if(m1 + m2 != m3)
		return 0;

	else{
		/* define an array and of sums and find that minimal element to
		 * to determine whether to return zero or perform more calculations*/
		double minimum_array[] = {j1+j2-j3, j1-j2+j3, -j1+j2+j3, j1+j2+j3+1, j1-fabs(m1), j2-fabs(m2), j3-fabs(m3)};

		double minimum = array_min(minimum_array, 7);

		if (minimum < 0)
			return 0;

		else{
			/* Compute two arguments sqrtres and sumres to find the Clebsch
			 * Gordon Coefficient of the coupled system using their product
			 *
			 *
			 * We use the precomuted factorial array to compute the sqrtarg
			 * and sum arg*/


			long double tmp0 = factorial[(int)(j1+j2+j3+1)];
			long double tmp1 = factorial[(int)(j1+m1)] * factorial[(int)(j1-m1)]; 
			long double tmp2 = factorial[(int)(j2+m2)] * factorial[(int)(j2-m2)];
			long double tmp3 = factorial[(int)(j3+m3)] * factorial[(int)(j3-m3)];
			long double tmp4 = factorial[(int)(j1+j2-j3)] * factorial[(int)(j1-j2+j3)] *
				           factorial[(int)(-j1+j2+j3)]; 

			long double sqrtarg = (tmp1/tmp0*tmp2*tmp3*tmp4) * (2*j3 + 1)  ;
			double sqrtres = sqrt(sqrtarg);

			/* Define two arrays of sums to determine the looping procedure for sumres
			 * the max of the first array will determine the ending value of the loop
			 * and the min of the second array will determine the initial value*/
			double min_arr[] = {j1+m2-j3, j2-m1-j3, 0};
			double max_arr[] = {j2+m2, j1-m1, j1+j2-j3};

			double sumres = 0;
			
			/* the array max and min return double precision values by default, so we must
			 * broadcast those values to integers see array_ops.h for more details on the
			 * implementations.*/
			int vmin = (int)array_max(min_arr, 3);
			int vmax = (int)array_min(max_arr, 3);

			/* Use the loop to compute the recrprocal value of a factorial product by indexing
			 * the precomputed array of factorial values, then add or subtract that value depending
			 * on the index of the term in the sum (this depends on the value of v at that iteration*/
			for(int v=vmin; v<=vmax; ++v){
				long int value = (factorial[(int)(v)] * factorial[(int)(j1+j2-j3-v)] *
						factorial[(int)(j1-m1-v)] * factorial[(int)(j2+m2-v)] *
						factorial[(int)(j3-j2+m1+v)] * factorial[(int)(j3-j1-m2+v)]);
				sumres += pow(-1, v) / value;
			}

			// the result is the product of sumres and sqrt res
			double result = sqrtres * sumres;

			return result;
		}
	}
}

/****************************************************************
FUNCTION NAME : wigner_D
RETURN TYPE : complex
PARAMETERS : double alpha: the first euler rotation angle
	     double beta: the second euler rotation angle
	     double gamma: the third euler rotation angle
	     double j: integer/half-integer parameter - total angular momentum
	     double m: integer/half-integer parameter - z-component of angular momentum
	     double mp: integer/half-integer parameter - z-component of angular momentum after coordinate rotation
	     long int *factorial: pointer to first value in an array of precomputed factorial values
DESCRIPTION : this function transforms the wave function of a quantum mechanical system with angular momentum j and its z projection
m by coordinate rotation defined by the euler rotation angles alpha, beta, and gamma
****************************************************************/
double complex wigner_D(double alpha, double beta, double gamma, double j, double m, double mp, long long int *factorial){
	
	double complex result = 0+0*I;

	/* 
	 * a series of complex summations involving binomail coefficients, factorials, and complex exponentials
	 * is used to calculate the result
	 *
	 * if beta is about Pi/2 we can simplify the calculation by approximating the beta term to be 1*/
	if( fabs(beta - M_PI/2) < pow(10,-10) ){
		/* Varshalovich Eq. (5), Section 4.16, Page 113.
        	 j, m, and mp here are J, M, and M', respectively, in Eq. (5). */	
		for(int k=0; k<=2*j; ++k){
			
			if( (k > (j+mp)) || (k > (j-m)))
				break;

			else if( (k < (mp-m)))
				continue;

			else {  
				result += (pow(-1, k) * binomial(j+mp, k, factorial) *
						binomial(j-mp, k+m-mp, factorial));
			}
		}
		result *= (pow(-1, m-mp) * sqrt ((double)(factorial[(int)(j+m)] * factorial[(int)(j-m)]) /
						 (double)(factorial[(int)(j+mp)] * factorial[(int)(j-mp)])) /
						 pow(2,j));
		result *= cexp(-m*alpha*I) * cexp(-mp*gamma*I);
	}

	else{
		/* Varshalovich Eq. (10), Section 4.16, Page 113.
        	   m, mpp, and mp here are M, m, and M', respectively, in Eq. (10). */
		for(double mpp=-j; mpp<(j+1); ++mpp){
			double temp1 =0 ;
			
			for(int k=0; k<=2*j; ++k){
				
				if( (k>(j+mpp)) || (k>(j-m)) )
					break;

				else if( k< (mpp-m))
					continue;

				else{  
					temp1 += ( pow(-1,k) * binomial(j+mpp, k, factorial) *
						   binomial(j-mpp, k+m-mpp, factorial));
				}
			}

			temp1 *= ( pow(-1, m-mpp) * sqrt( (double)(factorial[(int)(j+m)] * factorial[(int)(j-m)]) /
						          (double)(factorial[(int)(j+mpp)] * factorial[(int)(j-mpp)])) /
							   pow(2,j));





			double temp2 = 0;

			for(int k=0; k<=2*j; ++k){
				
				if( (k>(j-mp)) || (k>(j-mpp)) )
					break;

				else if( k < (-mp-mpp) )
					continue;

				else{
					temp2 += ( pow(-1,k) * binomial(j-mp, k, factorial) *
						   binomial(j+mp, k+mpp+mp, factorial));
				}

			}

			temp2 *= ( pow(-1, mpp+mp) * sqrt( (double)(factorial[(int)(j+mpp)] * factorial[(int)(j-mpp)]) /
							   (double)(factorial[(int)(j-mp)] * factorial[(int)(j+mp)])) /
						     pow(2,j));

			result += temp1 * cexp(-mpp*beta*I) * temp2;
		}
		
		// emperical normalization
		result *= cpow(I, 2*j-m-mp) * pow(-1, 2*m);
		result *= cexp(-m*alpha*I) * cexp(-mp*gamma*I);
	}

	return result;
}
