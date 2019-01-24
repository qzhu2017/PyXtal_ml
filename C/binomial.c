#include "binomial.h"
/****************************************************************
FUNCTION NAME : binomial
RETURN TYPE : int
PARAMETERS : int n: n-term of the computation of the binomial coefficient
	     int k: k-term of the computation of the binomial coefficient
	     int *factorial: a pointer to the first element of an array of precomuted factorial values
DESCRIPTION : the coefficient of the x^k term in the polynomial expansion of the binomial power (1+x)^n
	      we say "n choose k"
****************************************************************/
int binomial(int n, int k, long long int *factorial){

	int c = factorial[n] / (factorial[k] * factorial[n-k]);
	return c;
}
