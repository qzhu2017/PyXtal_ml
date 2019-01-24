#include <stdio.h>
#include <stdlib.h>
#include "factorial.h"
/****************************************************************
FUNCTION NAME : factorial
RETURN TYPE : int
PARAMETERS : int n
DESCRIPTION : this function outputs the factorial of an integer
	using recursion
****************************************************************/
long long int Factorial(int n)
{
	if (n == 0)
		return 1;
	
	else if (n < 0) {
		printf("Factorial not defined for negative numbers\n");
		exit(0);
			}
	
	else
		return (n * Factorial(n-1));
}

/****************************************************************
FUNCTION NAME : factorial_values
RETURN TYPE : int *
PARAMETERS : int n_max: the maximal value of n (used in bispectrum calculation this is j_max)
DESCRIPTION : this function returns an array of integers corresponding to the factorial values
of factorial(0) to factorial(3*n_max+1) in ascending order
****************************************************************/
void factorial_values(long long int *f_arr, int size){
	for(int k=0; k<(size); ++k){
		f_arr[k] = Factorial(k);
	}
}
