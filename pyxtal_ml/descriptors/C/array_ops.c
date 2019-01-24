#include "array_ops.h"
#include <stdio.h>
/****************************************************************
FUNCTION NAME :  array_min
RETURN TYPE :  double
PARAMETERS : double *a: a pointer to the first element of an array of doubles
DESCRIPTION :  This function outputs the minimal element of a
one-dimensional array of doubles
****************************************************************/
double array_min(double *a, int size) {
	double min = a[0]; // temporary minimal element
	int c; // incrimenting pointer
	/*loop over all elements using incimenting pointer
	  if the array element pointed at by c is less than
	  the current minimum, assign that value to min */
	for (c = 0; c < size; ++c) {
		if (a[c] < min) {
			min = a[c];
		}
	}

	return min;
}

/****************************************************************
FUNCTION NAME : array_max
RETURN TYPE : double
PARAMETERS : double *a: a pointer to the first element of an array of doubles
DESCRIPTION : this function outputs the maximal element of a
one-dimensional array of doubles
****************************************************************/
double array_max(double *a, int size) {
	double max = a[0]; // temporary maximal element
	int c; // incimenting pointer
	/*loop over all elements using incimenting pointer
	  if the array element pointed at by c is greater
	  than the current maximum, assign that value to max */
	for (c = 0; c < size; ++c) {
		if (a[c] > max) {
			max = a[c];
		}
	}

	return max;
}

/****************************************************************
FUNCTION NAME : max_ab(a, b)
RETURN TYPE : double
PARAMETERS :  double a 
	      double b 
DESCRIPTION :  this function returns the maximum value of two
double precision input values
****************************************************************/

double Max_ab(double a, double b){

	if ( a >= b){
		return a;
	}
	else{
		return b;
	}
}

/****************************************************************
FUNCTION NAME : Min_ab(a, b)
RETURN TYPE : double
PARAMETERS :  double a 
	      double b
DESCRIPTION : this function returns the minimum value of two
double precision input values
****************************************************************/

double Min_ab(double a, double b){

	if ( a >= b){
		return b;
	}
	else{
		return a;
	}
}

/****************************************************************
FUNCTION NAME : disp_1d_array
****************************************************************/

void disp_1d_array(double *array, int size){
	
	for(int i=0; i<size; i++){
		printf("%16.4f", array[i]);
		if ((i+1) % 5 == 0){
			printf("\n");
		}
	}
	printf("\n");
}

/****************************************************************
FUNCTION NAME : disp_1d_array_int
****************************************************************/

void disp_1d_array_int(long long int *array, int size){
	
	for(int i=0; i<size; i++){
		printf("%lld, ", array[i]);
		if ((i+1) % 5 == 0){
			printf("\n");
		}
	}
	printf("\n");
}
