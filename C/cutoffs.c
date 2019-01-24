#include <math.h>
#include "cutoffs.h"

/****************************************************************
FUNCTION NAME : cosine_cutoff
RETURN TYPE : double
PARAMETERS : doublr r, double RC
DESCRIPTION : this function takes two values:
	r: the magnitude of the position vector
	Rc: a cutoff radius
and returns the value of a cutoff function
****************************************************************/
double cosine_cutoff(double r, double Rc){
	if(r > Rc)
		return 0;

	else
		return 0.5 * (cos(M_PI * r / Rc) + 1);
}
