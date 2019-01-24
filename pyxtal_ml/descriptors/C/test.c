#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "spherical_harmonics.h"
#include "factorial.h"
#include "angular_momentum.h"
#include "bispectrum.h"
#include "binomial.h"
#include "array_ops.h"

void test_factorial(){
	printf("\ntesting factorial\n");
	long long int returned[7];
	long long int expected[7] = {1, 1, 2, 6, 24, 120, 720};
	factorial_values(returned, 7);
	printf("Returned: ");
	disp_1d_array_int(returned, 7);
	printf("Expected: ");
	disp_1d_array_int(expected, 7);

}

void test_binomial(){
	
	printf("\ntesting binomial\n");
	long long int Factorial[5];
	int n=4, k=3;
	long long int expected=4, returned;

	factorial_values(Factorial, 5);
	returned = binomial(n, k, Factorial);
	printf("Returned: %lld\n", returned);
	printf("Expected: %lld\n", expected);
	if (returned == expected)
		printf("------Passed-------\n");
	else
		printf("------Failed-------\n");
}



void test_Wigner_D(){

	printf("\ntesting Wigner_D\n");
	long long int Factorial[20];
	factorial_values(Factorial, 20);
	double returned;
	double expected = 0.5000000000000001;
	complex double out;
	//out = wigner_D(M_PI/3, M_PI/2, M_PI/6, 0.5, 0.5, 0.5, Factorial);
	out = wigner_D(0.5235987755982989, -1.5707963267948966, -0.5235987755982989, 1.0, 1.0, 1.0, Factorial);
	returned = creal(out);
	printf("Returned: %1f\n", returned);
	printf("Expected: %1f\n", expected);
}


void test_coefficient_CG(){

	printf("\ntesting CG\n");
	long long int Factorial[20];
	factorial_values(Factorial, 20);
	double returned;
	double expected = -0.7071067811865476;
	returned = CG(0.5, -0.5, 0.5, 0.5, 0.0, -0.0, Factorial);
	printf("Returned: %1f\n", returned);
	printf("Expected: %1f\n", expected);
}

void test_U(){
	printf("\ntesting U\n");
	long long int Factorial[20];
	factorial_values(Factorial, 20);
	double returned;
	double expected = 0.9659258262890682;
	complex double out;
	out = U(0.5, 0.5, 0.5, 0.5235987755982989, -1.5707963267948966, -0.5235987755982989, Factorial);
	returned = creal(out);
	printf("Returned: %1f\n", returned);
	printf("Expected: %1f\n", expected);
}


void test_coefficient_C(){

	printf("\ntesting coefficient C\n");
	long long int Factorial[20];
	factorial_values(Factorial, 20);
	double returned;
	double expected = 3.5065154317897083;
	double complex out;
	double site[3] = {0.,     0.,     1.6775};
	double neighbors[9] = {1.228, 0.70898614, 1.6775, 0.0000, -1.41797225,  1.6775, -1.228,  0.70898614,  1.6775 };
	double G[3] = {6, 6, 6};
	out = c(0.0, 0.0, 0.0, site, 3, neighbors, 9, G, 3, 2.0, Factorial);
	returned = creal(out);
	printf("Returned: %1f\n", returned);
	printf("Expected: %1f\n", expected);
}

void test_bispectrum(){
	//QZ: for the purpose of test, let's just use a small cutoff, say 1.8
	printf("\ntesting bispectrum\n");
	int j_max = 6;
	double cutoff=2.0;
	double expected[] = {43.11488812879717, 36.7602370189890, 88.60852392089515, 28.04236048755574, 66.30964609657414, 
			     69.79931428405888, 19.1498691176388, 43.37058656821445, 41.84207556059405, 31.72688125614834, 
			     12.12415278048087, 24.9334410549403, 19.17287936767453, 10.45276513091704, 18.55128680572768, 
			     8.268474018308577, 14.3385521644087, 5.832920000584183, -0.49229838544607, 5.276053869508487, 
			     23.86181936981333, 7.82751205831162, 12.31585101212968, 2.750866714917436, -0.58195744517042, 
			     8.601450141179644, 16.5863564981417, 22.13311355387084, 10.02868057986819, 17.06297945972453, 
			     7.829043181072610, 7.40913318457892, 23.83490553138292, 33.58258062326649, 27.88741469319666, 
			     13.44763910061149, 25.1237523041558, 16.98430135133593, 18.67663888113001, 43.22802402147224, 
			     62.29235845797744, 56.0446195920753, 16.55050759592471, 32.71094140645340, 25.72034120378543, 
			     28.21722651250996, 58.8902943526339, 88.99857828324012, 88.29997822134610, 18.21337576431390, 
			     36.98776637986512, 30.6535971018384, 32.54063243736765, 65.48150619146337, 103.4733921129424, 
			     109.7431145465124, 18.0419657912239, 36.87304609460205, 30.47464910460771, 30.67253453204755, 
			     61.75054679133947, 101.892216069241, 113.0782035248681, 16.39886568333275, 33.13785312088485, 
			     26.06829504111912, 24.1549055887980, 50.46692580428515, 87.17132247192721, 99.65471638130106};

	//define the dynamical arrays F, B, N, G
	int dim_F = 3 * (j_max + 1);
	int dim_B = calc_B_length(j_max);
	int dim_N = 9;
	int dim_G = 3;

	long long int *F = malloc(dim_F * sizeof(long int));
	double *B = malloc(dim_B * sizeof(double));

	factorial_values(F, dim_F);
	//disp_1d_array_int(F, dim_F);

	double site[3] = {0.,     0.,     1.6775};
	double N[9] = {1.228, 0.70898614, 1.6775, 0.0000, -1.41797225,  1.6775, -1.228,  0.70898614,  1.6775 };
	double G[3] = {6, 6, 6};
	site_bispectrum(B, dim_B, j_max, site, N, dim_N, G, dim_G, cutoff, F);

	printf("Returned: (%d)\n", dim_B);
	disp_1d_array(B, dim_B);
	printf("Expected: (%d)\n", dim_B);
	disp_1d_array(expected, dim_B);
	free(B);
	free(F);
}

int main()
{
	//long long int Factorial[16];
	//factorial_values(Factorial, 16);
	//disp_1d_array_int(Factorial, 16);
	test_factorial();
	test_binomial();
	test_coefficient_CG();
	test_Wigner_D();
	test_U();
	test_coefficient_C();
	test_bispectrum();
}

