# Usage:
# make        # compile all binary
# make clean  # remove ALL binaries and objects

CC=gcc -O3 -fPIC
LINKERFLAG = -lm
LDFLAG = -shared #-Wl,-soname,
BIN1 = unit_test
BIN2 = B_test
LIB = lib_bispectrum.so


all: ${BIN1} ${BIN2} ${LIB}

${LIB}: lib_bispectrum.o bispectrum.o spherical_harmonics.o angular_momentum.o factorial.o array_ops.o cutoffs.o binomial.o
	$(CC) -o ${LIB} lib_bispectrum.o bispectrum.o spherical_harmonics.o angular_momentum.o factorial.o array_ops.o cutoffs.o binomial.o ${LDFLAG} 

${BIN2}: lib_bispectrum.o bispectrum.o spherical_harmonics.o angular_momentum.o factorial.o array_ops.o cutoffs.o binomial.o
	$(CC) ${LINKERFLAG} -o ${BIN2} lib_bispectrum.o bispectrum.o spherical_harmonics.o angular_momentum.o factorial.o array_ops.o cutoffs.o binomial.o

${BIN1}: test.o bispectrum.o spherical_harmonics.o angular_momentum.o factorial.o array_ops.o cutoffs.o binomial.o
	$(CC) ${LINKERFLAG} -o ${BIN1} test.o bispectrum.o spherical_harmonics.o angular_momentum.o factorial.o array_ops.o cutoffs.o binomial.o

lib_bispectrum.o: lib_bispectrum.c bispectrum.h factorial.h array_ops.h 
	$(CC) -c lib_bispectrum.c


test.o: test.c bispectrum.h factorial.h angular_momentum.h spherical_harmonics.h array_ops.h
	$(CC) -c test.c

bispectrum.o: bispectrum.c bispectrum.h cutoffs.h array_ops.h angular_momentum.h spherical_harmonics.h
	$(CC) -c bispectrum.c 

spherical_harmonics.o: spherical_harmonics.c spherical_harmonics.h cutoffs.h angular_momentum.h
	$(CC) -c spherical_harmonics.c

angular_momentum.o: angular_momentum.c binomial.h array_ops.h angular_momentum.h
	$(CC) -c angular_momentum.c

cutoffs.o: cutoffs.c cutoffs.h
	$(CC) -c cutoffs.c 

binomial.o: binomial.c binomial.h
	$(CC) -c binomial.c

factorial.o: factorial.c factorial.h
	$(CC) -c factorial.c 

array_ops.o: array_ops.c array_ops.h
	$(CC) -c array_ops.c

clean :
	@echo "Cleaning up..."
	rm -rvf *.o ${BIN1} ${BIN2} ${LIB}

