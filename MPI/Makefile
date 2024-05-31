all: sisRecom-mpi

sisRecom-mpi: sisRecom-mpi.c
	mpicc -fopenmp -o sisRecom-mpi sisRecom-mpi.c

run: sisRecom-mpi
	mpirun -np 2 ./sisRecom-mpi input.txt
