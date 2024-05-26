all: main sisRecom.o main.o run
main: sisRecom.o main.o
	mpicc -o main sisRecom.o main.o 

sisRecom.o: sisRecom.c sisRecom.h
	mpicc -g -c sisRecom.c
	
main.o: main.c sisRecom.h
	mpicc -g -c main.c -fopenmp 
run:
	./main

