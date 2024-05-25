all: main sisRecom.o main.o run
main: sisRecom.o main.o
	gcc -o main sisRecom.o main.o -fopenmp -lm

sisRecom.o: sisRecom.c sisRecom.h
	gcc -g -c sisRecom.c
	
main.o: main.c sisRecom.h
	gcc -g -c main.c -fopenmp 
run:
	./main

