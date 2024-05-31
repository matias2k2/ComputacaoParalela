# Projeto MPI - Recomendação de Itens

Este projeto implementa um algoritmo de recomendação de itens usando MPI (Message Passing Interface) para paralelização.

## Requisitos

- GCC (GNU Compiler Collection)
- OpenMPI (Implementação de MPI)

## Compilação(prompt de comando Linux)

Para compilar o projeto, use os seguintes comandos:

mpicc -o sisRecom-mpi  sisRecom-mpi.c.

Execução (prompt de comando Linux):

mpirun -np 2 ./sisRecom-mpi input.txt



