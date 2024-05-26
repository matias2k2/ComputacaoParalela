#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#define ALEATORIO ((double)rand() / (double)RAND_MAX)

void preenche_aleatorio_LR(double **L, double **R, int nU, int nI, int nF) {
    srand(0);
    double factor = 1.0 / (double)nF;
    for (int i = 0; i < nU; i++) {
        for (int j = 0; j < nF; j++) {
            L[i][j] = ALEATORIO * factor;
        }
    }

    for (int i = 0; i < nF; i++) {
        for (int j = 0; j < nI; j++) {
            R[i][j] = ALEATORIO * factor;
        }
    }
}

void inicializar_matriz(double **matriz, int numero_linhas, int numero_colunas) {
    for (int i = 0; i < numero_linhas; i++) {
        for (int j = 0; j < numero_colunas; j++) {
            matriz[i][j] = 0.0;
        }
    }
}

void calcular_B(double **B, double **L, double **R, int numero_linhasL, int numero_colunasR, int numero_colunasL) {
    for (int i = 0; i < numero_linhasL; i++) {
        for (int j = 0; j < numero_colunasR; j++) {
            B[i][j] = 0;
            for (int k = 0; k < numero_colunasL; k++) {
                B[i][j] += L[i][k] * R[k][j];
            }
        }
    }
}

void calcular_L_posterior(double **L, double **A, double **B, double **R, double alfa, int numero_linhas, int numero_colunas, int numero_caracteristicas, int start_row, int end_row) {
    double soma;
    for (int i = start_row; i < end_row; i++) {
        for (int k = 0; k < numero_caracteristicas; k++) {
            soma = 0.0;
            for (int j = 0; j < numero_colunas; j++) {
                if (A[i][j] != 0) {
                    soma += 2 * (A[i][j] - B[i][j]) * (-R[k][j]);
                }
            }
            L[i][k] = L[i][k] - (alfa * soma);
        }
    }
}

void calcular_R_posterior(double **R, double **A, double **B, double **L, double alfa, int numero_linhas, int numero_colunas, int numero_caracteristicas, int start_col, int end_col) {
    double soma;
    for (int k = 0; k < numero_caracteristicas; k++) {
        for (int j = start_col; j < end_col; j++) {
            soma = 0.0;
            for (int i = 0; i < numero_linhas; i++) {
                if (A[i][j] != 0) {
                    soma += 2 * (A[i][j] - B[i][j]) * (-L[i][k]);
                }
            }
            R[k][j] = R[k][j] - (alfa * soma);
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Status status;

    FILE *input_file;
    int numero_iteracoes, numero_linhas, numero_colunas, numero_caracteristicas, numero_elementos_diferentes_de_zero, linha, coluna;
    double alfa, valor;

    MPI_Init(&argc, &argv); /*START MPI */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); /* Determine rank of this processor */
    MPI_Comm_size(MPI_COMM_WORLD, &size); /* Determine total number of processors */

    double **matriz;
    if (rank == 0) {
        char nome_arquivo[100];
        printf("\nDigite o nome do seu ficheiro :  ");
        scanf("%s", nome_arquivo);

        input_file = fopen(nome_arquivo, "r");

        if (input_file == NULL) {
            printf("Erro ao abrir o arquivo.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        fscanf(input_file, "%d", &numero_iteracoes);
        fscanf(input_file, "%lf", &alfa);
        fscanf(input_file, "%d", &numero_caracteristicas);
        fscanf(input_file, "%d %d %d", &numero_linhas, &numero_colunas, &numero_elementos_diferentes_de_zero);

        matriz = (double **)malloc(numero_linhas * sizeof(double *));
        for (int i = 0; i < numero_linhas; i++) {
            matriz[i] = (double *)malloc(numero_colunas * sizeof(double));
        }
        inicializar_matriz(matriz, numero_linhas, numero_colunas);

        for (int i = 0; i < numero_elementos_diferentes_de_zero; i++) {
            fscanf(input_file, "%d %d %lf", &linha, &coluna, &valor);
            matriz[linha][coluna] = valor;
        }

        fclose(input_file);
    }

    MPI_Bcast(&numero_iteracoes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&alfa, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numero_caracteristicas, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numero_linhas, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numero_colunas, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numero_elementos_diferentes_de_zero, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        matriz = (double **)malloc(numero_linhas * sizeof(double *));
        for (int i = 0; i < numero_linhas; i++) {
            matriz[i] = (double *)malloc(numero_colunas * sizeof(double));
        }
    }

    for (int i = 0; i < numero_linhas; i++) {
        MPI_Bcast(matriz[i], numero_colunas, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    double **L = (double **)malloc(numero_linhas * sizeof(double *));
    for (int i = 0; i < numero_linhas; i++) {
        L[i] = (double *)malloc(numero_caracteristicas * sizeof(double));
    }
    inicializar_matriz(L, numero_linhas, numero_caracteristicas);

    double **R = (double **)malloc(numero_caracteristicas * sizeof(double *));
    for (int i = 0; i < numero_caracteristicas; i++) {
        R[i] = (double *)malloc(numero_colunas * sizeof(double));
    }
    inicializar_matriz(R, numero_caracteristicas, numero_colunas);

    preenche_aleatorio_LR(L, R, numero_linhas, numero_colunas, numero_caracteristicas);

    double **B = (double **)malloc(numero_linhas * sizeof(double *));
    for (int i = 0; i < numero_linhas; i++) {
        B[i] = (double *)malloc(numero_colunas * sizeof(double));
    }
    inicializar_matriz(B, numero_linhas, numero_colunas);

    int rows_per_process = (numero_linhas + size - 1) / size;  // Ensure balanced load
    int start_row = rank * rows_per_process;
    int end_row = (rank + 1) * rows_per_process;
    if (end_row > numero_linhas) {
        end_row = numero_linhas;
    }

    for (int iter = 1; iter <= numero_iteracoes; iter++) {
        calcular_B(B, L, R, numero_linhas, numero_colunas, numero_caracteristicas);
        MPI_Allgather(L[start_row], (end_row - start_row) * numero_caracteristicas, MPI_DOUBLE, L[0], rows_per_process * numero_caracteristicas, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgather(R[0], numero_caracteristicas * numero_colunas, MPI_DOUBLE, R[0], numero_caracteristicas * numero_colunas, MPI_DOUBLE, MPI_COMM_WORLD);

        calcular_L_posterior(L, matriz, B, R, alfa, numero_linhas, numero_colunas, numero_caracteristicas, start_row, end_row);
        calcular_R_posterior(R, matriz, B, L, alfa, numero_linhas, numero_colunas, numero_caracteristicas, 0, numero_colunas);
    }

    if (rank == 0) {
        double maior;
        int indice;
        for (int i = 0; i < numero_linhas; i++) {
            maior = 0.0;
            indice = -1;
            for (int j = 0; j < numero_colunas; j++) {
                if (matriz[i][j] == 0 && B[i][j] > maior) {
                    maior = B[i][j];
                    indice = j;
                }
            }
            printf("%d\n", indice);
        }
    }

    for (int i = 0; i < numero_linhas; i++) {
        free(L[i]);
        free(B[i]);
    }
    for (int i = 0; i < numero_caracteristicas; i++) {
        free(R[i]);
    }
    for (int i = 0; i < numero_linhas; i++) {
        free(matriz[i]);
    }
    free(L);
    free(R);
    free(B);
    free(matriz);

    MPI_Finalize();

    return 0;
}
