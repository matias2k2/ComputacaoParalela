#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#define ALEATORIO ((double)random() / (double)RAND_MAX)

void preenche_aleatorio_LR(int nU, int nI, int nF, double L[nU][nF], double R[nF][nI]) {
    srandom(0);
    int i, j;
    for(i = 0; i < nU; i++)
        for(j = 0; j < nF; j++)
            L[i][j] = ALEATORIO / nF;
    for(i = 0; i < nF; i++)
        for(j = 0; j < nI; j++)
            R[i][j] = ALEATORIO / nF;
}

void atualiza_LR(int start_row, int end_row, int nI, int nF, double L[][nF], double R[][nI], double B[][nI], double A[][nI], double alpha) {
    int i, j, k;
    double delta;

    #pragma omp parallel for private(j, k) schedule(static)
    for(i = start_row; i < end_row; i++) {
        for(j = 0; j < nI; j++) {
            B[i][j] = 0;
            for(k = 0; k < nF; k++)
                B[i][j] += L[i][k] * R[k][j];
        }
    }

    #pragma omp parallel for private(j, k, delta) schedule(static)
    for(i = start_row; i < end_row; i++) {
        for(j = 0; j < nI; j++) {
            if (A[i][j] != 0) {
                delta = A[i][j] - B[i][j];
                for(k = 0; k < nF; k++) {
                    L[i][k] -= alpha * 2 * delta * (-R[k][j]);
                    R[k][j] -= alpha * 2 * delta * (-L[i][k]);
                }
            }
        }
    }
}

void print_matrix(int nU, int nI, double B[nU][nI], double A[nU][nI]) {
    int i, j;
    for(i = 0; i < nU; i++) {
        int max_item = -1;
        double max_value = -1;
        for(j = 0; j < nI; j++) {
            if (A[i][j] == 0 && B[i][j] > max_value) {
                max_value = B[i][j];
                max_item = j;
            }
        }
        printf("%d\n", max_item);
    }
}

void print_matrix1(int nU, int nI, double B[nU][nI], double A[nU][nI]) {
    FILE *file = fopen("output.txt", "w");
    if (file == NULL) {
        fprintf(stderr, "Erro ao abrir o arquivo para escrita.\n");
        return;
    }

    int i, j;
    for(i = 0; i < nU; i++) {
        int max_item = -1;
        double max_value = -1;
        for(j = 0; j < nI; j++) {
            if (A[i][j] == 0 && B[i][j] > max_value) {
                max_value = B[i][j];
                max_item = j;
            }
        }
        fprintf(file, "%d\n", max_item);
    }

    fclose(file);
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int iteracoes, nU, nI, nF, nDiferentes, i, j;
    double alpha;
    double start_time, end_time;

    start_time = MPI_Wtime();

    if (argc != 2) {
        if (rank == 0)
            printf("Uso: %s <arquivo_de_entrada>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    FILE *file = fopen(argv[1], "r");
    if (file == NULL) {
        if (rank == 0)
            printf("Erro ao abrir o arquivo %s.\n", argv[1]);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        fscanf(file, "%d", &iteracoes);
        fscanf(file, "%lf", &alpha);
        fscanf(file, "%d", &nF);
        fscanf(file, "%d %d %d", &nU, &nI, &nDiferentes);
    }

    MPI_Bcast(&iteracoes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nF, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nU, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nI, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nDiferentes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double (*A)[nI] = malloc(nU * nI * sizeof(double));
    if (rank == 0) {
        for(i = 0; i < nU; i++) {
            for(j = 0; j < nI; j++) {
                A[i][j] = 0;
            }
        }
        for (i = 0; i < nDiferentes; i++) {
            int row, col;
            double val;
            fscanf(file, "%d %d %lf", &row, &col, &val);
            A[row][col] = val;
        }
        fclose(file);
    }

    MPI_Bcast(A, nU * nI, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double (*L)[nF] = malloc(nU * nF * sizeof(double));
    double (*B)[nI] = malloc(nU * nI * sizeof(double));
    double (*R)[nI] = malloc(nF * nI * sizeof(double));
    double (*L_local)[nF] = malloc((nU / size + 1) * nF * sizeof(double));
    double (*R_local)[nI] = malloc(nF * nI * sizeof(double));

    preenche_aleatorio_LR(nU, nI, nF, L, R);

    int chunk_size = (nU + size - 1) / size; // Tamanho do pedaço para cada processo
    int start_row = rank * chunk_size;// Primeira linha que o processo vai manipular
    int end_row = (rank + 1) * chunk_size;// Última linha que o processo vai manipular
    if (end_row > nU) end_row = nU;// Ajusta a última linha para não ultrapassar o limite

    MPI_Scatter(L, chunk_size * nF, MPI_DOUBLE, L_local, chunk_size * nF, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(R, nF * nI, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int iter = 0; iter < iteracoes; iter++) {
        atualiza_LR(0, end_row - start_row, nI, nF, L_local, R, B, A, alpha);
        MPI_Allgather(L_local, chunk_size * nF, MPI_DOUBLE, L, chunk_size * nF, MPI_DOUBLE, MPI_COMM_WORLD);
        
        double R_global[nF][nI];
        MPI_Allreduce(R, R_global, nF * nI, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        for (int i = 0; i < nF; i++) {
            for (int j = 0; j < nI; j++) {
                R[i][j] = R_global[i][j] / size;
            }
        }
    }

    MPI_Reduce(rank == 0 ? MPI_IN_PLACE : B, B, nU * nI, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        print_matrix(nU, nI, B, A);
        print_matrix1(nU, nI, B, A);
        end_time = MPI_Wtime();
        printf("Tempo de execução: %f segundos\n", end_time - start_time);
    }

    free(A);
    free(L);
    free(B);
    free(R);
    free(L_local);
    free(R_local);

    MPI_Finalize();
    return 0;
}
