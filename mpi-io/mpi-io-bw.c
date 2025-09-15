#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FILENAME "mpi_io_testfile.dat"
#define DATA_PER_PROC_MB 10  // Size of data per process in MB

int main(int argc, char **argv) {
    int rank, size;
    MPI_File fh;
    MPI_Status status;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Each process writes DATA_PER_PROC_MB of data
    size_t data_size_bytes = DATA_PER_PROC_MB * 1024 * 1024;
    char *buffer = (char *)malloc(data_size_bytes);
    if (!buffer) {
        fprintf(stderr, "Rank %d: Unable to allocate buffer.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    memset(buffer, 'A' + rank % 26, data_size_bytes);  // Fill buffer with character

    // Open the file for writing (create if not exists)
    MPI_File_open(MPI_COMM_WORLD, FILENAME, 
                  MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    MPI_Offset offset = rank * data_size_bytes;

    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // Perform collective write
    MPI_File_write_at_all(fh, offset, buffer, data_size_bytes, MPI_BYTE, &status);

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    // Calculate bandwidth
    double total_time = end_time - start_time;
    double total_MB_written = DATA_PER_PROC_MB * size;
    double bandwidth_MBps = total_MB_written / total_time;

    if (rank == 0) {
        printf("Total data written: %.2f MB\n", total_MB_written);
        printf("Time taken: %.4f seconds\n", total_time);
        printf("Aggregate write bandwidth: %.2f MB/s\n", bandwidth_MBps);
    }

    MPI_File_close(&fh);
    free(buffer);
    MPI_Finalize();
    return 0;
}

