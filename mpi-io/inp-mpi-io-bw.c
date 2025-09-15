#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_DATA_PER_PROC_MB 10  // Default data size per process in MB

int main(int argc, char **argv) {
    int rank, size;
    MPI_File fh;
    MPI_Status status;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check for correct input arguments
    if (argc < 2) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <output_file_path> [data_size_MB_per_proc]\n", argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const char *filename = argv[1];
    size_t data_per_proc_MB = (argc >= 3) ? atoi(argv[2]) : DEFAULT_DATA_PER_PROC_MB;
    size_t data_size_bytes = data_per_proc_MB * 1024 * 1024;

    // Allocate buffer
    char *buffer = (char *)malloc(data_size_bytes);
    if (!buffer) {
        fprintf(stderr, "Rank %d: Memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    memset(buffer, 'A' + (rank % 26), data_size_bytes);  // Fill with dummy data

    // Open file for writing
    int rc = MPI_File_open(MPI_COMM_WORLD, filename,
                           MPI_MODE_CREATE | MPI_MODE_WRONLY,
                           MPI_INFO_NULL, &fh);
    if (rc != MPI_SUCCESS) {
        if (rank == 0) fprintf(stderr, "Could not open file: %s\n", filename);
        free(buffer);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    MPI_Offset offset = rank * data_size_bytes;

    // Synchronize processes before timing
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // Collective write
    MPI_File_write_at_all(fh, offset, buffer, data_size_bytes, MPI_BYTE, &status);

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    // Calculate bandwidth
    double time_taken = end_time - start_time;
    double total_MB_written = data_per_proc_MB * size;
    double bandwidth_MBps = total_MB_written / time_taken;

    if (rank == 0) {
        printf("File: %s\n", filename);
        printf("Processes: %d\n", size);
        printf("Data per process: %zu MB\n", data_per_proc_MB);
        printf("Total data written: %.2f MB\n", total_MB_written);
        printf("Time taken: %.4f seconds\n", time_taken);
        printf("Aggregate bandwidth: %.2f MB/s\n", bandwidth_MBps);
    }

    MPI_File_close(&fh);
    free(buffer);
    MPI_Finalize();
    return 0;
}

