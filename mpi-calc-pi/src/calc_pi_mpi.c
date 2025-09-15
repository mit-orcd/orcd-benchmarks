#include "mpi.h"
#include <stdio.h>
#include <math.h>

int main(argc,argv)
int argc;
char *argv[];
{
    long long int n = 200000000000;
    int myid, numprocs;
    long long int i;
    double PI25DT = 3.141592653589793238462643;
    double mypi, pi, h, sum, x;
    double start_time;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    
    if (myid == 0) {
	start_time = MPI_Wtime();
    }
    h   = 1.0 / n;
    sum = 0.0;
    for (i = myid + 1; i <= n; i += numprocs) {
	x = h * ((double)i - 0.5);
	sum += 4.0 / (1.0 + x*x);
    }
    mypi = h * sum;
    MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0,
		   MPI_COMM_WORLD);
    
    if (myid == 0) {
	printf("time taken: %.8f\n", MPI_Wtime() - start_time);
	printf("pi is approximately %.16f, Error is %.16f\n", pi, fabs(pi - PI25DT));
    }
    MPI_Finalize();
    return 0;

}
