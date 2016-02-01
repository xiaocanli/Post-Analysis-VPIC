#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <float.h>
#include <mpi.h>
#include "particle_tags.h"
#include "tracked_particle.h"
#include "time_frame_info.h"

/******************************************************************************
 * Main of the program to retrieve particle trajectory.
 ******************************************************************************/
int main(int argc, char **argv){
    int mpi_size, mpi_rank;
    static const char *options="d:o:n:p:r:a";
    const int MAX_LEN = 200;
    int ntf, tinterval;
    extern char *optarg;
    int ratio_emax;  // The ratio of maximum energy and target energy
    int c, num_ptl, tstep;
    int *tags;

    MPI_Comm comm = MPI_COMM_WORLD;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(comm, &mpi_size);
    MPI_Comm_rank(comm, &mpi_rank);

    num_ptl = 10;
    char filepath[MAX_LEN];
    char filename_out[MAX_LEN];
    char particle[MAX_LEN];
    ratio_emax = 1;
    while ((c = getopt (argc, argv, options)) != -1){
        switch (c){
            case 'd':
                strcpy(filepath, optarg);
                break;
            case 'o':
                strcpy(filename_out, optarg);
                break;
            case 'n':
                num_ptl = atoi(optarg);
                break;
            case 'p':
                strcpy(particle, optarg);
                break;
            case 'r':
                ratio_emax = atoi(optarg);
                break;
            default:
                printf("Error option [%s]\n", optarg);
                exit(-1);
        }
    }

    ntf = 0;
    tinterval = 0;
    get_time_frame_info(&ntf, &tinterval, filepath);

    tags = (int *)malloc(num_ptl * sizeof(int));
    char filename[MAX_LEN];
    tstep = (ntf - 1) * tinterval;
    snprintf(filename, MAX_LEN, "%s%s%d%s%s%s", filepath, "T.",
            tstep, "/", particle, "_tracer_energy_sorted.h5p");
    get_particle_tags(filename, tstep, ratio_emax, num_ptl, tags);
    qsort(tags, num_ptl, sizeof(int), CompareInt32Value);
    track_particles(mpi_rank, mpi_size, ntf, tinterval,
            filepath, tags, num_ptl, filename_out, particle);
    free(tags);
    MPI_Finalize();
    return 0;
}
