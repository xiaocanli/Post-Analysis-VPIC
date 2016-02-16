/* 
 *
 * This transfers the particle trajectories in binary format to HDF5 format.
 * 
 */

#include "stdlib.h"
#include "hdf5.h"
#include <unistd.h>
#include <string.h>
#include <float.h>
#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <getopt.h>
#include "constants.h"
#include "vpic_data.h"
#include "get_data.h"
#include "qsort-parallel.h"
#include "particle_tags.h"
#include "tracked_particle.h"

int get_configuration(int argc, char **argv, int mpi_rank, char *filepath,
        char *species, int *tmax, int *tmin, int *tinterval, int *ncpus);
void print_help(void);
void read_header(FILE *fp, int *nptl, float *q_m, double *t);

/******************************************************************************
 * Main program to transfer the data
 ******************************************************************************/
int main(int argc, char **argv)
{
    int mpi_size, mpi_rank, ncpus, is_help;
    char *filepath, *species;
    int tmax, tmin, tinterval;
    double t0, t1;

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(comm, &mpi_size);
    MPI_Comm_rank(comm, &mpi_rank);

    t0 = MPI_Wtime();

    filepath = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    species = (char *)malloc(16 * sizeof(char));

    get_configuration(argc, argv, mpi_rank, filepath,
        species, &tmax, &tmin, &tinterval, &ncpus);

    free(species);
    free(filepath);

    MPI_Barrier(comm);
    t1 = MPI_Wtime();
    if(mpi_rank == 0) {
        printf("Overall time is [%f]s \n", (t1 - t0));
    }

    MPI_Finalize();
    return 0;
}

/******************************************************************************
 * Read the header, including
 * nptl: number of particles in this file
 * q_m: the charge to mass ratio
 * t: current simulation time for this time step
 ******************************************************************************/
void read_header(FILE *fp, int *nptl, float *q_m, double *t)
{
    float tmpf;
    double tmpd;
    fread(nptl, sizeof(int), 1, fp);
    fread(q_m, sizeof(float), 1, fp);
    fread(t, sizeof(double), 1, fp);
    fread(&tmpf, sizeof(float), 1, fp);
    fread(&tmpd, sizeof(double), 1, fp);
    /* printf("%d %f %f\n", *nptl, *q_m, *t); */
    printf("%d\n", *nptl);
}

/******************************************************************************
 * Get the analysis configuration.
 ******************************************************************************/
int get_configuration(int argc, char **argv, int mpi_rank, char *filepath,
        char *species, int *tmax, int *tmin, int *tinterval, int *ncpus)
{
    int c;
    static const char *options="e:s:i:f:p:n:";
    static struct option long_options[] = 
    {
        {"tmax", required_argument, 0, 'e'},
        {"tmin", required_argument, 0, 's'},
        {"tinterval", required_argument, 0, 'i'},
        {"filepath", required_argument, 0, 'f'},
        {"species", required_argument, 0, 'p'},
        {"ncpus", required_argument, 0, 'n'},
        {0, 0, 0, 0},
    };
    /* getopt_long stores the option index here. */
    int option_index = 0;
    extern char *optarg;

    /* Default values */
    *ncpus = 16;
    *tmin = 0;
    *tmax = 12;
    *tinterval = 6;
    *ncpus = 256;

    while ((c = getopt_long (argc, argv, options, long_options, &option_index)) != -1){
        switch (c){
            case 'e':
                *tmax = atoi(optarg);
                break;
            case 's':
                *tmin = atoi(optarg);
                break;
            case 'i':
                *tinterval = atoi(optarg);
                break;
            case 'f':
                strcpy(filepath, optarg);
                break;
            case 'p':
                strcpy(species, optarg);
                break;
            case 'n':
                *ncpus = atoi(optarg);
                break;
            case 'h':
                if (mpi_rank == 0) {
                    print_help();
                }
                return 1;
            default:
                printf("Error option [%s]\n", optarg);
                exit(-1);
        }
    }
    return 0;
}

/******************************************************************************
 * Print help information.
 ******************************************************************************/
void print_help(void)
{
    char *msg="Usage: %s [OPTION] \n\
               -h help (--help)\n\
               -e the particle output maximum time step \n\
               -s the particle output minimum time step \n\
               -i the particle output time interval \n\
               -n number of CPUs to use \n\
               -f file path saving the particle tracing data \n\
               -p  particle species for sorting \n";
    fprintf(stdout, msg, "binary_to_hdf5");
}

/******************************************************************************
 * Set dname_array. This is based on looking into the data directly. It is
 * not generated directly.
 ******************************************************************************/
dset_name_item *set_dname_array(int dataset_num)
{
    dset_name_item *dname_array;
    dname_array = (dset_name_item *)malloc(MAX_DATASET_NUM * sizeof(dset_name_item));
    snprintf(dname_array[0].dataset_name, NAME_MAX, "%s", "q");
    snprintf(dname_array[1].dataset_name, NAME_MAX, "%s", "x");
    snprintf(dname_array[2].dataset_name, NAME_MAX, "%s", "y");
    snprintf(dname_array[3].dataset_name, NAME_MAX, "%s", "z");
    snprintf(dname_array[4].dataset_name, NAME_MAX, "%s", "ux");
    snprintf(dname_array[5].dataset_name, NAME_MAX, "%s", "uy");
    snprintf(dname_array[6].dataset_name, NAME_MAX, "%s", "uz");
    snprintf(dname_array[7].dataset_name, NAME_MAX, "%s", "ex");
    snprintf(dname_array[8].dataset_name, NAME_MAX, "%s", "ey");
    snprintf(dname_array[9].dataset_name, NAME_MAX, "%s", "ez");
    snprintf(dname_array[10].dataset_name, NAME_MAX, "%s", "cbx");
    snprintf(dname_array[11].dataset_name, NAME_MAX, "%s", "cby");
    snprintf(dname_array[12].dataset_name, NAME_MAX, "%s", "cbz");

    dname_array[0].did = 0;
    dname_array[0].type_id = H5T_NATIVE_INT;
    dname_array[0].type_size = 4;

    for (int i = 1; i < dataset_num; i++) {
        dname_array[i].did = 0;
        /* dname_array[i].type_id = H5T_IEEE_F32LE; */
        dname_array[i].type_id = H5T_NATIVE_FLOAT;
        dname_array[i].type_size = 4;
    }
    return dname_array;
}

/******************************************************************************
 * Set the filenames.
 ******************************************************************************/
void set_filenames(char *filepath, int tstep, char *group_name,
        char *filename_sorted, char *filename_attribute)
{
    snprintf(group_name, MAX_FILENAME_LEN, "%s%d", "/Step#", tstep);
    snprintf(filename_sorted, MAX_FILENAME_LEN, "%s%s", filepath,
            "/electron_tracer_sorted.h5p");
    snprintf(filename_attribute, MAX_FILENAME_LEN, "%s%s", filepath,
            "/attribute");
}

/******************************************************************************
 * Get total number of particles for current MPI process.
 ******************************************************************************/
void get_nptl_tot(int mpi_rank, int mpi_size, int ncpus, char *filepath,
        char *filename, int tstep, unsigned long long *nptl_tot)
{
    int nptl;
    float q_m;
    double t;
    FILE *fp;
    for (int i = mpi_rank; i < ncpus; i += mpi_size) {
        snprintf(filename, MAX_FILENAME_LEN, "%s%s%d%s%d", filepath,
                "/T.", tstep, "/electron_tracer.", i);
        fp = fopen(filename, "rb");
        read_header(fp, &nptl, &q_m, &t);
        *nptl_tot += nptl;
        fclose(fp);
    }
}

/******************************************************************************
 * Get package data.
 ******************************************************************************/
void get_package_data(int mpi_rank, int mpi_size, int ncpus, char *filepath,
        char *filename, int tstep, int row_size, char *package_data)
{
    unsigned long long nptl_acc;
    int nptl;
    float q_m;
    double t;
    FILE *fp;
    char *data;

    nptl_acc = 0;
    for (int i = mpi_rank; i < ncpus; i += mpi_size) {
        snprintf(filename, MAX_FILENAME_LEN, "%s%s%d%s%d", filepath,
                "/T.", tstep, "/electron_tracer.", i);
        fp = fopen(filename, "rb");
        read_header(fp, &nptl, &q_m, &t);
        data = (char *)malloc(row_size*nptl*sizeof(char));
        fread(data, sizeof(char), row_size*nptl, fp);
        memcpy(package_data + nptl_acc*row_size, data, row_size*nptl);
        nptl_acc += nptl;
        free(data);
        fclose(fp);
    }
}
