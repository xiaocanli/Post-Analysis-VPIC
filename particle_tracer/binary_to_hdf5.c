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

void read_header(FILE *fp, int *nptl, float *q_m, double *t);
int get_configuration(int argc, char **argv, int mpi_rank, int *key_index,
        int *skew_data, int *verbose, int *write_result, int *collect_data,
        int *local_sort_threaded, int *local_sort_threads_num, char *filepath,
        char *species, int *tmax, int *tmin, int *tinterval, int *ux_kindex,
        int *ncpus, char *filename_traj, int *nptl_traj, float *ratio_max);
void set_filenames(char *filepath, int tstep, char *group_name,
        char *filename_sorted, char *filename_attribute);
void print_help(void);
dset_name_item *set_dname_array(int dataset_num);
void get_nptl_tot(int mpi_rank, int mpi_size, int ncpus, char *filepath,
        char *filename, int tstep, unsigned long long *nptl_tot);
void get_package_data(int mpi_rank, int mpi_size, int ncpus, char *filepath,
        char *filename, int tstep, int row_size, char *package_data);
char *get_sorted_data(char *filepath, int tstep, char *filename, char *group_name,
        char *filename_sorted, char *filename_attribute, int mpi_rank,
        int mpi_size, int ncpus, hsize_t rest_size, int row_size,
        int max_type_size, int key_index, int dataset_num, int key_value_type,
        int verbose, int local_sort_threaded, int local_sort_threads_num,
        int skew_data, int collect_data, int write_result, dset_name_item *dname_array,
        unsigned long long *rsize);

/******************************************************************************
 * Main program to transfer the data
 ******************************************************************************/
int main(int argc, char **argv)
{
    int mpi_size, mpi_rank, ncpus, is_help;
    unsigned long long rsize;
    double t0, t1;

    int key_index, skew_data, verbose, write_result, collect_data;
    int local_sort_threaded, local_sort_threads_num;
    int ntf, tmax, tmin, tinterval, tstep;
    int ux_kindex, nptl_traj;
    float ratio_max;

    char *filename, *group_name, *filename_sorted, *filename_attribute;
    char *filepath, *species, *filename_traj;
    int max_type_size, dataset_num, key_value_type, row_size, qindex;
    char *final_buff;
    hsize_t rest_size;
    dset_name_item *dname_array;

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(comm, &mpi_size);
    MPI_Comm_rank(comm, &mpi_rank);

    t0 = MPI_Wtime();

    /* Initial filenames */
    filename = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    group_name = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    filename_sorted = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    filename_attribute = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    filename_traj = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    filepath = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    species = (char *)malloc(16 * sizeof(char));

    /* Get configuration data */
    is_help = get_configuration(argc, argv, mpi_rank, &key_index, &skew_data,
            &verbose, &write_result, &collect_data, &local_sort_threaded,
            &local_sort_threads_num, filepath, species, &tmax, &tmin,
            &tinterval, &ux_kindex, &ncpus, filename_traj, &nptl_traj,
            &ratio_max);

    /* when -h flag is set to seek help of how to use this program */
    if (is_help) {
        MPI_Finalize();
        return 1;
    }

    /* The variables are required to use the sorting procedures for HDF5 */
    max_type_size = 4;
    dataset_num = 13;
    row_size = dataset_num * max_type_size;
    rest_size = 0;
    dname_array = set_dname_array(dataset_num);

    create_opic_data_type(row_size);

    ntf = (tmax - tmin) / tinterval + 1;

    /* Get the particle tags of high energy particles */
    char *tracked_particles, *tracked_particles_sum;
    int *tags;
    key_index = dataset_num + 1; // for energy sorting
    key_value_type = H5GS_FLOAT32;
    set_variable_data(max_type_size, 0, dataset_num, key_value_type, 0);
    tstep = tmax;
    final_buff = get_sorted_data(filepath, tstep, filename, group_name,
            filename_sorted, filename_attribute, mpi_rank, mpi_size, ncpus,
            rest_size, row_size, max_type_size, key_index, dataset_num,
            key_value_type, verbose, local_sort_threaded,
            local_sort_threads_num, skew_data, collect_data, write_result,
            dname_array, &rsize);
    qindex = 0;
    tracked_particles = (char *)malloc(ntf * nptl_traj * row_size);
    for (int j = 0; j < ntf*nptl_traj*row_size; j++) {
        tracked_particles[j] = 0;
    }
    if (mpi_rank == 0) {
        tracked_particles_sum = (char *)malloc(ntf * nptl_traj * row_size);
        for (int j = 0; j < ntf*nptl_traj*row_size; j++) {
            tracked_particles_sum[j] = 0;
        }
    }
    tags = (int *)malloc(nptl_traj * sizeof(int));
    get_tracked_particle_info(final_buff, qindex, row_size,
            rsize, 0, ntf, tags, nptl_traj, tracked_particles);
    if(collect_data == 1) {
        free(final_buff);
    }

    key_index = 0; // sort by particle tag
    key_value_type = H5GS_INT32;
    /* Set the variables for retrieving the data with actual datatypes. */
    set_variable_data(max_type_size, key_index, dataset_num, key_value_type,
            ux_kindex);

    for (int i = 1; i < ntf; i++) {
        tstep = i * tinterval;
        final_buff = get_sorted_data(filepath, tstep, filename, group_name,
                filename_sorted, filename_attribute, mpi_rank, mpi_size, ncpus,
                rest_size, row_size, max_type_size, key_index, dataset_num,
                key_value_type, verbose, local_sort_threaded,
                local_sort_threads_num, skew_data, collect_data, write_result,
                dname_array, &rsize);
        get_tracked_particle_info(final_buff, qindex, row_size,
                rsize, i, ntf, tags, nptl_traj, tracked_particles);
        if(collect_data == 1) {
            free(final_buff);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Reduce(tracked_particles, tracked_particles_sum, ntf*nptl_traj*row_size,
            MPI_CHAR, MPI_SUM, 0, MPI_COMM_WORLD);

    /* Save the particle data. */
    if (mpi_rank == 0) {
        save_tracked_particles(filename_traj, tracked_particles_sum, ntf,
                nptl_traj, row_size, dataset_num, max_type_size, dname_array,
                tags);
    }

    free(tags);
    free(tracked_particles);
    if (mpi_rank == 0) {
        free(tracked_particles_sum);
    }

    free_opic_data_type();
    free(dname_array);
    free(filename);
    free(group_name);
    free(filename_sorted);
    free(filename_attribute);
    free(filename_traj);
    free(filepath);
    free(species);

    MPI_Barrier(MPI_COMM_WORLD);		
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
int get_configuration(int argc, char **argv, int mpi_rank, int *key_index,
        int *skew_data, int *verbose, int *write_result, int *collect_data,
        int *local_sort_threaded, int *local_sort_threads_num, char *filepath,
        char *species, int *tmax, int *tmin, int *tinterval, int *ux_kindex,
        int *ncpus, char *filename_traj, int *nptl_traj, float *ratio_max)
{
    int c;
    static const char *options="k:hsvw:t:cb:i:u:n:m:o:l:r:";
    static struct option long_options[] = 
    {
        {"tmax", required_argument, 0, 'b'},
        {"tmin", required_argument, 0, 'm'},
        {"tinterval", required_argument, 0, 'i'},
        {"filepath", required_argument, 0, 1},
        {"species", required_argument, 0, 2},
        {0, 0, 0, 0},
    };
    /* getopt_long stores the option index here. */
    int option_index = 0;
    extern char *optarg;

    /* Default values */
    *key_index = 0;
    *skew_data = 0;
    *verbose = 0;
    *write_result = 1;
    *collect_data = 1;
    *local_sort_threaded = 0;
    *local_sort_threads_num = 16;
    *ux_kindex = 0;
    *ncpus = 16;
    *tmax = 14;
    *tinterval = 14;

    while ((c = getopt_long (argc, argv, options, long_options, &option_index)) != -1){
        switch (c){
            case 'k':
                *key_index = atoi(optarg);
                break;
            case 's':
                *skew_data = 1;
                break;
            case 'w':
                *write_result = 0;
                break;
            case 'v':
                *verbose = 1;
                break;
            case 't':
                *local_sort_threaded = 1;
                *local_sort_threads_num = atoi(optarg);
                break;
            case 'c':
                *collect_data = 0;
                break;
            case 'b':
                *tmax = atoi(optarg);
                break;
            case 'm':
                *tmin = atoi(optarg);
                break;
            case 'i':
                *tinterval = atoi(optarg);
                break;
            case 1:
                strcpy(filepath, optarg);
                break;
            case 2:
                strcpy(species, optarg);
                break;
            case 'u':
                *ux_kindex = atoi(optarg);
                break;
            case 'n':
                *ncpus = atoi(optarg);
                break;
            case 'o':
                strcpy(filename_traj, optarg);
                break;
            case 'l':
                *nptl_traj = atoi(optarg);
                break;
            case 'r':
                *ratio_max = atoi(optarg);
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
               -k the index key of the file \n\
               -s the data is in skew shape \n\
               -w won't write the sorted data \n\
               -b the particle output maximum time step \n\
               -m the particle output minimum time step \n\
               -i the particle output time interval \n\
               -u the key index of ux \n\
               -n number of CPUs to use \n\
               -o the particle trajectory output file name \n\
               -l the number of particles to track \n\
               -r the ratio of the maximum energy of all particles to that of tracked particles \n\
               --filepath file path saving the particle tracing data \n\
               --species  particle species for sorting \n\
               -v verbose  \n";
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

/******************************************************************************
 * Get the sorted data.
 ******************************************************************************/
char *get_sorted_data(char *filepath, int tstep, char *filename, char *group_name,
        char *filename_sorted, char *filename_attribute, int mpi_rank,
        int mpi_size, int ncpus, hsize_t rest_size, int row_size,
        int max_type_size, int key_index, int dataset_num, int key_value_type,
        int verbose, int local_sort_threaded, int local_sort_threads_num,
        int skew_data, int collect_data, int write_result, dset_name_item *dname_array,
        unsigned long long *rsize)
{
    unsigned long long nptl_tot;
    char *package_data, *final_buff;
    hsize_t my_data_size;
    
    set_filenames(filepath, tstep, group_name, filename_sorted,
            filename_attribute);
    /* Get the total number of particles for current mpi_rank */
    nptl_tot = 0;
    get_nptl_tot(mpi_rank, mpi_size, ncpus, filepath, filename, tstep,
            &nptl_tot);
    my_data_size = nptl_tot; 
    package_data = (char *)malloc(row_size*nptl_tot*sizeof(char));
    get_package_data(mpi_rank, mpi_size, ncpus, filepath, filename, tstep,
            row_size, package_data);

    /* master: In addition to slave's job, also responsible for samples and pivots */
    /* slave: (1) sorts. (2) samples (3) sends sample to master */
    /*        (4) receives pivots */
    /*        (5) sends/receives data to/from other processes based on pivots */
    /*        (6) sorts data again   (7) writes data to its location */
    if (mpi_rank==0){
        printf("Start master of parallel sorting ! \n");
        master(mpi_rank, mpi_size, package_data, my_data_size, rest_size,
                row_size, max_type_size, key_index, dataset_num, key_value_type,
                verbose, local_sort_threaded, local_sort_threads_num, skew_data,
                collect_data, write_result, group_name, filename_sorted,
                filename_attribute, dname_array, final_buff, rsize);
    }else{
        slave(mpi_rank, mpi_size, package_data, my_data_size, rest_size, row_size,
                max_type_size, key_index, dataset_num, key_value_type, verbose,
                local_sort_threaded, local_sort_threads_num, skew_data,
                collect_data, write_result, group_name, filename_sorted,
                filename_attribute, dname_array, final_buff, rsize);
    }

    free(package_data);

    return final_buff;
}
