#include "stdlib.h"
#include "hdf5.h"
#include <unistd.h>
#include <string.h>
#include <float.h>
#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <getopt.h>
void print_help();

/******************************************************************************
 * Get the analysis configuration.
 ******************************************************************************/
int get_configuration(int argc, char **argv, int mpi_rank, int *key_index,
        int *sort_key_only, int *skew_data, int *verbose, int *write_result,
        int *collect_data, int *weak_scale_test, int *weak_scale_test_length,
        int *local_sort_threaded, int *local_sort_threads_num, int *meta_data,
        char *filename, char *group_name, char *filename_sorted,
        char *filename_attribute, char *filename_meta, char *filepath,
        char *species, int *tmax, int *tmin, int *tinterval, int *multi_tsteps,
        int *ux_kindex, char *filename_traj, int *nptl_traj, float *ratio_emax,
        int *tracking_traj, int *load_tracer_meta, int *is_recreate)
{
    int c;
    static const char *options="f:o:a:g:m:k:hsvewl:t:c:b:i:pu:qr";
    static struct option long_options[] = 
    {
        {"tmax", required_argument, 0, 'b'},
        {"tinterval", required_argument, 0, 'i'},
        {"filepath", required_argument, 0, 1},
        {"species", required_argument, 0, 2},
        {"filename_traj", required_argument, 0, 3},
        {"nptl_traj", required_argument, 0, 4},
        {"ratio_emax", required_argument, 0, 5},
        {"tmin", required_argument, 0, 6},
        {"is_recreate", required_argument, 0, 7},
        {"load_tracer_meta", required_argument, 0, 'r'},
        {0, 0, 0, 0},
    };
    /* getopt_long stores the option index here. */
    int option_index = 0;
    extern char *optarg;

    /* Default values */
    *key_index = 1;  
    *sort_key_only = 0;
    *skew_data = 0;
    *verbose = 0;
    *write_result = 1;
    *collect_data = 1;
    *weak_scale_test = 0;
    *weak_scale_test_length = 1000000;
    *local_sort_threaded = 0;
    *local_sort_threads_num = 16;
    *meta_data = 0;
    *multi_tsteps = 0;
    *ux_kindex = 0;
    *nptl_traj = 10;
    *ratio_emax = 1;
    *tracking_traj = 0;
    *load_tracer_meta = 0;
    *is_recreate = 0; // Do not recreate a HDF5 file when it exists

    /* while ((c = getopt (argc, argv, options)) != -1){ */
    while ((c = getopt_long (argc, argv, options, long_options, &option_index)) != -1){
        switch (c){
            case 'f':
                strcpy(filename, optarg);
                /* strncpy(filename, optarg, NAME_MAX); */
                /* filename = strdup(optarg); */
                break;
            case 'o':
                strcpy(filename_sorted, optarg);
                /* strncpy(filename_sorted, optarg, NAME_MAX); */
                /* filename_sorted = strdup(optarg); */
                break;
            case 'a':
                strcpy(filename_attribute, optarg);
                /* strncpy(filename_attribute, optarg, NAME_MAX); */
                /* filename_attribute = strdup(optarg); */  
                break;
            case 'g':
                strcpy(group_name, optarg);
                /* strncpy(group_name, optarg, NAME_MAX); */
                /* group_name = strdup(optarg); */
                break;
            case 'm':
                *meta_data = 1;
                strcpy(filename_meta, optarg);
                break;
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
            case 'l':
                *weak_scale_test = 1; 
                *weak_scale_test_length = atoi(optarg);
                break;
            case 'e':
                *sort_key_only = 1;
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
            case 'i':
                *tinterval = atoi(optarg);
                break;
            case 1:
                strcpy(filepath, optarg);
                break;
            case 2:
                strcpy(species, optarg);
                break;
            case 3:
                strcpy(filename_traj, optarg);
                break;
            case 4:
                *nptl_traj = atoi(optarg);
                break;
            case 5:
                *ratio_emax = atof(optarg);
                break;
            case 6:
                *tmin = atoi(optarg);
                break;
            case 7:
                *is_recreate = atoi(optarg);
                break;
            case 'r':
                *load_tracer_meta = 1;
                break;
            case 'p':
                *multi_tsteps = 1;
                break;
            case 'u':
                *ux_kindex = atoi(optarg);
                break;
            case 'q':
                *tracking_traj = 1;
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
void print_help(){
    char *msg="Usage: %s [OPTION] \n\
               -h help (--help)\n\
               -f name of the file to sort \n\
               -g group path within HDF5 file to data set \n\
               -o name of the file to store sorted results \n\
               -a name of the attribute file to store sort table  \n\
               -k the index key of the file \n\
               -m the meta data is used determine particle position \n\
               -s the data is in skew shape \n\
               -w won't write the sorted data \n\
               -b the particle output maximum time step \n\
               -i the particle output time interval \n\
               -p run sorting for multiple time steps \n\
               -u the key index of ux \n\
               -q tracking the trajectories of particles\n\
               -r whether to load tracer meta data \n\
               --tmin the particle output minimum time step \n\
               --filepath file path saving the particle tracing data \n\
               --species particle species for sorting \n\
               --filename_traj output file for particle trajectories \n\
               --nptl_traj number of particles for trajectory tracking \n\
               --ratio_emax ratio of Emax of all particles to that of tracked ones \n\
               --is_recreate whether to recreate a HDF5 file \n\
               -e only sort the key  \n\
               -v verbose  \n\
               example: ./h5group-sorter -f testf.h5p  -g /testg  -o testg-sorted.h5p -a testf.attribute -k 0 \n";
    fprintf(stdout, msg, "h5group-sorter");
}
