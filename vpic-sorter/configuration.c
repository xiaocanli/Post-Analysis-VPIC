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
        char *filename_attribute, char *filename_meta, int *tmax, int *tinterval)
{
    int c;
    static const char *options="f:o:a:g:m:k:hsvewl:t:c:b:i:";
    static struct option long_options[] = 
    {
        {"tmax", required_argument, 0, 'b'},
        {"tinterval", required_argument, 0, 'i'},
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
               -e only sort the key  \n\
               -v verbose  \n\
               example: ./h5group-sorter -f testf.h5p  -g /testg  -o testg-sorted.h5p -a testf.attribute -k 0 \n";
    fprintf(stdout, msg, "h5group-sorter");
}
