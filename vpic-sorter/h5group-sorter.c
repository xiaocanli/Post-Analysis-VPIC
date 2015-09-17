/* 
 *
 * Generally, this file sort the dataset inside a HDF5 group
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
#include "constants.h"
#include "qsort-parallel.h"
#include "mpi_io.h"
#include "vpic_data.h"
#include "configuration.h"
#include "get_data.h"

/******************************************************************************
 * Main of the parallel sampling sort
 ******************************************************************************/
int main(int argc, char **argv){
    int mpi_size, mpi_rank;
    double t0, t1;
    int row_size, is_help;
    int key_index, key_value_type, sort_key_only, skew_data, verbose,
        write_result, collect_data, weak_scale_test, weak_scale_test_length,
        local_sort_threaded, local_sort_threads_num, dataset_num,
        max_type_size;
    char *filename, *group_name, *filename_sorted, *filename_attribute;
    hsize_t my_data_size, rest_size;

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(comm, &mpi_size);
    MPI_Comm_rank(comm, &mpi_rank);

    filename = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    group_name = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    filename_sorted = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    filename_attribute = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));

    t0 = MPI_Wtime();
    is_help = get_configuration(argc, argv, mpi_rank, &key_index,
            &sort_key_only, &skew_data, &verbose, &write_result,
            &collect_data, &weak_scale_test, &weak_scale_test_length,
            &local_sort_threaded, &local_sort_threads_num, filename,
            group_name, filename_sorted, filename_attribute);

    /* when -h flag is set to seek help of how to use this program */
    if (is_help) {
        MPI_Finalize();
        return 1;
    }

    /* Set the variables for retrieving the data with actual datatypes. */
    set_variable_data(max_type_size, key_index, dataset_num, key_value_type);

    char *package_data;
    dset_name_item *dname_array;
    dname_array = (dset_name_item *)malloc(MAX_DATASET_NUM * sizeof(dset_name_item));
    package_data = get_vpic_data_h5(mpi_rank, mpi_size, filename, group_name,
            weak_scale_test, weak_scale_test_length, sort_key_only, key_index,
            &row_size, &my_data_size, &rest_size, &dataset_num, &max_type_size,
            &key_value_type, dname_array);

    /* master:  also do slave's job. In addition, it is responsible for samples and pivots */
    /* slave:   (1) sorts. (2) samples (3) sends sample to master (4) receives pivots */
    /*          (5) sends/receives data to/from other processes based on pivots */
    /*          (6) sorts data again   (7) writes data to its location */
    create_opic_data_type(row_size);
    if (mpi_rank==0){
        printf("Start master of parallel sorting ! \n");
        master(mpi_rank, mpi_size, package_data, my_data_size, rest_size, row_size,
                max_type_size, key_index, dataset_num, key_value_type, verbose,
                local_sort_threaded, local_sort_threads_num, skew_data,
                collect_data, write_result, group_name, filename_sorted,
                filename_attribute, dname_array);
    }else{
        slave(mpi_rank, mpi_size, package_data, my_data_size, rest_size, row_size,
                max_type_size, key_index, dataset_num, key_value_type, verbose,
                local_sort_threaded, local_sort_threads_num, skew_data,
                collect_data, write_result, group_name, filename_sorted,
                filename_attribute, dname_array);
    }
    free_opic_data_type();
    free(dname_array);
    free(package_data);

    MPI_Barrier(MPI_COMM_WORLD);		
    t1 = MPI_Wtime();
    if(mpi_rank == 0)
        printf("Overall time is [%f]s \n", (t1 - t0));

    free(filename);
    free(group_name);
    free(filename_sorted);
    free(filename_attribute);
    MPI_Finalize();
    return 0;
}
