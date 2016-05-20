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
#include "meta_data.h"
#include "tracked_particle.h"
#include "particle_tags.h"

char* sorting_single_tstep(int mpi_size, int mpi_rank, int key_index,
        int sort_key_only, int skew_data, int verbose, int write_result,
        int collect_data, int weak_scale_test, int weak_scale_test_length,
        int local_sort_threaded, int local_sort_threads_num, int meta_data,
        int ux_kindex, char *filename, char *group_name, char *filename_sorted,
        char *filename_attribute, char *filename_meta,
        unsigned long long *rsize, int load_tracer_meta);
void set_filenames(int tstep, char *filepath, char *species, char *filename,
        char *group_name, char *filename_sorted, char *filename_attribute,
        char *filename_meta);

/******************************************************************************
 * Main of the parallel sampling sort
 ******************************************************************************/
int main(int argc, char **argv){
    int mpi_size, mpi_rank;
    double t0, t1;
    int is_help;
    int key_index, sort_key_only, skew_data, verbose, write_result,
        collect_data, weak_scale_test, weak_scale_test_length,
        local_sort_threaded, local_sort_threads_num, meta_data,
        load_tracer_meta;
    int tmin, tmax, tinterval; // Minimum, maximum time step and time interval
    int multi_tsteps, ux_kindex;
    char *filename, *group_name, *filename_sorted, *filename_attribute;
    char *filename_meta, *filepath, *species, *filename_traj;
    char *final_buff;
    float ratio_emax;
    unsigned long long rsize;
    int nptl_traj, tracking_traj;

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(comm, &mpi_size);
    MPI_Comm_rank(comm, &mpi_rank);

    filename = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    group_name = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    filename_sorted = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    filename_attribute = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    filename_meta = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    filename_traj = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    filepath = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    species = (char *)malloc(16 * sizeof(char));
    tmin = 0;
    tmax = 0;
    tinterval = 1;

    t0 = MPI_Wtime();
    is_help = get_configuration(argc, argv, mpi_rank, &key_index,
            &sort_key_only, &skew_data, &verbose, &write_result,
            &collect_data, &weak_scale_test, &weak_scale_test_length,
            &local_sort_threaded, &local_sort_threads_num, &meta_data,
            filename, group_name, filename_sorted, filename_attribute,
            filename_meta, filepath, species, &tmax, &tmin, &tinterval,
            &multi_tsteps, &ux_kindex, filename_traj, &nptl_traj,
            &ratio_emax, &tracking_traj, &load_tracer_meta);

    /* when -h flag is set to seek help of how to use this program */
    if (is_help) {
        MPI_Finalize();
        return 1;
    }

    int ntf, mtf, tstep;
    mtf = tmin / tinterval;
    ntf = tmax / tinterval + 1;

    /* Get the particle tags from sorted-by-energy data of the last time frame */
    /* Then sort the tags */
    int *tags, qindex;
    int row_size, dataset_num, max_type_size, key_value_type;
    char *tracked_particles, *tracked_particles_sum, *package_data;
    dset_name_item *dname_array;
    hsize_t my_data_size, rest_size;
    if (tracking_traj) {
        tags = (int *)malloc(nptl_traj * sizeof(int));
        char filename_ene[MAX_FILENAME_LEN];
        tstep = (ntf - 1) * tinterval;
        snprintf(filename_ene, MAX_FILENAME_LEN, "%s%s%d%s%s%s", filepath, "/T.",
                tstep, "/", species, "_tracer_energy_sorted.h5p");
        get_particle_tags(filename_ene, tstep, ratio_emax, nptl_traj, tags);
        qsort(tags, nptl_traj, sizeof(int), CompareInt32Value);
        snprintf(group_name, MAX_FILENAME_LEN, "%s%d", "/Step#", tstep);
        dname_array = (dset_name_item *)malloc(MAX_DATASET_NUM * sizeof(dset_name_item));
        package_data = get_vpic_pure_data_h5(mpi_rank, mpi_size, filename_ene,
            group_name, &row_size, &my_data_size, &rest_size, &dataset_num,
            &max_type_size, &key_value_type, dname_array);
        free(package_data);
        qindex = get_dataset_index("q", dname_array, dataset_num);
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
    }

    if (multi_tsteps) {
        for (int i = mtf; i < ntf; i++) {
            tstep = i * tinterval;
            if (mpi_rank == 0) printf("%d\n", tstep);
            set_filenames(tstep, filepath, species, filename, group_name,
                    filename_sorted, filename_attribute, filename_meta);
            final_buff = sorting_single_tstep(mpi_size, mpi_rank, key_index,
                    sort_key_only, skew_data, verbose, write_result, collect_data,
                    weak_scale_test, weak_scale_test_length, local_sort_threaded,
                    local_sort_threads_num, meta_data, ux_kindex, filename,
                    group_name, filename_sorted, filename_attribute,
                    filename_meta, &rsize, load_tracer_meta);
            if (tracking_traj) {
                get_tracked_particle_info(final_buff, qindex, row_size,
                        rsize, i, ntf, tags, nptl_traj, tracked_particles);
            }
            if(collect_data == 1) {
                free(final_buff);
            }
        }
    } else {
        final_buff = sorting_single_tstep(mpi_size, mpi_rank, key_index,
                sort_key_only, skew_data, verbose, write_result, collect_data,
                weak_scale_test, weak_scale_test_length, local_sort_threaded,
                local_sort_threads_num, meta_data, ux_kindex, filename,
                group_name, filename_sorted, filename_attribute,
                filename_meta, &rsize, load_tracer_meta);
        if(collect_data == 1) {
            free(final_buff);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (tracking_traj) {
        MPI_Reduce(tracked_particles, tracked_particles_sum,
                ntf*nptl_traj*row_size, MPI_CHAR, MPI_SUM, 0, MPI_COMM_WORLD);

        /* Save the particle data. */
        if (mpi_rank == 0) {
            save_tracked_particles(filename_traj, tracked_particles_sum, ntf,
                    nptl_traj, row_size, dataset_num, max_type_size,
                    dname_array, tags);
        }
        free(tracked_particles);
        if (mpi_rank == 0) {
            free(tracked_particles_sum);
        }
        free(tags);
        free(dname_array);
    }

    MPI_Barrier(MPI_COMM_WORLD);		
    t1 = MPI_Wtime();
    if(mpi_rank == 0) {
        printf("Overall time is [%f]s \n", (t1 - t0));
    }

    free(filename);
    free(group_name);
    free(filename_sorted);
    free(filename_attribute);
    free(filename_meta);
    free(filename_traj);
    free(filepath);
    free(species);
    MPI_Finalize();
    return 0;
}

/******************************************************************************
 * Read the particle data and sort the data using one key. This is only for
 * a single time step.
 ******************************************************************************/
char* sorting_single_tstep(int mpi_size, int mpi_rank, int key_index,
        int sort_key_only, int skew_data, int verbose, int write_result,
        int collect_data, int weak_scale_test, int weak_scale_test_length,
        int local_sort_threaded, int local_sort_threads_num, int meta_data,
        int ux_kindex, char *filename, char *group_name, char *filename_sorted,
        char *filename_attribute, char *filename_meta,
        unsigned long long *rsize, int load_tracer_meta) {
    int max_type_size, dataset_num, key_value_type, row_size;
    hsize_t my_data_size, rest_size, my_offset;
    char *package_data, *final_buff;
    dset_name_item *dname_array;
    dname_array = (dset_name_item *)malloc(MAX_DATASET_NUM * sizeof(dset_name_item));
    package_data = get_vpic_data_h5(mpi_rank, mpi_size, filename, group_name,
            weak_scale_test, weak_scale_test_length, sort_key_only, key_index,
            &row_size, &my_data_size, &rest_size, &dataset_num, &max_type_size,
            &key_value_type, dname_array, &my_offset);

    /* Set the variables for retrieving the data with actual datatypes. */
    set_variable_data(max_type_size, key_index, dataset_num, key_value_type,
            ux_kindex);

    /* We have to use the meta data to calculate the particle position */
    /* But when using the binary output, the position is already calculated, */
    /* so we don't have to load the meta data. */
    if (load_tracer_meta) {
        calc_particle_positions(mpi_rank, my_offset, row_size, max_type_size,
                my_data_size, filename_meta, group_name, dname_array,
                dataset_num, package_data);
    }

    /* master:  also do slave's job. In addition, it is responsible for samples and pivots */
    /* slave:   (1) sorts. (2) samples (3) sends sample to master (4) receives pivots */
    /*          (5) sends/receives data to/from other processes based on pivots */
    /*          (6) sorts data again   (7) writes data to its location */
    create_opic_data_type(row_size);
    if (mpi_rank==0){
        printf("Start master of parallel sorting ! \n");
        final_buff = master(mpi_rank, mpi_size, package_data, my_data_size,
                rest_size, row_size, max_type_size, key_index, dataset_num,
                key_value_type, verbose, local_sort_threaded, local_sort_threads_num,
                skew_data, collect_data, write_result, group_name,
                filename_sorted, filename_attribute, dname_array, rsize);
    }else{
        final_buff = slave(mpi_rank, mpi_size, package_data, my_data_size,
                rest_size, row_size, max_type_size, key_index, dataset_num,
                key_value_type, verbose, local_sort_threaded, local_sort_threads_num,
                skew_data, collect_data, write_result, group_name,
                filename_sorted, filename_attribute, dname_array, rsize);
    }

    free_opic_data_type();

    free(dname_array);
    free(package_data);
    return final_buff;
}

/******************************************************************************
 * Set filenames.
 ******************************************************************************/
void set_filenames(int tstep, char *filepath, char *species, char *filename,
        char *group_name, char *filename_sorted, char *filename_attribute,
        char *filename_meta)
{
    snprintf(group_name, MAX_FILENAME_LEN, "%s%d", "/Step#", tstep);
    snprintf(filename, MAX_FILENAME_LEN, "%s%s%d%s%s%s", filepath,
            "/T.", tstep, "/", species, "_tracer.h5p");
    snprintf(filename_sorted, MAX_FILENAME_LEN, "%s%s%d%s%s%s",
            filepath, "/T.", tstep, "/", species, "_tracer_sorted.h5p");
    snprintf(filename_attribute, MAX_FILENAME_LEN, "%s", "attribute");
    snprintf(filename_meta, MAX_FILENAME_LEN, "%s%s%d%s%s%s", filepath,
            "/T.", tstep, "/grid_metadata_", species, "_tracer.h5p");
}
