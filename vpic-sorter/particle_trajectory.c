#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <float.h>
#include <mpi.h>
#include "hdf5.h"
#include "time_frame_info.h"
#include "particle_tags.h"
#include "constants.h"
#include "vpic_data.h"
#include "get_data.h"
#include "package_data.h"

void track_particles(int mpi_rank, int mpi_size, int ntf, int tinterval,
        char *filepath, int *tags, int num_ptl, char *filename_out);
void get_tracked_particle_info(char *package_data, int qindex, int row_size,
        hsize_t my_data_size, int ct, int ntf, int *tags, int num_ptl, 
        char *tracked_particles);
int CompareInt32Value (const void * a, const void * b);
void save_tracked_particles(char *filename_out, char *tracked_particles,
        int ntf, int num_ptl, int row_size, int dataset_num, int max_type_size,
        dset_name_item *dname_array, int *tags);

/******************************************************************************
 * Main of the program to retrieve particle trajectory.
 ******************************************************************************/
int main(int argc, char **argv){
    int mpi_size, mpi_rank;
    static const char *options="d:o:n:a";
    const int MAX_LEN = 200;
    int ntf, tinterval;
    extern char *optarg;
    int c, num_ptl, tstep;
    int *tags;

    MPI_Comm comm = MPI_COMM_WORLD;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(comm, &mpi_size);
    MPI_Comm_rank(comm, &mpi_rank);

    num_ptl = 10;
    char filepath[MAX_LEN];
    char filename_out[MAX_LEN];
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
    snprintf(filename, MAX_LEN, "%s%s%d%s", filepath, "T.",
            tstep, "/electron_tracer_energy_sorted.h5p");
    get_particle_tags(filename, tstep, num_ptl, tags);
    qsort(tags, num_ptl, sizeof(int), CompareInt32Value);
    track_particles(mpi_rank, mpi_size, ntf, tinterval,
            filepath, tags, num_ptl, filename_out);
    free(tags);
    MPI_Finalize();
    return 0;
}

/******************************************************************************
 * Get the index of one dataset.
 ******************************************************************************/
int get_dataset_index(char *dname, dset_name_item *dname_array, int dataset_num)
{
    int i = 0;
    for (i = 0; i < dataset_num; i++) {
        if (strcmp(dname, dname_array[i].dataset_name) == 0)
            break;
    }
    return i;
}

/******************************************************************************
 * Track particles
 ******************************************************************************/
void track_particles(int mpi_rank, int mpi_size, int ntf, int tinterval,
        char *filepath, int *tags, int num_ptl, char *filename_out)
{
    int i, row_size, dataset_num, max_type_size, key_value_type;
    hsize_t my_data_size, rest_size;
    dset_name_item *dname_array;
    char *package_data;
    double t0, t1;
    char *tracked_particles, *tracked_particles_sum;
    int tstep, qindex;
    hsize_t j;

    dname_array = (dset_name_item *)malloc(MAX_DATASET_NUM * sizeof(dset_name_item));
    char filename[MAX_FILENAME_LEN];
    char group_name[MAX_FILENAME_LEN];

    t0 = MPI_Wtime();

    tstep = 0;
    snprintf(filename, MAX_FILENAME_LEN, "%s%s%d%s", filepath, "T.",
            tstep, "/electron_tracer_sorted.h5p");
    snprintf(group_name, MAX_FILENAME_LEN, "%s%d", "/Step#", tstep);
    package_data = get_vpic_pure_data_h5(mpi_rank, mpi_size, filename,
            group_name, &row_size, &my_data_size, &rest_size, &dataset_num,
            &max_type_size, &key_value_type, dname_array);

    set_variable_data(max_type_size, 0, dataset_num, key_value_type);
    qindex = get_dataset_index("q", dname_array, dataset_num);

    tracked_particles = (char *)malloc(ntf * num_ptl * row_size);
    for (j = 0; j < ntf*num_ptl*row_size; j++) {
        tracked_particles[j] = 0;
    }
    if (mpi_rank == 0) {
        tracked_particles_sum = (char *)malloc(ntf * num_ptl * row_size);
        for (j = 0; j < ntf*num_ptl*row_size; j++) {
            tracked_particles_sum[j] = 0;
        }
    }
    get_tracked_particle_info(package_data, qindex, row_size,
            my_data_size, 0, ntf, tags, num_ptl, tracked_particles);
    free(package_data);

    for (i = 1; i < ntf; i++) {
        tstep = i * tinterval;
        snprintf(filename, MAX_FILENAME_LEN, "%s%s%d%s", filepath, "T.",
                tstep, "/electron_tracer_sorted.h5p");
        snprintf(group_name, MAX_FILENAME_LEN, "%s%d", "/Step#", tstep);
        package_data = get_vpic_pure_data_h5(mpi_rank, mpi_size, filename,
                group_name, &row_size, &my_data_size, &rest_size, &dataset_num,
                &max_type_size, &key_value_type, dname_array);
        get_tracked_particle_info(package_data, qindex, row_size,
                my_data_size, i, ntf, tags, num_ptl, tracked_particles);
        free(package_data);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Reduce(tracked_particles, tracked_particles_sum, ntf*num_ptl*row_size,
            MPI_CHAR, MPI_SUM, 0, MPI_COMM_WORLD);

    /* Save the particle data. */
    if (mpi_rank == 0) {
        save_tracked_particles(filename_out, tracked_particles_sum, ntf, num_ptl,
                row_size, dataset_num, max_type_size, dname_array, tags);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    if(mpi_rank == 0)
        printf("Overall time is [%f]s \n", (t1 - t0));

    free(tracked_particles);
    if (mpi_rank == 0) {
        free(tracked_particles_sum);
    }
    free(dname_array);
}

/******************************************************************************
 * Get tracked particles information.
 ******************************************************************************/
void get_tracked_particle_info(char *package_data, int qindex, int row_size,
        hsize_t my_data_size, int ct, int ntf, int *tags, int num_ptl, 
        char *tracked_particles)
{
    hsize_t i;
    int qvalue, qvalue_tracked, iptl;
    /* Make sure tracked qvalue is not smaller than the 1st qvalue in the data */
    qvalue = getInt32Value(qindex, package_data);
    iptl = 0;
    while (tags[iptl] < qvalue && iptl < num_ptl) {
        iptl++;
    }
    if (iptl < num_ptl) {
        qvalue_tracked = tags[iptl];
    } else {
        qvalue_tracked = -1;
    }
    for (i = 0; i < my_data_size; i++) {
        qvalue = getInt32Value(qindex, package_data + i*row_size);
        if (qvalue == qvalue_tracked) {
            memcpy(tracked_particles + (iptl * ntf + ct) * row_size,
                    package_data + i*row_size, row_size);
            if (iptl >= num_ptl-1) {
                break;
            } else {
                qvalue_tracked = tags[++iptl];
            }
        }
    }
}

/******************************************************************************
 * Compare the value "int32" type
 ******************************************************************************/
int CompareInt32Value (const void * a, const void * b)
{
    return ( *(int*)a - *(int*)b );
}

/******************************************************************************
 * Write data from HDF5 file using one process.
 ******************************************************************************/
void write_data_serial_h5(hid_t file_id, char *gname, int dataset_num, int rank,
        dset_name_item *dname_array, hsize_t *dimsf, hsize_t *count,
        hsize_t *offset, int my_data_size, int row_size, int max_type_size,
        char *data)
{
    hid_t group_id;
    hid_t filespace, memspace;
    hid_t typeid;
    int i;

    /* Create a group */
    group_id = H5Gcreate2(file_id, gname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    char *temp_data;
    temp_data = (char *)malloc(max_type_size * my_data_size);
    if(temp_data == NULL){
        printf("Memory allocation fails ! \n");
        exit(-1);
    }

    for (i = 0; i < dataset_num; i++) {
        filespace = H5Screate_simple(rank, dimsf, NULL);
        dname_array[i].did = H5Dcreate2(group_id, dname_array[i].dataset_name,
                dname_array[i].type_id, filespace, H5P_DEFAULT, H5P_DEFAULT,
                H5P_DEFAULT);

        memspace = H5Screate_simple(rank, count, NULL);
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);
        unpackage(data, i, my_data_size, temp_data, row_size,
                dname_array[i].type_size, max_type_size);
        H5Dwrite(dname_array[i].did, dname_array[i].type_id, memspace,
                filespace, H5P_DEFAULT, data);
        typeid = H5Dget_type(dname_array[i].did);
        switch (H5Tget_class(typeid)){
            case H5T_INTEGER:
                if(H5Tequal(typeid, H5T_STD_I32LE) == TRUE){
                    H5Dwrite(dname_array[i].did, H5T_NATIVE_INT, memspace,
                            filespace, H5P_DEFAULT, temp_data);
                }else if(H5Tequal(typeid, H5T_STD_I64LE) == TRUE){
                    H5Dwrite(dname_array[i].did, H5T_NATIVE_LLONG, memspace,
                            filespace, H5P_DEFAULT, temp_data);
                }else if(H5Tequal(typeid, H5T_STD_I8LE) == TRUE){
                    H5Dwrite(dname_array[i].did, H5T_NATIVE_CHAR, memspace,
                            filespace, H5P_DEFAULT, temp_data);
                }else if(H5Tequal(typeid, H5T_STD_I16LE) == TRUE){
                    H5Dwrite(dname_array[i].did, H5T_NATIVE_SHORT, memspace,
                            filespace, H5P_DEFAULT, temp_data);
                }
                break;
            case H5T_FLOAT:
                if(H5Tequal(typeid, H5T_IEEE_F32LE) == TRUE){
                    H5Dwrite(dname_array[i].did, H5T_NATIVE_FLOAT, memspace,
                            filespace, H5P_DEFAULT, temp_data);
                }else if(H5Tequal(typeid, H5T_IEEE_F64LE) == TRUE){
                    H5Dwrite(dname_array[i].did, H5T_NATIVE_DOUBLE, memspace,
                            filespace, H5P_DEFAULT, temp_data);
                }
                break;
            default:
                break;
        }

        /* Close/release resources */
        H5Dclose(dname_array[i].did);
        H5Sclose(memspace);
        H5Sclose(filespace);
    }
    free(temp_data);
    H5Gclose(group_id);
}

/******************************************************************************
 * Save the tracked particle data.
 ******************************************************************************/
void save_tracked_particles(char *filename_out, char *tracked_particles,
        int ntf, int num_ptl, int row_size, int dataset_num, int max_type_size,
        dset_name_item *dname_array, int *tags)
{
    hid_t file_id;
    hsize_t dimsf[1], count[1], offset[1];
    char *temp_data;
    char gname[MAX_FILENAME_LEN];
    int i, rank;
    temp_data = (char *)malloc(ntf * row_size);
    file_id = H5Fcreate(filename_out, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    
    rank = 1;
    dimsf[0] = ntf;
    count[0] = ntf;
    offset[0] = 0;
    for (i = 0; i < num_ptl; i++) {
        memcpy(temp_data, tracked_particles + i*row_size*ntf, row_size*ntf);
        snprintf(gname, MAX_FILENAME_LEN, "%s%d", "/Particle#", tags[i]);
        write_data_serial_h5(file_id, gname, dataset_num, rank, dname_array,
                dimsf, count, offset, ntf, row_size, max_type_size, temp_data);
    }

    H5Fclose(file_id);
    free(temp_data);
}
