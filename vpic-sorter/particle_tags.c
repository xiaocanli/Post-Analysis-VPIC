#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <float.h>
#include <omp.h>
#include <stdio.h>
#include <math.h>
#include "hdf5.h"

void read_data_serial_h5(int n, char *fname, char *gname, char *dname, int *data);

/******************************************************************************
 * Get the number of digits of an integer.
 ******************************************************************************/
int get_int_len(int value)
{
    int l = !value;
    while(value){ l++; value/=10; }
    return l;
}

/******************************************************************************
 * Get particle tags from the last time frame
 * Input:
 *  filename: the HDF5 filename.
 *  tstep: current time step.
 *  num_ptl: number of high energy particles to read.
 * Output:
 *  tags: particle tags.
 ******************************************************************************/
void get_particle_tags(char *filename, int tstep, int num_ptl, int *tags)
{
    int step_len = get_int_len(tstep);
    char gname[step_len+6];
    snprintf(gname, sizeof(gname), "%s%d", "Step#", tstep);
    read_data_serial_h5(num_ptl, filename, gname, "q", tags);
    /* for (int i = 0; i < num_ptl; i++) { */
    /*     printf("%d\n", tags[i]); */
    /* } */
}

/******************************************************************************
 * Read data from HDF5 file using one MPI process.
 *
 * Input:
 *  n: number of data to read.
 *  fname: the name of the HDF5 file.
 *  gname: the name of the group.
 *  dname dataset name.
 * Output:
 *  data: the read data from the file.
 ******************************************************************************/
void read_data_serial_h5(int n, char *fname, char *gname, char *dname, int *data)
{
    const int rank = 1;
    hid_t file_id, group_id, dset_id;
    hid_t filespace, memspace;
    hsize_t count[rank], offset[rank];
    hsize_t dims_out[rank];

    count[0] = n;

    /* Open the existing file. */
    file_id = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    /* Open a group */
    group_id = H5Gopen1(file_id, gname);

    /* Open a dataset and get its dataspace */
    dset_id = H5Dopen(group_id, dname, H5P_DEFAULT);
    filespace = H5Dget_space(dset_id);
    H5Sget_simple_extent_dims(filespace, dims_out, NULL);
    offset[0] = dims_out[0] - n;

    memspace = H5Screate_simple(rank, count, NULL);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);
    H5Dread(dset_id, H5T_NATIVE_INT, memspace, filespace, H5P_DEFAULT, data);

    /* Close/release resources */
    H5Dclose(dset_id);
    H5Sclose(memspace);
    H5Sclose(filespace);
    H5Gclose(group_id);
    H5Fclose(file_id);
}
