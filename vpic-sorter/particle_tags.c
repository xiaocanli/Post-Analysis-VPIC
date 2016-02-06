#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <float.h>
#include <omp.h>
#include <stdio.h>
#include <math.h>
#include "hdf5.h"

void read_data_h5(int n, hsize_t offset, char *fname, char *gname,
        char *dname, int data_type, void *data);
void get_ene_max(char *fname, char *gname, double *emax);
void get_particle_offset(char *fname, char *gname, double emax,
        int ratio_emax, hsize_t *poffset);

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
void get_particle_tags(char *filename, int tstep, float ratio_emax,
        int num_ptl, int *tags)
{
    double emax;
    int step_len = get_int_len(tstep);
    char gname[step_len+6];
    hsize_t poffset = 0;
    snprintf(gname, sizeof(gname), "%s%d", "Step#", tstep);
    get_ene_max(filename, gname, &emax);
    get_particle_offset(filename, gname, emax, ratio_emax, &poffset);
    read_data_h5(num_ptl, poffset, filename, gname, "q", H5T_NATIVE_INT, tags);
    /* for (int i = 0; i < num_ptl; i++) { */
    /*     printf("%d\n", tags[i]); */
    /* } */
}

/******************************************************************************
 * Get the energy of the most energetic particle.
 *
 * Input:
 *  fname: the name of the HDF5 file.
 *  gname: the name of the group.
 *
 * Output:
 *  emax: the maximum energy.
 ******************************************************************************/
void get_ene_max(char *fname, char *gname, double *emax)
{
    float ux[1], uy[1], uz[1];
    int count, offset;
    count = 1;
    offset = 0;
    read_data_h5(count, offset, fname, gname, "Ux", H5T_NATIVE_FLOAT, ux);
    read_data_h5(count, offset, fname, gname, "Uy", H5T_NATIVE_FLOAT, uy);
    read_data_h5(count, offset, fname, gname, "Uz", H5T_NATIVE_FLOAT, uz);
    *emax = sqrt(1.0 + ux[0]*ux[0] + uy[0]*uy[0] + uz[0]*uz[0]) - 1.0;
}

/******************************************************************************
 * Get the offset of the particle having the energy ~ emax / ratio_emax.
 *
 * Input:
 *  fname: the name of the HDF5 file.
 *  gname: the name of the group.
 *  emax: the maximum energy.
 *  ratio_emax: the ratio of the maximum energy to the target energy.
 * Output:
 *  poffset: the offset of the target particle from the end of the file.
 ******************************************************************************/
void get_particle_offset(char *fname, char *gname, double emax,
        int ratio_emax, hsize_t *poffset)
{
    float *ux, *uy, *uz;
    double ene, target_ene;
    const int rank = 1;
    hid_t file_id, group_id, dset_ux, dset_uy, dset_uz;
    hid_t filespace, memspace;
    hsize_t count[rank], offset[rank];
    hsize_t dims_out[rank];

    /* Open the existing file. */
    file_id = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    /* Open a group */
    group_id = H5Gopen1(file_id, gname);

    /* Open a dataset */
    dset_ux = H5Dopen(group_id, "Ux", H5P_DEFAULT);
    dset_uy = H5Dopen(group_id, "Uy", H5P_DEFAULT);
    dset_uz = H5Dopen(group_id, "Uz", H5P_DEFAULT);
    
    /* ux */
    filespace = H5Dget_space(dset_ux);
    H5Sget_simple_extent_dims(filespace, dims_out, NULL);
    count[0] = dims_out[0];
    offset[0] = 0;
    memspace = H5Screate_simple(rank, count, NULL);
    ux = (float *)malloc(sizeof(float)*count[0]);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);
    H5Dread(dset_ux, H5T_NATIVE_FLOAT, memspace, filespace, H5P_DEFAULT, ux);
    H5Sclose(filespace);
    
    /* uy */
    filespace = H5Dget_space(dset_uy);
    uy = (float *)malloc(sizeof(float)*count[0]);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);
    H5Dread(dset_uy, H5T_NATIVE_FLOAT, memspace, filespace, H5P_DEFAULT, uy);
    H5Sclose(filespace);

    /* uz */
    filespace = H5Dget_space(dset_uz);
    uz = (float *)malloc(sizeof(float)*count[0]);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);
    H5Dread(dset_uz, H5T_NATIVE_FLOAT, memspace, filespace, H5P_DEFAULT, uz);
    H5Sclose(filespace);

    hsize_t i;
    target_ene = emax / ratio_emax;
    ene = 10.0 * emax;
    *poffset = 0;
    while (ene > target_ene) {
        i = dims_out[0] - (*poffset) - 1;
        ene = sqrt(ux[i]*ux[i] + uy[i]*uy[i] + uz[i]*uz[i] + 1.0) - 1.0;
        (*poffset)++;
    }

    /* Close/release resources */
    H5Sclose(memspace);
    H5Dclose(dset_ux);
    H5Dclose(dset_uy);
    H5Dclose(dset_uz);
    H5Gclose(group_id);
    H5Fclose(file_id);

    free(ux);
    free(uy);
    free(uz);
}

/******************************************************************************
 * Read data from HDF5 file using one MPI process.
 *
 * Input:
 *  n: number of data to read.
 *  offset: the offset from the end of the file, not the beginning.
 *  fname: the name of the HDF5 file.
 *  gname: the name of the group.
 *  dname dataset name.
 *  data_type: the data type for the dataset.
 * Output:
 *  data: the read data from the file.
 ******************************************************************************/
void read_data_h5(int n, hsize_t offset, char *fname, char *gname,
        char *dname, int data_type, void *data)
{
    const int rank = 1;
    hid_t file_id, group_id, dset_id;
    hid_t filespace, memspace;
    hsize_t count[rank], offset_file[rank];
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
    memspace = H5Screate_simple(rank, count, NULL);

    offset_file[0] = dims_out[0] - offset - n;
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset_file, NULL, count, NULL);
    H5Dread(dset_id, data_type, memspace, filespace, H5P_DEFAULT, data);

    /* Close/release resources */
    H5Dclose(dset_id);
    H5Sclose(memspace);
    H5Sclose(filespace);
    H5Gclose(group_id);
    H5Fclose(file_id);
}
