#include "stdlib.h"
#include <unistd.h>
#include <string.h>
#include <float.h>
#include <stdio.h>
/* #define NDEBUG */
#include <assert.h>
#include "hdf5.h"
#include "constants.h"
#include "vpic_data.h"
#include "get_data.h"
#include "tracked_particle.h"

/******************************************************************************
 * Read data from HDF5 file using one process.
 *
 * Input:
 *  dset_id: the data_set id.
 *  my_offset: the offset from the beginning.
 *  my_data_size: the size of the data.
 *
 * Output:
 *  data: the read data from the file.
 ******************************************************************************/
void read_data_serial_h5(hid_t dset_id, hsize_t my_offset, hsize_t my_data_size,
        void *data)
{
    hid_t   dataspace, memspace, typeid;
    int     rank;
    hid_t   plist2_id;	

    //Create the memory space & hyperslab for each process
    dataspace = H5Dget_space(dset_id);
    rank = H5Sget_simple_extent_ndims(dataspace);
    memspace =  H5Screate_simple(rank, &my_data_size, NULL);
    H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, &my_offset, NULL,
            &my_data_size, NULL);	

    plist2_id = H5P_DEFAULT;
    typeid = H5Dget_type(dset_id);

    switch (H5Tget_class(typeid)){
        case H5T_INTEGER:
            if(H5Tequal(typeid, H5T_STD_I32LE) == TRUE){
                H5Dread(dset_id, H5T_NATIVE_INT, memspace, dataspace, plist2_id, data);
            }else if(H5Tequal(typeid, H5T_STD_I64LE) == TRUE){
                H5Dread(dset_id, H5T_NATIVE_LLONG, memspace, dataspace, plist2_id, data);
            }else if(H5Tequal(typeid, H5T_STD_I8LE) == TRUE){
                H5Dread(dset_id, H5T_NATIVE_CHAR, memspace, dataspace, plist2_id, data);
            }else if(H5Tequal(typeid, H5T_STD_I16LE) == TRUE){
                H5Dread(dset_id, H5T_NATIVE_SHORT, memspace, dataspace, plist2_id, data);
            }
            break;
        case H5T_FLOAT:
            if(H5Tequal(typeid, H5T_IEEE_F32LE) == TRUE){
                H5Dread(dset_id, H5T_NATIVE_FLOAT, memspace, dataspace, plist2_id, data);
            }else if(H5Tequal(typeid, H5T_IEEE_F64LE) == TRUE){
                H5Dread(dset_id, H5T_NATIVE_DOUBLE, memspace, dataspace, plist2_id, data);
            }
            break;
        default:
            break;
    }

    H5Pclose(plist2_id);
    H5Sclose(dataspace);
}

/******************************************************************************
 * Open the file, group and datasets.
 *
 * Input:
 *  fname: file name
 *  gname: group_name
 *
 * Output:
 *  file_id: the file handler.
 *  group_id: the group handler.
 *  dname_array: the dataset attributes and handlers.
 *  dims_out: the size of the data.
 *  dataset_num: the number of datasets.
 ******************************************************************************/
void open_file_group_dset(char *fname, char *gname, hid_t *file_id,
        hid_t *group_id, dset_name_item *dname_array, hsize_t *dims_out,
        int *dataset_num)
{
    int is_all_dset, max_type_size, key_index;
    hid_t dataspace;
    *file_id = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    *group_id = H5Gopen(*file_id, gname, H5P_DEFAULT);
    is_all_dset = 1;
    *dataset_num = 0;
    key_index = 0;
    open_dataset_h5(*group_id, is_all_dset, key_index, dname_array, dataset_num,
            &max_type_size);
    dataspace = H5Dget_space(dname_array[0].did);
    H5Sget_simple_extent_dims(dataspace, dims_out, NULL);
    H5Sclose(dataspace);
}

/******************************************************************************
 * Read the meta data for VPIC simulation. The procedure assumes that all MPI
 * processes of the PIC simulations have the same nx, ny, nz, dx, dy, dz. It
 * can easily modified if the grid is nonuniform.
 *
 * Input:
 *  dname_array: the dataset attributes and handlers.
 *  dims_out: the size of the data.
 *
 * Output:
 *  cell_sizes: the cell sizes.
 *  grid_dims: the number of grid points for each MPI process.
 *  np_local: number of particles for each MPI process.
 *  x0, y0, z0: the bottom left corner coordinates.
 ******************************************************************************/
void read_vpic_meta_data_h5(int dataset_num, hsize_t *dims_out,
        dset_name_item *dname_array, float *cell_sizes, int *grid_dims,
        int *np_local, float *x0, float *y0, float *z0)
{
    int i;
    hsize_t my_offset, my_data_size;

    my_offset = 0;
    my_data_size = 1;
    read_data_serial_h5(dname_array[0].did, my_offset, my_data_size, &cell_sizes[0]);
    read_data_serial_h5(dname_array[1].did, my_offset, my_data_size, &cell_sizes[1]);
    read_data_serial_h5(dname_array[2].did, my_offset, my_data_size, &cell_sizes[2]);
    read_data_serial_h5(dname_array[4].did, my_offset, my_data_size, &grid_dims[0]);
    read_data_serial_h5(dname_array[5].did, my_offset, my_data_size, &grid_dims[1]);
    read_data_serial_h5(dname_array[6].did, my_offset, my_data_size, &grid_dims[2]);

    my_data_size = dims_out[0];
    read_data_serial_h5(dname_array[3].did, my_offset, my_data_size, np_local);
    read_data_serial_h5(dname_array[7].did, my_offset, my_data_size, x0);
    read_data_serial_h5(dname_array[8].did, my_offset, my_data_size, y0);
    read_data_serial_h5(dname_array[9].did, my_offset, my_data_size, z0);

    for (i = 0; i < dataset_num; i++) {
        H5Dclose(dname_array[i].did);
    }
}

/******************************************************************************
 * Calculate the absolute values of the particle spatial positions.
 *
 * Input:
 *  mpi_rank: the rank of current MPI process.
 *  my_offset: the offset from the beginning of the file in data numbers.
 *  filename_meta: the meta file name.
 *  group_name: the group name.
 *  my_data_size: the data size on current MPI process.
 *  row_size: the size of one record of all dataset.
 *  max_type_size the maximum data size of all datasets.
 *
 * Input & output:
 *  package_data: the package includes all of the particle information.
 ******************************************************************************/
void calc_particle_positions(int mpi_rank, hsize_t my_offset, int row_size,
        int max_type_size, hsize_t my_data_size, char* filename_meta,
        char *group_name, char *package_data)
{
    float cell_sizes[3];
    int grid_dims[3];
    int dataset_num;
    int *np_local;
    long int *np_global;
    float *x0, *y0, *z0;
    int dim;
    dset_name_item *dname_array;
    int j, xindex, yindex, zindex, icell_index;
    hsize_t i;

    hid_t file_id, group_id;
    hsize_t dims_out[1];

    /* Get the data size */
    if (mpi_rank == 0) {
        dname_array = (dset_name_item *)malloc(MAX_DATASET_NUM *
                sizeof(dset_name_item));
        open_file_group_dset(filename_meta, group_name, &file_id, &group_id,
                dname_array, dims_out, &dataset_num);
        dim = (int)dims_out[0];
        xindex = get_dataset_index("dX", dname_array, dataset_num);
        yindex = get_dataset_index("dY", dname_array, dataset_num);
        zindex = get_dataset_index("dZ", dname_array, dataset_num);
        icell_index = get_dataset_index("i", dname_array, dataset_num);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&xindex, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&yindex, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&zindex, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&icell_index, 1, MPI_INT, 0, MPI_COMM_WORLD);

    np_local = (int *)malloc(dim * sizeof(int));
    np_global = (long int *)malloc(dim * sizeof(long int));
    x0 = (float *)malloc(dim * sizeof(float));
    y0 = (float *)malloc(dim * sizeof(float));
    z0 = (float *)malloc(dim * sizeof(float));

    /* Read the data and broadcast to all MPI processes */
    if (mpi_rank == 0) {
        read_vpic_meta_data_h5(dataset_num, dims_out, dname_array,
                cell_sizes, grid_dims, np_local, x0, y0, z0);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(cell_sizes, 3, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(grid_dims, 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(np_local, dim, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(x0, dim, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(y0, dim, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(z0, dim, MPI_FLOAT, 0, MPI_COMM_WORLD);

    np_global[0] = np_local[0];
    for (i = 1; i < dim; i++) {
        np_global[i] = np_local[i] + np_global[i-1];
    }

    /* Find right corners. */
    int startp, endp;
    int *nptl;
    hsize_t offset;
    offset = my_offset;
    i = 0;
    while (offset > np_global[i]) {
        i++;
    }
    startp = i;
    offset += my_data_size;
    while (offset > np_global[i]) {
        i++;
    }
    endp = i;
    nptl = (int *)malloc((endp - startp + 1) * sizeof(int));
    for (i = 0; i < endp - startp + 1; i++) {
        nptl[i] = np_local[i + startp];
    }

    /* Correction for the first and last */
    nptl[0] = np_global[startp] - my_offset;
    if (endp >= 1) {
        nptl[endp - startp] = my_offset + my_data_size - np_global[endp - 1];
    } else {
        nptl[endp - startp] = my_offset + my_data_size;
    }

    /* test */
    int sum = 0;
    for (i = 0; i < endp - startp + 1; i++) {
        sum += nptl[i];
    }
    assert(sum == my_data_size);

    /* do the calculation */
    float deltax, deltay, deltaz;
    float px, py, pz;
    float x0c, y0c, z0c;
    int icell, ix, iy, iz, nxg, nyg;
    offset = 0;
    nxg = grid_dims[0] + 2; // with ghost cells.
    nyg = grid_dims[1] + 2;
    for (j = 0; j < endp - startp + 1; j++) {
        x0c = x0[j+startp];
        y0c = y0[j+startp];
        z0c = z0[j+startp];
        for (i = 0; i < nptl[j]; i++) {
            deltax = getFloat32Value(xindex, package_data + offset*row_size);
            deltay = getFloat32Value(yindex, package_data + offset*row_size);
            deltaz = getFloat32Value(zindex, package_data + offset*row_size);
            icell = getInt32Value(icell_index, package_data + offset*row_size);
            iz = icell / (nxg*nyg);               // [1, nzg-2]
            iy = (icell - iz*nxg*nyg) / nxg;      // [1, nyg-2]
            ix = icell - iz*nxg*nyg - iy*nxg;     // [1, nxg-2]
            px = x0c + ((ix - 1) + (deltax + 1)*0.5) * cell_sizes[0];
            py = y0c + ((iy - 1) + (deltay + 1)*0.5) * cell_sizes[1];
            pz = z0c + ((iz - 1) + (deltaz + 1)*0.5) * cell_sizes[2];
            memcpy(package_data + offset*row_size + max_type_size*xindex,
                    &px, max_type_size);
            memcpy(package_data + offset*row_size + max_type_size*yindex,
                    &py, max_type_size);
            memcpy(package_data + offset*row_size + max_type_size*zindex,
                    &pz, max_type_size);
            offset++;
        }
    }

    free(nptl);
    free(x0);
    free(y0);
    free(z0);
    free(np_local);
    free(np_global);
    if (mpi_rank == 0) {
        free(dname_array);
        H5Gclose(group_id);
        H5Fclose(file_id);
    }
}
