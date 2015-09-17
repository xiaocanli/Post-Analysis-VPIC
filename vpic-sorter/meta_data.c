#include "stdlib.h"
#include <unistd.h>
#include <string.h>
#include <float.h>
#include <stdio.h>
#include "hdf5.h"
#include "constants.h"
#include "vpic_data.h"

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
 * Read the meta data for VPIC simulation.
 ******************************************************************************/
void read_vpic_meta_data(char *filename, char *group_name)
{
    int is_all_dset, dataset_num, max_type_size, key_index;
    hid_t file_id, group_id;
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    group_id = H5Gopen(file_id, group_name, H5P_DEFAULT);

    is_all_dset = 1;
    dataset_num = 0;
    key_index = 0;
    dset_name_item *dname_array;
    dname_array = (dset_name_item *)malloc(MAX_DATASET_NUM * sizeof(dset_name_item));
    open_dataset_h5(group_id, is_all_dset, key_index, dname_array,
            &dataset_num, &max_type_size);

    H5Gclose(group_id);
    H5Fclose(file_id);
    free(dname_array);
}
