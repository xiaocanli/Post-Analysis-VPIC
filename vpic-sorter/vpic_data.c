#include "stdlib.h"
#include "hdf5.h"
#include <unistd.h>
#include <string.h>
#include <float.h>
#include <omp.h>
#include <stdio.h>
#include <math.h>
#include "mpi_io.h"
#include "package_data.h"
#include "constants.h"
#include "vpic_data.h"

void print_sorting_key_info(int mpi_rank, int sort_key_only, int key_index,
        dset_name_item *dname_array, int dataset_num, int *key_value_type);
void open_file_group_h5(char *filename, char *group_name, hid_t *plist_id,
        hid_t *file_id, hid_t *gid);
void partition_data_h5(dset_name_item *dname_array, int mpi_rank, int mpi_size,
        hsize_t *dims_out, hsize_t *my_data_size, hsize_t *rest_size,
        hsize_t *my_offset);
void partition_data_weak_test_h5(dset_name_item *dname_array, int mpi_rank,
        int mpi_size, int weak_scale_test_length, hsize_t *dims_out,
        hsize_t *my_data_size, hsize_t *rest_size, hsize_t *my_offset);
void read_dataset_h5(size_t row_count, int row_size, int max_type_size,
        hsize_t my_data_size, int mpi_rank, int mpi_size, int dataset_num,
        dset_name_item *dname_array, hsize_t my_offset, char *package_data);
int getIndexDataType(hid_t did);
int getDataType(hid_t dtid);

/******************************************************************************
 * Open the HDF5 file, retrieve the attributes and read the data.
 ******************************************************************************/
char* get_vpic_data_h5(int mpi_rank, int mpi_size, char *filename,
        char *group_name, int weak_scale_test, int weak_scale_test_length,
        int sort_key_only, int key_index, int *row_size, hsize_t *my_data_size,
        hsize_t *rest_size, int *dataset_num, int *max_type_size,
        int *key_value_type, dset_name_item *dname_array, hsize_t *my_offset)
{
    int is_all_dset;

    hid_t plist_id, file_id, gid;
    hsize_t dims_out[1];

    open_file_group_h5(filename, group_name, &plist_id, &file_id, &gid);
    /* whether to read all of the dataset. */
    if (sort_key_only == 1) {
        is_all_dset = 0;
    } else {
        is_all_dset = 1;
    }
    *dataset_num = 0;
    open_dataset_h5(gid, is_all_dset, key_index, dname_array,
            dataset_num, max_type_size);
    print_sorting_key_info(mpi_rank, sort_key_only, key_index,
            dname_array, *dataset_num, key_value_type);
    if (weak_scale_test == 1) {
        partition_data_weak_test_h5(dname_array, mpi_rank, mpi_size,
                weak_scale_test_length, dims_out, my_data_size,
                rest_size, my_offset);
    } else {
        partition_data_h5(dname_array, mpi_rank, mpi_size, dims_out,
                my_data_size, rest_size, my_offset);
    }

    /* Compute the size of "package_data" */ 
    /* Use max_type_size to ensure the memory alignment! */ 
    size_t row_count;
    *row_size = (*max_type_size) * (*dataset_num);  //The size of each row
    row_count = *my_data_size;                 //The total size of rows 
    if(row_count != *my_data_size || *my_data_size > ULONG_MAX){
        printf("Size of too big to size_t type !\n");
        exit(-1);
    }

    if(mpi_rank == 0 || mpi_rank == (mpi_size -1)){
        printf(" Data patition (based on key) My rank: %d, ", mpi_rank);
        printf("file size: %lu, ", (unsigned long)dims_out[0]);
        printf("my_data_size: %lu, ", (unsigned long)(*my_data_size));
        printf("my_offset: %lu, ", (unsigned long)(*my_offset));
        printf("row_size %d, ", *row_size);
        printf("char size %ld ,", sizeof(char));
        printf("dataset_num = %d, ", *dataset_num);
        printf("row_count = %ld, ", row_count);
        printf("max type size %d, ", *max_type_size);
        printf("sizeof(MPI_CHAR)=%ld \n ", sizeof(MPI_CHAR));
    }

    char *package_data;
    package_data = (char *)malloc(row_count * (*row_size) * sizeof(char));

    read_dataset_h5(row_count, *row_size, *max_type_size, *my_data_size,
            mpi_rank, mpi_size, *dataset_num, dname_array, *my_offset,
            package_data);

    /* free(package_data); */

    H5Pclose(plist_id);
    H5Gclose(gid);
    H5Fclose(file_id);
    return package_data;
}

/******************************************************************************
 * Open the HDF5 file, retrieve the attributes and read the data. This procedure
 * reads the data, but doesn't deal with the sorting information, the weak-scale
 * test.
 ******************************************************************************/
char* get_vpic_pure_data_h5(int mpi_rank, int mpi_size, char *filename,
        char *group_name, int *row_size, hsize_t *my_data_size,
        hsize_t *rest_size, int *dataset_num, int *max_type_size,
        int *key_value_type, dset_name_item *dname_array)
{
    int is_all_dset, key_index;
    char *package_data;

    hid_t plist_id, file_id, gid;
    hsize_t dims_out[1];
    hsize_t my_offset;

    open_file_group_h5(filename, group_name, &plist_id, &file_id, &gid);
    /* whether to read all of the dataset. */
    is_all_dset = 1;
    *dataset_num = 0;
    key_index = 0;
    open_dataset_h5(gid, is_all_dset, key_index, dname_array,
            dataset_num, max_type_size);
    partition_data_h5(dname_array, mpi_rank, mpi_size, dims_out,
            my_data_size, rest_size, &my_offset);

    /* Compute the size of "package_data" */ 
    /* Use max_type_size to ensure the memory alignment! */ 
    size_t row_count;
    *row_size = (*max_type_size) * (*dataset_num);  //The size of each row
    row_count = *my_data_size;                 //The total size of rows 
    if(row_count != *my_data_size || *my_data_size > ULONG_MAX){
        printf("Size of too big to size_t type !\n");
        exit(-1);
    }

    if(mpi_rank == 0 || mpi_rank == (mpi_size -1)){
        printf(" Data patition (based on key) My rank: %d, ", mpi_rank);
        printf("file size: %lu, ", (unsigned long)dims_out[0]);
        printf("my_data_size: %lu, ", (unsigned long)(*my_data_size));
        printf("my_offset: %lu, ", (unsigned long)my_offset);
        printf("row_size %d, ", *row_size);
        printf("char size %ld ,", sizeof(char));
        printf("dataset_num = %d, ", *dataset_num);
        printf("row_count = %ld, ", row_count);
        printf("max type size %d, ", *max_type_size);
        printf("sizeof(MPI_CHAR)=%ld \n ", sizeof(MPI_CHAR));
    }

    package_data = (char *)malloc(row_count * (*row_size) * sizeof(char));

    read_dataset_h5(row_count, *row_size, *max_type_size, *my_data_size,
            mpi_rank, mpi_size, *dataset_num, dname_array, my_offset,
            package_data);

    H5Pclose(plist_id);
    H5Gclose(gid);
    H5Fclose(file_id);
    return package_data;
}

/******************************************************************************
 * Get the data type of dataset.
 ******************************************************************************/
int getIndexDataType(hid_t did)
{
    hid_t dtid  = H5Dget_type(did);
    switch (H5Tget_class(dtid)){
        case H5T_INTEGER:
            if(H5Tequal(dtid, H5T_STD_I32LE) == TRUE){
                //printf("Key is in Int type ! \n");
                return H5GS_INT32;
            }else if(H5Tequal(dtid, H5T_STD_I64LE) == TRUE){
                //printf("Key is in Long Long type ! \n");
                return H5GS_INT64;
            }
            //Only support int and long  long now
            //else if(H5Tequal(dtid, H5T_STD_I8LE) == TRUE){
            //  return H5T_STD_I8LE;
            //}else if(H5Tequal(dtid, H5T_STD_I16LE) == TRUE){
            //  return H5T_STD_I16LE;
            //}
            return H5GS_INT32;
            break;
        case H5T_FLOAT:
            if(H5Tequal(dtid, H5T_IEEE_F32LE) == TRUE){
                //printf("Key is in Float type ! \n");
                return H5GS_FLOAT32;
            }else if(H5Tequal(dtid, H5T_IEEE_F64LE) == TRUE){
                //printf("Key is in Double type ! \n");
                return H5GS_FLOAT64;
            }
            return H5GS_FLOAT32;
            break;
        default:
            printf("Not support this type of key now !");
            exit(-1);
            break;
    }
}

int getDataType(hid_t dtid)
{
    switch (H5Tget_class(dtid)){
        case H5T_INTEGER:
            if(H5Tequal(dtid, H5T_STD_I32LE) == TRUE){
                return H5T_STD_I32LE;
            }else if(H5Tequal(dtid, H5T_STD_I64LE) == TRUE){
                return H5T_STD_I64LE;
            }
            //Only support int and long long now
            //else if(H5Tequal(dtid, H5T_STD_I8LE) == TRUE){
            //  return H5T_STD_I8LE;
            //}else if(H5Tequal(dtid, H5T_STD_I16LE) == TRUE){
            //  return H5T_STD_I16LE;
            //}
            return H5T_STD_I32LE;
            break;
        case H5T_FLOAT:
            if(H5Tequal(dtid, H5T_IEEE_F32LE) == TRUE){
                return H5T_IEEE_F32LE;
            }else if(H5Tequal(dtid, H5T_IEEE_F64LE) == TRUE){
                return H5T_IEEE_F64LE;
            }
            return H5T_IEEE_F32LE;
            break;
        case H5T_COMPOUND:
            return H5T_COMPOUND;
            break;
        case H5T_STRING:
            return H5T_STRING;
            break;
        default:
            break;
    }
    return 0;
}


/******************************************************************************
 * Print the sorting key information.
 ******************************************************************************/
void print_sorting_key_info(int mpi_rank, int sort_key_only, int key_index,
        dset_name_item *dname_array, int dataset_num, int *key_value_type)
{
    if (sort_key_only == 1){
        if(dataset_num == 1){
            key_index = 0; //The first one is the key
        }else{
            printf("Sort the key only but we found two keys !\n");
            exit(-1);
        }
    }
    //Get the type of key
    if(mpi_rank == 0) {
        if (key_index >= dataset_num) {
            printf("Key is %s ", "Particle energy");
        } else {
            printf("Key is %s ", dname_array[key_index].dataset_name);
        }
    }

    if (key_index >= dataset_num) {
        *key_value_type = H5GS_FLOAT64;
    } else {
        *key_value_type = getIndexDataType(dname_array[key_index].did);
    }

    if(mpi_rank == 0) {
        switch(*key_value_type){
            case H5GS_INT32:
                printf(", in INT32 type !\n");
                break;
            case H5GS_INT64:
                printf(", in INT64 type !\n");
                break;
            case H5GS_FLOAT32:
                printf(", in FLOAT32 type !\n");
                break;
            case H5GS_FLOAT64:
                printf(", in FLOAT64 type !\n");
                break;
        }
    }
}

/******************************************************************************
 * Open HDF5 file and group using parallel HDF5.
 ******************************************************************************/
void open_file_group_h5(char *filename, char *group_name, hid_t *plist_id,
        hid_t *file_id, hid_t *gid)
{
    MPI_Info info = MPI_INFO_NULL;
    *plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(*plist_id, MPI_COMM_WORLD, info);
    MPI_Barrier(MPI_COMM_WORLD);

    *file_id = H5Fopen(filename, H5F_ACC_RDONLY, *plist_id);
    *gid = H5Gopen(*file_id, group_name, H5P_DEFAULT);
}

/******************************************************************************
 * Open HDF5 datasets using parallel HDF5.
 ******************************************************************************/
void open_dataset_h5(hid_t gid, int is_all_dset, int key_index,
        dset_name_item *dname_array, int *dataset_num, int *max_type_size)
{
    hsize_t num_obj;
    hid_t typeid, dset_id;
    int i, obj_class, dset_id_start, dset_id_stop;
    char obj_name[NAME_MAX + 1];
    //Find all dataset in "group_name"
    H5Gget_num_objs(gid, &num_obj);
    if (is_all_dset == 1) {
        dset_id_start = 0;
        dset_id_stop = num_obj;
    } else {
        dset_id_start = key_index;
        dset_id_stop = key_index + 1;
    }
    *max_type_size = 0;
    for (i = dset_id_start; i < dset_id_stop; i++) {
        obj_class = H5Gget_objtype_by_idx(gid, i);
        H5Gget_objname_by_idx(gid, i, obj_name, NAME_MAX);
        /* Deal with object based on its obj_class. */
        switch(obj_class) {
            case H5G_DATASET:
                strncpy(dname_array[*dataset_num].dataset_name, obj_name, NAME_MAX);
                /* Open the dataset. */
                dset_id = H5Dopen(gid, obj_name, H5P_DEFAULT);
                dname_array[*dataset_num].did = dset_id;
                typeid = H5Dget_type(dset_id);
                dname_array[*dataset_num].type_size = H5Tget_size(typeid);
                /* Identify the type with the maximum size */
                if (*max_type_size < dname_array[*dataset_num].type_size){
                    *max_type_size = dname_array[*dataset_num].type_size;
                }
                dname_array[*dataset_num].type_id = getDataType(typeid);
                (*dataset_num)++;
                break;
            default:
                printf("Unknown object class %d!", obj_class);
        }
    }
}

/******************************************************************************
 * Partition the data evenly among cores, assuming other datasets have the
 * same size and shape.
 ******************************************************************************/
void partition_data_h5(dset_name_item *dname_array, int mpi_rank, int mpi_size,
        hsize_t *dims_out, hsize_t *my_data_size, hsize_t *rest_size,
        hsize_t *my_offset)
{
    hid_t dataspace;
    dataspace = H5Dget_space(dname_array[0].did);
    H5Sget_simple_extent_dims(dataspace, dims_out, NULL);
    *my_data_size = 0;
    *rest_size = dims_out[0] % mpi_size;
    if (mpi_rank ==  (mpi_size - 1)){
        *my_data_size = dims_out[0]/mpi_size + (*rest_size);
    }else{
        *my_data_size = dims_out[0]/mpi_size;
    }
    *my_offset = mpi_rank * (dims_out[0]/mpi_size);
    H5Sclose(dataspace);
}

/******************************************************************************
 * Partition the data evenly among cores, assuming other datasets have the
 * same size and shape.
 ******************************************************************************/
void partition_data_weak_test_h5(dset_name_item *dname_array, int mpi_rank,
        int mpi_size, int weak_scale_test_length, hsize_t *dims_out,
        hsize_t *my_data_size, hsize_t *rest_size, hsize_t *my_offset)
{
    hid_t dataspace;
    dataspace = H5Dget_space(dname_array[0].did);
    H5Sget_simple_extent_dims(dataspace, dims_out, NULL);
    *my_data_size = 0;
    if((*my_data_size) * mpi_size > dims_out[0]){
        printf("File is too small for scale test");
        printf("test legnth %lld, ", *my_data_size);
        printf("(mpi size %d, ", mpi_size);
        printf("data size %lld !\n", dims_out[0]);
        exit(-1);
    }
    *rest_size = 0;
    *my_data_size = weak_scale_test_length;
    *my_offset = mpi_rank * (*my_data_size);
    H5Sclose(dataspace);
}

/******************************************************************************
 * Read data from each dataset.
 ******************************************************************************/
void read_dataset_h5(size_t row_count, int row_size, int max_type_size,
        hsize_t my_data_size, int mpi_rank, int mpi_size, int dataset_num,
        dset_name_item *dname_array, hsize_t my_offset, char *package_data)
{
    double t1, t2;
    int i;
    if(package_data == NULL){
        printf("Memory allocation fails for package_data! \n");
        exit(-1);
    }

    //Allocate buf for one dataset
    char *temp_data;
    temp_data = (char *)malloc(max_type_size * my_data_size);
    if(temp_data == NULL){
        printf("Memory allocation fails ! \n");
        exit(-1);
    }

    //Read each dataset seperately and then combine each row into a package 
    if(mpi_rank == 0){
        printf("Read and package the data ! \n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    for (i = 0 ; i < dataset_num; i++){
        //Read one dataset
        get_one_file_data(dname_array[i].did, mpi_rank, mpi_size,
                temp_data, my_offset, my_data_size);
        //Merge the data into "package_data"
        package(package_data, i, row_count, temp_data, row_size,
                dname_array[i].type_size, max_type_size);
        H5Dclose(dname_array[i].did);
        if(mpi_rank == 0)
            printf("%d,  %s , type id (%d), type size (%d)\n ", i, 
                    dname_array[i].dataset_name, dname_array[i].type_id,
                    dname_array[i].type_size);
    }
    free(temp_data);
    MPI_Barrier(MPI_COMM_WORLD);

    t2 = MPI_Wtime();
    if(mpi_rank == 0){
        printf("Reading data takes [%f]s, for each dataset [%f]s \n",
                (t2-t1), (t2-t1)/dataset_num);
    }	    
}
