#include "stdlib.h"
#include "hdf5.h"
#include <unistd.h>
#include <string.h>
#include <float.h>
#include <omp.h>
#include <stdio.h>
#include <math.h>
#include "constants.h"
#include "mpi_io.h"
#include "package_data.h"
#include "vpic_data.h"
#include "get_data.h"

/* The metadata table of the sorted data */
/* We use float to represent (min, max) for different types */
typedef struct{
    float   c_min;
    float   c_max;
    unsigned long long c_start_offset;
    unsigned long long c_end_offset;
}SDSSortTableElem;

MPI_Datatype SDS_DATA_TYPE;

/******************************************************************************
 * Create SDS_DATA_TYPE used in mpi_io.
 ******************************************************************************/
void create_SDS_data_type() {
    /* create a type for SDSSortTableElem*/
    const int    nitems2=4;
    int          blocklengths2[4] = {1,1,1,1};
    MPI_Datatype types2[4] = {MPI_FLOAT, MPI_FLOAT, MPI_UNSIGNED_LONG_LONG,
                              MPI_UNSIGNED_LONG_LONG};
    MPI_Aint     offsets2[4];

    offsets2[0] = offsetof(SDSSortTableElem, c_min);
    offsets2[1] = offsetof(SDSSortTableElem, c_max);
    offsets2[2] = offsetof(SDSSortTableElem, c_start_offset);
    offsets2[3] = offsetof(SDSSortTableElem, c_end_offset);

    MPI_Type_create_struct(nitems2, blocklengths2, offsets2, types2, &SDS_DATA_TYPE);
    MPI_Type_commit(&SDS_DATA_TYPE);
}

/******************************************************************************
 * Read the data from "dset_id" based on "my_offset" and "my_data_size"
 ******************************************************************************/
void get_one_file_data(hid_t dset_id, int mpi_rank, int mpi_size, void *data,
        hsize_t my_offset, hsize_t my_data_size){
    hid_t   dataspace, memspace, typeid;
    int     rank;
    hid_t   plist2_id;	

    //Create the memory space & hyperslab for each process
    dataspace = H5Dget_space(dset_id);
    rank      = H5Sget_simple_extent_ndims(dataspace);
    memspace =  H5Screate_simple(rank, &my_data_size, NULL);
    H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, &my_offset, NULL, &my_data_size, NULL);	

    plist2_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist2_id, H5FD_MPIO_COLLECTIVE);
    //plist2_id = H5P_DEFAULT;
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

            //H5Dread(dset_id, H5T_NATIVE_FLOAT, memspace, dataspace, plist2_id, data);
            break;
        default:
            break;
    }

    H5Pclose(plist2_id);
    H5Sclose(dataspace);
}


/******************************************************************************
 * Write the results to the file
 ******************************************************************************/
int write_result_file(int mpi_rank, int mpi_size, char *data,
        hsize_t my_data_size, int row_size, int dataset_num, int max_type_size,
        int key_index, char *group_name, char *filename_sorted,
        char *filename_attribute, dset_name_item *dname_array,
        char is_recreate)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Info info = MPI_INFO_NULL;
    hid_t plist_id, plist_id2, file_id, group_id;
    hid_t dataspace_id, filespace, memspace;
    herr_t   status;
    hsize_t  i, *size_vector;
    hsize_t  file_size, my_offset[1], count[1];

    hsize_t                   my_start_address, my_end_address;
    SDSSortTableElem         *global_metadata_array;
    SDSSortTableElem          my_metadata_array;
    unsigned long long       *global_metadata_array2;
    long long                *global_metadata_array1;
    double                    my_min_value, my_max_value;
    //long long                *e_data, *x_data, *y_data, *z_data;


    double t4, t6, t7, t8;
    MPI_Barrier(MPI_COMM_WORLD);

    // printf("my_rank %d, write file\n ", mpi_rank);

    //Gather the size of data belonging to each proccess
    size_vector = malloc(mpi_size * sizeof(hsize_t));

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allgather(&my_data_size, 1, MPI_UNSIGNED_LONG_LONG, size_vector, 1,
            MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);

    file_size = 0;
    for (i = 0; i < mpi_size; i++) {
        file_size = file_size + size_vector[i];
        //if (mpi_rank == 0) {
        //    printf("%d,  size_vector %lu, file size %lu \n ", 
        //            i, (unsigned long)size_vector[i], (unsigned long)file_size);
        //}
    }

    my_start_address = 0;
    for (i = 0; i < mpi_rank; i++) {
        my_start_address = my_start_address + size_vector[i];
    }

    my_offset[0] = my_start_address;
    my_end_address  = my_start_address + my_data_size;

    if(mpi_rank == 0 || mpi_rank == (mpi_size -1)){
        printf("My rank: %d, my data size %llu, ", mpi_rank, my_data_size);
        printf("Total size: %llu, ", (unsigned long long)file_size);
        printf("SA %llu, EA %llu \n", my_start_address, my_end_address);
    }

    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, comm, info);
    //H5Pset_libver_bounds (plist_id, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);

    if( access( filename_sorted, F_OK ) != -1 && !is_recreate) {
        file_id = H5Fopen(filename_sorted, H5F_ACC_RDWR, plist_id);
    } else {
        file_id = H5Fcreate(filename_sorted, H5F_ACC_TRUNC, H5P_DEFAULT,
                plist_id);
    }
    H5Pclose(plist_id);

    H5L_info_t link_buff;
    if (is_recreate) {
        group_id = H5Gcreate(file_id, group_name, H5P_DEFAULT,
                H5P_DEFAULT, H5P_DEFAULT);
    } else {
        status = H5Eset_auto1(NULL, NULL);
        status = H5Gget_objinfo (file_id, group_name, 0, NULL);
        /* status = H5Lget_info(file_id, group_name, &link_buff, H5P_DEFAULT); */
        if (status != 0) {
            group_id = H5Gcreate(file_id, group_name, H5P_DEFAULT,
                    H5P_DEFAULT, H5P_DEFAULT);
        } else {
            group_id = H5Gopen(file_id, group_name, H5P_DEFAULT);
        }
        status = H5Eset_auto1(NULL, NULL);
    }

    count[0] = my_data_size;
    memspace = H5Screate_simple(1, count, NULL);

    plist_id2 = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id2, H5FD_MPIO_COLLECTIVE);

    char *temp_data;
    temp_data = (char *)malloc(max_type_size * my_data_size);
    if(temp_data == NULL){
        printf("Memory allocation fails ! \n");
        exit(-1);
    }

    hid_t  e_dpid;
    e_dpid = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_attr_phase_change(e_dpid, 0, 0);

    char    dataset_name_t[NAME_MAX];

    hid_t typeid;
    dataspace_id = H5Screate_simple(1, &file_size, NULL);

    MPI_Barrier(MPI_COMM_WORLD);
    t6 = MPI_Wtime();

    for (i = 0; i < dataset_num; i++){
        sprintf(dataset_name_t, "%s/%s", group_name, dname_array[i].dataset_name);
        //printf("Dataset name %s \n", dataset_name_t);
        dname_array[i].did = H5Dcreate(file_id, dataset_name_t,
                dname_array[i].type_id, dataspace_id, H5P_DEFAULT,
                H5P_DEFAULT, H5P_DEFAULT);
        if((dname_array[i].did < 0)){
            printf("Error in creating dataset ! \n");
            exit(-1);
        }
        filespace = H5Dget_space(dname_array[i].did);
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, my_offset, NULL, count, NULL);

        unpackage(data, i, my_data_size, temp_data, row_size,
                dname_array[i].type_size, max_type_size);

        typeid = H5Dget_type(dname_array[i].did);
        switch (H5Tget_class(typeid)){
            case H5T_INTEGER:
                if(H5Tequal(typeid, H5T_STD_I32LE) == TRUE){
                    H5Dwrite(dname_array[i].did, H5T_NATIVE_INT, memspace,
                            filespace, plist_id2, temp_data);
                }else if(H5Tequal(typeid, H5T_STD_I64LE) == TRUE){
                    H5Dwrite(dname_array[i].did, H5T_NATIVE_LLONG, memspace,
                            filespace, plist_id2, temp_data);
                }else if(H5Tequal(typeid, H5T_STD_I8LE) == TRUE){
                    H5Dwrite(dname_array[i].did, H5T_NATIVE_CHAR, memspace,
                            filespace, plist_id2, temp_data);
                }else if(H5Tequal(typeid, H5T_STD_I16LE) == TRUE){
                    H5Dwrite(dname_array[i].did, H5T_NATIVE_SHORT, memspace,
                            filespace, plist_id2, temp_data);
                }
                break;
            case H5T_FLOAT:
                if(H5Tequal(typeid, H5T_IEEE_F32LE) == TRUE){
                    H5Dwrite(dname_array[i].did, H5T_NATIVE_FLOAT, memspace,
                            filespace, plist_id2, temp_data);
                }else if(H5Tequal(typeid, H5T_IEEE_F64LE) == TRUE){
                    H5Dwrite(dname_array[i].did, H5T_NATIVE_DOUBLE, memspace,
                            filespace, plist_id2, temp_data);
                }
                break;
            default:
                break;
        }
        H5Sclose(filespace);
        H5Dclose(dname_array[i].did);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t7 = MPI_Wtime();
    if(mpi_rank == 0 ){
        printf(" ---Write file, takes [%f]s, ", (t7-t6));
        printf("for each dataset takes [%f]s )\n", (t7-t6)/dataset_num);
    }

    H5Sclose(dataspace_id);
    H5Pclose(e_dpid);
    H5Pclose(plist_id2);
    H5Sclose(memspace);
    H5Gclose(group_id);
    H5Fclose(file_id);

    MPI_Barrier(MPI_COMM_WORLD);
    t8 = MPI_Wtime();
    if(mpi_rank == 0 ){
        printf(" ---Close output file,  takes [%f]s)\n", (t8-t7));
    }


    free(temp_data);
    free(size_vector);

    // my_min_value = getLongLongValue(key_index, data);
    //my_max_value = getLongLongValue(key_index, data + (my_data_size - 1) * row_size);
    my_min_value = get_value_double(key_index, data);
    my_max_value = get_value_double(key_index, data + (my_data_size - 1) * row_size);

    //printf("Start SDS meta data ! \n");
    /* Write SDS metadata to the file */
    global_metadata_array = (SDSSortTableElem *)malloc(mpi_size * sizeof(SDSSortTableElem));
    global_metadata_array2 = (unsigned long long *)malloc(mpi_size * 2 * sizeof(unsigned long long));
    global_metadata_array1 = (long long *) malloc(mpi_size * 2 * sizeof(long long));

    my_metadata_array.c_min = my_min_value;
    my_metadata_array.c_max = my_max_value;
    my_metadata_array.c_start_offset = my_start_address;
    my_metadata_array.c_end_offset   = my_end_address;


    MPI_Barrier(MPI_COMM_WORLD);
    if (mpi_rank == (mpi_size -1) || mpi_rank == 0){
        printf("rank %d:  %f, %f, start: %llu,  enb: %llu \n", mpi_rank,
                my_min_value, my_max_value, (unsigned long long)my_start_address,
                (unsigned long long)my_end_address);
    }

    create_SDS_data_type();
    MPI_Allgather(&my_metadata_array, 1, SDS_DATA_TYPE, global_metadata_array,
            1, SDS_DATA_TYPE, MPI_COMM_WORLD);
    MPI_Type_free(&SDS_DATA_TYPE);

    for (i=0; i < mpi_size; i++){
        global_metadata_array1[i*2]   = global_metadata_array[i].c_min;
        global_metadata_array1[i*2+1] = global_metadata_array[i].c_max;
        global_metadata_array2[i*2]   = global_metadata_array[i].c_start_offset;
        global_metadata_array2[i*2+1] = global_metadata_array[i].c_end_offset;
    }

    if (mpi_rank == 0){
        FILE *file_ptr;
        file_ptr =fopen(filename_attribute, "w");
        if (!file_ptr){
            printf("Can't create attribute file [%s]. \n", filename_attribute);
            return -1;
        }else{
            printf("Create attribute file successful ! \n");
        }

        /* for (i=0; i < mpi_size; i++){ */
        /*     printf("%f %f %llu %llu, count %lld \n", */
        /*             global_metadata_array[i].c_min, global_metadata_array[i].c_max, */
        /*             global_metadata_array[i].c_start_offset, */
        /*             global_metadata_array[i].c_end_offset, */
        /*             (global_metadata_array[i].c_end_offset - */
        /*              global_metadata_array[i].c_start_offset)); */ 
        /*     fprintf(file_ptr,"%f %f %llu %llu\n", global_metadata_array[i].c_min, */
        /*             global_metadata_array[i].c_max, */
        /*             global_metadata_array[i].c_start_offset, */
        /*             global_metadata_array[i].c_end_offset); */ 
        /* } */ 
        fclose(file_ptr);
    }
    free(global_metadata_array);

    MPI_Barrier(MPI_COMM_WORLD);
    t4 = MPI_Wtime();
    if(mpi_rank == 0 ){
        printf(" ---Write metadata takes [%f]s)\n", (t4-t8));
    }

    //MPI_Info_free(&info);
    return 0;
}
