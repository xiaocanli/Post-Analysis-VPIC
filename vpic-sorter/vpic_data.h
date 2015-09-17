#include "dset_name_item.h"

//Find the type of dataset
int getDataType (hid_t dtid);
int getIndexDataType(hid_t did);
char* get_vpic_data_h5(int mpi_rank, int mpi_size, char *filename,
        char *group_name, int weak_scale_test, int weak_scale_test_length,
        int sort_key_only, int key_index, int *row_size, hsize_t *my_data_size,
        hsize_t *rest_size, int *dataset_num, int *max_type_size,
        int *key_value_type, dset_name_item *dname_array);

char* get_vpic_pure_data_h5(int mpi_rank, int mpi_size, char *filename,
        char *group_name, int *row_size, hsize_t *my_data_size,
        hsize_t *rest_size, int *dataset_num, int *max_type_size,
        int *key_value_type, dset_name_item *dname_array);

void open_dataset_h5(hid_t gid, int is_all_dset, int key_index,
        dset_name_item *dname_array, int *dataset_num, int *max_type_size);
