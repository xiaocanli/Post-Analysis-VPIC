#include "dset_name_item.h"

//Read the data from "dset_id" based on "my_offset" and "my_data_size"
void get_one_file_data(hid_t dset_id, int mpi_rank, int mpi_size, 
        void *data, hsize_t my_offset, hsize_t my_data_size);

//Write sorted datat to the file
int write_result_file(int mpi_rank, int mpi_size, char *data,
        hsize_t my_data_size, int row_size, int dataset_num, int max_type_size,
        int key_index, char *group_name, char *filename_sorted,
        char *filename_attribute, dset_name_item *dname_array,
        char is_recreate);

void create_SDS_data_type();
