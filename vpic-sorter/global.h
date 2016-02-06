#include "constants.h"

extern char *filename;
extern char *group_name;
extern char *filename_sorted;
extern char *filename_attribute;

MPI_Status   Stat;
//This is the struct type for orginal data
extern MPI_Datatype OPIC_DATA_TYPE;
//This is the struct type for sorted metadata table
extern MPI_Datatype SDS_DATA_TYPE;

//Actually number of datasets to sort
extern int dataset_num;
//The index of the key in the sort
//Other datasets will be sorted based on the key
extern int key_index;
//The value type of the key (e.g, float, int..)
//Used in comparison function
extern int key_value_type;

//Only sort the key (no extra values)
extern int sort_key_only;
//Data is in skew
extern int skew_data;
//Print more infor
extern int verbose;

//Write results to disk
extern int write_result;

//collect_data in phase2
extern int collect_data;

//This is only for test
extern int weak_scale_test;
extern int weak_scale_test_length;

//Use the local_sort with openmp 
extern int local_sort_threaded;
extern int local_sort_threads_num;

//The struct to store the information of datasets
typedef struct{
    char  dataset_name[NAME_MAX];
    hid_t did;
    hid_t type_id;
    int   type_size;
    //int   package_offset;
}dset_name_item;
extern dset_name_item dname_array[MAX_DATASET_NUM];

//For different data types, we use the one with the maximum size 
//to store the data
extern int max_type_size;

//
//This is the buffer to store the data
//Use the package_data to store different types (e/g. int, float)
//
extern char *package_data;

//Get the index-th data in the row (row_data) as double
double get_value_double(int index, char *row_data);
