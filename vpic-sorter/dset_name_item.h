#include <hdf5.h>

#ifndef DSET_NAME_ITEM_H
#define DSET_NAME_ITEM_H

//The struct to store the information of datasets
typedef struct{
    char  dataset_name[NAME_MAX];
    hid_t did;
    hid_t type_id;
    int   type_size;
    //int   package_offset;
}dset_name_item;

#endif
