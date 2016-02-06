#include "hdf5.h"
#include <string.h>
#include <stdio.h>

/******************************************************************************
 * Package data in different types into "package_data"
 ******************************************************************************/
void package(char *p_data, int file_index, size_t my_data_size,
        char *my_data, int row_size, int t_size, int max_type_size)
{
    int i, p_offset;
    char *t_address, *s_address;

    p_offset = max_type_size * file_index;

    for (i = 0; i < my_data_size; i++){
        t_address = (char *) (p_data + i * row_size + p_offset);
        s_address = (char *) (my_data + i * t_size);
        memcpy(t_address, s_address, t_size);
    }
}

/******************************************************************************
 * Unpackage "package_data"data into different types
 ******************************************************************************/
void unpackage(char *p_data, int file_index, size_t my_data_size,
        char *my_data, int row_size, int t_size, int max_type_size)
{
    int i, p_offset;

    p_offset = max_type_size * file_index;

    for (i = 0; i < my_data_size; i++){
        memcpy(my_data + i * t_size , p_data + i * row_size + p_offset, t_size);
    }
}
