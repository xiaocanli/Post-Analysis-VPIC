#include "stdlib.h"
#include <unistd.h>
#include <string.h>
#include <float.h>
#include <stdio.h>
#include "constants.h"

int max_type_size;
int key_index;
int dataset_num;
int key_value_type;
int ux_key_index;

/******************************************************************************
 * Set the external variables
 ******************************************************************************/
void set_variable_data(int type_size_max, int key, int dset_num,
        int key_data_type, int ux_kindex)
{
    max_type_size = type_size_max;
    key_index = key;
    dataset_num = dset_num;
    key_value_type = key_data_type;
    ux_key_index = ux_kindex;
}

/******************************************************************************
 * Get the data at index.
 ******************************************************************************/
int getInt32Value(int index, char *row_data){
    int  p_offset;
    int  value, *p;

    p_offset = max_type_size * index;

    p = (int *)(row_data + p_offset);
    value = *p;

    return value;
}

long long getInt64Value(int index, char *row_data){
    int  p_offset;
    long long value, *p;

    p_offset = max_type_size * index;

    p = (long long *)(row_data + p_offset);
    value = *p;

    return value;
} 

float getFloat32Value(int index, char *row_data){
    int   p_offset;
    float value, *p, *u;

    if (index >= dataset_num) {
        /* Using particle energy as the sorting key. */
        value = 0.0;
        // Ux
        p_offset = ux_key_index * max_type_size;
        u = (float *)(row_data + p_offset);
        value += (*u) * (*u);
        // Uy
        p_offset += max_type_size;
        u = (float *)(row_data + p_offset);
        value += (*u) * (*u);
        // Uz
        p_offset += max_type_size;
        u = (float *)(row_data + p_offset);
        value += (*u) * (*u);
    } else {
        p_offset = max_type_size * index;
        p = (float *)(row_data + p_offset);
        value = *p;
    }

    return value;
}

double getFloat64Value(int index, char *row_data){
    int    p_offset;
    double value, *p, u1;
    float *u;

    if (index >= dataset_num) {
        /* Using particle energy as the sorting key. */
        value = 0.0;
        // Ux
        p_offset = ux_key_index * max_type_size;
        u = (float *)(row_data + p_offset);
        u1 = (double)(*u);
        value += u1 * u1;
        // Uy
        p_offset += max_type_size;
        u = (float *)(row_data + p_offset);
        u1 = (double)(*u);
        value += u1 * u1;
        // Uz
        p_offset += max_type_size;
        u = (float *)(row_data + p_offset);
        u1 = (double)(*u);
        value += u1 * u1;
    } else {
        p_offset = max_type_size * index;
        p = (double *)(row_data + p_offset);
        value = *p;
    }

    return value;
}

/******************************************************************************
 * Get the index-th data in the row (row_data) as double
 ******************************************************************************/
double get_value_double(int index, char *row_data){
    double value;
    value = 0.0;
    switch(key_value_type){
        case H5GS_INT64:
            value = (double) getInt64Value(index, row_data);
            break;
        case H5GS_INT32:
            value = (double) getInt32Value(index, row_data);
            break;
        case H5GS_FLOAT32:
            value = (double) getFloat32Value(index, row_data);
            break;
        case H5GS_FLOAT64:
            value = (double) getFloat64Value(index, row_data);
            break;
        default:
            printf("Un-supported data type ! \n");
            break;
    }
    return value;
}
