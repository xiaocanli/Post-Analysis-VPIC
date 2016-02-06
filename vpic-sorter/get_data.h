int getInt32Value(int index, char *row_data);
long long getInt64Value(int index, char *row_data);
float getFloat32Value(int index, char *row_data);
double getFloat64Value(int index, char *row_data);
double get_value_double(int index, char *row_data);
void set_variable_data(int type_size_max, int key, int dset_num,
        int key_data_type, int ux_kindex);
