//Package and unpackage data in different types into "package_data"
void package(char *p_data, int file_index, size_t my_data_size, 
        char *my_data, int row_size, int t_size, int max_type_size);
void unpackage(char *p_data, int file_index, size_t my_data_size,
        char *my_data, int row_size, int t_size, int max_type_size);
