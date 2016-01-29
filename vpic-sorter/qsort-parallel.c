#include "stdlib.h"
#include "hdf5.h"
#include <unistd.h>
#include <string.h>
#include <float.h>
#include <omp.h>
#include <stdio.h>
#include <math.h>
#include "constants.h"
#include "qsort-parallel.h"
#include "vpic_data.h"
#include "mpi_io.h"
#include "get_data.h"

int max_type_size;
int key_index;
int dataset_num;
int key_value_type;
int verbose;
int local_sort_threaded, local_sort_threads_num;
char *group_name, *filename_sorted, *filename_attribute;
dset_name_item *dname_array;
MPI_Datatype OPIC_DATA_TYPE;

int skewed_data_partition(int mpi_rank, int mpi_size, char *data,
        int64_t my_data_size, char *pivots, int row_size, double dest_pivot,
        int dest, double *cur_value, int *nelem, int *previous_ii);
void check_load_balance(int mpi_rank, int mpi_size, unsigned long long rsize);
char *exchange_data(int mpi_rank, int mpi_size, char *data, int *scount,
        int64_t my_data_size, int row_size, int collect_data, int write_result, 
        unsigned long long *rsize);

/******************************************************************************
 * Compare the key in "long long" type
 ******************************************************************************/
int CompareInt64Key(const void *a, const void *b){
    long long v1, v2;
    v1 = getInt64Value(key_index, (char *)a);
    v2 = getInt64Value(key_index, (char *)b);

    if(v1 == v2){
        return 0;
    }

    if(v1 < v2){
        return -1;
    }else{
        return 1;
    }
}

/******************************************************************************
 * Compare the key in "int32" type
 ******************************************************************************/
int CompareInt32Key(const void *a, const void *b){
    int v1, v2;
    v1 = getInt32Value(key_index, (char *)a);
    v2 = getInt32Value(key_index, (char *)b);

    if(v1 == v2){
        return 0;
    }

    if(v1 < v2){
        return -1;
    }else{
        return 1;
    }
}


/******************************************************************************
 * Compare the key in "floa32t" type
 ******************************************************************************/
int CompareFloat32Key(const void *a, const void *b){
    float v1, v2;
    v1 = getFloat32Value(key_index, (char *)a);
    v2 = getFloat32Value(key_index, (char *)b);

    if(v1 == v2){
        return 0;
    }

    if(v1 < v2){
        return -1;
    }else{
        return 1;
    }
}


/******************************************************************************
 * Compare the key in "float64" type
 ******************************************************************************/
int CompareFloat64Key(const void *a, const void *b){
    double v1, v2;
    v1 = getFloat64Value(key_index, (char *)a);
    v2 = getFloat64Value(key_index, (char *)b);

    if(v1 == v2){
        return 0;
    }

    if(v1 < v2){
        return -1;
    }else{
        return 1;
    }
}


/******************************************************************************
 * Phase 1 on paralle sampling sorting.
 * Sort the data first and select the samples
 ******************************************************************************/
int phase1(int mpi_rank, int mpi_size, char *data, int64_t my_data_size,
        char *sample_of_rank0, int row_size)
{
    char *my_sample;
    int i, pass, temp_index, sample_size;
    double t1, t2;

    t1 = MPI_Wtime();

    //Sort data based on the type of key
    if(local_sort_threaded == 0){
        qsort_type(data, my_data_size, row_size);
    }else{
        openmp_sort(data, my_data_size, local_sort_threads_num, row_size);
    }

    t2 = MPI_Wtime();
    if(mpi_rank == 0){
        printf("Master : qsort take %f secs to sort %ld data \n", (t2-t1), my_data_size);
    }
    //Choose sample
    my_sample = malloc(mpi_size * row_size);
    if(my_sample == NULL){
        printf("Memory allocation  fails ! \n");
        exit(-1);
    }
    pass = my_data_size/mpi_size;
    for(i = 0; i < mpi_size; i++){
        //my_sample[i] = data[pass*(i)];
        temp_index = pass*(i);
        memcpy(my_sample+i*row_size, data+temp_index*row_size, row_size);
    }

    //If it is master, just pass my_sample through parameter
    //If it is slave, use MPI_send
    if (mpi_rank != 0){
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Send(my_sample, mpi_size, OPIC_DATA_TYPE, 0, 2, MPI_COMM_WORLD);
    }else{
        //  sample_of_rank0[i] = my_sample[i];
        memcpy(sample_of_rank0, my_sample, row_size * mpi_size);
        printf("First 5 samples at root :");
        if (mpi_size >= 5) {
            sample_size = 5;
        } else {
            sample_size = mpi_size;
        }
        for (i=0; i< sample_size; i++) {
            printf("%f ", get_value_double(key_index, my_sample + i*row_size));
        }
        printf("\n");
    }

    free(my_sample);
    return 0;
}


/******************************************************************************
 * Phase 2 of the parallel sampling sorting
 * 1, Receive the pilots from the master
 * 2, Exchange the data 
 * 3, Sort the data again 
 ******************************************************************************/
char *phase2(int mpi_rank, int mpi_size, char *data, int64_t my_data_size,
        char *pivots, int rest_size, int row_size, int skew_data,
        int collect_data, int write_result, unsigned long long *rsize)
{
    int dest, k;
    double t1, t2; 
    int *scount;
    int ii, nelem, previous_ii;
    double dest_pivot, cur_value;
    char *final_buff;

    scount = malloc(mpi_size * sizeof(int));

    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime(); 	 

    for(k = 0; k < mpi_size; k++) {
        scount[k] = 0;
    }

    //Start from dest 0 (mpi_rank 0)
    dest = 0;

    previous_ii = 0;
    cur_value = 0.0;
    for(k = 0; k < mpi_size; k++){
        if(dest != (mpi_size -1)){
            dest_pivot = get_value_double(key_index, pivots + dest*row_size);
        }else{
            dest_pivot = higest_double;
        }
        nelem = 0;

        //Does the data is skewed 
        if(skew_data == 1){ 
            skewed_data_partition(mpi_rank, mpi_size, data,
                    my_data_size, pivots, row_size, dest_pivot,
                    dest, &cur_value, &nelem, &previous_ii);
        }else{
            //Find the data starting from previous_ii;
            for (ii=previous_ii; ii < my_data_size; ii++){
                cur_value = get_value_double(key_index, data+ii*row_size);
                previous_ii = ii;

                if(cur_value <= dest_pivot){
                    nelem++;
                }else{
                    break;
                }
            }
        }
        scount[dest] = nelem;
        dest = (dest + 1) % mpi_size;
        if(previous_ii+1 >= my_data_size && cur_value <= dest_pivot)
            break;
    }

    t2 = MPI_Wtime(); 	 

    if(mpi_rank == 0 || mpi_rank == (mpi_size -1)) {
        printf("Data partition ends, my_rank %d, ", mpi_rank);
        printf("taking [%f]s \n", (t2-t1));
    }

    /* echange the data and sort it again. */
    final_buff = exchange_data(mpi_rank, mpi_size, data, scount, my_data_size,
            row_size, collect_data, write_result, rsize);

    free(scount);

    return final_buff;
}

/******************************************************************************
 * Set external variables in this file.
 ******************************************************************************/
void set_external_variables(int type_size_max, int index_key, int dset_num,
        int key_data_type, int verbosity, int omp_threaded, int omp_threads_num,
        char *gname, char *fname_sorted, char *fname_attribute,
        dset_name_item *dataname_array)
{
    /* external variables */
    max_type_size = type_size_max;
    key_index = index_key;
    dataset_num = dset_num;
    key_value_type = key_data_type;
    verbose = verbosity;
    local_sort_threaded = omp_threaded;
    local_sort_threads_num = omp_threads_num;
    group_name = (char *)malloc(NAME_MAX * sizeof(char));
    strcpy(group_name, gname);
    filename_sorted = (char *)malloc(NAME_MAX * sizeof(char));
    strcpy(filename_sorted, fname_sorted);
    filename_attribute = (char *)malloc(NAME_MAX * sizeof(char));
    strcpy(filename_attribute, fname_attribute);
    dname_array = (dset_name_item *)malloc(MAX_DATASET_NUM * sizeof(dset_name_item));
    memcpy(dname_array, dataname_array, MAX_DATASET_NUM * sizeof(dset_name_item));
}

/******************************************************************************
 * Free external variables in this file.
 ******************************************************************************/
void free_external_variable()
{
    free(group_name);
    free(filename_sorted);
    free(filename_attribute);
    free(dname_array);
}

/******************************************************************************
 * Master does slave's job, and also gather and sort pivots
 ******************************************************************************/
char *master(int mpi_rank, int mpi_size, char *data, int64_t my_data_size,
        int rest_size, int row_size, int type_size_max, int index_key,
        int dset_num, int key_data_type, int verbosity, int omp_threaded,
        int omp_threads_num, int skew_data, int collect_data, int write_result,
        char *gname, char *fname_sorted, char *fname_attribute,
        dset_name_item *dataname_array, unsigned long long *rsize)
{
    char *all_samp, *temp_samp, *final_buff;
    char *pivots;
    double t1, t2; 
    int i;
    MPI_Status Stat;

    set_external_variables(type_size_max, index_key, dset_num, key_data_type,
            verbosity, omp_threaded, omp_threads_num, gname, fname_sorted,
            fname_attribute, dataname_array);

    /* All samples */
    all_samp  = malloc(mpi_size * mpi_size * row_size);
    temp_samp = malloc(mpi_size * row_size);
    pivots    = malloc((mpi_size - 1)*row_size);

    t1 = MPI_Wtime();
    printf("Phase1 is running .... \n"); 
    /* Sort its own data, and select samples */
    phase1(0, mpi_size, data, my_data_size, temp_samp, row_size);
    t2 = MPI_Wtime();
    printf("Master's phase1 taking [%f]s \n", (t2-t1));

    memcpy(all_samp, temp_samp, mpi_size * row_size);

    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    printf("Master receives samples ....\n");
    //Gather sample from slaves
    for(i = 1; i < mpi_size; i++){
        //printf("rank 0 -> I am waiting %d \n", i);
        MPI_Recv(temp_samp, mpi_size, OPIC_DATA_TYPE, MPI_ANY_SOURCE,
                2, MPI_COMM_WORLD, &Stat);
        //for(j = 0; j < mpi_size; j++){
        //all_samp[i*mpi_size + j] = temp_samp[j];
        memcpy(all_samp + i * mpi_size * row_size, temp_samp,  mpi_size * row_size);
        //}
    }
    free(temp_samp);
    t2 = MPI_Wtime();
    printf("Receiving samples taks [%f]s \n", (t2-t1));

    //Sort samples and choose pivots
    qsort_type(all_samp, (mpi_size * mpi_size), row_size);

    if(verbose){
        printf("All samples: ");
        for(i = 0; i < mpi_size * mpi_size; i++){
            printf("%lf, ", get_value_double(key_index, all_samp + row_size*i));
        }
        printf("\n");
    }

    int rou;
    double previous_pivot = lowest_double, cur_pivot;
    rou = mpi_size / 2;
    printf("Pivots : ");
    for(i = 0; i < (mpi_size-1); i++){
        memcpy(pivots+row_size*i, all_samp + (mpi_size*(i+1) + rou -1)*row_size, row_size);
        cur_pivot = get_value_double(key_index,   pivots+row_size*i);
        if(previous_pivot ==  cur_pivot && i < 60){
            printf(" (Same pivot)");
        }
        previous_pivot = cur_pivot;
        if(i < 60)
            printf("%f, ", cur_pivot);
    }
    printf("\n");
    free(all_samp);

    //To all slaves
    MPI_Bcast(pivots, (mpi_size-1), OPIC_DATA_TYPE, 0, MPI_COMM_WORLD);

    printf("Phase2 is running... \n");
    //To sort and write sorted file

    final_buff = phase2(0, mpi_size, data, my_data_size, pivots, rest_size,
            row_size, skew_data, collect_data, write_result, rsize);
    free(pivots);

    free_external_variable();

    return final_buff;
}

/******************************************************************************
 * Do sort and sample
 ******************************************************************************/
char *slave(int mpi_rank, int mpi_size, char *data, int64_t my_data_size,
        int rest_size, int row_size, int type_size_max, int index_key,
        int dset_num, int key_data_type, int verbosity, int omp_threaded,
        int omp_threads_num, int skew_data, int collect_data, int write_result,
        char *gname, char *fname_sorted, char *fname_attribute,
        dset_name_item *dataname_array, unsigned long long *rsize)
{
    char  *pivots;
    char *final_buff;
    set_external_variables(type_size_max, index_key, dset_num, key_data_type,
            verbosity, omp_threaded, omp_threads_num, gname, fname_sorted,
            fname_attribute, dataname_array);

    strcpy(group_name, gname);
    strcpy(filename_sorted, fname_sorted);
    strcpy(filename_attribute, fname_attribute);
    memcpy(dname_array, dataname_array, sizeof(dset_name_item));

    phase1(mpi_rank, mpi_size, data, my_data_size, NULL, row_size);

    //Receive pivots from masters
    pivots = malloc((mpi_size-1) * row_size);
    MPI_Bcast(pivots, (mpi_size-1), OPIC_DATA_TYPE, 0, MPI_COMM_WORLD);

    final_buff = phase2(mpi_rank, mpi_size, data, my_data_size, pivots,
            rest_size, row_size, skew_data, collect_data, write_result,
            rsize);
    free(pivots);

    free_external_variable();

    return final_buff;
}

/******************************************************************************
 * Sort the data based on the type
 ******************************************************************************/
int qsort_type(void *data, int64_t my_data_size, size_t row_size){
    switch(key_value_type){
        case H5GS_INT64:
            qsort(data, my_data_size, row_size, CompareInt64Key);
            break;
        case H5GS_INT32:
            qsort(data, my_data_size, row_size, CompareInt32Key);
            break;
        case H5GS_FLOAT32:
            qsort(data, my_data_size, row_size, CompareFloat32Key);
            break;
        case H5GS_FLOAT64:
            qsort(data, my_data_size, row_size, CompareFloat64Key);
            break;
        default:
            printf("Un-supported data type ! \n");
            exit(-1);
            break;
    }
    return 0;
}

/******************************************************************************
 ******************************************************************************/
int pivots_replicated(char *pivots, int dest, int *dest_pivot_replicated_size,
        int *dest_pivot_replicated_rank, int mpi_size, int mpi_rank,
        int row_size, double *p_value_head){
    int replicated = 0;
    int replicated_size = 1; //At least have one 
    double dest_pivot;
    if(dest != (mpi_size - 1)){
        dest_pivot= get_value_double(key_index, pivots + dest*row_size);
    }else{
        dest_pivot = higest_double;
    }

    double next_pivot;
    int i;
    i = dest - 1;
    while(i >= 0){
        if(i != (mpi_size - 1)){
            next_pivot = get_value_double(key_index, pivots + i*row_size);
        }else{
            next_pivot = higest_double;
        }

        if(next_pivot == dest_pivot){
            i--;
            replicated = 1;
            replicated_size++;
        }else{
            break;
        }
    }

    if(i >= 0){
        *p_value_head = next_pivot;
    }else{
        *p_value_head = lowest_double;
    }

    //Rank starts from 0
    *dest_pivot_replicated_rank = replicated_size - 1;

    i= dest + 1;
    while(i < mpi_size){
        if(i != (mpi_size - 1)){
            next_pivot = get_value_double(key_index, pivots + i*row_size);
        }else{
            next_pivot = higest_double;
        }
        if(next_pivot == dest_pivot){
            i++;
            replicated = 1;
            replicated_size++;
        }else{
            break;
        }
    }

    *dest_pivot_replicated_size = replicated_size;

    return replicated;
}

/******************************************************************************
 ******************************************************************************/
void rank_pivot(char *pivots, char *data,  int64_t my_data_size,
        int dest_pivot, int *rank_less, int *rank_equal, int row_size,
        int mpi_size, double p_value_head){
    double dest_pivot_value;    
    if(dest_pivot != (mpi_size - 1)){
        dest_pivot_value = get_value_double(key_index, pivots+dest_pivot*row_size);
    }else{
        dest_pivot_value = higest_double;
    }
    int    less_count = 0 , equal_count = 0;
    double cur_value;

    int i;
    for (i = 0 ; i < my_data_size; i++){
        cur_value = get_value_double(key_index, data+i*row_size);

        if(cur_value < dest_pivot_value && cur_value > p_value_head ){
            less_count++;
        }

        if(cur_value == dest_pivot_value){
            equal_count++;
        }

        if( cur_value > dest_pivot_value){
            break;
        }
    }

    *rank_less  =  less_count;
    *rank_equal =  equal_count;

}

/******************************************************************************
 * Merge sorted lists A and B into list A.  A must have dim >= m+n
 ******************************************************************************/
void merge(char *A, char *B, int m, int n, int row_size) 
{
    int i = 0, j = 0, k = 0;
    int size = m + n;
    double v1, v2;

    char *C = (char *)malloc(size * row_size);
    v1 = getFloat64Value(key_index, A+i*row_size);
    v2 = getFloat64Value(key_index, B+j*row_size);
    while (i < m && j < n) {
        if (v1 <= v2){
            memcpy(C+k*row_size, A+i*row_size, row_size);
            i++;
            v1 = getFloat64Value(key_index, A+i*row_size);
        }else {
            memcpy(C+k*row_size, B+i*row_size, row_size);
            j++;
            v2 = getFloat64Value(key_index, B+j*row_size);
        }
        k++;
    }

    if (i < m){
        //for (p = i; p < m; p++,k++){
        //  memcpy(C+k*row_size, A+p*row_size, row_size);
        //}
        memcpy(C+k*row_size, A+i*row_size, (m-i)*row_size);
    }else{
        //for(p = j; p < n; p++,k++){
        //  memcpy(C+k*row_size, B+p*row_size, row_size);
        //}
        memcpy(C+k*row_size, B+j*row_size, (n-j)*row_size);
    }

    //for( i=0; i<size; i++ ) 
    //  memcpy(A+i*row_size, C+i*row_size, row_size);
    memcpy(A, C, size * row_size);
    free(C);
}

/******************************************************************************
 * Merges N sorted sub-sections of array a into final, fully sorted array a
 ******************************************************************************/
void arraymerge(char *a, int size, int *index, int N, int row_size)
{
    /*
    int i, thread_size; 

    while(N>1){
        thread_size = size/N; //Check (size % N != 0)
        for( i=0; i< N; i++ ){ 
            index[i]=i * thread_size; 
        }
        index[N]=size;

#pragma omp parallel for private(i) 

        for( i=0; i<N; i+=2 ) {
            merge(a+(index[i]*row_size), a+(index[i+1]*row_size), 
                    index[i+1]-index[i], index[i+2]-index[i+1], row_size);
        }
        N /= 2;
    }
    */
}

/******************************************************************************
 * Quicksort using openmp
 ******************************************************************************/
int openmp_sort(char *data, int size, int threads, size_t row_size)
{
    /*
    int i;
    omp_set_num_threads(threads);

    //int threads = mic_threads * mpi_size;
    //printf("Using [%d] threads each node \n", threads);

    //omp_set_num_threads(threads);
    int *index       = (int *)malloc((threads+1)*sizeof(int));
    int  thread_size = size/threads; //We might comeback to check data_size%threads
    printf(" size of each thread %d \n", thread_size);

    for(i=0; i<threads; i++){
    index[i]=i * thread_size; 
    printf("%d ", index[i]);
    }
    index[threads]=size;
    printf("%d (index)\n", index[threads]);


    // Main parallel sort loop 
    double start = omp_get_wtime();

#pragma omp parallel for private(i)

    for(i=0; i<threads; i++){ 
        //qsort(a+index[i], index[i+1]-index[i], sizeof(int), CmpInt);
        qsort_type(data+(index[i]*row_size), index[i+1]-index[i], row_size);
        //qsort(data+index[i], index[i+1]-index[i], row_size, CompareInt32Key);
    }

    //printf("Sorting is done ! \n");
    double middle = omp_get_wtime();

    // Merge sorted array pieces 
    if(threads>1 ) 
    arraymerge(data, size, index, threads, row_size);

    double end = omp_get_wtime();
    printf("sort time = %g s, ", end - start);
    printf("of which %g is sort time , %g s is merge time\n", middle-start, end-middle);

    */
    return 0;
}

/******************************************************************************
 * Skewed data partition.
 ******************************************************************************/
int skewed_data_partition(int mpi_rank, int mpi_size, char *data,
        int64_t my_data_size, char *pivots, int row_size, double dest_pivot,
        int dest, double *cur_value, int *nelem, int *previous_ii) {
    int dest_pivot_replicated = 0;
    int dest_pivot_replicated_size;
    int dest_pivot_replicated_rank;
    double p_value_head;
    int rank_equal, rank_less, partition_size;
    double my_pivot;
    int ii;

    if(mpi_rank != (mpi_size -1)){
        my_pivot = get_value_double(key_index, pivots + mpi_rank *row_size);;
    }else{
        my_pivot = higest_double; //This is the maximum value
    }

    //Send data (<< dest_pivot) to my self
    if(dest_pivot == my_pivot && mpi_rank == dest){
        //Find the data starting from previous_ii;
        for (ii = *previous_ii; ii < my_data_size; ii++){
            *cur_value = get_value_double(key_index, data+ii*row_size);
            //The data is sorted and we could skip when it bigger values appear.
            *previous_ii =  ii;
            if(*cur_value <= dest_pivot){
                (*nelem)++;
            }else{
                break;
            }
        }
    }else if (dest_pivot == my_pivot  && mpi_rank !=  dest ){
        //Send nothing to the one who has same pivot as me
        *nelem = 0;
    }else{
        //"dest_pivot != my_pivot";
        dest_pivot_replicated = pivots_replicated(pivots, dest,
                &dest_pivot_replicated_size, &dest_pivot_replicated_rank,
                mpi_size, mpi_rank, row_size, &p_value_head);
        if(dest_pivot_replicated == 0){
            //Find the data starting from previous_ii;
            if(verbose && mpi_rank == 1 ) //|| mpi_rank  == (mpi_size -1))
                printf("I am here: %d , %f \n", *previous_ii, dest_pivot);

            for (ii = *previous_ii; ii < my_data_size; ii++){
                *cur_value = get_value_double(key_index, data+ii*row_size);
                *previous_ii =  ii;
                //The data is sorted and we could skip when it bigger values appear.
                if(*cur_value <= dest_pivot){
                    //memcpy(send_buff+nelem*row_size, data+ii*row_size, row_size);
                    (*nelem)++;
                }else{
                    break;
                }
            }
        }else{
            rank_pivot(pivots, data, my_data_size, dest, &rank_less,
                    &rank_equal, row_size, mpi_rank, p_value_head);

            if(rank_less == 0){
                //replciated pivots are the smallest values. do equally parition the data
                if(rank_equal % dest_pivot_replicated_size == 0){
                    partition_size = rank_equal / dest_pivot_replicated_size;
                }else{
                    if(dest_pivot_replicated_rank == (dest_pivot_replicated_size - 1)) {
                        partition_size = rank_equal / dest_pivot_replicated_size +
                            rank_equal % dest_pivot_replicated_size;
                    }else{
                        partition_size = rank_equal / dest_pivot_replicated_size;
                    }
                }
            }else{
                if(dest_pivot_replicated_rank == 0){
                    if(rank_less < rank_equal/(dest_pivot_replicated_size - 1)){
                        partition_size = (rank_equal + rank_less)/(dest_pivot_replicated_size);
                    }else{
                        partition_size = rank_less;
                    }
                }else{
                    if(dest_pivot_replicated_rank != (dest_pivot_replicated_size - 1)){
                        if(rank_less < rank_equal/(dest_pivot_replicated_size - 1)){
                            partition_size = (rank_equal + rank_less)/(dest_pivot_replicated_size);
                        }else{
                            partition_size = rank_equal / dest_pivot_replicated_size;
                        }
                    }else{
                        if(rank_less < rank_equal/(dest_pivot_replicated_size - 1)){
                            partition_size = (rank_equal + rank_less)/(dest_pivot_replicated_size)
                                + (rank_equal + rank_less)%(dest_pivot_replicated_size);
                        }else{
                            partition_size = rank_equal / dest_pivot_replicated_size +
                                rank_equal % dest_pivot_replicated_size;
                        }
                    }
                }
            }
            if(mpi_rank == 0) {
                printf("rank less %d, ", rank_less);
                printf("rank equal %d, ", rank_equal);
                printf("partition_size %d, ", partition_size);
                printf("(my rank %d, dest = %d, ", mpi_rank, dest);
                printf("p_value_head %f, dest_pivot %f ) \n", p_value_head, dest_pivot);
            }

            for (ii = *previous_ii ; ii < my_data_size; ii++){
                *cur_value = get_value_double(key_index, data+ii*row_size);
                *previous_ii =  ii;

                if(*nelem > partition_size)
                    break;

                //When the dest pivot is one of replicated keys
                if(*cur_value <= dest_pivot ){
                    (*nelem)++;
                }else{
                    break;
                }
            }
        }//end of replicated = 1
    }
    return 0;
}

/******************************************************************************
 * Load balance check.
 ******************************************************************************/
void check_load_balance(int mpi_rank, int mpi_size, unsigned long long rsize) {
    unsigned long long load, sum_load, sq_load, sumsq_load, max_load, min_load;
    double var_load;
    load = rsize;
    MPI_Allreduce(&load, &max_load, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&load, &min_load, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&load, &sum_load, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    sq_load = load - (sum_load / mpi_size);
    sq_load = sq_load * sq_load;
    MPI_Allreduce(&sq_load, &sumsq_load, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if(mpi_size > 1) {
        var_load  = sumsq_load / (mpi_size-1);
    }else {
        var_load = 0;
    }

    if(mpi_rank == 0 || mpi_rank == (mpi_size -1)) {
        printf("(max %llu, min %llu, var %f) \n", max_load, min_load, var_load);
    }
}

/******************************************************************************
 * Exchange the data among all processes.
 ******************************************************************************/
char *exchange_data(int mpi_rank, int mpi_size, char *data, int *scount,
        int64_t my_data_size, int row_size, int collect_data, int write_result,
        unsigned long long *rsize)
{
    int *sdisp, *rcount, *rdisp;
    unsigned long long ssize = 0;
    double t1, t2;
    char *final_buff;
    int i;

    t1 = MPI_Wtime();

    *rsize = 0;

    rcount = malloc(mpi_size * sizeof(int));
    sdisp  = malloc(mpi_size * sizeof(int));
    rdisp  = malloc(mpi_size * sizeof(int));

    //Figure out all-to-all vector parameters
    MPI_Alltoall(scount, 1, MPI_INT, rcount, 1,MPI_INT, MPI_COMM_WORLD);
    if(mpi_rank == 0)
        printf("Done for assigning data based on pivots !\n");

    sdisp[0] = 0;
    for(i = 1; i < mpi_size; i++){
        sdisp[i] = sdisp[i-1] + scount[i-1];
        if(verbose && mpi_rank == 1 ){ //|| mpi_rank  == (mpi_size -1))
            printf("mpi rank (%d) scount[%d]   %d rcount[%d] %d \n ",
                    mpi_rank, i-1,  scount[i-1], i-1, rcount[i-1]);
        }
    }

    rdisp[0] = 0;
    for(i = 1; i < mpi_size; i++)
        rdisp[i] = rdisp[i-1] + rcount[i-1];

    for (i = 0; i < mpi_size; i++){
        *rsize = *rsize + (unsigned long long)rcount[i];
        ssize = ssize + (unsigned long long)scount[i];
    }

    if(ssize != my_data_size){
        printf("At rank %d, ", mpi_rank);
        printf("Size mismatch after assigning on pivots: ");
        printf("send size (%lld), my_data_size (%ld) ! \n", ssize, my_data_size);
        exit(-1);
    }

    if(collect_data == 1){
        final_buff = malloc((*rsize) * row_size);  
        if(final_buff == NULL){
            printf("Allocation for final_buff fails !\n");
            exit(0);
        }
        MPI_Alltoallv(data, scount, sdisp, OPIC_DATA_TYPE, final_buff,
                rcount, rdisp, OPIC_DATA_TYPE, MPI_COMM_WORLD);
    } else {
        final_buff = malloc(row_size);  
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();

    if(mpi_rank == 0 || mpi_rank == (mpi_size -1)) {
        printf("Exchange data ends, my_rank %d, ", mpi_rank);
        printf("orgi_size %ld, new_size %llu, ", my_data_size, *rsize);
        printf("taking [%f]s\n", (t2-t1));
    }

    free(rdisp);
    free(sdisp);
    free(rcount);

    check_load_balance(mpi_rank, mpi_size, *rsize);

    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();

    if(collect_data == 1){
        if(local_sort_threaded == 0){
            qsort_type(final_buff, *rsize, row_size);
        }else{
            openmp_sort(final_buff, *rsize, local_sort_threads_num, row_size);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    if(mpi_rank == 0 ){
        printf("Sorting aftering takes [%f]s)\n", (t2-t1));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();

    /* For test, we only consider the sorting time */
    if(write_result == 1) {
        write_result_file(mpi_rank, mpi_size, final_buff, *rsize, row_size,
                dataset_num, max_type_size, key_index, group_name,
                filename_sorted, filename_attribute, dname_array);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    if(mpi_rank == 0 ){
        printf("Write result file takes [%f]s)\n", (t2-t1));
    }
    return final_buff;
}

/******************************************************************************
 * VPIC data type in bytes.
 ******************************************************************************/
void create_opic_data_type(int row_size)
{
    MPI_Type_contiguous(row_size, MPI_BYTE, &OPIC_DATA_TYPE);
    MPI_Type_commit(&OPIC_DATA_TYPE);
}

/******************************************************************************
 * free VPIC data type.
 ******************************************************************************/
void free_opic_data_type()
{
    free(OPIC_DATA_TYPE);
}
