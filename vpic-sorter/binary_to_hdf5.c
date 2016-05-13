/* 
 *
 * This transfers the particle trajectories in binary format to HDF5 format.
 * 
 */

#include "stdlib.h"
#include "hdf5.h"
#include <unistd.h>
#include <string.h>
#include <float.h>
#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <getopt.h>
#include "constants.h"
#include "mpi_io.h"
#include "get_data.h"

int get_configuration(int argc, char **argv, int mpi_rank, char *fpath_binary,
        char *fpath_hdf5, char *species, int *tmax, int *tmin, int *tinterval,
        int *ncpus, int *dataset_num);
void print_help(void);
void read_header(FILE *fp, int *nptl, float *q_m, double *t);
dset_name_item *set_dname_array_13(int dataset_num);
dset_name_item *set_dname_array_8(int dataset_num);
void set_jobs(int mpi_rank, int mpi_size, int njobs_tot, int *njobs, int *offset);
void save_np_pic(int *np_local, char *fpath_hdf5, char *species, int ncpus,
        int tstep);
void get_package_data(int njobs, int offset, char *fpath_binary,
        char *fname_binary, char *species, int tstep, int row_size,
        char *package_data);

/******************************************************************************
 * Main program to transfer the data
 ******************************************************************************/
int main(int argc, char **argv)
{
    int mpi_size, mpi_rank, ncpus;
    char *fpath_binary, *fpath_hdf5, *species, *fname_binary, *fname_hdf5;
    char *group_name;
    int tmax, tmin, tinterval, tstep;
    int dataset_num;
    double t0, t1;

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(comm, &mpi_size);
    MPI_Comm_rank(comm, &mpi_rank);

    t0 = MPI_Wtime();

    fpath_binary = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    fpath_hdf5 = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    fname_binary = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    fname_hdf5 = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    species = (char *)malloc(32 * sizeof(char));
    group_name = (char *)malloc(32 * sizeof(char));

    get_configuration(argc, argv, mpi_rank, fpath_binary, fpath_hdf5,
        species, &tmax, &tmin, &tinterval, &ncpus, &dataset_num);

    /* The variables are required to use the sorting procedures for HDF5 */
    dset_name_item *dname_array;
    int max_type_size, row_size;
    max_type_size = 4;
    row_size = dataset_num * max_type_size;
    if (dataset_num == 13) {
        dname_array = set_dname_array_13(dataset_num);
    } else {
        dname_array = set_dname_array_8(dataset_num);
    }

    int nptl;
    float q_m;
    double t;
    FILE *fp;
    int *np_local; // number of particles in each PIC mpi_rank
    int *np_all;   // for all PIC mpi_ranks
    int njobs, offset;
    long int nptl_tot;  // Total number of particles for current mpi_rank
    
    // Not actually necessary, but it is easies to use the existing code
    int key_index, key_value_type;
    key_index = 0;
    key_value_type = H5GS_FLOAT32;

    set_jobs(mpi_rank, mpi_size, ncpus, &njobs, &offset);
    np_local = (int *)malloc(sizeof(int) * njobs);
    np_all = (int *)malloc(sizeof(int) * ncpus);
    for (tstep = tmin; tstep <= tmax; tstep += tinterval) {
        if (mpi_rank == 0) printf("%d\n", tstep);
        for (int i = 0; i < njobs; i++) {
            np_local[i] = 0;
        }
        for (int i = 0; i < ncpus; i++) {
            np_all[i] = 0;
        }
        nptl_tot = 0;
        // Get the particle number for each PIC mpi_rank
        for (int icpu = offset; icpu < offset+njobs; icpu++) {
            snprintf(fname_binary, MAX_FILENAME_LEN, "%s%s%d%s%s%s%d",
                    fpath_binary, "/T.", tstep, "/", species, "_tracer.", icpu);
            fp = fopen(fname_binary, "rb");
            read_header(fp, &nptl, &q_m, &t);
            nptl_tot += nptl;
            np_local[icpu-offset] = nptl;
            fclose(fp);
        }
        // Gather and save the particle number for each PIC mpi_rank
        MPI_Gather(np_local, njobs, MPI_INT, np_all, njobs, MPI_INT, 0,
                MPI_COMM_WORLD);
        if (mpi_rank == 0) {
            save_np_pic(np_all, fpath_hdf5, species, ncpus, tstep);
        }
        // Get the tracer data
        char *package_data = (char *)malloc(row_size * nptl_tot);
        get_package_data(njobs, offset, fpath_binary, fname_binary, species,
                tstep, row_size, package_data);
        hsize_t my_data_size = nptl_tot;

        // Not actually necessary, but it is easies to use the existing code
        set_variable_data(max_type_size, 0, dataset_num, key_value_type, 0);
        snprintf(fname_hdf5, MAX_FILENAME_LEN, "%s%s%d%s%s%s",
                fpath_hdf5, "/T.", tstep, "/", species, "_tracer.h5p");
        snprintf(group_name, 32, "%s%d", "/Step#", tstep);

        write_result_file(mpi_rank, mpi_size, package_data, my_data_size,
                row_size, dataset_num, max_type_size, key_index, group_name,
                fname_hdf5, "attribute", dname_array);
        free(package_data);
    }

    free(np_all);
    free(np_local);

    free(dname_array);
    free(species);
    free(group_name);
    free(fname_hdf5);
    free(fname_binary);
    free(fpath_hdf5);
    free(fpath_binary);

    MPI_Barrier(comm);
    t1 = MPI_Wtime();
    if(mpi_rank == 0) {
        printf("Overall time is [%f]s \n", (t1 - t0));
    }

    MPI_Finalize();
    return 0;
}

/******************************************************************************
 * Save the number of particles in each PIC mpi_rank
 ******************************************************************************/
void save_np_pic(int *np_local, char *fpath_hdf5, char *species, int ncpus,
        int tstep)
{
    char *fname_meta, *group_name;
    hsize_t dimsf[1], count[1], offset[1];
    hid_t file_id, group_id, dset_id, filespace, memspace;
    int rank;
    fname_meta = (char *)malloc(MAX_FILENAME_LEN * sizeof(char));
    group_name = (char *)malloc(32 * sizeof(char));
    snprintf(fname_meta, MAX_FILENAME_LEN, "%s%s%d%s%s%s",
            fpath_hdf5, "/T.", tstep, "/grid_metadata_", 
            species, "_tracer.h5p");
    snprintf(group_name, 32, "%s%d", "/Step#", tstep);
    file_id = H5Fcreate(fname_meta, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    group_id = H5Gcreate2(file_id, group_name, H5P_DEFAULT, H5P_DEFAULT,
            H5P_DEFAULT);

    rank = 1;
    dimsf[0] = ncpus;
    count[0] = ncpus;
    offset[0] = 0;
    filespace = H5Screate_simple(rank, dimsf, NULL);
    memspace = H5Screate_simple(rank, count, NULL);
    dset_id = H5Dcreate2(group_id, "np_local", H5T_NATIVE_INT, filespace,
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);
    H5Dwrite(dset_id, H5T_NATIVE_INT, memspace, filespace,
            H5P_DEFAULT, np_local);
    H5Sclose(filespace);
    H5Sclose(memspace);
    H5Dclose(dset_id);
    H5Gclose(group_id);
    H5Fclose(file_id);
    free(group_name);
    free(fname_meta);
}

/******************************************************************************
 * Set number of jobs for each mpi_rank
 ******************************************************************************/
void set_jobs(int mpi_rank, int mpi_size, int njobs_tot, int *njobs, int *offset)
{
    int nleft;
    *njobs = njobs_tot / mpi_size;
    *offset = (*njobs) * mpi_rank;
    nleft = njobs_tot % mpi_size;
    if (mpi_rank < nleft) {
        (*njobs) += 1;
        (*offset) += mpi_rank;
    } else {
        (*offset) += nleft;
    }
}

/******************************************************************************
 * Read the header, including
 * nptl: number of particles in this file
 * q_m: the charge to mass ratio
 * t: current simulation time for this time step
 ******************************************************************************/
void read_header(FILE *fp, int *nptl, float *q_m, double *t)
{
    float tmpf;
    double tmpd;
    fread(nptl, sizeof(int), 1, fp);
    fread(q_m, sizeof(float), 1, fp);
    fread(t, sizeof(double), 1, fp);
    fread(&tmpf, sizeof(float), 1, fp);
    fread(&tmpd, sizeof(double), 1, fp);
    /* printf("%d %f %f\n", *nptl, *q_m, *t); */
}

/******************************************************************************
 * Get the analysis configuration.
 ******************************************************************************/
int get_configuration(int argc, char **argv, int mpi_rank, char *fpath_binary,
        char *fpath_hdf5, char *species, int *tmax, int *tmin, int *tinterval,
        int *ncpus, int *dataset_num)
{
    int c;
    static const char *options="e:s:i:f:p:n:d:m:";
    static struct option long_options[] = 
    {
        {"tmax", required_argument, 0, 'e'},
        {"tmin", required_argument, 0, 's'},
        {"tinterval", required_argument, 0, 'i'},
        {"fpath_binary", required_argument, 0, 'f'},
        {"fpath_hdf5", required_argument, 0, 'd'},
        {"species", required_argument, 0, 'p'},
        {"ncpus", required_argument, 0, 'n'},
        {"dataset_num", required_argument, 0, 'm'},
        {0, 0, 0, 0},
    };
    /* getopt_long stores the option index here. */
    int option_index = 0;
    extern char *optarg;

    /* Default values */
    *ncpus = 16;
    *tmin = 0;
    *tmax = 12;
    *tinterval = 6;
    *ncpus = 256;
    *dataset_num = 8;

    while ((c = getopt_long (argc, argv, options, long_options, &option_index)) != -1){
        switch (c){
            case 'e':
                *tmax = atoi(optarg);
                break;
            case 's':
                *tmin = atoi(optarg);
                break;
            case 'i':
                *tinterval = atoi(optarg);
                break;
            case 'f':
                strcpy(fpath_binary, optarg);
                break;
            case 'd':
                strcpy(fpath_hdf5, optarg);
                break;
            case 'p':
                strcpy(species, optarg);
                break;
            case 'n':
                *ncpus = atoi(optarg);
                break;
            case 'm':
                *dataset_num = atoi(optarg);
                break;
            case 'h':
                if (mpi_rank == 0) {
                    print_help();
                }
                return 1;
            default:
                printf("Error option [%s]\n", optarg);
                exit(-1);
        }
    }
    return 0;
}

/******************************************************************************
 * Print help information.
 ******************************************************************************/
void print_help(void)
{
    char *msg="Usage: %s [OPTION] \n\
               -h help (--help)\n\
               -e the particle output maximum time step \n\
               -s the particle output minimum time step \n\
               -i the particle output time interval \n\
               -n number of CPUs to use \n\
               -n number of datasets \n\
               -f file path of the binary data \n\
               -f file path to save the HDF5 data \n\
               -p  particle species for sorting \n";
    fprintf(stdout, msg, "binary_to_hdf5");
}

/******************************************************************************
 * Set dname_array. This is based on looking into the data directly. It is
 * not generated directly. This function assumes there are 8 datasets and they
 * are in order as described below.
 ******************************************************************************/
dset_name_item *set_dname_array_8(int dataset_num)
{
    dset_name_item *dname_array;
    dname_array = (dset_name_item *)malloc(MAX_DATASET_NUM * sizeof(dset_name_item));
    snprintf(dname_array[0].dataset_name, NAME_MAX, "%s", "Ux");
    snprintf(dname_array[1].dataset_name, NAME_MAX, "%s", "Uy");
    snprintf(dname_array[2].dataset_name, NAME_MAX, "%s", "Uz");
    snprintf(dname_array[3].dataset_name, NAME_MAX, "%s", "dX");
    snprintf(dname_array[4].dataset_name, NAME_MAX, "%s", "dY");
    snprintf(dname_array[5].dataset_name, NAME_MAX, "%s", "dZ");
    snprintf(dname_array[6].dataset_name, NAME_MAX, "%s", "i");
    snprintf(dname_array[7].dataset_name, NAME_MAX, "%s", "q");

    dname_array[6].did = 0;
    dname_array[6].type_id = H5T_NATIVE_INT;
    dname_array[6].type_size = 4;
    dname_array[7].did = 0;
    dname_array[7].type_id = H5T_NATIVE_INT;
    dname_array[7].type_size = 4;

    for (int i = 0; i < 6; i++) {
        dname_array[i].did = 0;
        dname_array[i].type_id = H5T_NATIVE_FLOAT;
        dname_array[i].type_size = 4;
    }
    return dname_array;
}

/******************************************************************************
 * Set dname_array. This is based on looking into the data directly. It is
 * not generated directly. This function assumes there are 13 datasets.
 ******************************************************************************/
dset_name_item *set_dname_array_13(int dataset_num)
{
    dset_name_item *dname_array;
    dname_array = (dset_name_item *)malloc(MAX_DATASET_NUM * sizeof(dset_name_item));
    snprintf(dname_array[0].dataset_name, NAME_MAX, "%s", "q");
    snprintf(dname_array[1].dataset_name, NAME_MAX, "%s", "x");
    snprintf(dname_array[2].dataset_name, NAME_MAX, "%s", "y");
    snprintf(dname_array[3].dataset_name, NAME_MAX, "%s", "z");
    snprintf(dname_array[4].dataset_name, NAME_MAX, "%s", "ux");
    snprintf(dname_array[5].dataset_name, NAME_MAX, "%s", "uy");
    snprintf(dname_array[6].dataset_name, NAME_MAX, "%s", "uz");
    snprintf(dname_array[7].dataset_name, NAME_MAX, "%s", "ex");
    snprintf(dname_array[8].dataset_name, NAME_MAX, "%s", "ey");
    snprintf(dname_array[9].dataset_name, NAME_MAX, "%s", "ez");
    snprintf(dname_array[10].dataset_name, NAME_MAX, "%s", "cbx");
    snprintf(dname_array[11].dataset_name, NAME_MAX, "%s", "cby");
    snprintf(dname_array[12].dataset_name, NAME_MAX, "%s", "cbz");

    dname_array[0].did = 0;
    dname_array[0].type_id = H5T_NATIVE_INT;
    dname_array[0].type_size = 4;

    for (int i = 1; i < dataset_num; i++) {
        dname_array[i].did = 0;
        /* dname_array[i].type_id = H5T_IEEE_F32LE; */
        dname_array[i].type_id = H5T_NATIVE_FLOAT;
        dname_array[i].type_size = 4;
    }
    return dname_array;
}

/******************************************************************************
 * Set the filenames.
 ******************************************************************************/
void set_filenames(char *filepath, int tstep, char *group_name,
        char *filename_sorted, char *filename_attribute)
{
    snprintf(group_name, MAX_FILENAME_LEN, "%s%d", "/Step#", tstep);
    snprintf(filename_sorted, MAX_FILENAME_LEN, "%s%s", filepath,
            "/electron_tracer_sorted.h5p");
    snprintf(filename_attribute, MAX_FILENAME_LEN, "%s%s", filepath,
            "/attribute");
}

/******************************************************************************
 * Get package data of all the data in the tracer files.
 * We use char datatype for the package data
 ******************************************************************************/
void get_package_data(int njobs, int offset, char *fpath_binary,
        char *fname_binary, char *species, int tstep, int row_size,
        char *package_data)
{
    unsigned long long nptl_acc;
    int nptl;
    float q_m;
    double t;
    FILE *fp;
    char *data;

    nptl_acc = 0;
    for (int i = offset; i < offset+njobs; i++) {
        snprintf(fname_binary, MAX_FILENAME_LEN, "%s%s%d%s%s%s%d",
                fpath_binary, "/T.", tstep, "/", species, "_tracer.", i);
        fp = fopen(fname_binary, "rb");
        read_header(fp, &nptl, &q_m, &t);
        data = (char *)malloc(row_size*nptl*sizeof(char));
        fread(data, sizeof(char), row_size*nptl, fp);
        memcpy(package_data + nptl_acc*row_size, data, row_size*nptl);
        nptl_acc += nptl;
        free(data);
        fclose(fp);
    }
}
