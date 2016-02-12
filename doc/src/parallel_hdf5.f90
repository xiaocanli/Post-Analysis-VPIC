!*******************************************************************************
! Module for reading and writing HDF5 file in parallel
!*******************************************************************************
program parallel_hdf5
    use constants, only: fp, dp
    use analysis_management, only: init_analysis, end_analysis
    use mpi_module
    use hdf5
    implicit none

    call init_analysis

    integer(hid_t) :: file_id
    integer(hid_t), allocatable, dimension(:) :: group_id, dset_id
    integer(hid_t) :: filespace, memspace, plist_id
    integer(hsize_t), allocatable, dimension(:) :: dset_dims, dcount, offset
    integer :: rank
    character(len=256) :: filename, groupname

    filename = "../../../tracer/T.0/electron_tracer.h5p"
    groupname = "/Step#0"

    call open_hdf5_parallel(filename, groupname)

    call end_analysis

    contains

    !---------------------------------------------------------------------------
    ! Open hdf5 file in parallel
    !---------------------------------------------------------------------------
    subroutine open_hdf5_parallel(filename, groupname)
        use mpi_info_module, only: fileinfo
        implicit none
        character(*), intent(in) :: filename, groupname
        integer :: error
        call h5open_f(error)
        call h5pcreate_f(H5P_FILE_ACCESS_F, plist_id, error)
        call h5pset_fapl_mpio_f(plist_id, MPI_COMM_WORLD, info, error)
        call h5fopen_f (filename, H5F_ACC_RDWR_F, file_id, error, &
            access_prp=plist_id)
        call h5pclose_f(plist_id, error)
        call h5fclose_f(file_id, error)
    end subroutine open_hdf5_parallel

    !---------------------------------------------------------------------------
    ! Read hdf5 file in parallel
    !---------------------------------------------------------------------------
    subroutine read_hdf5_parallel
        implicit none
    end subroutine read_hdf5_parallel
end program parallel_hdf5
