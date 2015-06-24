!*******************************************************************************
! MPI data type for parallel I/O.
!*******************************************************************************
module mpi_datatype_test
    use mpi_module
    implicit none
    private
    public datatype, set_mpi_datatype

    integer :: datatype

    contains

    !---------------------------------------------------------------------------
    ! Create a MPI data type and commit it.    
    !---------------------------------------------------------------------------
    subroutine set_mpi_datatype(sizes, subsizes, starts)
        implicit none
        integer, dimension(:), intent(in) :: sizes, subsizes, starts
        integer :: sz

        sz = size(sizes)

        call MPI_TYPE_CREATE_SUBARRAY(sz, sizes, subsizes, starts, &
                MPI_ORDER_FORTRAN, MPI_REAL, datatype, ierror)
        call MPI_TYPE_COMMIT(datatype, ierror)
    end subroutine set_mpi_datatype

end module mpi_datatype_test


!*******************************************************************************
! Set MPI_INFO for parallel I/O.
!*******************************************************************************
module  mpi_info_module_test
    use mpi_module
    implicit none
    private
    public fileinfo, set_mpi_info
    integer :: fileinfo     ! MPI_INFO object

    contains

    !---------------------------------------------------------------------------
    ! Create a MPI_INFO object and have proper settings for ROMIO's data-sieving
    ! and collective buffering.
    !
    ! Outputs:
    !   fileinfo: the MPI_INFO.
    !---------------------------------------------------------------------------
    subroutine set_mpi_info
        implicit none
        call MPI_INFO_CREATE(fileinfo, ierror)
        !! Disable ROMIO's data-sieving
        !call MPI_INFO_SET(fileinfo, "romio_ds_read", "disable", ierror)
        !call MPI_INFO_SET(fileinfo, "romio_ds_write", "disable", ierror)
        !! Enable ROMIO's collective buffering
        !call MPI_INFO_SET(fileinfo, "romio_cb_read", "enable", ierror)
        !call MPI_INFO_SET(fileinfo, "romio_cb_write", "enable", ierror)

        ! For panfs parallel file system.
        call MPI_INFO_SET(fileinfo, "panfs_concurrent_write", "1", ierror)
    end subroutine set_mpi_info

end module  mpi_info_module_test


!*******************************************************************************
! This contains one subroutine to open field data file using MPI I/O, one
! subroutine to read data using MPI I/O.
!*******************************************************************************
module mpi_io_module_test
    use mpi_module
    implicit none
    private
    public open_data_mpi_io, read_data_mpi_io, write_data_mpi_io

    contains

    !---------------------------------------------------------------------------
    ! Open one data file using MPI/IO.
    ! Input:
    !   fname: file name.
    !   amode: file access mode.
    !   fileinfo: MPI_INFO
    ! Output:
    !   fh: file handler.
    !---------------------------------------------------------------------------
    subroutine open_data_mpi_io(fname, amode, fileinfo, fh)
        implicit none
        character(*), intent(in) :: fname
        integer, intent(in) :: amode, fileinfo
        integer, intent(out) :: fh
        call MPI_FILE_OPEN(MPI_COMM_WORLD, fname, amode, &
            fileinfo, fh, ierror)
        if (ierror /= 0) then
            call MPI_ERROR_STRING(ierror, err_msg, err_length, ierror2)
            print*, "Error in MPI_FILE_OPEN: ", trim(err_msg)
        endif
    end subroutine open_data_mpi_io

    !---------------------------------------------------------------------------
    ! Read data from files using MPI/IO.
    ! Input:
    !   fh: file handler.
    !   datatype: MPI data type.
    !   subsizes: the sub-sizes of the data in current MPI process.
    !   disp: displacement form the beginning of the file (in bytes).
    !   offset: offset from current file view (in data etypes (e.g. int, real)).
    ! Output:
    !   rdata: the data read from the file.
    !---------------------------------------------------------------------------
    subroutine read_data_mpi_io(fh, datatype, subsizes, disp, offset, rdata)
        use constants, only: fp
        implicit none
        integer, intent(in) :: fh, datatype
        integer, dimension(3), intent(in) :: subsizes
        integer(kind=MPI_OFFSET_KIND), intent(in) :: disp, offset
        real(fp), dimension(:, :, :), intent(out) :: rdata
        !integer :: datatype1

        !call MPI_TYPE_CREATE_SUBARRAY(3, sizes_ghost, subsizes_ghost, starts_ghost, &
        !    MPI_ORDER_FORTRAN, MPI_REAL, datatype1, ierror)
        !call MPI_TYPE_COMMIT(datatype1, ierror)
        !
        call MPI_FILE_SET_VIEW(fh, disp, MPI_REAL, datatype, 'native', &
            MPI_INFO_NULL, ierror)
        if (ierror /= 0) then
            call MPI_ERROR_STRING(ierror, err_msg, err_length, ierror2)
            print*, "Error in MPI_FILE_SET_VIEW: ", trim(err_msg)
        endif

        call MPI_FILE_READ_AT_ALL(fh, offset, rdata, &
            subsizes(1)*subsizes(2)*subsizes(3), MPI_REAL, status, ierror)
        if (ierror /= 0) then
            call MPI_ERROR_STRING(ierror, err_msg, err_length, ierror2)
            print*, "Error in MPI_FILE_READ: ", trim(err_msg)
        endif
        !call MPI_TYPE_FREE(datatype1, ierror)
    end subroutine read_data_mpi_io

    !---------------------------------------------------------------------------
    ! Write data to files using MPI/IO.
    ! Input:
    !   fh: file handler.
    !   datatype: MPI data type.
    !   subsizes: the sub-sizes of the data in current MPI process.
    !   disp: displacement form the beginning of the file (in bytes).
    !   offset: offset from current file view (in data etypes (e.g. int, real)).
    ! Output:
    !   wdata: the data to write to file.
    !---------------------------------------------------------------------------
    subroutine write_data_mpi_io(fh, datatype, subsizes, disp, offset, wdata)
        use constants, only: fp
        implicit none
        integer, intent(in) :: fh, datatype
        integer, dimension(3), intent(in) :: subsizes
        integer(kind=MPI_OFFSET_KIND), intent(in) :: disp, offset
        real(fp), dimension(:,:,:), intent(in) :: wdata
        call MPI_FILE_SET_VIEW(fh, disp, MPI_REAL, datatype, 'native', &
            MPI_INFO_NULL, ierror)
        if (ierror /= 0) then
            call MPI_ERROR_STRING(ierror, err_msg, err_length, ierror2)
            print*, "Error in MPI_FILE_SET_VIEW: ", trim(err_msg)
        endif

        call MPI_FILE_WRITE_AT_ALL(fh, offset, wdata, &
                                   subsizes(1)*subsizes(2)*subsizes(3), &
                                   MPI_REAL, status, ierror)
        if (ierror /= 0) then
            call MPI_ERROR_STRING(ierror, err_msg, err_length, ierror2)
            print*, "Error in MPI_FILE_READ: ", trim(err_msg)
        endif
    end subroutine write_data_mpi_io

end module mpi_io_module_test