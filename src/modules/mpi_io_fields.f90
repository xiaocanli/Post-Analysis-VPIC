!*******************************************************************************
! MPI data type for parallel I/O.
!*******************************************************************************
module mpi_datatype
    use mpi_module
    use picinfo, only: domain
    use mpi_topology, only: ht, htg
    implicit none
    private
    public sizes_ghost, subsizes_ghost, starts_ghost, &
           sizes_nghost, subsizes_nghost, starts_nghost, &
           filetype_ghost, filetype_nghost, set_mpi_datatype
    ! Two kinds: one with ghost cells, the other without.
    integer :: sizes_ghost(3), subsizes_ghost(3), starts_ghost(3)
    integer :: sizes_nghost(3), subsizes_nghost(3), starts_nghost(3)
    integer :: filetype_ghost, filetype_nghost

    contains

    !---------------------------------------------------------------------------
    ! Create a MPI data type and commit it. The data type will be different for
    ! data input and data output since they have arrays with different sizes.
    ! Updates:
    !   sizes_ghost, subsizes_ghost, starts_ghost: the sizes for
    !       MPI_TYPE_CREATE_SUBARRAY with ghost cells.
    !   sizes_nghost, subsizes_nghost, starts_nghost: the sizes for
    !       MPI_TYPE_CREATE_SUBARRAY without ghost cells.
    !   filetype_ghost, filetype_nghost: the file types of the created array.
    !---------------------------------------------------------------------------
    subroutine set_mpi_datatype
        implicit none

        sizes_ghost(1) = domain%nx
        sizes_ghost(2) = domain%ny
        sizes_ghost(3) = domain%nz
        subsizes_ghost(1) = htg%nx
        subsizes_ghost(2) = htg%ny
        subsizes_ghost(3) = htg%nz
        starts_ghost(1) = htg%start_x
        starts_ghost(2) = htg%start_y
        starts_ghost(3) = htg%start_z

        call MPI_TYPE_CREATE_SUBARRAY(3, sizes_ghost, subsizes_ghost, &
            starts_ghost, MPI_ORDER_FORTRAN, MPI_REAL, filetype_ghost, ierror)
        call MPI_TYPE_COMMIT(filetype_ghost, ierror)

        sizes_nghost = sizes_ghost
        subsizes_nghost(1) = ht%nx
        subsizes_nghost(2) = ht%ny
        subsizes_nghost(3) = ht%nz
        starts_nghost(1) = ht%start_x
        starts_nghost(2) = ht%start_y
        starts_nghost(3) = ht%start_z

        call MPI_TYPE_CREATE_SUBARRAY(3, sizes_nghost, subsizes_nghost, &
            starts_nghost, MPI_ORDER_FORTRAN, MPI_REAL, filetype_nghost, ierror)
        call MPI_TYPE_COMMIT(filetype_nghost, ierror)
    end subroutine set_mpi_datatype

end module mpi_datatype

!*******************************************************************************
! Set MPI_INFO for parallel I/O.
!*******************************************************************************
module  mpi_info_object
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
        ! Enable ROMIO's data-sieving
        !call MPI_INFO_SET(fileinfo, "romio_ds_read", "enable", ierror)
        !call MPI_INFO_SET(fileinfo, "romio_ds_write", "enable", ierror)
        !! Enable ROMIO's collective buffering
        !call MPI_INFO_SET(fileinfo, "romio_cb_read", "enable", ierror)
        !call MPI_INFO_SET(fileinfo, "romio_cb_write", "enable", ierror)
        !! Disable ROMIO's collective buffering
        !call MPI_INFO_SET(fileinfo, "romio_cb_read", "disable", ierror)
        !call MPI_INFO_SET(fileinfo, "romio_cb_write", "disable", ierror)
        call MPI_INFO_SET(fileinfo, "panfs_concurrent_write", "1", ierror)
    end subroutine set_mpi_info
end module  mpi_info_object


!*******************************************************************************
! This contains one subroutine to open field data file using MPI I/O, one
! subroutine to read data using MPI I/O.
!*******************************************************************************
module mpi_io_module
    use mpi_module
    implicit none
    private
    public open_data_mpi_io, read_data_mpi_io, write_data_mpi_io, save_field

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
    !   filetype: MPI data type.
    !   subsizes: the sub-sizes of the data in current MPI process.
    !   disp: displacement form the beginning of the file (in bytes).
    !   offset: offset from current file view (in data etypes (e.g. int, real)).
    ! Output:
    !   rdata: the data read from the file.
    !---------------------------------------------------------------------------
    subroutine read_data_mpi_io(fh, filetype, subsizes, disp, offset, rdata)
        use constants, only: fp
        implicit none
        integer, intent(in) :: fh, filetype
        integer, dimension(3), intent(in) :: subsizes
        integer(kind=MPI_OFFSET_KIND), intent(in) :: disp, offset
        real(fp), dimension(:, :, :), intent(out) :: rdata
        !integer :: filetype1

        !call MPI_TYPE_CREATE_SUBARRAY(3, sizes_ghost, subsizes_ghost, starts_ghost, &
        !    MPI_ORDER_FORTRAN, MPI_REAL, filetype1, ierror)
        !call MPI_TYPE_COMMIT(filetype1, ierror)
        !
        call MPI_FILE_SET_VIEW(fh, disp, MPI_REAL, filetype, 'native', &
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
        !call MPI_TYPE_FREE(filetype1, ierror)
    end subroutine read_data_mpi_io

    !---------------------------------------------------------------------------
    ! Write data to files using MPI/IO.
    ! Input:
    !   fh: file handler.
    !   filetype: MPI data type.
    !   subsizes: the sub-sizes of the data in current MPI process.
    !   disp: displacement form the beginning of the file (in bytes).
    !   offset: offset from current file view (in data etypes (e.g. int, real)).
    ! Output:
    !   wdata: the data to write to file.
    !---------------------------------------------------------------------------
    subroutine write_data_mpi_io(fh, filetype, subsizes, disp, offset, wdata)
        use constants, only: fp
        implicit none
        integer, intent(in) :: fh, filetype
        integer, dimension(3), intent(in) :: subsizes
        integer(kind=MPI_OFFSET_KIND), intent(in) :: disp, offset
        real(fp), dimension(:,:,:), intent(in) :: wdata
        call MPI_FILE_SET_VIEW(fh, disp, MPI_REAL, filetype, 'native', &
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

    !---------------------------------------------------------------------------
    ! Save the fields data to a file use MPI/IO.
    ! Input:
    !   fdata: the field data to save to disk.
    !   varname: the field variable name.
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_field(fdata, varname, ct)
        use constants, only: fp
        use picinfo, only: domain
        use parameters, only: it1
        use path_info, only: outputpath
        use particle_info, only: species, ibtag
        use mpi_datatype, only: filetype_nghost, subsizes_nghost
        use mpi_info_object, only: fileinfo
        use mpi_topology, only: range_out
        implicit none
        real(fp), dimension(:, :, :), intent(in) :: fdata
        character(*), intent(in) :: varname
        integer, intent(in) :: ct
        character(len=150) :: fname
        integer :: fh ! File handler
        integer :: ixl, iyl, izl, ixh, iyh, izh
        real(fp), allocatable, dimension(:,:,:) :: data_nghost
        integer(kind=MPI_OFFSET_KIND) :: disp, offset

        allocate(data_nghost(subsizes_nghost(1), subsizes_nghost(2), &
                subsizes_nghost(3)))
        ixl = range_out%ixl
        ixh = range_out%ixh
        iyl = range_out%iyl
        iyh = range_out%iyh
        izl = range_out%izl
        izh = range_out%izh
        ! print*, (ixh-ixl+1) /= subsizes_nghost(1), &
        !         (iyh-iyl+1) /= subsizes_nghost(2), &
        !         (izh-izl+1) /= subsizes_nghost(3)
        ! print*, ixh > subsizes_ghost(1), iyh > subsizes_ghost(2), &
        !         izh > subsizes_ghost(3)
        data_nghost = fdata(ixl:ixh,iyl:iyh,izl:izh)

        disp = domain%nx * domain%ny * domain%nz * sizeof(MPI_REAL) * (ct-it1)
        offset = 0 

        fname = trim(adjustl(outputpath))//varname//ibtag//'_'//species//'.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDWR+MPI_MODE_CREATE, fileinfo, fh)
        call write_data_mpi_io(fh, filetype_nghost, subsizes_nghost, &
            disp, offset, data_nghost)
        call MPI_FILE_CLOSE(fh, ierror)

        deallocate(data_nghost)
    end subroutine save_field

end module mpi_io_module
