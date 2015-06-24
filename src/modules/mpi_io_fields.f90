!*******************************************************************************
! MPI data type for parallel I/O.
!*******************************************************************************
module mpi_datatype_fields
    use mpi_module
    use picinfo, only: domain
    use mpi_topology, only: ht, htg
    implicit none
    private
    public sizes_ghost, subsizes_ghost, starts_ghost, &
           sizes_nghost, subsizes_nghost, starts_nghost, &
           filetype_ghost, filetype_nghost, set_mpi_datatype_fields
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
    subroutine set_mpi_datatype_fields
        use mpi_datatype_module, only: set_mpi_datatype
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

        filetype_ghost = set_mpi_datatype(sizes_ghost, &
                subsizes_ghost, starts_ghost)

        sizes_nghost = sizes_ghost
        subsizes_nghost(1) = ht%nx
        subsizes_nghost(2) = ht%ny
        subsizes_nghost(3) = ht%nz
        starts_nghost(1) = ht%start_x
        starts_nghost(2) = ht%start_y
        starts_nghost(3) = ht%start_z

        filetype_nghost = set_mpi_datatype(sizes_nghost, &
                subsizes_nghost, starts_nghost)

    end subroutine set_mpi_datatype_fields

end module mpi_datatype_fields


!*******************************************************************************
! This contains one subroutine to open field data file using MPI I/O, one
! subroutine to read data using MPI I/O.
!*******************************************************************************
module mpi_io_fields
    use mpi_module
    implicit none
    private
    public save_field

    contains

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
        use mpi_datatype_fields, only: filetype_nghost, subsizes_nghost
        use mpi_info_module, only: fileinfo
        use mpi_topology, only: range_out
        use mpi_io_module, only: open_data_mpi_io, write_data_mpi_io
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

end module mpi_io_fields
