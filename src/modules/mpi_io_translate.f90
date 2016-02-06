!*******************************************************************************
! Module for mpi_io. It includes the routine to set the MPI datatype and
! MPI_INFO.
!*******************************************************************************
module mpi_io_translate
    implicit none
    private
    public datatype, set_mpi_io, write_data
    integer :: datatype

    contains

    !---------------------------------------------------------------------------
    ! Set MPI datatype and MPI_INFO.
    !---------------------------------------------------------------------------
    subroutine set_mpi_io
        use mpi_module
        use topology_translate, only: ht
        use picinfo, only: domain
        use mpi_datatype_module, only: set_mpi_datatype
        use mpi_info_module, only: set_mpi_info
        implicit none
        integer :: sizes(3), subsizes(3), starts(3)

        ! size of the global matrix
        sizes(1) = domain%nx
        sizes(2) = domain%ny
        sizes(3) = domain%nz

        ! size of the chunck seen by each process
        subsizes(1) = ht%nx
        subsizes(2) = ht%ny
        subsizes(3) = ht%nz

        ! where each chunck starts
        starts(1) = ht%ix*ht%nx
        starts(2) = ht%iy*ht%ny
        starts(3) = ht%iz*ht%nz

        datatype = set_mpi_datatype(sizes, subsizes, starts)
        call set_mpi_info
    end subroutine set_mpi_io

    !---------------------------------------------------------------------------
    ! Write data to file using MPI I/O.
    !---------------------------------------------------------------------------
    subroutine write_data(fname, fdata, tindex, output_record)
        use mpi_module
        use constants, only: fp, dp
        use topology_translate, only: ht
        use mpi_info_module, only: fileinfo
        use configuration_translate, only: output_format
        implicit none
        integer, intent(in) :: tindex, output_record
        real(fp), intent(in), dimension(:,:,:) :: fdata
        character(*), intent(in) :: fname
        integer(kind=MPI_OFFSET_KIND) :: disp, offset
        character(len=150) :: cfname
        real(dp) :: mp_elapsed
        integer :: fh

        mp_elapsed = MPI_WTIME()

        disp = 0
        if (output_format == 1) then
            offset = (output_record - 1) * ht%nx * ht%ny * ht%nz
            cfname = trim(fname) // '.gda'
        else
           offset = 0
           write(cfname, "(I0)") tindex
           cfname = trim(fname) // '_' // trim(cfname) // '.gda'
        endif

        call MPI_FILE_OPEN(MPI_COMM_WORLD, cfname, &
                          MPI_MODE_RDWR+MPI_MODE_CREATE, fileinfo, fh, ierror)
        if (ierror /= 0 ) then
            call MPI_Error_string(ierror, err_msg, err_length,ierror2)
            print *, "Error in MPI_FILE_OPEN:", trim(err_msg)
        endif

        call MPI_FILE_SET_VIEW(fh, disp, MPI_REAL4, datatype, &
                              'native', MPI_INFO_NULL, ierror)
        if (ierror /= 0 ) then
            call MPI_Error_string(ierror, err_msg, err_length, ierror2)
            print *, "Error in MPI_FILE_SET_VIEW:", trim(err_msg)
        endif

        if (myid==master) print *, "writing data to file ", trim(cfname)

        call MPI_FILE_WRITE_AT_ALL(fh, offset, fdata, ht%nx*ht%ny*ht%nz, &
                                   MPI_REAL4, status, ierror)

        if (ierror /= 0 ) then
            call MPI_Error_string(ierror, err_msg, err_length, ierror2)
            print *, "Error in MPI_FILE_WRITE:", trim(err_msg)
        endif

        call MPI_FILE_CLOSE(fh, ierror)

        mp_elapsed = MPI_WTIME() - mp_elapsed

        if (myid==master) write(*,'(A, F5.1)') " => time(s):", mp_elapsed
    end subroutine write_data

end module mpi_io_translate
