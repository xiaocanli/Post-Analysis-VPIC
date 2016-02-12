!*******************************************************************************
!*******************************************************************************
! Module of the configuration setup for this analysis.
! httx: number of domains in x (converter)
! htty: the same in y
! httz: the same in z
! tindex_start: time slice to start at
! tindex_stop: time slice to stop at. Set to a large number to process all.
! output_format: output format. 2=file per slice, 1=all slices in one file
! append_to_files: set to 1 to append to existing files, anything else to
!                  start at the beginning.
!*******************************************************************************
module configuration_translate
    implicit none
    private
    public httx, htty, httz, tindex_start, tindex_stop, output_format, &
           append_to_files, read_configuration
    integer :: httx, htty, httz, tindex_start, tindex_stop
    integer :: output_format, append_to_files

    contains

    !---------------------------------------------------------------------------
    ! Read the configuration from file and broadcast to all MPI processes.
    !---------------------------------------------------------------------------
    subroutine read_configuration
        use mpi_module
        implicit none
        namelist /datum/ httx, htty, httz, tindex_start, tindex_stop, &
                         output_format, append_to_files 

        ! Read the configuration file
        if (myid==master) then
            open(unit=10, file='config_files/conf.dat', &
                 form='formatted', status='old')
            read(10, datum)
            close(10)
        endif

        ! Broadcast to all MPI processes.
        call MPI_BCAST(httx, 1, MPI_INTEGER, master, MPI_COMM_WORLD, ierr)
        call MPI_BCAST(htty, 1, MPI_INTEGER, master, MPI_COMM_WORLD, ierr)
        call MPI_BCAST(httz, 1, MPI_INTEGER, master, MPI_COMM_WORLD, ierr)

        call MPI_BCAST(tindex_start, 1, MPI_INTEGER, master, MPI_COMM_WORLD, ierr)
        call MPI_BCAST(tindex_stop, 1, MPI_INTEGER, master, MPI_COMM_WORLD, ierr)

        call MPI_BCAST(output_format, 1, MPI_INTEGER, master, MPI_COMM_WORLD, ierr)
        call MPI_BCAST(append_to_files, 1, MPI_INTEGER, master, MPI_COMM_WORLD, ierr)

        if (myid == master) then
            ! Echo this information
            print *, "---------------------------------------------------"
            print *, "Configuration to translate data"
            write(*, "(A,I0,A,I0,A,I0)") " Topology tx, ty, tz = ", &
                httx, ', ', htty, ', ', httz
            write(*, "(A,I0,A,I0)") " tindex_start, tindex_stop = ", &
                tindex_start, ', ', tindex_stop
            write(*, "(A,I0)") " output_format: ", output_format
            write(*, "(A,I0)") " append_to_files: ", append_to_files
            print *, "---------------------------------------------------"
        endif
    end subroutine read_configuration

end module configuration_translate


!*******************************************************************************
! Module includes routines to do mapping between 1D index and 3D indices on
! a 3D grid.
!*******************************************************************************
module rank_index_mapping
    implicit none
    private
    public rank_to_index, index_to_rank

    contains
    !---------------------------------------------------------------------------
    ! Map a 1D index to a 3D grid.
    ! Inputs:
    !   rank: the 1D index.
    !   nx, ny, nz: the sizes of the 3D grid.
    ! Outputs:
    !   ix, iy, iz: the 3D indices.
    !---------------------------------------------------------------------------
    subroutine rank_to_index(rank, nx, ny, nz, ix, iy, iz) 
        implicit none
        integer, intent(in) :: rank, nx, ny, nz
        integer, intent(out) :: ix, iy, iz

        iz = rank / (nx*ny)
        iy = (rank - iz*nx*ny) / nx
        ix = rank - iz*nx*ny - iy*nx
    end subroutine rank_to_index

    !---------------------------------------------------------------------------
    ! Map 3D indices one a 3D grid to a 1D index.
    !---------------------------------------------------------------------------
    subroutine index_to_rank(ix, iy, iz, nx, ny, nz, rank)
        implicit none
        integer, intent(in) :: ix, iy, iz, nx, ny, nz
        integer, intent(out) :: rank
        integer :: iix, iiy, iiz

        iix = mod(ix, nx)
        iiy = mod(iy, ny)
        iiz = mod(iz, nz)

        rank = iix + nx*(iiy + ny*iiz) + 1 
    end subroutine index_to_rank

end module rank_index_mapping
