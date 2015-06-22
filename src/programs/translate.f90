!*******************************************************************************
! Parallel conversion; Using Bill's data type at LANL
! 
! This code convert VPIC output into gda files, which are "bricks" of data
! 
!*******************************************************************************
program translate
    use mpi_module
    use constants, only: dp
    use configuration_translate, only: tindex_start, tindex_stop
    use time_info, only: nout, output_record
    use emfields, only: read_emfields, write_emfields
    use particle_fields, only: read_particle_fields, set_current_density_zero, &
            calc_current_density, calc_absJ, write_current_densities, &
            adjust_particle_fields, write_particle_fields
    use path_info, only: rootpath
    implicit none
    integer :: tindex, tindex_new
    character(len=150) :: fname
    real(dp) :: mp_elapsed
    logical :: dfile

    call init_analysis

    mp_elapsed = MPI_WTIME()

    ! Loop over time slices
    dfile= .true.
    tindex = tindex_start
    do while(dfile)
        if (myid==master) print *, " Time slice: ", tindex

        ! EMF
        call read_emfields(tindex)
        call write_emfields(tindex, output_record)

        ! Particle fields
        call read_particle_fields(tindex, 'e')
        call calc_current_density
        call adjust_particle_fields('e')
        call write_particle_fields(tindex, output_record, 'e')
        call read_particle_fields(tindex, 'H')
        call calc_current_density
        call adjust_particle_fields('H')
        call write_particle_fields(tindex, output_record, 'i')
        call calc_absJ
        call write_current_densities(tindex, output_record)
        ! Avoid accumulation in calc_current_density
        call set_current_density_zero

        ! Might as well just wait here
        call MPI_BARRIER(MPI_COMM_WORLD, ierror)

        ! Check if there is another time slice to read
        dfile = .false.
        tindex_new = tindex + nout
        if (tindex_new <= tindex_stop) then
            write(fname, "(A,I0,A,I0,A)") &
                trim(adjustl(rootpath))//"fields/T.", tindex_new, &
                "/fields.", tindex_new, ".0"
            inquire(file=trim(fname), exist=dfile)
        endif
        tindex = tindex_new     
        if (dfile) output_record = output_record + 1
    enddo

    mp_elapsed = MPI_WTIME() - mp_elapsed

    if (myid==master) then
        write(*,'(A, F6.1)') " Total time used (s): ", mp_elapsed
    endif

    call end_analysis

    contains

    !---------------------------------------------------------------------------
    ! Initialize the analysis.
    !---------------------------------------------------------------------------
    subroutine init_analysis
        use mpi_module
        use path_info, only: get_file_paths
        use picinfo, only: read_domain, broadcast_pic_info, write_pic_info
        use configuration_translate, only: read_configuration
        use topology, only: set_topology, set_start_stop_cells
        use time_info, only: get_nout, adjust_tindex_start, set_output_record
        use mpi_io_translate, only: set_mpi_io
        use emfields, only: init_emfields
        use particle_fields, only: init_particle_fields
        implicit none

        call MPI_INIT(ierr)
        call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
        call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)

        call get_file_paths
        if (myid == master) then
            call read_domain
            !call write_pic_info
        endif
        call broadcast_pic_info
        call read_configuration
        call set_topology
        call set_start_stop_cells
        call get_nout
        call adjust_tindex_start
        call set_output_record
        call set_mpi_io
        call init_emfields
        call init_particle_fields

    end subroutine init_analysis

    !---------------------------------------------------------------------------
    ! End the analysis by free the memory.
    !---------------------------------------------------------------------------
    subroutine end_analysis
        use mpi_module
        use topology, only: free_start_stop_cells
        use mpi_io_translate, only: filetype, fileinfo
        use emfields, only: free_emfields
        use particle_fields, only: free_particle_fields
        implicit none
        call free_particle_fields
        call free_emfields
        call free_start_stop_cells
        call MPI_TYPE_FREE(filetype, ierror)
        call MPI_INFO_FREE(fileinfo, ierror)
        call MPI_FINALIZE(ierr)
    end subroutine end_analysis

end program translate
