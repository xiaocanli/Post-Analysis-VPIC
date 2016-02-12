!*******************************************************************************
! Main program.
!*******************************************************************************
program parspec
    use mpi_module
    use path_info, only: get_file_paths
    use picinfo, only: read_domain, broadcast_pic_info
    use particle_frames, only: get_particle_frames, nt, tinterval
    use spectrum_config, only: read_spectrum_config, set_spatial_range_de
    use particle_energy_spectrum, only: init_energy_spectra, &
            free_energy_spectra, calc_energy_spectra, &
            set_energy_spectra_zero
    use parameters, only: get_start_end_time_points, get_inductive_flag
    implicit none
    integer :: ct
    ! Initialize Message Passing
    call MPI_INIT(ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)

    call get_file_paths
    if (myid==master) then
        call get_particle_frames
    endif
    call MPI_BCAST(nt, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
    call MPI_BCAST(tinterval, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)

    if (myid==master) then
        call read_domain
    endif
    call broadcast_pic_info
    call get_start_end_time_points
    call get_inductive_flag
    call read_spectrum_config
    call set_spatial_range_de

    call init_energy_spectra

    do ct = 1, 10
        call calc_energy_spectra(ct, 'e')
        call set_energy_spectra_zero
    enddo

    do ct = 1, 10
        call calc_energy_spectra(ct, 'h')
        call set_energy_spectra_zero
    enddo

    call free_energy_spectra
    call MPI_FINALIZE(ierr)
end program parspec
