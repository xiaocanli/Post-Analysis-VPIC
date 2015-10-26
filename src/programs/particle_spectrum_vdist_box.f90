!*******************************************************************************
! Main program to calculate particle spectrum and velocity distributions in a
! user defined box.
!*******************************************************************************
program particle_spectrum_vdist_box
    use mpi_module
    use constants, only: dp
    use path_info, only: get_file_paths
    use picinfo, only: read_domain, broadcast_pic_info, domain
    use particle_frames, only: get_particle_frames, nt, tinterval
    use spectrum_config, only: read_spectrum_config, set_spatial_range_de, &
            calc_pic_mpi_ids, tframe, init_pic_mpi_ranks, free_pic_mpi_ranks, &
            calc_pic_mpi_ranks, calc_velocity_interval
    use velocity_distribution, only: init_velocity_bins, free_velocity_bins, &
            init_vdist_2d, set_vdist_2d_zero, free_vdist_2d, init_vdist_1d, &
            set_vdist_1d_zero, free_vdist_1d, calc_vdist_2d, calc_vdist_1d
    use particle_energy_spectrum, only: init_energy_spectra, &
            free_energy_spectra, calc_energy_spectra, &
            set_energy_spectra_zero
    use parameters, only: get_start_end_time_points, get_inductive_flag, &
            get_relativistic_flag
    use magnetic_field, only: init_magnetic_fields, free_magnetic_fields, &
            read_magnetic_fields
    use particle_info, only: species, get_ptl_mass_charge
    implicit none
    integer :: ct, ct_field, ratio_particle_field
    real(dp) :: mp_elapsed

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
    call get_relativistic_flag
    call read_spectrum_config
    call calc_velocity_interval
    call set_spatial_range_de
    call calc_pic_mpi_ids
    call init_pic_mpi_ranks
    call calc_pic_mpi_ranks

    call init_energy_spectra
    call init_velocity_bins
    call init_vdist_2d
    call init_vdist_1d
    call init_magnetic_fields

    mp_elapsed = MPI_WTIME()

    ! Ratio of particle output interval to fields output interval
    ratio_particle_field = domain%Particle_interval / domain%fields_interval
    ct_field = ratio_particle_field * tframe
    call read_magnetic_fields(ct_field)

    species = 'e'
    call get_ptl_mass_charge(species)
    call calc_spectrum_vdist(tframe, 'e')
    call set_energy_spectra_zero
    call set_vdist_2d_zero
    call set_vdist_1d_zero

    species = 'i'
    call get_ptl_mass_charge(species)
    call calc_spectrum_vdist(tframe, 'h')

    mp_elapsed = MPI_WTIME() - mp_elapsed

    if (myid==master) then
        write(*,'(A, F6.1)') " Total time used (s): ", mp_elapsed
    endif

    call free_magnetic_fields
    call free_vdist_1d
    call free_vdist_2d
    call free_velocity_bins
    call free_pic_mpi_ranks
    call free_energy_spectra
    call MPI_FINALIZE(ierr)

    contains

    !---------------------------------------------------------------------------
    ! Calculate spectrum and velocity distributions.
    !---------------------------------------------------------------------------
    subroutine calc_spectrum_vdist(ct, species)
        use mpi_module
        use constants, only: fp
        use particle_frames, only: tinterval
        use spectrum_config, only: nbins
        use particle_file, only: check_existence
        use particle_energy_spectrum, only: save_particle_spectra, &
                sum_spectra_over_mpi, calc_energy_bins
        use velocity_distribution, only: sum_vdist_1d_over_mpi, &
                sum_vdist_2d_over_mpi, save_vdist_1d, save_vdist_2d
        implicit none
        integer, intent(in) :: ct
        character(len=1), intent(in) :: species
        integer :: tindex
        logical :: is_exist

        call calc_energy_bins

        tindex = ct * tinterval
        call set_energy_spectra_zero
        call check_existence(tindex, species, is_exist)
        if (is_exist) then
            call calc_distributions_mpi(tindex, species)
            call sum_vdist_1d_over_mpi
            call sum_vdist_2d_over_mpi
            call sum_spectra_over_mpi
            call save_vdist_1d(ct, species)
            call save_vdist_2d(ct, species)
            call save_particle_spectra(ct, species)
        endif
    end subroutine calc_spectrum_vdist

    !---------------------------------------------------------------------------
    ! Calculate spectrum and velocity distributions for multiple PIC MPI ranks.
    !---------------------------------------------------------------------------
    subroutine calc_distributions_mpi(tindex, species)
        use mpi_module
        use picinfo, only: domain
        use file_header, only: pheader
        use spectrum_config, only: spatial_range, tot_pic_mpi, pic_mpi_ranks
        use particle_file, only: open_particle_file, check_particle_in_range, &
                close_particle_file, fh
        implicit none
        character(len=1), intent(in) :: species
        integer, intent(in) :: tindex
        character(len=50) :: cid
        logical :: isrange
        integer :: np, iptl
        integer :: IOstatus

        ! Read particle data in parallel to generate distributions
        do np = 0, tot_pic_mpi-numprocs, numprocs
            write(cid, "(I0)") myid + pic_mpi_ranks(np+1)
            call open_particle_file(tindex, species, cid)
            isrange = check_particle_in_range(spatial_range)

            if (isrange) then
                ! Loop over particles
                do iptl = 1, pheader%dim, 1
                    IOstatus = single_particle_energy_vel(fh)
                    if (IOstatus /= 0) exit
                enddo
            endif
            call close_particle_file
        enddo
    end subroutine calc_distributions_mpi

    !---------------------------------------------------------------------------
    ! Calculate energy and velocities for one single particle and update the
    ! distribution arrays.
    !---------------------------------------------------------------------------
    function single_particle_energy_vel(fh) result(IOstatus)
        use particle_module, only: ptl, calc_particle_energy, px, py, pz, &
                calc_ptl_coord, calc_para_perp_velocity
        use particle_energy_spectrum, only: update_energy_spectrum
        use velocity_distribution, only: update_vdist_1d, update_vdist_2d
        use spectrum_config, only: spatial_range
        use constants, only: fp
        implicit none
        integer, intent(in) :: fh
        integer :: IOstatus

        read(fh, IOSTAT=IOstatus) ptl
        if (IOstatus == 0) then
            call calc_ptl_coord

            if ((px >= spatial_range(1, 1)) .and. (px <= spatial_range(2, 1)) .and. &
                (py >= spatial_range(1, 2)) .and. (py <= spatial_range(2, 2)) .and. &
                (pz >= spatial_range(1, 3)) .and. (pz <= spatial_range(2, 3))) then

                call calc_particle_energy
                call calc_para_perp_velocity
                call update_energy_spectrum
                call update_vdist_2d
                call update_vdist_1d
            endif
        endif

    end function single_particle_energy_vel

end program particle_spectrum_vdist_box
