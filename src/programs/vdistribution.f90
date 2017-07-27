!*******************************************************************************
! Main program.
!*******************************************************************************
program vdistribution
    use mpi_module
    use path_info, only: get_file_paths
    use picinfo, only: read_domain, broadcast_pic_info, domain
    use particle_frames, only: get_particle_frames, nt, tinterval
    use spectrum_config, only: read_spectrum_config, set_spatial_range_de, &
            calc_pic_mpi_ids, tframe, init_pic_mpi_ranks, free_pic_mpi_ranks, &
            calc_pic_mpi_ranks, calc_velocity_interval
    use velocity_distribution, only: init_velocity_bins, free_velocity_bins, &
            init_vdist_2d, set_vdist_2d_zero, free_vdist_2d, init_vdist_1d, &
            set_vdist_1d_zero, free_vdist_1d, calc_vdist_2d, calc_vdist_1d
    use parameters, only: get_start_end_time_points, get_inductive_flag, &
            get_relativistic_flag
    use magnetic_field, only: init_magnetic_fields, free_magnetic_fields, &
            read_magnetic_fields
    use particle_info, only: species, get_ptl_mass_charge
    implicit none
    integer :: ct_field, ratio_particle_field, ct, ct_start, ct_end
    character(len=256) :: rootpath
    character(len=64) :: spect_config_name

    ! Initialize Message Passing
    call MPI_INIT(ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)

    call get_cmd_args
    call get_file_paths(rootpath)
    if (myid==master) then
        call get_particle_frames(rootpath)
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
    call read_spectrum_config(spect_config_name)
    call calc_velocity_interval
    call set_spatial_range_de
    call calc_pic_mpi_ids
    call init_pic_mpi_ranks
    call calc_pic_mpi_ranks

    call init_velocity_bins
    call init_vdist_2d
    call init_vdist_1d

    call init_magnetic_fields

    ! Ratio of particle output interval to fields output interval
    ratio_particle_field = domain%Particle_interval / domain%fields_interval

    do ct = ct_start, ct_end
        ct_field = ratio_particle_field * ct
        call read_magnetic_fields(ct_field)

        species = 'e'
        call get_ptl_mass_charge(species)
        call calc_vdist_2d(ct, 'e')
        call calc_vdist_1d(ct, 'e')
        call set_vdist_2d_zero
        call set_vdist_1d_zero

        species = 'i'
        call get_ptl_mass_charge(species)
        call calc_vdist_2d(ct, 'h')
        call calc_vdist_1d(ct, 'h')
        call set_vdist_2d_zero
        call set_vdist_1d_zero
    enddo

    call free_magnetic_fields

    call free_vdist_1d
    call free_vdist_2d
    call free_velocity_bins
    call free_pic_mpi_ranks
    call MPI_FINALIZE(ierr)

    contains

    !<--------------------------------------------------------------------------
    !< Get commandline arguments
    !<--------------------------------------------------------------------------
    subroutine get_cmd_args
        use flap                                !< FLAP package
        use penf
        implicit none
        type(command_line_interface) :: cli     !< Command Line Interface (CLI).
        integer(I4P)                 :: error   !< Error trapping flag.
        call cli%init(progname = 'translate', &
            authors     = 'Xiaocan Li', &
            help        = 'Usage: ', &
            description = 'Calculate velocity distributions', &
            examples    = ['vdistribution -rp simulation_root_path &
                                          -sc spectra_config_file &
                                          -st starting_time_frame &
                                          -et ending_time_frame'])
        call cli%add(switch='--rootpath', switch_ab='-rp', &
            help='simulation root path', required=.true., act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--spect_config', switch_ab='-sc', &
            help='particle spectrum configurationfile', &
            required=.false., act='store', def='config_files/spectrum_config.dat', error=error)
        if (error/=0) stop
        call cli%add(switch='--starting_time', switch_ab='-st', &
            help='starting time frame', required=.false., act='store', &
            def='1', error=error)
        if (error/=0) stop
        call cli%add(switch='--ending_time', switch_ab='-et', &
            help='ending time frame', required=.false., act='store', &
            def='1', error=error)
        if (error/=0) stop
        call cli%get(switch='-rp', val=rootpath, error=error)
        if (error/=0) stop
        call cli%get(switch='-sc', val=spect_config_name, error=error)
        if (error/=0) stop
        call cli%get(switch='-st', val=ct_start, error=error)
        if (error/=0) stop
        call cli%get(switch='-et', val=ct_end, error=error)
        if (error/=0) stop

        if (myid == 0) then
            print '(A,A)', 'The simulation rootpath: ', trim(adjustl(rootpath))
            print '(A,A)', 'The spectrum configuration filename: ', trim(adjustl(spect_config_name))
            print '(A,I0,A,I0)', 'Starting and ending time frames: ', &
                ct_start, ', ', ct_end
        endif
    end subroutine get_cmd_args

end program vdistribution
