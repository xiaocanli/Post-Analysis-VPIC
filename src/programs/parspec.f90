!*******************************************************************************
! Main program.
!*******************************************************************************
program parspec
    use mpi_module
    use constants, only: dp
    use path_info, only: get_file_paths
    use picinfo, only: read_domain, broadcast_pic_info
    use particle_frames, only: get_particle_frames, nt, tinterval
    use spectrum_config, only: read_spectrum_config, set_spatial_range_de, &
            calc_pic_mpi_ids, tframe, init_pic_mpi_ranks, free_pic_mpi_ranks, &
            calc_pic_mpi_ranks
    use particle_energy_spectrum, only: init_energy_spectra, &
            free_energy_spectra, calc_energy_spectra, &
            set_energy_spectra_zero, init_maximum_energy, free_maximum_energy, &
            set_maximum_energy_zero, get_maximum_energy_global, &
            save_maximum_energy, init_emax_pic_mpi, free_emax_pic_mpi
    use particle_maximum_energy, only: distribute_pic_mpi, init_emax_array, &
            free_emax_array, set_emax_datatype, free_emax_datatype
    use parameters, only: get_start_end_time_points, get_inductive_flag, &
            get_relativistic_flag
    use mpi_info_module, only: set_mpi_info
    implicit none
    integer :: ct
    real(dp) :: mp_elapsed
    integer :: is_emax_cell
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
    call set_spatial_range_de
    call calc_pic_mpi_ids
    call init_pic_mpi_ranks
    call calc_pic_mpi_ranks

    call init_energy_spectra
    call init_maximum_energy(nt)

    ! Get the maximum energy in each cell
    if (is_emax_cell == 1) then
        call distribute_pic_mpi
        call init_emax_array
        call set_emax_datatype
        call init_emax_pic_mpi
        call set_mpi_info
    endif

    mp_elapsed = MPI_WTIME()

    do ct = 1, nt
        call calc_energy_spectra(ct, 'e', is_emax_cell)
        call set_energy_spectra_zero
    enddo
    call get_maximum_energy_global(nt)
    if (myid == master) then
        call save_maximum_energy(nt, 'e')
    endif
    call set_maximum_energy_zero

    do ct = 1, nt
        call calc_energy_spectra(ct, 'h', is_emax_cell)
        call set_energy_spectra_zero
    enddo
    call get_maximum_energy_global(nt)
    if (myid == master) then
        call save_maximum_energy(nt, 'h')
    endif

    mp_elapsed = MPI_WTIME() - mp_elapsed

    if (myid==master) then
        write(*,'(A, F6.1)') " Total time used (s): ", mp_elapsed
    endif

    if (is_emax_cell == 1) then
        call free_emax_datatype
        call free_emax_pic_mpi
        call free_emax_array
    endif

    call free_pic_mpi_ranks
    call free_maximum_energy
    call free_energy_spectra
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
            description = 'Merge VPIC simulation output from all MPI processes', &
            examples    = ['translate -rp simulation_root_path -ec 0/1 &
                                      -sc spectra_config_file'])
        call cli%add(switch='--rootpath', switch_ab='-rp', &
            help='simulation root path', required=.true., act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--emax_cell', switch_ab='-ec', &
            help='whether to calculate maximum energy in each cell', &
            required=.false., act='store', def='0', error=error)
        if (error/=0) stop
        call cli%add(switch='--spect_config', switch_ab='-sc', &
            help='particle spectrum configurationfile', &
            required=.false., act='store', def='config_files/spectrum_config.dat', error=error)
        if (error/=0) stop
        call cli%get(switch='-rp', val=rootpath, error=error)
        if (error/=0) stop
        call cli%get(switch='-ec', val=is_emax_cell, error=error)
        if (error/=0) stop
        call cli%get(switch='-sc', val=spect_config_name, error=error)
        if (error/=0) stop

        if (myid == 0) then
            print '(A,A)', 'The simulation rootpath: ', trim(adjustl(rootpath))
            if (is_emax_cell == 1) then
                print '(A)', 'Calculate maximum particle energy in each cell'
            endif
            print '(A,A)', 'The spectrum configuration filename: ', trim(adjustl(spect_config_name))
        endif
    end subroutine get_cmd_args

end program parspec
