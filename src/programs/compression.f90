!*******************************************************************************
! The main procedure to calculate the compressional and shear heating terms.
!*******************************************************************************
program compression
    use mpi_module
    use particle_info, only: species, ibtag
    implicit none
    character(len=256) :: rootpath
    logical :: use_exb_drift  ! Use ExB drift as single fluid velocity
    logical :: save_2d_fields ! Whether to save 2D fields

    call init_analysis

    ibtag = '00'
    species = 'e'
    call commit_analysis
    species = 'i'
    call commit_analysis

    call end_analysis

    contains

    !---------------------------------------------------------------------------
    ! Initialize the analysis by reading the PIC simulation domain information,
    ! get file paths for the field data and the outputs.
    !---------------------------------------------------------------------------
    subroutine init_analysis
        use path_info, only: get_file_paths
        use mpi_topology, only: set_mpi_topology
        use mpi_datatype_fields, only: set_mpi_datatype_fields
        use mpi_info_module, only: fileinfo, set_mpi_info
        use picinfo, only: read_domain, broadcast_pic_info, &
                get_total_time_frames, get_energy_band_number, &
                read_thermal_params, calc_energy_interval, nbands
        use parameters, only: get_start_end_time_points, get_inductive_flag, &
                tp2, get_relativistic_flag
        use configuration_translate, only: read_configuration
        use time_info, only: get_nout

        implicit none

        call MPI_INIT(ierr)
        call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
        call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)

        call get_cmd_args

        call get_file_paths(rootpath)
        if (myid == master) then
            call read_domain
        endif
        call broadcast_pic_info
        call get_start_end_time_points
        call get_inductive_flag
        call get_relativistic_flag
        call read_configuration
        call get_total_time_frames(tp2)
        call get_energy_band_number
        call read_thermal_params
        if (nbands > 0) then
            call calc_energy_interval
        endif
        call set_mpi_topology   ! MPI topology
        call set_mpi_datatype_fields
        call set_mpi_info
    end subroutine init_analysis

    !---------------------------------------------------------------------------
    ! Finalizing the analysis by release the memory, MPI data types, MPI info.
    !---------------------------------------------------------------------------
    subroutine end_analysis
        use mpi_module
        use mpi_datatype_fields, only: filetype_ghost, filetype_nghost
        use mpi_info_module, only: fileinfo
        implicit none

        call MPI_TYPE_FREE(filetype_ghost, ierror)
        call MPI_TYPE_FREE(filetype_nghost, ierror)
        call MPI_INFO_FREE(fileinfo, ierror)
        call MPI_FINALIZE(ierr)
    end subroutine end_analysis

    !---------------------------------------------------------------------------
    ! Doing the analysis for one species.
    !---------------------------------------------------------------------------
    subroutine commit_analysis
        use mpi_module
        use mpi_topology, only: htg
        use particle_info, only: species, ibtag, get_ptl_mass_charge
        use para_perp_pressure, only: init_para_perp_pressure, &
                free_para_perp_pressure, calc_para_perp_pressure, &
                calc_ppara_pperp_single, calc_real_para_perp_pressure
        use pic_fields, only: open_pic_fields, init_pic_fields, &
                free_pic_fields, close_pic_fields_file, &
                read_pic_fields
        use saving_flags, only: get_saving_flags
        use neighbors_module, only: init_neighbors, free_neighbors, get_neighbors
        use compression_shear, only: init_compression_shear, &
                free_compression_shear, calc_compression_shear, &
                save_compression_shear, save_tot_compression_shear, &
                calc_exb_drift
        use pressure_tensor, only: init_scalar_pressure, init_div_ptensor, &
                free_scalar_pressure, free_div_ptensor, calc_scalar_pressure, &
                calc_div_ptensor
        use usingle, only: init_usingle, open_velocity_density_files, &
                free_usingle, close_velocity_density_files, &
                read_velocity_density, calc_usingle
        use parameters, only: tp1, tp2
        implicit none
        integer :: input_record, output_record

        call get_ptl_mass_charge(species)

        call init_pic_fields
        call init_para_perp_pressure
        call get_saving_flags

        call open_pic_fields(species)

        call init_neighbors(htg%nx, htg%ny, htg%nz)
        call get_neighbors

        call init_scalar_pressure
        call init_div_ptensor
        call init_compression_shear(use_exb_drift)
        call init_usingle(species)
        call open_velocity_density_files(species)
        do input_record = tp1, tp2
            if (myid==master) print*, input_record
            output_record = input_record - tp1 + 1
            call read_pic_fields(input_record)
            call read_velocity_density(input_record, species)
            call calc_usingle(species)
            if (use_exb_drift) then
                call calc_exb_drift
            endif
            call calc_real_para_perp_pressure(input_record)
            call calc_scalar_pressure
            call calc_div_ptensor
            call calc_compression_shear(use_exb_drift)
            if (save_2d_fields) then
                call save_compression_shear(input_record)
            endif
            call save_tot_compression_shear(input_record, use_exb_drift)
        enddo
        call close_velocity_density_files(species)
        call free_usingle(species)
        call free_compression_shear(use_exb_drift)
        call free_div_ptensor
        call free_scalar_pressure

        call free_neighbors
        call free_para_perp_pressure
        call free_pic_fields
        call close_pic_fields_file
    end subroutine commit_analysis

    !<--------------------------------------------------------------------------
    !< Get commandline arguments
    !<--------------------------------------------------------------------------
    subroutine get_cmd_args
        use flap                                !< FLAP package
        use penf
        implicit none
        type(command_line_interface) :: cli     !< Command Line Interface (CLI).
        integer(I4P)                 :: error   !< Error trapping flag.
        call cli%init(progname = 'compression', &
            authors     = 'Xiaocan Li', &
            help        = 'Usage: ', &
            description = 'Calculate compression related energy conversion', &
            examples    = ['compression -rp simulation_root_path'])
        call cli%add(switch='--rootpath', switch_ab='-rp', &
            help='simulation root path', required=.true., act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--use_exb_drift', switch_ab='-uexb', &
            help='whether to use ExB drift as the single fluid velocity', &
            required=.false., act='store_true', def='.false.', error=error)
        if (error/=0) stop
        call cli%add(switch='--save_2d_fields', switch_ab='-s2', &
            help='whether to save 2D fields', required=.false., &
            act='store_true', def='.false.', error=error)
        if (error/=0) stop
        call cli%get(switch='-rp', val=rootpath, error=error)
        if (error/=0) stop
        call cli%get(switch='-uexb', val=use_exb_drift, error=error)
        if (error/=0) stop
        call cli%get(switch='-s2', val=save_2d_fields, error=error)
        if (error/=0) stop

        if (myid == 0) then
            print '(A,A)', 'The simulation rootpath: ', trim(adjustl(rootpath))
            if (use_exb_drift) then
                print '(A)', 'ExB drift will be used as single fluid velocity'
            endif
            if (save_2d_fields) then
                print '(A)', '2D fields will be saved'
            endif
        endif
    end subroutine get_cmd_args

end program compression
