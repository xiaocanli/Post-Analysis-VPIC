!*******************************************************************************
! The main procedure. 
!*******************************************************************************
program dissipation
    use mpi_module
    use particle_info, only: species, ibtag, get_ptl_mass_charge
    implicit none
    character(len=256) :: rootpath
    integer :: ct

    ibtag = '00'
    ct = 1

    call init_analysis
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
    ! This subroutine does the analysis.
    !---------------------------------------------------------------------------
    subroutine commit_analysis
        use mpi_topology, only: htg
        use particle_info, only: species
        use para_perp_pressure, only: init_para_perp_pressure, &
                free_para_perp_pressure, save_averaged_para_perp_pressure
        use pic_fields, only: open_pic_fields, init_pic_fields, &
                free_pic_fields, close_pic_fields_file
        use saving_flags, only: get_saving_flags
        use neighbors_module, only: init_neighbors, free_neighbors, get_neighbors
        use compression_shear, only: init_div_v, free_div_v
        use configuration_translate, only: output_format
        implicit none

        call get_ptl_mass_charge(species)
        call init_pic_fields
        call init_para_perp_pressure
        call init_div_v  ! For compression related current density.
        call get_saving_flags

        if (output_format == 1) then
            call open_pic_fields(species)
        endif

        call init_neighbors(htg%nx, htg%ny, htg%nz)
        call get_neighbors

        call energy_conversion_from_current

        call free_neighbors
        call free_para_perp_pressure
        call free_pic_fields
        if (output_format == 1) then
            call close_pic_fields_file
        endif
        call free_div_v
    end subroutine commit_analysis

    !---------------------------------------------------------------------------
    ! This subroutine calculates the energy conversion through electric current.
    !---------------------------------------------------------------------------
    subroutine energy_conversion_from_current
        use mpi_module
        use mpi_topology, only: htg
        use picinfo, only: domain
        use parameters, only: tp1, tp2, inductive, is_rel
        use particle_info, only: ibtag, species
        use pic_fields, only: open_pic_fields, read_pic_fields, &
            close_pic_fields_file
        use pic_fields, only: vfields_fh, ufields_fh, nrho_fh
        use inductive_electric_field, only: calc_inductive_e, &
            init_inductive, free_inductive
        use pre_post_hydro, only: init_pre_post_v, &
            read_pre_post_v, free_pre_post_v, &
            init_pre_post_density, read_pre_post_density, free_pre_post_density
        use current_densities, only: init_current_densities, &
            free_current_densities, set_current_densities_to_zero, &
            init_ava_current_densities, free_avg_current_densities, &
            save_averaged_current
        use para_perp_pressure, only: save_averaged_para_perp_pressure
        use jdote_module, only: init_jdote, free_jdote, &
            init_jdote_total, free_jdote_total, save_jdote_total
        use configuration_translate, only: output_format

        implicit none

        integer :: input_record, output_record
        integer :: tindex

        if (inductive == 1) then
            call init_inductive(species)
        endif

        ! Calculate electric current due to all kinds of terms.
        ! And calculate energy conversion due to j.E.
        call init_current_densities
        call init_ava_current_densities
        call init_pre_post_v(htg%nx, htg%ny, htg%nz)
        call init_pre_post_density
        call init_jdote
        call init_jdote_total
        do input_record = tp1, tp2
            if (myid==master) print*, input_record
            output_record = input_record - tp1 + 1
            if (output_format /= 1) then
                tindex = domain%fields_interval * (input_record - tp1) 
                call open_pic_fields(species, tindex)
                output_record = 1
                call read_pic_fields(tp1)
            else
                call read_pic_fields(input_record)
            endif
            if (inductive == 1) then
                call calc_inductive_e(input_record, species)
            endif
            ! if (is_rel == 1) then
            !     call read_pre_post_v(input_record, ufields_fh)
            ! else
            !     call read_pre_post_v(input_record, vfields_fh)
            ! endif
            call read_pre_post_density(input_record, nrho_fh)
            call calc_energy_conversion(input_record)
            call set_current_densities_to_zero
            if (output_format /= 1) then
                call close_pic_fields_file
            endif
        enddo

        if (myid == master) then
            call save_averaged_current
            call save_jdote_total
            call save_averaged_para_perp_pressure
        endif

        call free_jdote_total
        call free_jdote
        call free_pre_post_v
        call free_pre_post_density
        call free_avg_current_densities
        call free_current_densities

        if (inductive == 1) then
            call free_inductive(species)
        endif
    end subroutine energy_conversion_from_current

    !---------------------------------------------------------------------------
    ! This subroutine calculates energy conversion for one time frame.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine calc_energy_conversion(ct)
        use para_perp_pressure, only: calc_para_perp_pressure, &
                calc_real_para_perp_pressure
        use current_densities, only: calc_current_densities
        implicit none
        integer, intent(in) :: ct
        call calc_real_para_perp_pressure(ct)
        ! call calc_para_perp_pressure(ct)
        call calc_current_densities(ct)
    end subroutine calc_energy_conversion

    !<--------------------------------------------------------------------------
    !< Get commandline arguments
    !<--------------------------------------------------------------------------
    subroutine get_cmd_args
        use flap                                !< FLAP package
        use penf
        implicit none
        type(command_line_interface) :: cli     !< Command Line Interface (CLI).
        integer(I4P)                 :: error   !< Error trapping flag.
        call cli%init(progname = 'dissipation', &
            authors     = 'Xiaocan Li', &
            help        = 'Usage: ', &
            description = 'Calculate energy conversion due to different drift terms', &
            examples    = ['dissipation -rp simulation_root_path'])
        call cli%add(switch='--rootpath', switch_ab='-rp', &
            help='simulation root path', required=.true., act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--species', switch_ab='-sp', &
            help='particle species', required=.false., act='store', &
            def='e', error=error)
        if (error/=0) stop
        call cli%get(switch='-rp', val=rootpath, error=error)
        if (error/=0) stop
        call cli%get(switch='-sp', val=species, error=error)
        if (error/=0) stop

        if (myid == 0) then
            print '(A,A)', 'The simulation rootpath: ', trim(adjustl(rootpath))
            print '(A,A)', 'Partical species: ', species
        endif
    end subroutine get_cmd_args

end program dissipation
