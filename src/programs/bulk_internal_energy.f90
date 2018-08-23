!*******************************************************************************
! Program to calculate the bulk flow energy and internal energy.
!*******************************************************************************
program bulk_flow_energy
    use mpi_module
    use constants, only: fp, dp
    use particle_info, only: species, get_ptl_mass_charge
    implicit none
    character(len=256) :: rootpath

    call init_analysis

    call get_ptl_mass_charge(species)
    call commit_analysis

    call end_analysis

    contains

    !---------------------------------------------------------------------------
    ! Commit analysis.
    !---------------------------------------------------------------------------
    subroutine commit_analysis
        use mpi_topology, only: htg
        use picinfo, only: domain
        use pic_fields, only: vx, vy, vz, ux, uy, uz, pxx, pyy, pzz, num_rho, &
            open_velocity_field_files, open_pressure_tensor_files, &
            init_velocity_fields, init_pressure_tensor, &
            read_velocity_fields, read_pressure_tensor, &
            free_velocity_fields, free_pressure_tensor, &
            close_velocity_field_files, close_pressure_tensor_files, &
            open_number_density_file, init_number_density, &
            read_number_density, free_number_density, &
            close_number_density_file
        use particle_info, only: species, ptl_mass
        use parameters, only: tp1, tp2, is_rel
        use statistics, only: get_average_and_total
        use configuration_translate, only: output_format
        implicit none
        real(fp), allocatable, dimension(:, :) :: bulk_energy, internal_energy
        real(fp), dimension(4) :: bene_tot, iene_tot
        real(fp) :: avg
        logical :: dir_e
        integer :: tframe, tindex, posf, nframes
        character(len=256) :: fname

        bene_tot = 0.0  ! Bulk energy
        iene_tot = 0.0  ! Internal energy
        nframes = tp2 - tp1 + 1
        if (myid == master) then
            allocate(bulk_energy(4, nframes))
            allocate(internal_energy(4, nframes))
            bulk_energy = 0.0
            internal_energy = 0.0
        endif

        call init_velocity_fields(htg%nx, htg%ny, htg%nz)
        call init_pressure_tensor(htg%nx, htg%ny, htg%nz)
        call init_number_density(htg%nx, htg%ny, htg%nz)

        if (output_format == 1) then
            call open_velocity_field_files(species)
            call open_pressure_tensor_files(species)
            call open_number_density_file(species)
        endif

        do tframe = tp1, tp2
            if (myid == master) then
                print*, "Time frame: ", tframe
            endif
            if (output_format /= 1) then
                tindex = domain%fields_interval * (tframe - tp1)
                call open_velocity_field_files(species, tindex)
                call read_velocity_fields(tp1)
                call close_number_density_file
                call open_pressure_tensor_files(species, tindex)
                call read_pressure_tensor(tp1)
                call close_pressure_tensor_files
                call open_number_density_file(species, tindex)
                call read_number_density(tp1)
                call close_velocity_field_files
            else
                call read_velocity_fields(tframe)
                call read_pressure_tensor(tframe)
                call read_number_density(tframe)
            endif

            call get_average_and_total(0.5*vx*ux*ptl_mass*num_rho, &
                    avg, bene_tot(1))
            call get_average_and_total(0.5*vy*uy*ptl_mass*num_rho, &
                    avg, bene_tot(2))
            call get_average_and_total(0.5*vz*uz*ptl_mass*num_rho, &
                    avg, bene_tot(3))
            call get_average_and_total(&
                (sqrt(1.0+ux*ux+uy*uy+uz*uz) - 1.0)*ptl_mass*num_rho, &
                avg, bene_tot(4))

            call get_average_and_total(0.5*pxx, avg, iene_tot(1))
            call get_average_and_total(0.5*pyy, avg, iene_tot(2))
            call get_average_and_total(0.5*pzz, avg, iene_tot(3))
            iene_tot(4) = sum(iene_tot(1:3))
            if (myid == master) then
                bulk_energy(:, tframe-tp1+1) = bene_tot
                internal_energy(:, tframe-tp1+1) = iene_tot
            endif
        enddo

        if (output_format == 1) then
            call close_number_density_file
            call close_pressure_tensor_files
            call close_velocity_field_files
        endif

        call free_number_density
        call free_velocity_fields
        call free_pressure_tensor

        if (myid == master) then
            dir_e = .false.
            inquire(file='./data/bulk_internal_energy/.', exist=dir_e)
            if (.not. dir_e) then
                call system('mkdir -p ./data/bulk_internal_energy/')
            endif
            print*, "Saving bulk and internal energy..."

            fname = 'data/bulk_internal_energy/bulk_internal_energy_'//species//'.dat'
            open(unit=62, file=fname, access='stream', status='unknown', &
                form='unformatted', action='write')
            posf = 1
            write(62, pos=posf) bulk_energy
            posf = posf + 4 * nframes * 4
            write(62, pos=posf) internal_energy
            close(62)
        endif

        if (myid == master) then
            deallocate(bulk_energy)
            deallocate(internal_energy)
        endif
    end subroutine commit_analysis

    !<--------------------------------------------------------------------------
    !< Initialize the analysis
    !<--------------------------------------------------------------------------
    subroutine init_analysis
        use mpi_topology, only: set_mpi_topology, htg
        use mpi_datatype_fields, only: set_mpi_datatype_fields
        use mpi_info_module, only: set_mpi_info
        use particle_info, only: get_ptl_mass_charge
        use path_info, only: get_file_paths
        use picinfo, only: read_domain, broadcast_pic_info, &
                get_total_time_frames, get_energy_band_number, &
                read_thermal_params, calc_energy_interval, nbands, &
                write_pic_info, domain
        use configuration_translate, only: read_configuration
        use topology_translate, only: set_topology, set_start_stop_cells
        use mpi_io_translate, only: set_mpi_io
        use parameters, only: get_relativistic_flag, get_start_end_time_points, tp2
        use neighbors_module, only: init_neighbors, get_neighbors
        implicit none
        integer :: nx, ny, nz

        call MPI_INIT(ierr)
        call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
        call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)

        call get_cmd_args

        call get_file_paths(rootpath)
        if (myid == master) then
            call read_domain
        endif
        call broadcast_pic_info
        call get_ptl_mass_charge(species)
        call get_start_end_time_points
        call get_relativistic_flag
        ! call get_energy_band_number
        call read_thermal_params
        if (nbands > 0) then
            call calc_energy_interval
        endif
        call read_configuration
        call get_total_time_frames(tp2)
        call set_topology
        call set_start_stop_cells
        call set_mpi_io

        call set_mpi_topology(1)   ! MPI topology
        call set_mpi_datatype_fields
        call set_mpi_info

        call init_neighbors(htg%nx, htg%ny, htg%nz)
        call get_neighbors

    end subroutine init_analysis

    !<--------------------------------------------------------------------------
    !< End the analysis and free memory
    !<--------------------------------------------------------------------------
    subroutine end_analysis
        use topology_translate, only: free_start_stop_cells
        use mpi_io_translate, only: datatype
        use mpi_info_module, only: fileinfo
        use neighbors_module, only: free_neighbors
        use mpi_datatype_fields, only: filetype_ghost, filetype_nghost
        implicit none
        call free_neighbors
        call free_start_stop_cells
        call MPI_TYPE_FREE(datatype, ierror)
        call MPI_INFO_FREE(fileinfo, ierror)
        call MPI_TYPE_FREE(filetype_ghost, ierror)
        call MPI_TYPE_FREE(filetype_nghost, ierror)
        call MPI_FINALIZE(ierr)
    end subroutine end_analysis

    !<--------------------------------------------------------------------------
    !< Get commandline arguments
    !<--------------------------------------------------------------------------
    subroutine get_cmd_args
        use flap                                !< FLAP package
        use penf
        implicit none
        type(command_line_interface) :: cli     !< Command Line Interface (CLI).
        integer(I4P)                 :: error   !< Error trapping flag.
        call cli%init(progname    = 'Calculate bulk and internal energy', &
                      authors     = 'Xiaocan Li', &
                      help        = 'Usage: ', &
                      description = 'Calculate bulk and internal energy', &
                      examples    = ['bulk_internal_energy -rp simulation_root_path'])
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

end program bulk_flow_energy
