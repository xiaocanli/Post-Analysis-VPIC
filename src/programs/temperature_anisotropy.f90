!<******************************************************************************
!< Calculate the temperature anisotropy
!<******************************************************************************
program temperature_anisotropy
    use mpi_module
    use constants, only: fp, dp
    use particle_info, only: species, get_ptl_mass_charge
    implicit none
    integer :: nbins_temp, nbins_beta, nbins_tratio
    real(fp) :: temp_min, temp_max, beta_min, beta_max, tratio_min, tratio_max
    real(fp) :: temp_min_log, beta_min_log, tratio_min_log
    real(fp) :: dtemp_log, dbeta_log, dtratio_log
    real(fp), allocatable, dimension(:) :: ftpara_local, ftperp_local
    real(fp), allocatable, dimension(:) :: ftpara_global, ftperp_global
    real(fp), allocatable, dimension(:) :: fbpara_local, fbperp_local
    real(fp), allocatable, dimension(:) :: fbpara_global, fbperp_global
    real(fp), allocatable, dimension(:) :: ftratio_local, ftratio_global
    real(fp), allocatable, dimension(:) :: temp_bins_edge, beta_bins_edge, tratio_bins_edge
    real(fp), allocatable, dimension(:, :) :: ftratio_bpara_local, ftratio_bpara_global
    character(len=256) :: rootpath
    integer :: ct

    ct = 1

    call init_analysis
    call init_dists
    call calc_bins_edge
    call calc_temparature_anisotropy
    call free_dists
    call end_analysis

    contains

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
    !< Initialize the distributions and bins
    !<--------------------------------------------------------------------------
    subroutine init_dists
        implicit none
        allocate(ftpara_local(nbins_temp))
        allocate(ftperp_local(nbins_temp))
        allocate(fbpara_local(nbins_beta))
        allocate(fbperp_local(nbins_beta))
        allocate(ftratio_local(nbins_tratio))
        allocate(ftratio_bpara_local(nbins_tratio, nbins_beta))
        allocate(temp_bins_edge(nbins_temp + 1))
        allocate(beta_bins_edge(nbins_beta + 1))
        allocate(tratio_bins_edge(nbins_tratio + 1))
        if (myid == master) then
            allocate(ftpara_global(nbins_temp))
            allocate(ftperp_global(nbins_temp))
            allocate(fbpara_global(nbins_beta))
            allocate(fbperp_global(nbins_beta))
            allocate(ftratio_global(nbins_tratio))
            allocate(ftratio_bpara_global(nbins_tratio, nbins_beta))
        endif
        temp_bins_edge = 0.0
        beta_bins_edge = 0.0
        tratio_bins_edge = 0.0
        call set_dists_zero
    end subroutine init_dists

    !<--------------------------------------------------------------------------
    !< Initialize the distributions and bins
    !<--------------------------------------------------------------------------
    subroutine set_dists_zero
        implicit none
        ftpara_local = 0.0
        ftperp_local = 0.0
        fbpara_local = 0.0
        fbperp_local = 0.0
        ftratio_local = 0.0
        ftratio_bpara_local = 0.0
        if (myid == master) then
            ftpara_global = 0.0
            ftperp_global = 0.0
            fbpara_global = 0.0
            fbperp_global = 0.0
            ftratio_global = 0.0
            ftratio_bpara_global = 0.0
        endif
    end subroutine set_dists_zero

    !<--------------------------------------------------------------------------
    !< Free the distributions and bins
    !<--------------------------------------------------------------------------
    subroutine free_dists
        implicit none
        deallocate(ftpara_local, ftperp_local)
        deallocate(fbpara_local, fbperp_local)
        deallocate(ftratio_local, ftratio_bpara_local)
        deallocate(temp_bins_edge, beta_bins_edge, tratio_bins_edge)
        if (myid == master) then
            deallocate(ftpara_global, ftperp_global)
            deallocate(fbpara_global, fbperp_global)
            deallocate(ftratio_global, ftratio_bpara_global)
        endif
    end subroutine free_dists

    !<--------------------------------------------------------------------------
    !< Calculate the bins edge
    !<--------------------------------------------------------------------------
    subroutine calc_bins_edge
        implicit none
        integer :: i
        temp_min_log = log10(temp_min)
        beta_min_log = log10(beta_min)
        tratio_min_log = log10(tratio_min)
        dtemp_log = (log10(temp_max) - temp_min_log) / nbins_temp
        dbeta_log = (log10(beta_max) - beta_min_log) / nbins_beta
        dtratio_log = (log10(tratio_max) - tratio_min_log) / nbins_tratio
        do i = 0, nbins_temp
            temp_bins_edge(i+1) = 10**(temp_min_log + dtemp_log * i)
        enddo
        do i = 0, nbins_beta
            beta_bins_edge(i+1) = 10**(beta_min_log + dbeta_log * i)
        enddo
        do i = 0, nbins_tratio
            tratio_bins_edge(i+1) = 10**(tratio_min_log + dtratio_log * i)
        enddo
    end subroutine calc_bins_edge

    !<--------------------------------------------------------------------------
    !< Get commandline arguments
    !<--------------------------------------------------------------------------
    subroutine get_cmd_args
        use flap                                !< FLAP package
        use penf
        implicit none
        type(command_line_interface) :: cli     !< Command Line Interface (CLI).
        integer(I4P)                 :: error   !< Error trapping flag.
        call cli%init(progname    = 'temperature_anisotropy', &
                      authors     = 'Xiaocan Li', &
                      help        = 'Usage: ', &
                      description = 'Calculate temperature anisotropy', &
                      examples    = ['temperature_anisotropy -rp simulation_root_path'])
        call cli%add(switch='--rootpath', switch_ab='-rp', &
            help='simulation root path', required=.true., act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--species', switch_ab='-sp', &
            help='particle species', required=.false., act='store', &
            def='e', error=error)
        if (error/=0) stop
        call cli%add(switch='--nbins_temp', switch_ab='-nt', &
            help='Number of bins for temperature', &
            required=.false., def='700', act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--nbins_beta', switch_ab='-nb', &
            help='Number of bins for plasma beta', &
            required=.false., def='600', act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--nbins_tratio', switch_ab='-nr', &
            help='Number of bins for the ratio of parallel and perpendicular temperature', &
            required=.false., def='400', act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--temp_min', switch_ab='-tl', &
            help='Minimum temperature for histogram', &
            required=.false., def='1E-5', act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--temp_max', switch_ab='-th', &
            help='Minimum temperature for histogram', &
            required=.false., def='100.0', act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--beta_min', switch_ab='-bl', &
            help='Minimum beta for histogram', &
            required=.false., def='1E-3', act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--beta_max', switch_ab='-bh', &
            help='Minimum beta for histogram', &
            required=.false., def='1000.0', act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--tratio_min', switch_ab='-rl', &
            help='Minimum temperature ratio for histogram', &
            required=.false., def='1E-2', act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--tratio_max', switch_ab='-rh', &
            help='Minimum temperature ratio for histogram', &
            required=.false., def='100.0', act='store', error=error)
        if (error/=0) stop
        call cli%get(switch='-rp', val=rootpath, error=error)
        if (error/=0) stop
        call cli%get(switch='-sp', val=species, error=error)
        if (error/=0) stop
        call cli%get(switch='-nt', val=nbins_temp, error=error)
        if (error/=0) stop
        call cli%get(switch='-nb', val=nbins_beta, error=error)
        if (error/=0) stop
        call cli%get(switch='-nr', val=nbins_tratio, error=error)
        if (error/=0) stop
        call cli%get(switch='-tl', val=temp_min, error=error)
        if (error/=0) stop
        call cli%get(switch='-th', val=temp_max, error=error)
        if (error/=0) stop
        call cli%get(switch='-bl', val=beta_min, error=error)
        if (error/=0) stop
        call cli%get(switch='-bh', val=beta_max, error=error)
        if (error/=0) stop
        call cli%get(switch='-rl', val=tratio_min, error=error)
        if (error/=0) stop
        call cli%get(switch='-rh', val=tratio_max, error=error)
        if (error/=0) stop

        if (myid == 0) then
            print '(A,A)', 'The simulation rootpath: ', trim(adjustl(rootpath))
            print '(A,A)', 'Partical species: ', species
            print '(A, I0)', 'Number of bins for temperature: ', nbins_temp
            print '(A, 2E10.3)', 'Minimum and maximum temperature: ', &
                temp_min, temp_max
            print '(A, I0)', 'Number of bins for plasma beta: ', nbins_beta
            print '(A, 2E10.3)', 'Minimum and maximum plasma beta: ', &
                beta_min, beta_max
            print '(A, I0)', 'Number of bins for temperature ratio: ', nbins_tratio
            print '(A, 2E10.3)', 'Minimum and maximum temperature ratio: ', &
                tratio_min, tratio_max
        endif
    end subroutine get_cmd_args

    !<--------------------------------------------------------------------------
    !< Calculate the parallel and perpendicular temperature
    !<--------------------------------------------------------------------------
    subroutine calc_temparature_anisotropy
        use mpi_topology, only: htg
        use picinfo, only: domain
        use particle_info, only: species
        use para_perp_pressure, only: init_para_perp_pressure, &
            free_para_perp_pressure, calc_real_para_perp_pressure
        use pic_fields, only: init_magnetic_fields, init_pressure_tensor, &
            init_number_density, free_magnetic_fields, free_pressure_tensor, &
            free_number_density, open_magnetic_field_files, &
            open_pressure_tensor_files, open_number_density_file, &
            close_magnetic_field_files, close_pressure_tensor_files, &
            close_number_density_file, read_magnetic_fields, &
            read_pressure_tensor, read_number_density, &
            interp_bfield_node, shift_pressure_tensor, shift_number_density
        use saving_flags, only: get_saving_flags
        use configuration_translate, only: output_format
        use parameters, only: tp1, tp2
        implicit none
        integer :: tframe, nframes, posf, fh1, tindex
        real(fp), allocatable, dimension(:, :) :: jdote, jdote_tot
        character(len=256) :: fname
        logical :: dir_e

        call get_ptl_mass_charge(species)
        call get_saving_flags
        call init_magnetic_fields(htg%nx, htg%ny, htg%nz)
        call init_pressure_tensor(htg%nx, htg%ny, htg%nz)
        call init_number_density(htg%nx, htg%ny, htg%nz)
        call init_para_perp_pressure

        if (output_format == 1) then
            call open_magnetic_field_files
            call open_pressure_tensor_files(species)
            call open_number_density_file(species)
        endif

        do tframe = tp1, tp2
            if (myid==master) print*, tframe
            if (output_format /= 1) then
                tindex = domain%fields_interval * (tframe - tp1)
                call open_magnetic_field_files(tindex)
                call read_magnetic_fields(tp1)
                call close_magnetic_field_files
                if (myid == master) print*, "Finished reading magnetic fields"
                call open_pressure_tensor_files(species, tindex)
                call read_pressure_tensor(tp1)
                call close_pressure_tensor_files
                if (myid == master) print*, "Finished reading pressure tensor"
                call open_number_density_file(species, tindex)
                call read_number_density(tp1)
                call close_number_density_file
                if (myid == master) print*, "Finished reading number density"
            else
                call read_magnetic_fields(tframe)
                if (myid == master) print*, "Finished reading magnetic fields"
                call read_pressure_tensor(tframe)
                if (myid == master) print*, "Finished reading pressure tensor"
                call read_number_density(tframe)
                if (myid == master) print*, "Finished reading number density"
            endif
            call interp_bfield_node
            call shift_pressure_tensor
            call shift_number_density
            call calc_real_para_perp_pressure(tframe)
            call set_dists_zero
            call calc_temparature_anisotropy_single
            call save_temperature_anisotropy(tframe)
        enddo

        if (output_format == 1) then
            call close_magnetic_field_files
            call close_pressure_tensor_files
            call close_number_density_file
        endif

        call free_para_perp_pressure
        call free_magnetic_fields
        call free_pressure_tensor
        call free_number_density
    end subroutine calc_temparature_anisotropy

    !<--------------------------------------------------------------------------
    !< Calculate the parallel and perpendicular temperature for a single frame
    !<--------------------------------------------------------------------------
    subroutine calc_temparature_anisotropy_single
        use mpi_topology, only: ht
        use para_perp_pressure, only: ppara, pperp
        use pic_fields, only: absB, num_rho
        implicit none
        real(fp) :: tpara, tperp, beta_para, beta_perp, tratio, irho, ib2
        integer :: itemp_para, itemp_perp, ibeta_para, ibeta_perp, itratio
        integer :: i, j, k

        do k = 1, ht%nz
            do j = 1, ht%ny
                do i = 1, ht%nx
                    irho = 1.0 / num_rho(i, j, k)
                    ib2 = 2.0 / absB(i, j, k)**2
                    tpara = ppara(i, j, k) * irho
                    tperp = pperp(i, j, k) * irho
                    tratio = tperp / tpara
                    beta_para = ppara(i, j, k) * ib2
                    beta_perp = pperp(i, j, k) * ib2
                    itemp_para = (log10(tpara) - temp_min_log) / dtemp_log + 1
                    itemp_perp = (log10(tperp) - temp_min_log) / dtemp_log + 1
                    ibeta_para = (log10(beta_para) - beta_min_log) / dbeta_log + 1
                    ibeta_perp = (log10(beta_perp) - beta_min_log) / dbeta_log + 1
                    itratio = (log10(tratio) - tratio_min_log) / dtratio_log + 1
                    if (itemp_para > 0 .and. itemp_para <= nbins_temp) then
                        ftpara_local(itemp_para) = ftpara_local(itemp_para) + 1
                    endif
                    if (itemp_perp > 0 .and. itemp_perp <= nbins_temp) then
                        ftperp_local(itemp_perp) = ftperp_local(itemp_perp) + 1
                    endif
                    if (ibeta_para > 0 .and. ibeta_para <= nbins_beta) then
                        fbpara_local(ibeta_para) = fbpara_local(ibeta_para) + 1
                    endif
                    if (ibeta_perp > 0 .and. ibeta_perp <= nbins_beta) then
                        fbperp_local(ibeta_perp) = fbperp_local(ibeta_perp) + 1
                    endif
                    if (itratio > 0 .and. itratio <= nbins_tratio) then
                        ftratio_local(itratio) = ftratio_local(itratio) + 1
                        if (ibeta_para > 0 .and. ibeta_para <= nbins_beta) then
                            ftratio_bpara_local(itratio, ibeta_para) = &
                                ftratio_bpara_local(itratio, ibeta_para) + 1
                        endif
                    endif
                enddo
            enddo
        enddo

        call MPI_REDUCE(ftpara_local, ftpara_global, nbins_temp, MPI_REAL, &
            MPI_SUM, master, MPI_COMM_WORLD, ierr)
        call MPI_REDUCE(ftperp_local, ftperp_global, nbins_temp, MPI_REAL, &
            MPI_SUM, master, MPI_COMM_WORLD, ierr)
        call MPI_REDUCE(fbpara_local, fbpara_global, nbins_beta, MPI_REAL, &
            MPI_SUM, master, MPI_COMM_WORLD, ierr)
        call MPI_REDUCE(fbperp_local, fbperp_global, nbins_beta, MPI_REAL, &
            MPI_SUM, master, MPI_COMM_WORLD, ierr)
        call MPI_REDUCE(ftratio_local, ftratio_global, nbins_tratio, MPI_REAL, &
            MPI_SUM, master, MPI_COMM_WORLD, ierr)
        call MPI_REDUCE(ftratio_bpara_local, ftratio_bpara_global, &
            nbins_tratio * nbins_beta, MPI_REAL, MPI_SUM, master, MPI_COMM_WORLD, ierr)
    end subroutine calc_temparature_anisotropy_single

    !<--------------------------------------------------------------------------
    !< Save temperature anisotropy
    !<--------------------------------------------------------------------------
    subroutine save_temperature_anisotropy(tindex)
        implicit none
        integer, intent(in) :: tindex
        integer :: fh1, posf
        character(len=16) :: tindex_str
        character(len=256) :: fname
        logical :: dir_e
        if (myid == master) then
            inquire(file='./data/temperature_anisotropy/.', exist=dir_e)
            if (.not. dir_e) then
                call system('mkdir -p ./data/temperature_anisotropy/')
            endif
            print*, "Saving temperature anisotropy resutls..."

            fh1 = 66
            write(tindex_str, "(I0)") tindex - 1

            ! Parallel temperature
            fname = 'data/temperature_anisotropy/ftpara_'//species
            fname = trim(fname)//"_"//trim(tindex_str)//'.gda'
            open(unit=fh1, file=fname, access='stream', status='unknown', &
                form='unformatted', action='write')
            posf = 1
            write(fh1, pos=posf) nbins_temp + 0.0
            posf = posf + 4
            write(fh1, pos=posf) temp_bins_edge
            posf = posf + 4 * (nbins_temp + 1)
            write(fh1, pos=posf) ftpara_global
            close(fh1)

            ! Perpendicular temperature
            fname = 'data/temperature_anisotropy/ftperp_'//species
            fname = trim(fname)//"_"//trim(tindex_str)//'.gda'
            open(unit=fh1, file=fname, access='stream', status='unknown', &
                form='unformatted', action='write')
            posf = 1
            write(fh1, pos=posf) nbins_temp + 0.0
            posf = posf + 4
            write(fh1, pos=posf) temp_bins_edge
            posf = posf + 4 * (nbins_temp + 1)
            write(fh1, pos=posf) ftperp_global
            close(fh1)

            ! Parallel plasma beta
            fname = 'data/temperature_anisotropy/fbpara_'//species
            fname = trim(fname)//"_"//trim(tindex_str)//'.gda'
            open(unit=fh1, file=fname, access='stream', status='unknown', &
                form='unformatted', action='write')
            posf = 1
            write(fh1, pos=posf) nbins_beta + 0.0
            posf = posf + 4
            write(fh1, pos=posf) beta_bins_edge
            posf = posf + 4 * (nbins_beta + 1)
            write(fh1, pos=posf) fbpara_global
            close(fh1)

            ! Perpendicular plasma beta
            fname = 'data/temperature_anisotropy/fbperp_'//species
            fname = trim(fname)//"_"//trim(tindex_str)//'.gda'
            open(unit=fh1, file=fname, access='stream', status='unknown', &
                form='unformatted', action='write')
            posf = 1
            write(fh1, pos=posf) nbins_beta + 0.0
            posf = posf + 4
            write(fh1, pos=posf) beta_bins_edge
            posf = posf + 4 * (nbins_beta + 1)
            write(fh1, pos=posf) fbperp_global
            close(fh1)

            ! Temperature ratio
            fname = 'data/temperature_anisotropy/ftratio_'//species
            fname = trim(fname)//"_"//trim(tindex_str)//'.gda'
            open(unit=fh1, file=fname, access='stream', status='unknown', &
                form='unformatted', action='write')
            posf = 1
            write(fh1, pos=posf) nbins_tratio + 0.0
            posf = posf + 4
            write(fh1, pos=posf) tratio_bins_edge
            posf = posf + 4 * (nbins_tratio + 1)
            write(fh1, pos=posf) ftratio_global
            close(fh1)

            ! Temperature ratio vs. parallel plasma beta
            fname = 'data/temperature_anisotropy/ftratio_bpara_'//species
            fname = trim(fname)//"_"//trim(tindex_str)//'.gda'
            open(unit=fh1, file=fname, access='stream', status='unknown', &
                form='unformatted', action='write')
            posf = 1
            write(fh1, pos=posf) nbins_tratio + 0.0
            posf = posf + 4
            write(fh1, pos=posf) nbins_beta + 0.0
            posf = posf + 4
            write(fh1, pos=posf) tratio_bins_edge
            posf = posf + 4 * (nbins_tratio + 1)
            write(fh1, pos=posf) beta_bins_edge
            posf = posf + 4 * (nbins_beta + 1)
            write(fh1, pos=posf) ftratio_bpara_global
            close(fh1)
        endif
    end subroutine save_temperature_anisotropy

end program temperature_anisotropy
