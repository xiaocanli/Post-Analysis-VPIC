!<******************************************************************************
!< Calculate the distribution of magnetic curvature, which is in terms of 1/di,
!< and di is the ion inertial length.
!<******************************************************************************
program magnetic_curvature_distribution
    use mpi_module
    use constants, only: fp, dp
    use particle_info, only: species, get_ptl_mass_charge
    implicit none
    integer :: nbins_kappa
    real(fp) :: kappa_min, kappa_max, kappa_min_log, dkappa_log
    real(fp), allocatable, dimension(:) :: fkappa_local, fkappa_global
    real(fp), allocatable, dimension(:) :: kappa_bins_edge
    real(fp), allocatable, dimension(:, :, :) :: tmpx, tmpy, tmpz, stmp
    character(len=256) :: rootpath
    integer :: ct

    ct = 1
    species = 'e'

    call init_analysis
    call init_dists
    call calc_bins_edge
    call calc_kappa
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
        allocate(fkappa_local(nbins_kappa))
        allocate(kappa_bins_edge(nbins_kappa+ 1))
        if (myid == master) then
            allocate(fkappa_global(nbins_kappa))
        endif
        kappa_bins_edge = 0.0
        call set_dists_zero
    end subroutine init_dists

    !<--------------------------------------------------------------------------
    !< Initialize the distributions and bins
    !<--------------------------------------------------------------------------
    subroutine set_dists_zero
        implicit none
        fkappa_local = 0.0
        if (myid == master) then
            fkappa_global = 0.0
        endif
    end subroutine set_dists_zero

    !<--------------------------------------------------------------------------
    !< Free the distributions and bins
    !<--------------------------------------------------------------------------
    subroutine free_dists
        implicit none
        deallocate(fkappa_local)
        deallocate(kappa_bins_edge)
        if (myid == master) then
            deallocate(fkappa_global)
        endif
    end subroutine free_dists

    !<--------------------------------------------------------------------------
    !< Calculate the bins edge
    !<--------------------------------------------------------------------------
    subroutine calc_bins_edge
        implicit none
        integer :: i
        kappa_min_log = log10(kappa_min)
        dkappa_log = (log10(kappa_max) - kappa_min_log) / nbins_kappa
        do i = 0, nbins_kappa
            kappa_bins_edge(i+1) = 10**(kappa_min_log + dkappa_log * i)
        enddo
    end subroutine calc_bins_edge

    !<--------------------------------------------------------------------------
    !< Initialize temporary data
    !<--------------------------------------------------------------------------
    subroutine init_tmp_data(nx, ny, nz)
        implicit none
        integer, intent(in) :: nx, ny, nz
        allocate(tmpx(nx, ny, nz))
        allocate(tmpy(nx, ny, nz))
        allocate(tmpz(nx, ny, nz))
        allocate(stmp(nx, ny, nz))

        tmpx = 0.0
        tmpy = 0.0
        tmpz = 0.0
        stmp = 0.0
    end subroutine init_tmp_data

    !<--------------------------------------------------------------------------
    !< Free temporary data
    !<--------------------------------------------------------------------------
    subroutine free_tmp_data
        implicit none
        deallocate(tmpx, tmpy, tmpz)
        deallocate(stmp)
    end subroutine free_tmp_data

    !<--------------------------------------------------------------------------
    !< Get commandline arguments
    !<--------------------------------------------------------------------------
    subroutine get_cmd_args
        use flap                                !< FLAP package
        use penf
        implicit none
        type(command_line_interface) :: cli     !< Command Line Interface (CLI).
        integer(I4P)                 :: error   !< Error trapping flag.
        call cli%init(progname    = 'kappa_dist', &
                      authors     = 'Xiaocan Li', &
                      help        = 'Usage: ', &
                      description = 'Calculate the distribution of magnetic curvature', &
                      examples    = ['kappa_dist -rp simulation_root_path'])
        call cli%add(switch='--rootpath', switch_ab='-rp', &
            help='simulation root path', required=.true., act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--nbins_kappa', switch_ab='-nk', &
            help='Number of bins for kappa', &
            required=.false., def='500', act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--kappa_min', switch_ab='-kl', &
            help='Minimum kappa for histogram (in 1/di)', &
            required=.false., def='1E-2', act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--kappa_max', switch_ab='-kh', &
            help='Minimum kappa for histogram (in 1/di)', &
            required=.false., def='1E3', act='store', error=error)
        if (error/=0) stop
        call cli%get(switch='-rp', val=rootpath, error=error)
        if (error/=0) stop
        call cli%get(switch='-nk', val=nbins_kappa, error=error)
        if (error/=0) stop
        call cli%get(switch='-kl', val=kappa_min, error=error)
        if (error/=0) stop
        call cli%get(switch='-kh', val=kappa_max, error=error)
        if (error/=0) stop

        if (myid == 0) then
            print '(A,A)', 'The simulation rootpath: ', trim(adjustl(rootpath))
            print '(A, I0)', 'Number of bins for kappa: ', nbins_kappa
            print '(A, 2E10.3)', 'Minimum and maximum temperature: ', &
                kappa_min, kappa_max
        endif
    end subroutine get_cmd_args

    !<--------------------------------------------------------------------------
    !< Calculate the distribution of magnetic curvature
    !<--------------------------------------------------------------------------
    subroutine calc_kappa
        use mpi_topology, only: htg
        use picinfo, only: domain
        use pic_fields, only: init_magnetic_fields, free_magnetic_fields, &
            open_magnetic_field_files, close_magnetic_field_files, &
            read_magnetic_fields
        use configuration_translate, only: output_format
        use parameters, only: tp1, tp2
        implicit none
        integer :: tframe, nframes, posf, fh1, tindex
        character(len=256) :: fname
        logical :: dir_e

        call init_magnetic_fields(htg%nx, htg%ny, htg%nz)
        call init_tmp_data(htg%nx, htg%ny, htg%nz)

        if (output_format == 1) then
            call open_magnetic_field_files
        endif

        do tframe = tp1, tp2
            if (myid==master) print*, tframe
            if (output_format /= 1) then
                tindex = domain%fields_interval * (tframe - tp1)
                call open_magnetic_field_files(tindex)
                call read_magnetic_fields(tp1)
                call close_magnetic_field_files
                if (myid == master) print*, "Finished reading magnetic fields"
            else
                call read_magnetic_fields(tframe)
                if (myid == master) print*, "Finished reading magnetic fields"
            endif
            call set_dists_zero
            call calc_kappa_single
            call save_kappa(tframe)
        enddo

        if (output_format == 1) then
            call close_magnetic_field_files
        endif

        call free_tmp_data
        call free_magnetic_fields
    end subroutine calc_kappa

    !<--------------------------------------------------------------------------
    !< Calculate the distribution of magnetic curvature for a single frame
    !<--------------------------------------------------------------------------
    subroutine calc_kappa_single
        use pic_fields, only: bx, by, bz, absB, interp_bfield_node_ghost
        use neighbors_module, only: ixl, iyl, izl, ixh, iyh, izh, idx, idy, idz
        use mpi_topology, only: htg, ht
        use picinfo, only: mime
        implicit none
        real(fp) :: kappax, kappay, kappaz, kappa, smime
        integer :: ix, iy, iz, ikappa

        smime = sqrt(mime)

        ! Interpolation magnetic field to nodes including ghost cells
        call interp_bfield_node_ghost

        ! Normalize magnetic field (not ideal but OK)
        bx = bx / absB
        by = by / absB
        bz = bz / absB

        do ix = 1, htg%nx
            tmpx(ix, :, :) = bx(ix, :, :) * (bx(ixh(ix), :, :) - bx(ixl(ix), :, :)) * idx(ix)
            tmpy(ix, :, :) = bx(ix, :, :) * (by(ixh(ix), :, :) - by(ixl(ix), :, :)) * idx(ix)
            tmpz(ix, :, :) = bx(ix, :, :) * (bz(ixh(ix), :, :) - bz(ixl(ix), :, :)) * idx(ix)
        enddo

        do iy = 1, htg%ny
            tmpx(:, iy, :) = tmpx(:, iy, :) + &
                by(:, iy, :) * (bx(:, iyh(iy), :) - bx(:, iyl(iy), :)) * idy(iy)
            tmpy(:, iy, :) = tmpy(:, iy, :) + &
                by(:, iy, :) * (by(:, iyh(iy), :) - by(:, iyl(iy), :)) * idy(iy)
            tmpz(:, iy, :) = tmpz(:, iy, :) + &
                by(:, iy, :) * (bz(:, iyh(iy), :) - bz(:, iyl(iy), :)) * idy(iy)
        enddo

        do iz = 1, htg%nz
            tmpx(:, :, iz) = tmpx(:, :, iz) + &
                bz(:, :, iz) * (bx(:, :, izh(iz)) - bx(:, :, izl(iz))) * idz(iz)
            tmpy(:, :, iz) = tmpy(:, :, iz) + &
                bz(:, :, iz) * (by(:, :, izh(iz)) - by(:, :, izl(iz))) * idz(iz)
            tmpz(:, :, iz) = tmpz(:, :, iz) + &
                bz(:, :, iz) * (bz(:, :, izh(iz)) - bz(:, :, izl(iz))) * idz(iz)
        enddo

        ! Shift kappa fields to remove ghost cells at lower end along x-, y-, and z-directions.
        if (ht%ix > 0) then
            tmpx(1:ht%nx, :, :) = tmpx(2:ht%nx+1, :, :)
            tmpy(1:ht%nx, :, :) = tmpy(2:ht%nx+1, :, :)
            tmpz(1:ht%nx, :, :) = tmpz(2:ht%nx+1, :, :)
        endif
        if (ht%iy > 0) then
            tmpx(:, 1:ht%ny, :) = tmpx(:, 2:ht%ny+1, :)
            tmpy(:, 1:ht%ny, :) = tmpy(:, 2:ht%ny+1, :)
            tmpz(:, 1:ht%ny, :) = tmpz(:, 2:ht%ny+1, :)
        endif
        if (ht%iz > 0) then
            tmpx(:, :, 1:ht%nz) = tmpx(:, :, 2:ht%nz+1)
            tmpy(:, :, 1:ht%nz) = tmpy(:, :, 2:ht%nz+1)
            tmpz(:, :, 1:ht%nz) = tmpz(:, :, 2:ht%nz+1)
        endif

        do iz = 1, ht%nz
            do iy = 1, ht%ny
                do ix = 1, ht%nx
                    kappax = tmpx(ix, iy, iz)
                    kappay = tmpy(ix, iy, iz)
                    kappaz = tmpz(ix, iy, iz)
                    kappa = sqrt(kappax**2 + kappay**2 + kappaz**2) * smime
                    ikappa = (log10(kappa) - kappa_min_log) / dkappa_log + 1
                    if (ikappa > 0 .and. ikappa <= nbins_kappa) then
                        fkappa_local(ikappa) = fkappa_local(ikappa) + 1
                    endif
                enddo
            enddo
        enddo

        call MPI_REDUCE(fkappa_local, fkappa_global, nbins_kappa, MPI_REAL, &
            MPI_SUM, master, MPI_COMM_WORLD, ierr)
    end subroutine calc_kappa_single

    !<--------------------------------------------------------------------------
    !< Save kappa
    !<--------------------------------------------------------------------------
    subroutine save_kappa(tindex)
        implicit none
        integer, intent(in) :: tindex
        integer :: fh1, posf
        character(len=16) :: tindex_str
        character(len=256) :: fname
        logical :: dir_e
        if (myid == master) then
            inquire(file='./data/kappa_dist/.', exist=dir_e)
            if (.not. dir_e) then
                call system('mkdir -p ./data/kappa_dist/')
            endif
            print*, "Saving kappa distribution resutls..."

            fh1 = 66
            write(tindex_str, "(I0)") tindex - 1

            fname = 'data/kappa_dist/fkappa_'//trim(tindex_str)//'.gda'
            open(unit=fh1, file=fname, access='stream', status='unknown', &
                form='unformatted', action='write')
            posf = 1
            write(fh1, pos=posf) nbins_kappa + 0.0
            posf = posf + 4
            write(fh1, pos=posf) kappa_bins_edge
            posf = posf + 4 * (nbins_kappa + 1)
            write(fh1, pos=posf) fkappa_global
            close(fh1)
        endif
    end subroutine save_kappa

end program magnetic_curvature_distribution
