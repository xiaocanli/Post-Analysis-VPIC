!<******************************************************************************
!< Calculate v.kappa and its distribution
!<******************************************************************************
program vdot_kappa
    use mpi_module
    use constants, only: fp, dp
    use time_info, only: output_record_initial => output_record
    use particle_info, only: get_ptl_mass_charge, ptl_mass, species
    implicit none
    character(len=256) :: rootpath
    real(fp) :: vkappa_min, vkappa_max
    integer :: nbins
    logical :: with_nrho
    real(fp), allocatable, dimension(:, :, :) :: vkappa
    real(fp), allocatable, dimension(:, :, :) :: tmpx, tmpy, tmpz, stmp
    real(fp), allocatable, dimension(:, :, :) :: vsx, vsy, vsz
    real(fp), allocatable, dimension(:) :: vkappa_bins_edge
    real(fp), allocatable, dimension(:) :: fvkappa_local, fvkappa_global
    real(fp) :: pmass_e, pmass_i
    real(fp) :: dvkappa_log, vkappa_min_log

    call init_analysis
    call get_ptl_mass_charge('e')
    pmass_e = ptl_mass
    call get_ptl_mass_charge('i')
    pmass_i = ptl_mass
    call commit_analysis
    call end_analysis

    contains

    !<--------------------------------------------------------------------------
    !< Commit the analysis
    !<--------------------------------------------------------------------------
    subroutine commit_analysis
        use mpi_topology, only: htg
        use pic_fields, only: init_magnetic_fields, init_vfields, &
            init_number_density, free_magnetic_fields, free_vfields, &
            free_number_density
        implicit none

        call init_magnetic_fields(htg%nx, htg%ny, htg%nz)
        call init_vfields(htg%nx, htg%ny, htg%nz)
        call init_number_density(htg%nx, htg%ny, htg%nz)
        call init_vkappa(htg%nx, htg%ny, htg%nz)
        call init_tmp_data(htg%nx, htg%ny, htg%nz)
        call init_vsingle(htg%nx, htg%ny, htg%nz)

        call init_dists
        call eval_vkappa
        call free_dists

        call free_magnetic_fields
        call free_vfields
        call free_number_density
        call free_vkappa
        call free_tmp_data
        call free_vsingle
    end subroutine commit_analysis

    !<--------------------------------------------------------------------------
    !< Initialize bins and distributions
    !<--------------------------------------------------------------------------
    subroutine init_dists
        implicit none
        integer :: i
        allocate(vkappa_bins_edge(nbins*2 + 5))
        allocate(fvkappa_local(nbins*2 + 4))
        allocate(fvkappa_global(nbins*2 + 4))
        vkappa_bins_edge = 0
        dvkappa_log = (log10(vkappa_max) - log10(vkappa_min)) / nbins
        vkappa_min_log = log10(vkappa_min)
        do i = nbins+4, nbins*2+5
            vkappa_bins_edge(i) = 10**(dvkappa_log * (i - nbins - 4) + vkappa_min_log)
        enddo
        vkappa_bins_edge(1:nbins+2) = vkappa_bins_edge(nbins*2+5:nbins+4:-1)
        vkappa_bins_edge(nbins+3) = 0

        call set_dists_zero
    end subroutine init_dists

    !<--------------------------------------------------------------------------
    !< Free bins and distributions
    !<--------------------------------------------------------------------------
    subroutine free_dists
        implicit none
        deallocate(vkappa_bins_edge)
        deallocate(fvkappa_local, fvkappa_global)
    end subroutine free_dists

    !<--------------------------------------------------------------------------
    !< Reset distributions to 0
    !<--------------------------------------------------------------------------
    subroutine set_dists_zero
        implicit none
        fvkappa_local = 0.0
        fvkappa_global = 0.0
    end subroutine set_dists_zero

    !<--------------------------------------------------------------------------
    !< Evaluate v.kappa
    !<--------------------------------------------------------------------------
    subroutine eval_vkappa
        use picinfo, only: domain
        use configuration_translate, only: tindex_start
        use configuration_translate, only: output_format
        use pic_fields, only: open_magnetic_field_files, open_vfield_files, &
            open_number_density_file, close_magnetic_field_files, &
            close_vfield_files, close_number_density_file, &
            read_magnetic_fields, read_vfields, read_number_density, &
            vfields_fh, nrho_fh
        use parameters, only: tp1
        implicit none
        integer :: tframe, tindex, out_record, tp1_local, tp2_local
        integer, dimension(3) :: vfields_fh_e, vfields_fh_i
        integer :: nrho_fh_e, nrho_fh_i
        logical :: dfile

        out_record = output_record_initial
        if (output_format == 1) then
            call open_magnetic_field_files
            call open_vfield_files('e')
            call open_number_density_file('e')
            ! Save file handlers so we open both electron and ion files
            vfields_fh_e = vfields_fh
            nrho_fh_e = nrho_fh
            call open_vfield_files('i')
            call open_number_density_file('i')
            vfields_fh_i = vfields_fh
            nrho_fh_i = nrho_fh
            call adjust_tp2(tp1_local, tp2_local)
        endif

        if (output_format /= 1) then
            dfile= .true.
            tindex = tindex_start

            do while (dfile)
                if (myid==master) print *, " Time slice: ", tindex
                call open_magnetic_field_files(tindex)
                call read_magnetic_fields(tp1)
                call close_magnetic_field_files
                if (myid == master) print*, "Finished reading magnetic fields"

                call open_vfield_files('e', tindex)
                call open_number_density_file('e', tindex)
                call read_vfields(tp1)
                call read_number_density(tp1)
                call close_vfield_files
                call close_number_density_file
                if (myid == master) print*, "Finished reading electron velocity and density fields"
                call vsingle_tmp

                call open_vfield_files('i', tindex)
                call open_number_density_file('i', tindex)
                call read_vfields(tp1)
                call read_number_density(tp1)
                call close_vfield_files
                call close_number_density_file
                if (myid == master) print*, "Finished reading ion velocity and density fields"
                call vsingle_final

                call eval_vkappa_single
                call write_vkappa(1, 1, tindex)
                call get_vkappa_dist
                call save_vkappa_dist(tindex)

                tindex = tindex + domain%fields_interval
                call check_file_existence(tindex, dfile)
                if (dfile) out_record = out_record + 1
            enddo
        else
            do tframe = tp1_local, tp2_local
                if (myid==master) print*, tframe
                call read_magnetic_fields(tframe)

                vfields_fh = vfields_fh_e
                nrho_fh = nrho_fh_e
                call read_vfields(tframe)
                call read_number_density(tframe)
                call vsingle_tmp

                vfields_fh = vfields_fh_i
                nrho_fh = nrho_fh_i
                call read_vfields(tframe)
                call read_number_density(tframe)
                call vsingle_final

                call eval_vkappa_single
                call write_vkappa(tframe, tp1_local, 0)
                call get_vkappa_dist
                call save_vkappa_dist(0)
            enddo
        endif

        if (output_format == 1) then
            call close_magnetic_field_files
            vfields_fh = vfields_fh_e
            nrho_fh = nrho_fh_e
            call close_vfield_files
            call close_number_density_file
            vfields_fh = vfields_fh_i
            nrho_fh = nrho_fh_i
            call close_vfield_files
            call close_number_density_file
        endif
    end subroutine eval_vkappa

    !<--------------------------------------------------------------------------
    !< Evaluate temporary single fluid velocity
    !<--------------------------------------------------------------------------
    subroutine vsingle_tmp
        use pic_fields, only: vx, vy, vz, num_rho
        implicit none
        vsx = vx
        vsy = vy
        vsz = vz
        tmpx = num_rho  ! For temporary use
    end subroutine vsingle_tmp

    !<--------------------------------------------------------------------------
    !< Evaluate final single fluid velocity
    !<--------------------------------------------------------------------------
    subroutine vsingle_final
        use pic_fields, only: vx, vy, vz, num_rho
        implicit none
        ! tmpx here is electron number density
        stmp = 1.0 / (tmpx * pmass_e + num_rho * pmass_i)
        vsx = (vsx * tmpx * pmass_e + vx * num_rho * pmass_i) * stmp
        vsy = (vsy * tmpx * pmass_e + vy * num_rho * pmass_i) * stmp
        vsz = (vsz * tmpx * pmass_e + vz * num_rho * pmass_i) * stmp

        if (with_nrho) then
            if (species == 'e') then
                stmp = tmpx
            else
                stmp = num_rho
            endif
        endif
    end subroutine vsingle_final

    !<--------------------------------------------------------------------------
    !< Evaluate v.kappa for a single time step
    !<--------------------------------------------------------------------------
    subroutine eval_vkappa_single
        use pic_fields, only: bx, by, bz, absB, interp_bfield_node_ghost
        use neighbors_module, only: ixl, iyl, izl, ixh, iyh, izh, idx, idy, idz
        use mpi_topology, only: htg
        implicit none
        integer :: ix, iy, iz

        ! Interpolation magnetic field to nodes
        call interp_bfield_node_ghost

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
        vkappa = (vsx * tmpx + vsy * tmpy + vsz * tmpz) / absB**2
        if (with_nrho) then
            vkappa = vkappa * stmp
        endif
    end subroutine eval_vkappa_single

    !<--------------------------------------------------------------------------
    !< Get the distribution of v.kappa
    !<--------------------------------------------------------------------------
    subroutine get_vkappa_dist
        use mpi_topology, only: range_out
        implicit none
        integer :: ixl, iyl, izl, ixh, iyh, izh
        integer :: ix, iy, iz, ivkappa
        real(fp) :: vkappa_tmp

        ixl = range_out%ixl
        ixh = range_out%ixh
        iyl = range_out%iyl
        iyh = range_out%iyh
        izl = range_out%izl
        izh = range_out%izh

        call set_dists_zero

        do iz = izl, izh
            do iy = iyl, iyh
                do ix = ixl, ixh
                    vkappa_tmp = vkappa(ix, iy, iz)
                    if (vkappa_tmp > 0) then
                        if (vkappa_tmp < vkappa_min) then
                            ivkappa = nbins + 3
                        else if (vkappa_tmp > vkappa_max) then
                            ivkappa = nbins*2 + 4
                        else
                            ivkappa = floor((log10(vkappa_tmp) - vkappa_min_log) / dvkappa_log)
                            ivkappa = nbins + ivkappa + 4
                        endif
                    else
                        if (-vkappa_tmp < vkappa_min) then
                            ivkappa = nbins + 2
                        else if (-vkappa_tmp > vkappa_max) then
                            ivkappa = 1
                        else
                            ivkappa = floor((log10(-vkappa_tmp) - vkappa_min_log) / dvkappa_log)
                            ivkappa = nbins - ivkappa + 1
                        endif
                    endif
                    fvkappa_local(ivkappa) = fvkappa_local(ivkappa) + 1
                enddo
            enddo
        enddo
        call MPI_REDUCE(fvkappa_local, fvkappa_global, (nbins+2)*2, &
            MPI_REAL, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
    end subroutine get_vkappa_dist

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
        call cli%init(progname    = 'vdote_kappa', &
                      authors     = 'Xiaocan Li', &
                      help        = 'Usage: ', &
                      description = 'Calculate v.kappa', &
                      examples    = ['vdot_kappa -rp simulation_root_path'])
        call cli%add(switch='--rootpath', switch_ab='-rp', &
            help='simulation root path', required=.true., &
            act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--with_nrho', switch_ab='-wn', &
            help='whether including number density', required=.false., &
            act='store_true', def='.false.', error=error)
        if (error/=0) stop
        call cli%add(switch='--species', switch_ab='-sp', &
            help="Particle species: 'e' or 'h'", required=.false., &
            act='store', def='e', error=error)
        if (error/=0) stop
        call cli%add(switch='--nbins', switch_ab='-nb', &
            help='Number of bins for vdot_kappa', &
            required=.false., def='100', act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--vkappa_min', switch_ab='-kl', &
            help='Minimum vkappa', &
            required=.false., def='1E-8', act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--vkappa_max', switch_ab='-kh', &
            help='Maximum vkappa', &
            required=.false., def='1E2', act='store', error=error)
        if (error/=0) stop
        call cli%get(switch='-rp', val=rootpath, error=error)
        if (error/=0) stop
        call cli%get(switch='-wn', val=with_nrho, error=error)
        if (error/=0) stop
        call cli%get(switch='-sp', val=species, error=error)
        if (error/=0) stop
        call cli%get(switch='-nb', val=nbins, error=error)
        if (error/=0) stop
        call cli%get(switch='-kl', val=vkappa_min, error=error)
        if (error/=0) stop
        call cli%get(switch='-kh', val=vkappa_max, error=error)
        if (error/=0) stop

        if (myid == 0) then
            print '(A,A)', 'The simulation rootpath: ', trim(adjustl(rootpath))
            print '(A,A)', 'Default particle species is: ', species
            if (with_nrho) then
                print '(A)', 'Calculate nv.kappa instead of v.kappa'
            endif
            print '(A,I0)', ' Number of bins for vkappa: ', nbins
            print '(A,E,E)', ' Minimum and maximum vdot_kappa: ', vkappa_min, vkappa_max
        endif
    end subroutine get_cmd_args

    !<--------------------------------------------------------------------------
    !< Initialize single fluid velocity
    !<--------------------------------------------------------------------------
    subroutine init_vsingle(nx, ny, nz)
        implicit none
        integer, intent(in) :: nx, ny, nz
        allocate(vsx(nx, ny, nz))
        allocate(vsy(nx, ny, nz))
        allocate(vsz(nx, ny, nz))

        vsx = 0.0
        vsy = 0.0
        vsz = 0.0
    end subroutine init_vsingle

    !<--------------------------------------------------------------------------
    !< Free single fluid velocity
    !<--------------------------------------------------------------------------
    subroutine free_vsingle
        implicit none
        deallocate(vsx, vsy, vsz)
    end subroutine free_vsingle

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
    !< Initialize v.kappa
    !<--------------------------------------------------------------------------
    subroutine init_vkappa(nx, ny, nz)
        implicit none
        integer, intent(in) :: nx, ny, nz
        allocate(vkappa(nx, ny, nz))

        vkappa = 0.0
    end subroutine init_vkappa

    !<--------------------------------------------------------------------------
    !< Free v.kappa
    !<--------------------------------------------------------------------------
    subroutine free_vkappa
        implicit none
        deallocate(vkappa)
    end subroutine free_vkappa

    !<---------------------------------------------------------------------------
    !< Check whether next time step exists
    !<---------------------------------------------------------------------------
    subroutine check_file_existence(tindex, dfile)
        implicit none
        integer, intent(in) :: tindex
        logical, intent(out) :: dfile
        character(len=256) :: fname
        dfile = .false.
        write(fname, "(A,I0,A)") trim(adjustl(rootpath))//"data/bx_", tindex, ".gda"
        inquire(file=trim(fname), exist=dfile)
    end subroutine check_file_existence

    !<--------------------------------------------------------------------------
    !< Adjust tp2 in case that it is too large
    !<--------------------------------------------------------------------------
    subroutine adjust_tp2(tp1_local, tp2_local)
        use path_info, only: filepath
        use picinfo, only: domain
        use parameters, only: tp1, tp2
        implicit none
        integer, intent(out) :: tp1_local, tp2_local
        character(len=256) :: fname
        integer(kind=8) :: filesize
        integer :: nt
        fname = trim(adjustl(filepath))//'bx.gda'
        inquire(file=trim(fname), size=filesize)
        nt = filesize / (domain%nx * domain%ny * domain%nz * 4)
        if ((tp2 - tp1 + 1) > nt) then
            tp2_local = tp1 + nt - 1
        else
            tp2_local = tp2
        endif
        tp1_local = tp1
    end subroutine adjust_tp2

    !---------------------------------------------------------------------------
    ! Write the v.kappa to a file use MPI/IO.
    !---------------------------------------------------------------------------
    subroutine write_vkappa(tframe, tstart, tindex)
        use constants, only: fp
        use picinfo, only: domain
        use path_info, only: outputpath
        use mpi_datatype_fields, only: filetype_nghost, subsizes_nghost
        use mpi_info_module, only: fileinfo
        use mpi_topology, only: range_out
        use mpi_io_module, only: open_data_mpi_io, write_data_mpi_io
        use configuration_translate, only: output_format
        implicit none
        integer, intent(in) :: tframe, tstart, tindex
        character(len=256) :: fname1
        integer :: fh
        integer :: ixl, iyl, izl, ixh, iyh, izh
        integer(kind=MPI_OFFSET_KIND) :: disp, offset
        character(len=16) :: cfname

        write(cfname, "(I0)") tindex

        ixl = range_out%ixl
        ixh = range_out%ixh
        iyl = range_out%iyl
        iyh = range_out%iyh
        izl = range_out%izl
        izh = range_out%izh

        disp = domain%nx * domain%ny * domain%nz * sizeof(MPI_REAL) * (tframe-tstart)
        offset = 0

        if (output_format /= 1) then
            if (with_nrho) then
                fname1 = trim(adjustl(outputpath))//'n'//species//'_vkappa_'//trim(cfname)//'.gda'
            else
                fname1 = trim(adjustl(outputpath))//'vkappa_'//trim(cfname)//'.gda'
            endif
        else
            if (with_nrho) then
                fname1 = trim(adjustl(outputpath))//'n'//species//'_vkappa.gda'
            else
                fname1 = trim(adjustl(outputpath))//'nvkappa.gda'
            endif
        endif
        call open_data_mpi_io(fname1, MPI_MODE_RDWR+MPI_MODE_CREATE, fileinfo, fh)
        call write_data_mpi_io(fh, filetype_nghost, subsizes_nghost, &
            disp, offset, vkappa(ixl:ixh, iyl:iyh, izl:izh))
        call MPI_FILE_CLOSE(fh, ierror)
    end subroutine write_vkappa

    !<--------------------------------------------------------------------------
    !< Save the distribution of vkappa
    !<--------------------------------------------------------------------------
    subroutine save_vkappa_dist(tindex)
        implicit none
        integer, intent(in) :: tindex
        integer :: fh1, posf
        character(len=16) :: tindex_str
        character(len=256) :: fname
        logical :: dir_e
        if (myid == master) then
            inquire(file='./data/vkappa_dist/.', exist=dir_e)
            if (.not. dir_e) then
                call system('mkdir -p ./data/vkappa_dist/')
            endif
            print*, "Saving the distribution of vdot_kappa..."

            fh1 = 66

            write(tindex_str, "(I0)") tindex
            fname = 'data/vkappa_dist/vkappa_dist'//'_'//species
            fname = trim(fname)//"_"//trim(tindex_str)//'.gda'
            open(unit=fh1, file=fname, access='stream', status='unknown', &
                form='unformatted', action='write')
            posf = 1
            write(fh1, pos=posf) (nbins*2 + 4.0)
            posf = posf + 4
            write(fh1, pos=posf) vkappa_bins_edge
            posf = posf + (nbins*2 + 4) * 5
            write(fh1, pos=posf) fvkappa_global
            close(fh1)
        endif
    end subroutine save_vkappa_dist

end program vdot_kappa
