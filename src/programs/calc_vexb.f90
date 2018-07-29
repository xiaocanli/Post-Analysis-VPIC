!<******************************************************************************
!< Calculate ExB drift velocity using electron information
!<******************************************************************************
program calc_vexb
    use mpi_module
    use constants, only: fp, dp
    use time_info, only: output_record_initial => output_record
    implicit none
    integer :: nvar, separated_pre_post, fd_tinterval
    character(len=256) :: rootpath
    character(len=1), parameter :: species = 'e'
    logical :: frequent_dump, use_emf
    real(fp), allocatable, dimension(:, :, :) :: vexb_x, vexb_y, vexb_z
    real(fp), allocatable, dimension(:, :, :) :: tmpx, tmpy, tmpz, stmp

    call init_analysis
    call commit_analysis
    call end_analysis

    contains

    !<--------------------------------------------------------------------------
    !< Commit the analysis
    !<--------------------------------------------------------------------------
    subroutine commit_analysis
        use mpi_topology, only: htg
        use pic_fields, only: init_magnetic_fields, init_electric_fields, &
            init_vfields, init_number_density, init_pressure_tensor, &
            free_magnetic_fields, free_electric_fields, &
            free_vfields, free_number_density, free_pressure_tensor
        implicit none

        call init_magnetic_fields(htg%nx, htg%ny, htg%nz)
        if (use_emf) then
            call init_electric_fields(htg%nx, htg%ny, htg%nz)
        else
            call init_vfields(htg%nx, htg%ny, htg%nz)
            call init_number_density(htg%nx, htg%ny, htg%nz)
            call init_pressure_tensor(htg%nx, htg%ny, htg%nz)
        endif
        call init_vexb(htg%nx, htg%ny, htg%nz)
        call init_tmp_data(htg%nx, htg%ny, htg%nz)

        call eval_exb(.false., 'no-suffic')
        if (frequent_dump) then
            call eval_exb(.true., 'pre')
            call eval_exb(.true., 'post')
        endif

        call free_magnetic_fields
        if (use_emf) then
            call free_electric_fields
        else
            call free_vfields
            call free_number_density
            call free_pressure_tensor
        endif
        call free_vexb
        call free_tmp_data
    end subroutine commit_analysis

    !<--------------------------------------------------------------------------
    !< Evaluate ExB drift velocity
    !< Args:
    !<  with_suffix: whether files have suffix
    !<  suffix: the suffix name
    !<--------------------------------------------------------------------------
    subroutine eval_exb(with_suffix, suffix)
        use picinfo, only: domain
        use configuration_translate, only: tindex_start
        use configuration_translate, only: output_format
        use pic_fields, only: open_magnetic_field_files, open_electric_field_files, &
            open_vfield_files, open_number_density_file, open_pressure_tensor_files, &
            close_magnetic_field_files, close_electric_field_files, &
            close_vfield_files, close_number_density_file, &
            close_pressure_tensor_files, read_magnetic_fields, &
            read_electric_fields, read_vfields, read_number_density, &
            read_pressure_tensor
        implicit none
        logical, intent(in) :: with_suffix
        character(*), intent(in) :: suffix
        integer :: tframe, tindex, out_record, tp1_local, tp2_local
        logical :: dfile

        out_record = output_record_initial
        if (output_format == 1) then
            if (with_suffix) then
                call open_magnetic_field_files(suffix)
                if (use_emf) then
                    call open_electric_field_files(suffix)
                else
                    call open_vfield_files('e', suffix)
                    call open_number_density_file('e', suffix)
                    call open_pressure_tensor_files('e', suffix)
                endif
            else
                call open_magnetic_field_files
                if (use_emf) then
                    call open_electric_field_files
                else
                    call open_vfield_files('e')
                    call open_number_density_file('e')
                    call open_pressure_tensor_files('e')
                endif
            endif
            call adjust_tp2(with_suffix, suffix, tp1_local, tp2_local)
        endif

        if (output_format /= 1) then
            dfile= .true.
            tindex = tindex_start
            if (suffix == "pre") then
                if (tindex > 0) tindex = tindex - fd_tinterval
            endif
            if (suffix == "post") then
                if (tindex > 0) then
                    tindex = tindex + fd_tinterval
                else
                    tindex = 1
                endif
            endif

            do while (dfile)
                if (myid==master) print *, " Time slice: ", tindex
                if (with_suffix) then
                    call open_magnetic_field_files(tindex, suffix)
                    if (use_emf) then
                        call open_electric_field_files(tindex, suffix)
                    else
                        call open_vfield_files('e', tindex, suffix)
                        call open_number_density_file('e', tindex, suffix)
                        call open_pressure_tensor_files('e', tindex, suffix)
                    endif
                else
                    call open_magnetic_field_files(tindex)
                    if (use_emf) then
                        call open_electric_field_files(tindex)
                    else
                        call open_vfield_files('e', tindex)
                        call open_number_density_file('e', tindex)
                        call open_pressure_tensor_files('e', tindex)
                    endif
                endif
                call read_magnetic_fields(tp1_local)
                if (use_emf) then
                    call read_electric_fields(tp1_local)
                else
                    call read_vfields(tp1_local)
                    call read_number_density(tp1_local)
                    call read_pressure_tensor(tp1_local)
                endif
                call close_magnetic_field_files
                if (use_emf) then
                    call close_electric_field_files
                else
                    call close_vfield_files
                    call close_number_density_file
                    call close_pressure_tensor_files
                endif
                if (use_emf) then
                    call eval_exb_emf
                else
                    call eval_exb_single
                endif
                call write_exb(1, 1, tindex, with_suffix, suffix)
                if (suffix == 'pre') then
                    if (tindex == 0) then
                        tindex = domain%fields_interval - fd_tinterval
                    else
                        tindex = tindex + domain%fields_interval
                    endif
                endif
                if (suffix == 'post') then
                    if (tindex == 1) then
                        tindex = domain%fields_interval + fd_tinterval
                    else
                        tindex = tindex + domain%fields_interval
                    endif
                endif
                call check_file_existence(with_suffix, suffix, tindex, dfile)
                if (dfile) out_record = out_record + 1
            enddo
        else
            do tframe = tp1_local, tp2_local
                if (myid==master) print*, tframe
                call read_magnetic_fields(tframe)
                if (use_emf) then
                    call read_electric_fields(tframe)
                    call eval_exb_emf
                else
                    call read_vfields(tframe)
                    call read_number_density(tframe)
                    call read_pressure_tensor(tframe)
                    call eval_exb_single
                endif
                call write_exb(tframe, tp1_local, 0, with_suffix, suffix)
            enddo
        endif

        if (output_format == 1) then
            call close_magnetic_field_files
            if (use_emf) then
                call close_electric_field_files
            else
                call close_vfield_files
                call close_number_density_file
                call close_pressure_tensor_files
            endif
        endif
    end subroutine eval_exb

    !<--------------------------------------------------------------------------
    !< Evaluate ExB drift velocity for a single time step
    !<--------------------------------------------------------------------------
    subroutine eval_exb_single
        use pic_fields, only: bx, by, bz, absB, num_rho, vx, vy, vz, &
            pxx, pxy, pxz, pyx, pyy, pyz, pzx, pzy, pzz, &
            interp_bfield_node_ghost
        use neighbors_module, only: ixl, iyl, izl, ixh, iyh, izh, idx, idy, idz
        use mpi_topology, only: htg
        implicit none
        integer :: ix, iy, iz

        call interp_bfield_node_ghost
        stmp = -1.0 / (num_rho * absB**2)
        do ix = 1, htg%nx
            tmpx(ix, :, :) = (pxx(ixh(ix), :, :) - pxx(ixl(ix), :, :)) * idx(ix)
            tmpy(ix, :, :) = (pxy(ixh(ix), :, :) - pxy(ixl(ix), :, :)) * idx(ix)
            tmpz(ix, :, :) = (pxz(ixh(ix), :, :) - pxz(ixl(ix), :, :)) * idx(ix)
        enddo

        do iy = 1, htg%ny
            tmpx(:, iy, :) = tmpx(:, iy, :) + (pyx(:, iyh(iy), :) - pyx(:, iyl(iy), :)) * idy(iy)
            tmpy(:, iy, :) = tmpy(:, iy, :) + (pyy(:, iyh(iy), :) - pyy(:, iyl(iy), :)) * idy(iy)
            tmpz(:, iy, :) = tmpz(:, iy, :) + (pyz(:, iyh(iy), :) - pyz(:, iyl(iy), :)) * idy(iy)
        enddo

        do iz = 1, htg%nz
            tmpx(:, :, iz) = tmpx(:, :, iz) + (pzx(:, :, izh(iz)) - pzx(:, :, izl(iz))) * idz(iz)
            tmpy(:, :, iz) = tmpy(:, :, iz) + (pzy(:, :, izh(iz)) - pzy(:, :, izl(iz))) * idz(iz)
            tmpz(:, :, iz) = tmpz(:, :, iz) + (pzz(:, :, izh(iz)) - pzz(:, :, izl(iz))) * idz(iz)
        enddo
        vexb_x = (tmpy * bz - tmpz * by) * stmp
        vexb_y = (tmpz * bx - tmpx * bz) * stmp
        vexb_z = (tmpx * by - tmpy * bx) * stmp

        stmp = (vx * bx + vy * by + vz * bz) / absB**2
        vexb_x = (vx - stmp * bx) + vexb_x
        vexb_y = (vy - stmp * by) + vexb_y
        vexb_z = (vz - stmp * bz) + vexb_z
    end subroutine eval_exb_single

    !<--------------------------------------------------------------------------
    !< Evaluate ExB drift velocity for a single time step using EMF
    !<--------------------------------------------------------------------------
    subroutine eval_exb_emf
        use pic_fields, only: bx, by, bz, absB, ex, ey, ez, &
            interp_emf_node_ghost
        implicit none
        integer :: ix, iy, iz

        call interp_emf_node_ghost
        vexb_x = (ey * bz - ez * by) / absB**2
        vexb_y = (ez * bx - ex * bz) / absB**2
        vexb_z = (ex * by - ey * bx) / absB**2
    end subroutine eval_exb_emf

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
        call cli%init(progname    = 'calc_vexb', &
                      authors     = 'Xiaocan Li', &
                      help        = 'Usage: ', &
                      description = 'Calculate ExB using electron hydro', &
                      examples    = ['fluid_drift_energization -rp simulation_root_path'])
        call cli%add(switch='--rootpath', switch_ab='-rp', &
            help='simulation root path', required=.true., &
            act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--separated_pre_post', switch_ab='-pp', &
            help='separated pre and post fields', required=.false., act='store', &
            def='1', error=error)
        if (error/=0) stop
        call cli%add(switch='--frequent_dump', switch_ab='-fd', &
            help='whether VPIC dumps fields frequently', required=.false., &
            act='store_true', def='.false.', error=error)
        if (error/=0) stop
        call cli%add(switch='--fd_tinterval', switch_ab='-ft', &
            help='Frame interval when dumping 3 continuous frames', &
            required=.false., def='1', act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--use_emf', switch_ab='-ue', &
            help='whether using only EMF to calculate ExB', required=.false., &
            act='store_true', def='.false.', error=error)
        if (error/=0) stop
        call cli%get(switch='-rp', val=rootpath, error=error)
        if (error/=0) stop
        call cli%get(switch='-pp', val=separated_pre_post, error=error)
        if (error/=0) stop
        call cli%get(switch='-fd', val=frequent_dump, error=error)
        if (error/=0) stop
        call cli%get(switch='-ft', val=fd_tinterval, error=error)
        if (error/=0) stop
        call cli%get(switch='-ue', val=use_emf, error=error)
        if (error/=0) stop

        if (myid == 0) then
            print '(A,A)', 'The simulation rootpath: ', trim(adjustl(rootpath))
            if (separated_pre_post) then
                print '(A)', 'Fields at previous and next time steps are saved separately'
            endif
            if (frequent_dump) then
                print '(A, I0)', 'Frame interval between previous and current step is: ', &
                    fd_tinterval
            endif
            if (use_emf) then
                print '(A)', 'We only use EMF to calculate ExB drift'
            endif
        endif
    end subroutine get_cmd_args

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
    !< Initialize ExB drift velocity
    !<--------------------------------------------------------------------------
    subroutine init_vexb(nx, ny, nz)
        implicit none
        integer, intent(in) :: nx, ny, nz
        allocate(vexb_x(nx, ny, nz))
        allocate(vexb_y(nx, ny, nz))
        allocate(vexb_z(nx, ny, nz))

        vexb_x = 0.0
        vexb_y = 0.0
        vexb_z = 0.0
    end subroutine init_vexb

    !<--------------------------------------------------------------------------
    !< Free temporary data
    !<--------------------------------------------------------------------------
    subroutine free_vexb
        implicit none
        deallocate(vexb_x, vexb_y, vexb_z)
    end subroutine free_vexb

    !<---------------------------------------------------------------------------
    !< Check whether next time step exists
    !<---------------------------------------------------------------------------
    subroutine check_file_existence(with_suffix, suffix, tindex, dfile)
        implicit none
        logical, intent(in) :: with_suffix
        character(*), intent(in) :: suffix
        integer, intent(in) :: tindex
        logical, intent(out) :: dfile
        character(len=256) :: fname
        dfile = .false.
        if (with_suffix) then
            write(fname, "(A,I0,A)") &
                trim(adjustl(rootpath))//"data/bx_"//trim(suffix)//"_", &
                tindex, ".gda"
        else
            write(fname, "(A,I0,A)") &
                trim(adjustl(rootpath))//"data/bx_", tindex, ".gda"
        endif
        inquire(file=trim(fname), exist=dfile)
    end subroutine check_file_existence

    !<--------------------------------------------------------------------------
    !< Adjust tp2 in case that it is too large
    !<--------------------------------------------------------------------------
    subroutine adjust_tp2(with_suffix, suffix, tp1_local, tp2_local)
        use path_info, only: filepath
        use picinfo, only: domain
        use parameters, only: tp1, tp2
        implicit none
        logical, intent(in) :: with_suffix
        character(*), intent(in) :: suffix
        integer, intent(out) :: tp1_local, tp2_local
        character(len=256) :: fname
        integer(kind=8) :: filesize
        integer :: nt
        if (with_suffix) then
            fname = trim(adjustl(filepath))//'bx_'//trim(suffix)//'.gda'
        else
            fname = trim(adjustl(filepath))//'bx.gda'
        endif
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
    ! Write the ExB drift data to a file use MPI/IO.
    !---------------------------------------------------------------------------
    subroutine write_exb(tframe, tstart, tindex, with_suffix, suffix)
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
        logical, intent(in) :: with_suffix
        character(*), intent(in) :: suffix
        character(len=256) :: fname1, fname2, fname3
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
            if (with_suffix) then
                fname1 = trim(adjustl(outputpath))//'vexb_x_'//trim(suffix)//'_'//trim(cfname)//'.gda'
                fname2 = trim(adjustl(outputpath))//'vexb_y_'//trim(suffix)//'_'//trim(cfname)//'.gda'
                fname3 = trim(adjustl(outputpath))//'vexb_z_'//trim(suffix)//'_'//trim(cfname)//'.gda'
            else
                fname1 = trim(adjustl(outputpath))//'vexb_x_'//trim(cfname)//'.gda'
                fname2 = trim(adjustl(outputpath))//'vexb_y_'//trim(cfname)//'.gda'
                fname3 = trim(adjustl(outputpath))//'vexb_z_'//trim(cfname)//'.gda'
            endif
        else
            if (with_suffix) then
                fname1 = trim(adjustl(outputpath))//'vexb_x_'//suffix//'.gda'
                fname2 = trim(adjustl(outputpath))//'vexb_y_'//suffix//'.gda'
                fname3 = trim(adjustl(outputpath))//'vexb_z_'//suffix//'.gda'
            else
                fname1 = trim(adjustl(outputpath))//'vexb_x.gda'
                fname2 = trim(adjustl(outputpath))//'vexb_y.gda'
                fname3 = trim(adjustl(outputpath))//'vexb_z.gda'
            endif
        endif
        call open_data_mpi_io(fname1, MPI_MODE_RDWR+MPI_MODE_CREATE, fileinfo, fh)
        call write_data_mpi_io(fh, filetype_nghost, subsizes_nghost, &
            disp, offset, vexb_x(ixl:ixh, iyl:iyh, izl:izh))
        call MPI_FILE_CLOSE(fh, ierror)

        call open_data_mpi_io(fname2, MPI_MODE_RDWR+MPI_MODE_CREATE, fileinfo, fh)
        call write_data_mpi_io(fh, filetype_nghost, subsizes_nghost, &
            disp, offset, vexb_y(ixl:ixh, iyl:iyh, izl:izh))
        call MPI_FILE_CLOSE(fh, ierror)

        call open_data_mpi_io(fname3, MPI_MODE_RDWR+MPI_MODE_CREATE, fileinfo, fh)
        call write_data_mpi_io(fh, filetype_nghost, subsizes_nghost, &
            disp, offset, vexb_z(ixl:ixh, iyl:iyh, izl:izh))
        call MPI_FILE_CLOSE(fh, ierror)
    end subroutine write_exb

end program calc_vexb
