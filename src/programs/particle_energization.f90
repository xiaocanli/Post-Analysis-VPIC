!!******************************************************************************
!! Module for calculating particle energization
!!******************************************************************************
program particle_energization
    use constants, only: fp, dp
    use mpi_module
    use path_info, only: set_filepath
    use particle_info, only: species, ptl_mass, ptl_charge
    use parameters, only: tp1
    use configuration_translate, only: output_format
    implicit none
    character(len=256) :: rootpath
    character(len=16) :: dir_emf, dir_hydro
    real :: start, finish, step1, step2
    integer :: tstart, tend, tinterval, tframe, fields_interval
    integer, parameter :: nbins = 60
    integer, parameter :: nvar = 4
    real(fp), parameter :: emin = 1E-4
    real(fp), parameter :: emax = 1E2
    real(fp), allocatable, dimension(:) :: ebins
    real(fp), allocatable, dimension(:, :) :: fbins, fbins_sum
    real(fp) :: de_log, emin_log
    integer :: i, tp_emf, tp_hydro
    logical :: is_translated_file

    call MPI_INIT(ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)

    call cpu_time(start)

    call get_cmd_args

    call init_analysis

    call calc_particle_energization

    call end_analysis

    call cpu_time(finish)
    if (myid == master) then
        print '("Time = ",f9.4," seconds.")',finish-start
    endif

    call MPI_FINALIZE(ierr)

    contains

    !<--------------------------------------------------------------------------
    !< Initialize energy bins and distributions
    !<--------------------------------------------------------------------------
    subroutine init_dists
        implicit none
        de_log = (log10(emax) - log10(emin)) / nbins
        emin_log = log10(emin)
        do i = 1, nbins + 1
            ebins(i) = emin * 10**(de_log * (i - 1))
        enddo
        ebins = ebins / ptl_mass
        fbins = 0.0
        fbins_sum = 0.0
    end subroutine init_dists

    !<--------------------------------------------------------------------------
    !< Calculate particle energization due to parallel and perpendicular
    !< electric field, compression and shear
    !<--------------------------------------------------------------------------
    subroutine calc_particle_energization
        use picinfo, only: domain
        use topology_translate, only: ht
        use mpi_topology, only: htg
        use pic_fields, only: init_electric_fields, init_magnetic_fields, &
            init_velocity_fields, free_electric_fields, free_magnetic_fields, &
            free_velocity_fields, open_electric_field_files, &
            open_magnetic_field_files, open_velocity_field_files, &
            read_electric_fields, read_magnetic_fields, read_velocity_fields, &
            close_electric_field_files, close_magnetic_field_files, &
            close_velocity_field_files
        use interpolation_emf, only: init_emfields, free_emfields
        use interpolation_vel_mom, only: init_vel_mom, free_vel_mom
        use interpolation_comp_shear, only: init_exb_drift, free_exb_drift, &
            read_exb_drift, init_comp_shear, free_comp_shear, calc_comp_shear, &
            init_comp_shear_single, free_comp_shear_single, &
            open_exb_drift_files, close_exb_drift_files
        implicit none
        integer :: dom_x, dom_y, dom_z
        real(fp) :: dx_domain, dy_domain, dz_domain

        call init_emfields
        call init_vel_mom
        call init_comp_shear_single
        if (is_translated_file) then
            call init_electric_fields(htg%nx, htg%ny, htg%nz)
            call init_magnetic_fields(htg%nx, htg%ny, htg%nz)
            call init_velocity_fields(htg%nx, htg%ny, htg%nz)
            call init_exb_drift(htg%nx, htg%ny, htg%nz)
            call init_comp_shear(htg%nx, htg%ny, htg%nz)
        endif

        allocate(ebins(nbins + 1))
        allocate(fbins(nbins + 1, nvar))
        allocate(fbins_sum(nbins + 1, nvar))

        call init_dists

        if (myid == master) then
            print '(A)', 'Finished initializing the analysis'
        endif

        if (is_translated_file .and. output_format == 1) then
            call set_filepath(dir_emf)
            call open_electric_field_files
            call open_magnetic_field_files
            call open_velocity_field_files(species)
            call open_exb_drift_files
        endif

        call cpu_time(step1)
        do tframe = tstart, tend, tinterval
            if (myid == master) print*, tframe
            tp_emf = tframe / fields_interval
            if (is_translated_file) then
                if (output_format /= 1) then
                    ! Fields at each time step are saved in different files
                    call set_filepath(dir_emf)
                    call open_electric_field_files(tframe)
                    call open_magnetic_field_files(tframe)
                    call open_velocity_field_files(species, tframe)
                    call open_exb_drift_files(tframe)
                    call read_electric_fields(tp1)
                    call read_magnetic_fields(tp1)
                    call read_velocity_fields(tp1)
                    call read_exb_drift(tp1)
                    call close_magnetic_field_files
                    call close_electric_field_files
                    call close_velocity_field_files
                    call close_exb_drift_files
                else
                    ! Fields at all time steps are saved in the same file
                    call read_electric_fields(tp_emf + 1)
                    call read_magnetic_fields(tp_emf + 1)
                    call read_velocity_fields(tp_emf + 1)
                    call read_exb_drift(tp_emf + 1)
                endif
                call calc_comp_shear(htg%nx, htg%ny, htg%nz)
            endif  ! is_translated_file
            dx_domain = domain%lx_de / domain%pic_tx
            dy_domain = domain%ly_de / domain%pic_ty
            dz_domain = domain%lz_de / domain%pic_tz
            do dom_z = ht%start_z, ht%stop_z
                do dom_y = ht%start_y, ht%stop_y
                    do dom_x = ht%start_x, ht%stop_x
                        call calc_particle_energization_single(tframe, &
                            dom_x, dom_y, dom_z, dx_domain, dy_domain, dz_domain)
                    enddo ! x
                enddo ! y
            enddo ! z
            call save_particle_energization(tframe)
            call cpu_time(step2)
            if (myid == master) then
                print '("Time for this step = ",f6.3," seconds.")', step2 - step1
            endif
            step1 = step2
        enddo  ! Time loop

        if (is_translated_file .and. output_format == 1) then
            call close_electric_field_files
            call close_magnetic_field_files
            call close_velocity_field_files
            call close_exb_drift_files
        endif

        deallocate(ebins, fbins, fbins_sum)

        if (is_translated_file) then
            call free_electric_fields
            call free_magnetic_fields
            call free_velocity_fields
            call free_exb_drift
            call free_comp_shear
        endif
        call free_emfields
        call free_vel_mom
        call free_comp_shear_single
    end subroutine calc_particle_energization

    !<--------------------------------------------------------------------------
    !< Calculate particle energization due to parallel and perpendicular
    !< electric field, compression and shear for particles in a single PIC MPI rank
    !<--------------------------------------------------------------------------
    subroutine calc_particle_energization_single(tindex, dom_x, dom_y, dom_z, &
            dx_domain, dy_domain, dz_domain)
        use picinfo, only: domain
        use topology_translate, only: ht
        use rank_index_mapping, only: index_to_rank
        use interpolation_emf, only: read_emfields_single
        use particle_module, only: ptl, calc_interp_param, &
            iex, jex, kex, iey, jey, key, iez, jez, kez, ibx, jbx, kbx, &
            iby, jby, kby, ibz, jbz, kbz, dx_ex, dy_ex, dz_ex, &
            dx_ey, dy_ey, dz_ey, dx_ez, dy_ez, dz_ez, dx_bx, dx_by, dx_bz, &
            dy_bx, dy_by, dy_bz, dz_bx, dz_by, dz_bz, ino, jno, kno, &
            dnx, dny, dnz, particle
        use interpolation_emf, only: trilinear_interp_only_bx, &
            trilinear_interp_only_by, trilinear_interp_only_bz, &
            trilinear_interp_ex, trilinear_interp_ey, trilinear_interp_ez, &
            set_emf, bx0, by0, bz0, ex0, ey0, ez0
        use interpolation_vel_mom, only: trilinear_interp_vel_mom, &
            set_vel_mom, vx0, vy0, vz0, ux0, uy0, uz0
        use interpolation_comp_shear, only: trilinear_interp_comp_shear, &
            set_comp_shear_single, divv0, sigmaxx0, sigmaxy0, sigmaxz0, &
            sigmayy0, sigmayz0, sigmazz0
        use file_header, only: set_v0header, pheader
        use particle_file, only: open_particle_file, close_particle_file, fh
        implicit none
        integer, intent(in) :: tindex, dom_x, dom_y, dom_z
        real(fp), intent(in) :: dx_domain, dy_domain, dz_domain
        integer :: IOstatus, ibin, n, iptl
        real(fp) :: x0, y0, z0
        real(fp) :: ex_para, ey_para, ez_para, edotb, ib2
        real(fp) :: gama, igama, dke_para, dke_perp
        real(fp) :: vx, vy, vz, ux, uy, uz
        real(fp) :: pxx, pxy, pxz, pyx, pyy, pyz, pzx, pzy, pzz
        real(fp) :: bxx, byy, bzz, bxy, bxz, byz
        real(fp) :: weight, pscalar, ppara, pperp, bbsigma
        real(fp) :: pdivv, pshear, ptensor_divv
        type(particle), allocatable, dimension(:) :: ptls
        character(len=16) :: cid

        call index_to_rank(dom_x, dom_y, dom_z, domain%pic_tx, &
            domain%pic_ty, domain%pic_tz, n)
        write(cid, "(I0)") n - 1

        ! Read particle data
        if (species == 'e') then
            call open_particle_file(tindex, species, cid)
        else
            call open_particle_file(tindex, 'h', cid)
        endif
        allocate(ptls(pheader%dim))
        read(fh, IOSTAT=IOstatus) ptls
        call close_particle_file

        if (is_translated_file) then
            x0 = dx_domain * dom_x
            y0 = dy_domain * dom_y
            z0 = dz_domain * dom_z
            call set_v0header(domain%pic_nx, domain%pic_ny, domain%pic_nz, &
                x0, y0, z0, real(domain%dx), real(domain%dy), real(domain%dz))
            call set_emf(dom_x, dom_y, dom_z, domain%pic_tx, domain%pic_ty, &
                domain%pic_tz, ht%start_x, ht%start_y, ht%start_z)
            call set_vel_mom(dom_x, dom_y, dom_z, domain%pic_tx, domain%pic_ty, &
                domain%pic_tz, ht%start_x, ht%start_y, ht%start_z)
            call set_comp_shear_single(dom_x, dom_y, dom_z, domain%pic_tx, &
                domain%pic_ty, domain%pic_tz, ht%start_x, ht%start_y, ht%start_z)
        else
            call read_emfields_single(tindex, n - 1)
        endif
        do iptl = 1, pheader%dim, 1
            ptl = ptls(iptl)
            call calc_interp_param
            call trilinear_interp_only_bx(ibx, jbx, kbx, dx_bx, dy_bx, dz_bx)
            call trilinear_interp_only_by(iby, jby, kby, dx_by, dy_by, dz_by)
            call trilinear_interp_only_bz(ibz, jbz, kbz, dx_bz, dy_bz, dz_bz)
            call trilinear_interp_ex(iex, jex, kex, dx_ex, dy_ex, dz_ex)
            call trilinear_interp_ey(iey, jey, key, dx_ey, dy_ey, dz_ey)
            call trilinear_interp_ez(iez, jez, kez, dx_ez, dy_ez, dz_ez)
            call trilinear_interp_vel_mom(ibx, jbx, kbx, dx_bx, dy_bx, dz_bx)
            call trilinear_interp_comp_shear(iby, jby, kby, dx_by, dy_by, dz_by)
            bxx = bx0**2
            byy = by0**2
            bzz = bz0**2
            bxy = bx0 * by0
            bxz = bx0 * bz0
            byz = by0 * bz0
            ib2 = 1.0 / (bxx + byy + bzz)
            edotb = ex0 * bx0 + ey0 * by0 + ez0 * bz0
            ux = ptl%vx  ! v in ptl is actually gamma*v
            uy = ptl%vy
            uz = ptl%vz
            gama = sqrt(1.0 + ux**2 + uy**2 + uz**2)
            igama = 1.0 / gama
            vx = ux * igama
            vy = uy * igama
            vz = uz * igama
            ex_para = edotb * bx0 * ib2
            ey_para = edotb * by0 * ib2
            ez_para = edotb * bz0 * ib2
            dke_para = (ex_para * vx + ey_para * vy + ez_para * vz) * ptl%q
            dke_perp = (ex0 * vx + ey0 * vy + ez0 * vz) * ptl%q - dke_para
            weight = abs(ptl%q)
            pxx = (vx - vx0) * (ux - ux0) * ptl_mass * weight
            pxy = (vx - vx0) * (uy - uy0) * ptl_mass * weight
            pxz = (vx - vx0) * (uz - uz0) * ptl_mass * weight
            pyx = (vy - vy0) * (ux - ux0) * ptl_mass * weight
            pyy = (vy - vy0) * (uy - uy0) * ptl_mass * weight
            pyz = (vy - vy0) * (uz - uz0) * ptl_mass * weight
            pzx = (vz - vz0) * (ux - ux0) * ptl_mass * weight
            pzy = (vz - vz0) * (uy - uy0) * ptl_mass * weight
            pzz = (vz - vz0) * (uz - uz0) * ptl_mass * weight
            pscalar = (pxx + pyy + pzz) / 3.0
            ppara = pxx * bxx + pyy * byy + pzz * bzz + &
                (pxy + pyx) * bxy + (pxz + pzx) * bxz + (pyz + pzy) * byz
            ppara = ppara * ib2
            pperp = 0.5 * (pscalar * 3 - ppara)
            bbsigma = sigmaxx0 * bxx + sigmayy0 * byy + sigmazz0 * bzz + &
                2.0 * (sigmaxy0 * bxy + sigmaxz0 * bxz + sigmayz0 * byz)
            bbsigma = bbsigma * ib2
            pdivv = -pscalar * divv0
            pshear = (pperp - ppara) * bbsigma

            ibin = floor((log10(gama - 1) - emin_log) / de_log)
            if (ibin > 0 .and. ibin < nbins + 1) then
                fbins(ibin+1, 1) = fbins(ibin+1, 1) + dke_para
                fbins(ibin+1, 2) = fbins(ibin+1, 2) + dke_perp
                fbins(ibin+1, 3) = fbins(ibin+1, 3) + pdivv
                fbins(ibin+1, 4) = fbins(ibin+1, 4) + pshear
            endif
        enddo
        deallocate(ptls)
    end subroutine calc_particle_energization_single

    !!--------------------------------------------------------------------------
    !! Save particle energization due to parallel and perpendicular electric field.
    !!--------------------------------------------------------------------------
    subroutine save_particle_energization(tindex)
        implicit none
        integer, intent(in) :: tindex
        integer :: fh1, posf
        character(len=16) :: tindex_str
        character(len=256) :: fname
        logical :: dir_e
        call MPI_REDUCE(fbins, fbins_sum, (nbins+1)*nvar, &
                MPI_REAL, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
        if (myid == master) then
            inquire(file='./data/particle_interp/.', exist=dir_e)
            if (.not. dir_e) then
                call system('mkdir -p ./data/particle_interp/')
            endif
            print*, "Saving particle based analysis resutls..."

            fh1 = 66

            write(tindex_str, "(I0)") tindex
            fname = 'data/particle_interp/particle_energization_'//species
            fname = trim(fname)//"_"//trim(tindex_str)//'.gda'
            open(unit=fh1, file=fname, access='stream', status='unknown', &
                form='unformatted', action='write')
            posf = 1
            write(fh1, pos=posf) nbins + 1.0
            posf = posf + 4
            write(fh1, pos=posf) nvar + 0.0
            posf = posf + 4
            write(fh1, pos=posf) ebins
            posf = posf + (nbins + 1) * 4
            write(fh1, pos=posf) fbins_sum
            close(fh1)
        endif
    end subroutine save_particle_energization

    !!--------------------------------------------------------------------------
    !! Initialize the analysis.
    !!--------------------------------------------------------------------------
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

        call get_file_paths(rootpath)
        if (myid == master) then
            call read_domain
        endif
        call broadcast_pic_info
        call get_ptl_mass_charge(species)
        call get_start_end_time_points
        call get_relativistic_flag
        call get_energy_band_number
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

    !!--------------------------------------------------------------------------
    !! End the analysis by free the memory.
    !!--------------------------------------------------------------------------
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
        call cli%init(progname = 'interpolation', &
            authors     = 'Xiaocan Li', &
            help        = 'Usage: ', &
            description = 'Interpolate fields at particle positions', &
            examples    = ['interpolation -rp rootpath'])
        call cli%add(switch='--rootpath', switch_ab='-rp', &
            help='simulation root path', required=.true., act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--translated_file', switch_ab='-tf', &
            help='whether using translated fields file', required=.false., &
            act='store_true', def='.false.', error=error)
        if (error/=0) stop
        call cli%add(switch='--tstart', switch_ab='-ts', &
            help='Starting time frame', required=.false., act='store', &
            def='0', error=error)
        if (error/=0) stop
        call cli%add(switch='--tend', switch_ab='-te', help='Last time frame', &
            required=.true., act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--tinterval', switch_ab='-ti', help='Time interval', &
            required=.true., act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--fields_interval', switch_ab='-fi', &
            help='Time interval for PIC fields', &
            required=.true., act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--species', switch_ab='-sp', &
            help="Particle species: 'e' or 'h'", required=.false., &
            act='store', def='e', error=error)
        if (error/=0) stop
        call cli%add(switch='--dir_emf', switch_ab='-de', &
            help='EMF data directory', required=.false., &
            act='store', def='data', error=error)
        if (error/=0) stop
        call cli%add(switch='--dir_hydro', switch_ab='-dh', &
            help='Hydro data directory', required=.false., &
            act='store', def='data', error=error)
        if (error/=0) stop
        call cli%get(switch='-rp', val=rootpath, error=error)
        if (error/=0) stop
        call cli%get(switch='-tf', val=is_translated_file, error=error)
        if (error/=0) stop
        call cli%get(switch='-ts', val=tstart, error=error)
        if (error/=0) stop
        call cli%get(switch='-te', val=tend, error=error)
        if (error/=0) stop
        call cli%get(switch='-ti', val=tinterval, error=error)
        if (error/=0) stop
        call cli%get(switch='-fi', val=fields_interval, error=error)
        if (error/=0) stop
        call cli%get(switch='-sp', val=species, error=error)
        if (error/=0) stop
        call cli%get(switch='-de', val=dir_emf, error=error)
        if (error/=0) stop
        call cli%get(switch='-dh', val=dir_hydro, error=error)
        if (error/=0) stop

        if (myid == 0) then
            print '(A,A)', 'The simulation rootpath: ', trim(adjustl(rootpath))
            print '(A,L1)', 'Whether using translated fields file: ', is_translated_file
            print '(A,I0,A,I0,A,I0)', 'Min, max and interval: ', &
                tstart, ' ', tend, ' ', tinterval
            print '(A,I0)', 'Time interval for electric and magnetic fields: ', &
                fields_interval
            if (species == 'e') then
                print '(A,A)', 'Particle: electron'
            else if (species == 'h' .or. species == 'i') then
                print '(A,A)', 'Particle: ion'
            endif
            print '(A,A)', 'EMF data directory: ', trim(dir_emf)
            print '(A,A)', 'Hydro data directory: ', trim(dir_hydro)
        endif
    end subroutine get_cmd_args
end program particle_energization
