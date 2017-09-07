!*******************************************************************************
! Program to calculate particle energy spectrum and velocity distributions
! between magnetic field lines
!*******************************************************************************
program particle_spectrum_vdist_fieldlines
    use mpi_module
    use constants, only: fp, dp
    use path_info, only: get_file_paths
    use picinfo, only: read_domain, broadcast_pic_info, domain
    use particle_frames, only: get_particle_frames, nt, tinterval
    use spectrum_config, only: read_spectrum_config, set_spatial_range_de, &
            calc_pic_mpi_ids, init_pic_mpi_ranks, free_pic_mpi_ranks, &
            calc_pic_mpi_ranks, calc_velocity_interval, set_time_frame
    use velocity_distribution, only: init_velocity_bins, free_velocity_bins, &
            init_vdist_2d, set_vdist_2d_zero, free_vdist_2d, init_vdist_1d, &
            set_vdist_1d_zero, free_vdist_1d, calc_vdist_2d, calc_vdist_1d
    use particle_energy_spectrum, only: init_energy_spectra, &
            free_energy_spectra, calc_energy_spectra, &
            set_energy_spectra_zero
    use parameters, only: get_start_end_time_points, get_inductive_flag, &
            get_relativistic_flag
    use magnetic_field, only: init_magnetic_fields, free_magnetic_fields, &
            read_magnetic_fields
    use particle_info, only: species, get_ptl_mass_charge
    implicit none
    real(dp), allocatable, dimension(:,:) :: fieldline1, fieldline2
    integer :: nps1, nps2 ! Number of field line points
    integer :: ct, ct_field, ratio_particle_field
    integer :: ix_top_left, ix_top_right, ix_bottom_left, ix_bottom_right
    character(len=128) :: filename_top, filename_bottom
    character(len=128) :: filepath
    real(dp) :: mp_elapsed
    real(dp) :: xlim(2), zlim(2)
    character(len=256) :: rootpath

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
    call read_spectrum_config
    call calc_velocity_interval

    call read_fieldlines_data
    call calc_fieldline_range
    call set_spatial_range_de(xlim, zlim)
    call set_time_frame(ct)

    call calc_pic_mpi_ids
    call init_pic_mpi_ranks
    call calc_pic_mpi_ranks

    call init_energy_spectra
    call init_velocity_bins
    call init_vdist_2d
    call init_vdist_1d
    call init_magnetic_fields

    mp_elapsed = MPI_WTIME()

    ! Ratio of particle output interval to fields output interval
    ratio_particle_field = domain%Particle_interval / domain%fields_interval
    ct_field = ratio_particle_field * ct
    call read_magnetic_fields(ct_field)

    call get_ptl_mass_charge(species)
    call calc_spectrum_vdist(ct, species)

    call free_fieldlines_data

    mp_elapsed = MPI_WTIME() - mp_elapsed

    if (myid==master) then
        write(*,'(A, F6.1)') " Total time used (s): ", mp_elapsed
    endif

    call free_magnetic_fields
    call free_vdist_1d
    call free_vdist_2d
    call free_velocity_bins
    call free_pic_mpi_ranks
    call free_energy_spectra

    call MPI_FINALIZE(ierr)

    contains

    !---------------------------------------------------------------------------
    ! Read the data points of the two field lines
    !---------------------------------------------------------------------------
    subroutine read_fieldlines_data
        implicit none
        integer :: fh1, fh2
        character(len=64) :: buff
        integer :: file_size
        fh1 = 15
        fh2 = 16
        if (myid==master) then
            inquire(file=filename_bottom, size=file_size)
            open(unit=fh1, file=filename_bottom, access='stream', &
                status='unknown', form='unformatted', action='read')
            nps1 = file_size / 16  ! The data type is double
            inquire(file=filename_top, size=file_size)
            open(unit=fh2, file=filename_top, access='stream', &
                status='unknown', form='unformatted', action='read')
            nps2 = file_size / 16  ! The data type is double
        endif
        call MPI_BCAST(nps1, 1, MPI_INTEGER, master, MPI_COMM_WORLD, ierr)
        call MPI_BCAST(nps2, 1, MPI_INTEGER, master, MPI_COMM_WORLD, ierr)
        call init_fieldlines_data
        if (myid==master) then
            read(fh1) fieldline1
            close(fh1)
            read(fh2) fieldline2
            close(fh2)
        endif
        call MPI_BCAST(fieldline1, nps1*2, MPI_DOUBLE, master, MPI_COMM_WORLD, ierr)
        call MPI_BCAST(fieldline2, nps2*2, MPI_DOUBLE, master, MPI_COMM_WORLD, ierr)
    end subroutine read_fieldlines_data

    !---------------------------------------------------------------------------
    ! Calculate the range of the field line data
    ! xlim, zlim are in the unit of the ion initial length di
    !---------------------------------------------------------------------------
    subroutine calc_fieldline_range
        implicit none
        xlim(1) = min(minval(fieldline1(1,:)), minval(fieldline2(1,:)))
        xlim(2) = max(maxval(fieldline1(1,:)), maxval(fieldline2(1,:)))
        zlim(1) = min(minval(fieldline1(2,:)), minval(fieldline2(2,:)))
        zlim(2) = max(maxval(fieldline1(2,:)), maxval(fieldline2(2,:)))
    end subroutine calc_fieldline_range

    !---------------------------------------------------------------------------
    ! Initialize the data points of the two field lines
    !---------------------------------------------------------------------------
    subroutine init_fieldlines_data
        implicit none
        allocate(fieldline1(2, nps1))
        allocate(fieldline2(2, nps2))
    end subroutine init_fieldlines_data

    !---------------------------------------------------------------------------
    ! Free the data points of the two field lines
    !---------------------------------------------------------------------------
    subroutine free_fieldlines_data
        implicit none
        deallocate(fieldline1)
        deallocate(fieldline2)
    end subroutine free_fieldlines_data

    !---------------------------------------------------------------------------
    ! Calculate the field lines segment that bounds current PIC MPI rank
    !---------------------------------------------------------------------------
    subroutine calc_fieldlines_range
        use picinfo, only: mime
        use file_header, only: v0
        implicit none
        real(fp) :: x0, y0, z0, x1, y1, z1, smime
        integer :: i, j

        smime = sqrt(mime)
        ! Corners of this MPI process's domain. Change it to ion skin depth
        x0 = v0%x0 / smime
        y0 = v0%y0 / smime
        z0 = v0%z0 / smime
        x1 = x0 + domain%pic_nx * domain%dx / smime
        y1 = y0 + domain%pic_ny * domain%dy / smime
        z1 = z0 + domain%pic_nz * domain%dz / smime

        ! Bottom field line
        if (fieldline1(1, 1) >= x0) then
            ix_bottom_left = 1
        else
            do i = 2, nps1
                if (fieldline1(1, i-1) < x0 .and. fieldline1(1, i) >= x0) then
                    ix_bottom_left = i - 1
                    exit
                endif
            enddo
            if (fieldline2(1, i) >= x1) then
                ix_bottom_right = i
            else
                do j = i, nps1
                    if (fieldline1(1, j-1) < x1 .and. fieldline1(1, j) >= x1) then
                        ix_bottom_right = j
                        exit
                    endif
                enddo
                ! Float number comparison might not work
                if (j == nps1) then
                    ix_bottom_right = nps1
                endif
            endif
        endif

        ! Top field line
        if (fieldline2(1, 1) >= x0) then
            ix_top_left = 1
        else
            do i = 2, nps2
                if (fieldline2(1, i-1) < x0 .and. fieldline2(1, i) >= x0) then
                    ix_top_left = i - 1
                    exit
                endif
            enddo
            if (fieldline2(1, i) >= x1) then
                ix_top_right = i
            else
                do j = i, nps2
                    if (fieldline2(1, j-1) < x1 .and. fieldline2(1, j) >= x1) then
                        ix_top_right = j
                        exit
                    endif
                enddo
                ! Float number comparison might not work
                if (j == nps2) then
                    ix_top_right = nps2
                endif
            endif
        endif
    end subroutine calc_fieldlines_range

    !---------------------------------------------------------------------------
    ! Calculate spectrum and velocity distributions.
    !---------------------------------------------------------------------------
    subroutine calc_spectrum_vdist(ct, species)
        use mpi_module
        use constants, only: fp
        use particle_frames, only: tinterval
        use spectrum_config, only: nbins
        use particle_file, only: check_existence
        use particle_energy_spectrum, only: save_particle_spectra, &
                sum_spectra_over_mpi, calc_energy_bins
        use velocity_distribution, only: sum_vdist_1d_over_mpi, &
                sum_vdist_2d_over_mpi, save_vdist_1d, save_vdist_2d
        implicit none
        integer, intent(in) :: ct
        character(len=1), intent(in) :: species
        integer :: tindex
        logical :: is_exist

        call calc_energy_bins

        tindex = ct * tinterval
        call set_energy_spectra_zero
        call check_existence(tindex, species, is_exist)
        if (is_exist) then
            call calc_distributions_mpi(tindex, species)
            call sum_vdist_1d_over_mpi
            call sum_vdist_2d_over_mpi
            call sum_spectra_over_mpi
            if (myid == master) then
                call save_vdist_1d(ct, species)
                call save_vdist_2d(ct, species)
                call save_particle_spectra(ct, species)
            endif
        endif
    end subroutine calc_spectrum_vdist

    !---------------------------------------------------------------------------
    ! Calculate spectrum and velocity distributions for multiple PIC MPI ranks.
    !---------------------------------------------------------------------------
    subroutine calc_distributions_mpi(tindex, species)
        use mpi_module
        use picinfo, only: domain
        use file_header, only: pheader
        use spectrum_config, only: spatial_range, tot_pic_mpi, pic_mpi_ranks
        use particle_file, only: open_particle_file, check_particle_in_range, &
                close_particle_file, fh
        implicit none
        character(len=1), intent(in) :: species
        integer, intent(in) :: tindex
        character(len=50) :: cid
        logical :: isrange
        integer :: np, iptl
        integer :: IOstatus
        integer :: ntot

        ! Read particle data in parallel to generate distributions
        do np = myid, tot_pic_mpi-1, numprocs
            write(cid, "(I0)") pic_mpi_ranks(np+1)
            call open_particle_file(tindex, species, cid)
            isrange = check_particle_in_range(spatial_range)
            call calc_fieldlines_range

            if (isrange) then
                ! Loop over particles
                do iptl = 1, pheader%dim, 1
                    IOstatus = single_particle_energy_vel(fh)
                    if (IOstatus /= 0) exit
                enddo
            endif
            call close_particle_file
        enddo
    end subroutine calc_distributions_mpi

    !---------------------------------------------------------------------------
    ! Calculate energy and velocities for one single particle and update the
    ! distribution arrays.
    !---------------------------------------------------------------------------
    function single_particle_energy_vel(fh) result(IOstatus)
        use picinfo, only: mime
        use particle_module, only: ptl, calc_particle_energy, px, py, pz, &
                calc_ptl_coord, calc_para_perp_velocity
        use particle_energy_spectrum, only: update_energy_spectrum
        use velocity_distribution, only: update_vdist_1d, update_vdist_2d
        use spectrum_config, only: spatial_range
        use constants, only: fp
        implicit none
        integer, intent(in) :: fh
        real(fp) :: px_di, py_di, pz_di, smime
        real(dp) :: zt, zb
        integer :: IOstatus

        read(fh, IOSTAT=IOstatus) ptl
        if (IOstatus == 0) then
            call calc_ptl_coord
            smime = sqrt(mime)
            px_di = px / smime
            pz_di = pz / smime
            call get_zpositions(px_di, zt, zb)

            if (pz_di <= zt .and. pz_di >= zb) then
                call calc_particle_energy
                call calc_para_perp_velocity
                call update_energy_spectrum
                call update_vdist_2d
                call update_vdist_1d
            endif
        endif

    end function single_particle_energy_vel

    !---------------------------------------------------------------------------
    ! Get the z position on the field lines with the particle x position
    !---------------------------------------------------------------------------
    subroutine get_zpositions(px_di, zt, zb)
        implicit none
        real(fp), intent(in) :: px_di
        real(dp), intent(out) :: zt, zb
        real(fp) :: delta_top, delta_bottom
        integer :: itop, ibottom
        real(dp) :: x
        integer :: i
        do i = ix_top_left, ix_top_right - 1
            if (fieldline2(1, i) <= px_di .and.  fieldline2(1, i+1) > px_di) then
                itop = i
                exit
            endif
        enddo
        x = fieldline2(1, i)
        delta_top = (px_di - x) / (fieldline2(1, i+1) - x)
        zt = fieldline2(2, i) * (1 - delta_top) + fieldline2(2, i+1) * delta_top

        do i = ix_bottom_left, ix_bottom_right - 1
            if (fieldline1(1, i) <= px_di .and.  fieldline1(1, i+1) > px_di) then
                ibottom = i
                exit
            endif
        enddo
        x = fieldline1(1, i)
        delta_bottom = (px_di - x) / (fieldline1(1, i+1) - x)
        zb = fieldline1(2, i) * (1 - delta_bottom) + &
             fieldline1(2, i+1) * delta_bottom
    end subroutine get_zpositions

    !---------------------------------------------------------------------------
    ! Get commandline arguments
    !---------------------------------------------------------------------------
    subroutine get_cmd_args
        use flap                                !< FLAP package
        use penf
        implicit none
        type(command_line_interface) :: cli     !< Command Line Interface (CLI).
        integer(I4P)                 :: error   !< Error trapping flag.
        call cli%init(progname = 'particle_spectrum_vdist_fieldlines', &
            authors     = 'Xiaocan Li', &
            help        = 'Usage: ', &
            description = 'Get particle spectrum and velocity distributions between two field lines', &
            examples    = ['particle_spectrum_vdist_fieldlines -rp simulation_root_path &
                            -ft filename_top -fb filename_bottom -fp filepath -t 40 -p e'])
        call cli%add(switch='--rootpath', switch_ab='-rp', &
            help='simulation root path', required=.true., act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--fieldline_top', switch_ab='-ft', help='a string', &
            required=.true., act='store', error=error) ; if (error/=0) stop
        call cli%add(switch='--fieldline_bottom', switch_ab='-fb', help='a string', &
            required=.true., act='store', error=error) ; if (error/=0) stop
        call cli%add(switch='--filepath', switch_ab='-fp', help='a string', &
            required=.true., act='store', error=error) ; if (error/=0) stop
        call cli%add(switch='--time_frame', switch_ab='-t', help='an integer', &
            required=.true., act='store', error=error) ; if (error/=0) stop
        call cli%add(switch='--species', switch_ab='-p', help='a string', &
            required=.false., def='e', act='store', error=error) ; if (error/=0) stop
        call cli%get(switch='-rp', val=rootpath, error=error)
        if (error/=0) stop
        call cli%get(switch='-ft', val=filename_top, error=error)
        if (error/=0) stop
        call cli%get(switch='-fb', val=filename_bottom, error=error)
        if (error/=0) stop
        call cli%get(switch='-fp', val=filepath, error=error)
        if (error/=0) stop
        call cli%get(switch='-t', val=ct, error=error)
        if (error/=0) stop
        call cli%get(switch='-p', val=species, error=error)
        if (error/=0) stop

        if (myid == 0) then
            print '(A,A)',   'The simulation rootpath: ', trim(adjustl(rootpath))
            print '(A)',     'filepath:        '//trim(adjustl(filepath))
            print '(A)',     'filename_top:    '//trim(adjustl(filename_top))
            print '(A)',     'filename_bottom: '//trim(adjustl(filename_bottom))
            print '(A, I0)', 'time_frame:      ', ct
            print '(A, A)',  'species:         ', species
        endif

        filename_top = trim(adjustl(filepath))//'/'//trim(adjustl(filename_top))
        filename_bottom = trim(adjustl(filepath))//'/'//trim(adjustl(filename_bottom))

    end subroutine get_cmd_args

end program particle_spectrum_vdist_fieldlines
