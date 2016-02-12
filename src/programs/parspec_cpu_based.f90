!*******************************************************************************
! Main program to get particle energy spectrum. The spectra are pre-calculated
! during PIC simulations for each MPI process. This program is going to sum
! over all the MPI processes to get the energy spectrum over the whole box.
!*******************************************************************************
program parspec_cpu_based
    use mpi_module
    use constants, only: fp, dp
    use path_info, only: get_file_paths
    use picinfo, only: read_domain, broadcast_pic_info
    use particle_frames, only: get_particle_frames_hydro, nt, tinterval
    use commandline_arguments, only: get_cmdline_arguments, is_emax_cell
    use mpi_info_module, only: set_mpi_info
    use particle_energy_spectrum, only: init_energy_spectra, &
            free_energy_spectra, set_energy_spectra_zero, &
            sum_spectra_over_mpi, save_particle_spectra, calc_energy_bins
    implicit none
    real(dp) :: mp_elapsed
    integer :: ct
    real(fp), allocatable, dimension(:) :: fcore

    ! Initialize Message Passing
    call MPI_INIT(ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)

    call get_cmdline_arguments
    call get_file_paths
    if (myid==master) then
        call get_particle_frames_hydro
    endif
    call MPI_BCAST(nt, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
    call MPI_BCAST(tinterval, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)

    if (myid==master) then
        call read_domain
    endif
    call broadcast_pic_info
    call get_spectra_config
    call init_energy_spectra
    call calc_energy_bins
    call init_spectrum_one_core

    mp_elapsed = MPI_WTIME()

    do ct = 1, nt
        if (myid == master) then
            print*, "electron", ct
        endif
        call get_energy_spectrum(ct, 'e')
        call sum_spectra_over_mpi
        if (myid == master) then
            call differentiate_spectrum
        endif
        if (myid == master) then
            call save_particle_spectra(ct, 'e')
        endif
        call set_energy_spectra_zero
    enddo

    do ct = 1, nt
        if (myid == master) then
            print*, "ion", ct
        endif
        call get_energy_spectrum(ct, 'h')
        call sum_spectra_over_mpi
        if (myid == master) then
            call differentiate_spectrum
        endif
        if (myid == master) then
            call save_particle_spectra(ct, 'h')
        endif
        call set_energy_spectra_zero
    enddo

    mp_elapsed = MPI_WTIME() - mp_elapsed

    if (myid==master) then
        write(*,'(A, F6.1)') " Total time used (s): ", mp_elapsed
    endif

    call free_spectrum_one_core
    call free_energy_spectra
    call MPI_FINALIZE(ierr)

    contains

    !---------------------------------------------------------------------------
    ! Get the energy spectra configuration from energy.cxx
    !---------------------------------------------------------------------------
    subroutine get_spectra_config
        use path_info, only: rootpath
        use read_config, only: get_variable
        use spectrum_config, only: nbins, emin, emax, dve, dlogve, &
                calc_energy_interval
        implicit none
        integer :: fh
        real(fp) :: tmp
        fh = 101
        open(unit=fh, file=trim(rootpath)//'energy.cxx', status='old')
        tmp = get_variable(fh, 'int nbin', '=')
        nbins = int(tmp)
        emin = get_variable(fh, 'float eminp', '=')
        emax = get_variable(fh, 'float emaxp', '=')
        close(fh)
        if (myid == master) then
            write(*, "(A,I0)") " Number of energy bins: ", nbins
            write(*, "(A,E12.4,E12.4)") " Energy range: ", emin, emax
        endif
        call calc_energy_interval
    end subroutine get_spectra_config

    !---------------------------------------------------------------------------
    ! Get the energy spectra from one time frame. The spectra are saved for
    ! each MPI process of the PIC simulation.
    !---------------------------------------------------------------------------
    subroutine get_energy_spectrum(ct, species)
        use particle_frames, only: is_frame0, tinterval
        use path_info, only: rootpath
        use picinfo, only: domain
        use particle_energy_spectrum, only: flog
        implicit none
        integer, intent(in) :: ct
        character(len=1), intent(in) :: species
        character(len=256) :: fpath, dataset, fname
        character(len=16) :: cid, ctindex
        character(len=1) :: ptl
        integer :: np, fh, tindex, stat, access
        if (species == 'e') then
            ptl = 'e'
        else
            ptl = 'H'
        endif
        if (is_frame0) then
            ! The zeroth frame is saved
            tindex = (ct - 1) * tinterval
        else
            tindex = ct * tinterval
        endif
        write(ctindex, "(I0)") tindex
        fpath = trim(rootpath)//"hydro/T."//trim(ctindex)//"/"
        ! Check whether the file can be accessed
        stat = access(trim(fpath), 'r')
        fh = 201
        if (stat == 0) then
            do np = myid, domain%nproc - 1, numprocs
                write(cid, "(I0)") np
                dataset = trim(fpath)//"spectrum-"//ptl//"hydro."
                fname = trim(dataset)//trim(ctindex)//"."//trim(cid)
                open(unit=fh, file=trim(fname), status='unknown', &
                     form='unformatted', access='stream', action='read')
                ! print*, trim(fname)
                read(fh) fcore
                flog = flog + fcore
                close(fh)
            enddo
        endif
    end subroutine get_energy_spectrum

    !---------------------------------------------------------------------------
    ! Initialize spectrum for one core.
    !---------------------------------------------------------------------------
    subroutine init_spectrum_one_core
        use spectrum_config, only: nbins
        implicit none
        allocate(fcore(nbins))
        fcore = 0.0
    end subroutine init_spectrum_one_core

    !---------------------------------------------------------------------------
    ! Free spectrum for one core.
    !---------------------------------------------------------------------------
    subroutine free_spectrum_one_core
        implicit none
        deallocate(fcore)
    end subroutine free_spectrum_one_core

    !---------------------------------------------------------------------------
    ! Differentiate the spectra w.r.t the energy.
    !---------------------------------------------------------------------------
    subroutine differentiate_spectrum
        use spectrum_config, only: nbins
        use particle_energy_spectrum, only: ebins_log, flogsum
        implicit none
        integer :: i
        flogsum(1) = flogsum(1) / ebins_log(1)
        do i = 2, nbins
            flogsum(i) = flogsum(i) / (ebins_log(i)-ebins_log(i-1))
        enddo
    end subroutine differentiate_spectrum
end program parspec_cpu_based
