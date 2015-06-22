!*******************************************************************************
! Module of particle energy spectrum.
!*******************************************************************************
module particle_energy_spectrum
    use constants, only: fp, dp
    use picinfo, only: domain 
    use path_info, only: rootpath
    implicit none
    private
    public f, fsum, flog, flogsum, ebins_lin, ebins_log
    public init_energy_spectra, free_energy_spectra, calc_energy_spectra, &
           set_energy_spectra_zero_single, calc_energy_bins, &
           init_energy_spectra_single, free_energy_spectra_single, &
           calc_energy_spectrum_single
    real(dp), allocatable, dimension(:) :: f, fsum, flog, flogsum
    real(dp), allocatable, dimension(:) :: ebins_lin, ebins_log

    contains

    !---------------------------------------------------------------------------
    ! Initialize particle energy spectrum.
    !---------------------------------------------------------------------------
    subroutine init_energy_spectra
        use mpi_module
        use spectrum_config, only: nbins
        implicit none
        call init_energy_spectra_single
        if (myid==master) then
            allocate(fsum(nbins))
            allocate(flogsum(nbins))
        endif
        call set_energy_spectra_zero
    end subroutine init_energy_spectra

    !---------------------------------------------------------------------------
    ! Initialize particle energy spectrum for current process.
    ! This can be used for both parallel routines and series routines.
    !---------------------------------------------------------------------------
    subroutine init_energy_spectra_single
        use spectrum_config, only: nbins
        implicit none
        allocate(f(nbins))
        allocate(flog(nbins))
        allocate(ebins_lin(nbins))
        allocate(ebins_log(nbins))
        ebins_lin = 0.0
        ebins_log = 0.0
        call set_energy_spectra_zero_single
    end subroutine init_energy_spectra_single

    !---------------------------------------------------------------------------
    ! Set the flux to zeros at the beginning of each time frame.
    !---------------------------------------------------------------------------
    subroutine set_energy_spectra_zero
        use mpi_module
        implicit none
        call set_energy_spectra_zero_single
        if (myid==master) then
            fsum = 0.0
            flogsum = 0.0
        endif
    end subroutine set_energy_spectra_zero

    !---------------------------------------------------------------------------
    ! Set the flux to zeros at the beginning of each time frame.
    ! This can be used for both parallel routines and series routines.
    !---------------------------------------------------------------------------
    subroutine set_energy_spectra_zero_single
        implicit none
        f = 0.0
        flog = 0.0
    end subroutine set_energy_spectra_zero_single

    !---------------------------------------------------------------------------
    ! Free the memory used by the particle energy flux.
    !---------------------------------------------------------------------------
    subroutine free_energy_spectra
        use mpi_module
        implicit none
        call free_energy_spectra_single
        if (myid==master) then
            deallocate(fsum, flogsum)
        endif
    end subroutine free_energy_spectra

    !---------------------------------------------------------------------------
    ! Free the memory used by the particle energy flux.
    ! This can be used for both parallel routines and series routines.
    !---------------------------------------------------------------------------
    subroutine free_energy_spectra_single
        implicit none
        deallocate(f, flog)
        deallocate(ebins_lin, ebins_log)
    end subroutine free_energy_spectra_single

    !---------------------------------------------------------------------------
    ! Calculate the energy bins for linear and logarithmic cases.
    !---------------------------------------------------------------------------
    subroutine calc_energy_bins
        use spectrum_config, only: nbins, dve, dlogve, emin
        implicit none
        integer :: i
        do i = 1, nbins
            ebins_lin(i) = dve * i
            ebins_log(i) = emin * 10**(dlogve*i)
        enddo
    end subroutine calc_energy_bins

    !---------------------------------------------------------------------------
    ! Get particle energy spectrum from individual particle information.
    ! Input:
    !   species: 'e' for electron. 'h' for others.
    !---------------------------------------------------------------------------
    subroutine calc_energy_spectra(species)
        use mpi_module
        use constants, only: fp
        use particle_frames, only: nt, tinterval
        use spectrum_config, only: nbins, dve, dlogve, emin
        implicit none
        character(len=1), intent(in) :: species
        character(len=100) :: fname
        integer :: i, it, tindex
        logical :: is_exist, dir_e

        if (myid == master) then
            inquire(file='./spectrum/.', exist=dir_e)
            if (.not. dir_e) then
                call system('mkdir spectrum')
            endif
        endif

        call calc_energy_bins

        do it = 1, 1
            tindex = it * tinterval
            call set_energy_spectra_zero
            call check_existence(tindex, species, is_exist)
            if (is_exist) then
                call calc_energy_spectrum_mpi(tindex, species)
                ! Sum over all nodes to get the total energy spectrum
                call MPI_REDUCE(f, fsum, nbins, MPI_DOUBLE_PRECISION, &
                    MPI_SUM, 0, MPI_COMM_WORLD, ierr)
                call MPI_REDUCE(flog, flogsum, nbins, MPI_DOUBLE_PRECISION, &
                    MPI_SUM, 0, MPI_COMM_WORLD, ierr)

                !  Now output the distribution on the master node
                if (myid==master) then
                    !print *," *** Finished Creating Spectrum ***"
                    write(fname, "(A,A1,A1,I0)") "spectrum/spectrum-", &
                                                 species, ".", it
                    open(unit=10, file=trim(fname), status='unknown')
                    do i=1, nbins
                        write(10, "(4e12.4)") ebins_lin(i), fsum(i), &
                            ebins_log(i), flogsum(i)
                    enddo
                    close(10)
                endif
            endif
        enddo
    end subroutine calc_energy_spectra

    !---------------------------------------------------------------------------
    ! Check the existence of the dataset. This is for the case that there is
    ! time gaps in the output files.
    ! Inputs:
    !   tindex: the time index, indicating the time step numbers in PIC simulation.
    !   species: 'e' for electron. 'h' for others.
    !---------------------------------------------------------------------------
    subroutine check_existence(tindex, species, existFlag)
        implicit none
        character(len=1), intent(in) :: species
        integer, intent(in) :: tindex
        logical, intent(out) :: existFlag
        character(len=20) :: ctindex
        character(len=150) :: dataset, fname

        write(ctindex, "(I0)") tindex
        dataset = trim(adjustl(rootpath))//"particle/T."//trim(ctindex)
        dataset = trim(adjustl(dataset))//"/"//species//"particle."
        fname = trim(dataset)//trim(ctindex)//".0"
        inquire(file=fname, exist=existFlag)
        if (.not. existFlag) then
            print*, fname, " doesn't exist." 
            print*, "There is probably a gap in the output."
        endif
    end subroutine check_existence

    !---------------------------------------------------------------------------
    ! Read particle data and calculate the energy spectrum for one time frame.
    ! This subroutine is used in parallel procedures.
    ! Input:
    !   tindex: the time index, indicating the time step numbers in PIC simulation.
    !   species: 'e' for electron. 'h' for others.
    !---------------------------------------------------------------------------
    subroutine calc_energy_spectrum_mpi(tindex, species)
        use mpi_module
        use file_header, only: read_boilerplate, read_particle_header, &
                               pheader, v0
        use spectrum_config, only: spatial_range
        use picinfo, only: domain
        implicit none
        character(len=1), intent(in) :: species
        integer, intent(in) :: tindex
        character(len=50) :: ctindex, cid
        character(len=150) :: dataset, fname
        real(fp) :: x0, y0, z0, x1, y1, z1
        logical :: isrange1, isrange2
        integer :: fh, np, iptl

        fh = 10
        write(ctindex, "(I0)") tindex
        dataset = trim(adjustl(rootpath))//"particle/T."//trim(ctindex)
        dataset = trim(adjustl(dataset))//"/"//species//"particle."
        ! Read particle data in parallel to generate distributions
        do np = 0, domain%nproc-numprocs, numprocs
            write(cid, "(I0)") myid + np
            fname = trim(dataset)//trim(ctindex)//"."//trim(cid)
            open(unit=fh, file=trim(fname), status='unknown', &
                 form='unformatted', access='stream', action='read')
            write(*, '(A,A,A,I0,A,I0)') "Reading --> ", trim(fname), &
                " Physical rank = ", myid, " np = ", np

            call read_boilerplate(fh)
            call read_particle_header(fh)

            ! Corners of this MPI process's domain
            x0 = v0%x0
            y0 = v0%y0
            z0 = v0%z0
            x1 = v0%x0 + domain%pic_nx * domain%dx
            y1 = v0%y0 + domain%pic_ny * domain%dy
            z1 = v0%z0 + domain%pic_nz * domain%dz

            ! Only if the corners are within the box.
            ! Shift one grid to cover boundary.
            isrange1 = x1 >= (spatial_range(1,1) - domain%dx) &
                 .and. x1 <= (spatial_range(2,1) + domain%dx) &
                 .and. y1 >= (spatial_range(1,2) - domain%dy) &
                 .and. y1 <= (spatial_range(2,2) + domain%dy) &
                 .and. z1 >= (spatial_range(1,3) - domain%dz) &
                 .and. z1 <= (spatial_range(2,3) + domain%dz)
            isrange2 = x0 >= (spatial_range(1,1) - domain%dx) &
                 .and. x0 <= (spatial_range(2,1) + domain%dx) &
                 .and. y0 >= (spatial_range(1,2) - domain%dy) &
                 .and. y0 <= (spatial_range(2,2) + domain%dy) &
                 .and. z0 >= (spatial_range(1,3) - domain%dz) &
                 .and. z0 <= (spatial_range(2,3) + domain%dz)

            if (isrange1 .or. isrange2) then
                ! Loop over particles
                do iptl = 1, pheader%dim, 1
                    call single_particle_energy(fh)
                enddo
            endif

            close(fh)
        enddo
    end subroutine calc_energy_spectrum_mpi

    !---------------------------------------------------------------------------
    ! Read particle data and calculate the energy spectrum for one time frame.
    ! This procedure is only use one CPU core.
    ! Input:
    !   tindex: the time index, indicating the time step numbers in PIC simulation.
    !   species: 'e' for electron. 'h' for others.
    !---------------------------------------------------------------------------
    subroutine calc_energy_spectrum_single(tindex, species)
        use file_header, only: read_boilerplate, read_particle_header, &
                               pheader, v0
        use spectrum_config, only: spatial_range, corners_mpi
        use picinfo, only: domain
        implicit none
        character(len=1), intent(in) :: species
        integer, intent(in) :: tindex
        character(len=50) :: ctindex, cid
        character(len=150) :: dataset, fname
        integer :: np, iptl, fh
        integer :: ix, iy, iz

        fh = 10
        write(ctindex, "(I0)") tindex
        dataset = trim(adjustl(rootpath))//"particle/T."//trim(ctindex)
        dataset = trim(adjustl(dataset))//"/"//species//"particle."

        ! Read particle data and update the spectra
        do iz = corners_mpi(1,3), corners_mpi(2,3)
            do iy = corners_mpi(1,2), corners_mpi(2,2)
                do ix = corners_mpi(1,1), corners_mpi(2,1)
                    np = ix + iy*domain%pic_tx + iz*domain%pic_tx*domain%pic_ty
                    write(cid, "(I0)") np
                    fname = trim(dataset)//trim(ctindex)//"."//trim(cid)
                    open(unit=fh, file=trim(fname), status='unknown', &
                         form='unformatted', access='stream', action='read')
                    write(*, '(A,A,A,I0)') "Reading --> ", trim(fname), &
                        " np = ", np

                    call read_boilerplate(fh)
                    call read_particle_header(fh)

                    ! Loop over particles
                    do iptl = 1, pheader%dim, 1
                        call single_particle_energy(fh)
                    enddo

                    close(fh)

                enddo ! Z
            enddo ! Y
        enddo ! X
    end subroutine calc_energy_spectrum_single

    !---------------------------------------------------------------------------
    ! Read one single particle information, check if it is in the spatial range,
    ! calculate its energy and put it into the flux arrays.
    ! Input:
    !   fh: file handler.
    !---------------------------------------------------------------------------
    subroutine single_particle_energy(fh)
        use particle_module, only: ptl, calc_particle_energy, px, py, pz, &
                                   calc_ptl_coord
        use spectrum_config, only: spatial_range
        use constants, only: fp
        implicit none
        integer, intent(in) :: fh

        read(fh) ptl
        call calc_ptl_coord

        if ((px >= spatial_range(1, 1)) .and. (px <= spatial_range(2, 1)) .and. &
            (py >= spatial_range(1, 2)) .and. (py <= spatial_range(2, 2)) .and. &
            (pz >= spatial_range(1, 3)) .and. (pz <= spatial_range(2, 3))) then

            call calc_particle_energy
            call update_energy_spectrum
        endif

    end subroutine single_particle_energy

    !---------------------------------------------------------------------------
    ! Update particle energy spectrum.
    !---------------------------------------------------------------------------
    subroutine update_energy_spectrum
        use particle_module, only: ke
        use spectrum_config, only: dve, dlogve, emin, nbins
        implicit none
        real(fp) :: rbin, shift, dke
        integer :: ibin, ibin1

        rbin = ke / dve
        ibin = int(rbin)
        ibin1 = ibin + 1
        
        if ((ibin >= 1) .and. (ibin1 <= nbins)) then 
            shift = rbin - ibin
            f(ibin)  = f(ibin) - shift + 1
            f(ibin1) = f(ibin1) + shift
        endif

        ! Exceptions
        if ( ibin .eq. 0) then
            ! Add lower energies to the 1st band.
            f(ibin1) = f(ibin1) + 1
        endif
        dke = ke * dlogve
     
        ! Logarithmic scale
        ibin = (log10(ke)-log10(emin))/dlogve + 1
        if ((ibin >= 1) .and. (ibin1 <= nbins)) then 
            flog(ibin)  = flog(ibin) + 1.0/dke
        endif
    end subroutine update_energy_spectrum
    
end module particle_energy_spectrum
