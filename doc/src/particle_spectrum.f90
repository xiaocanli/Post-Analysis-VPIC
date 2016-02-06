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
           set_energy_spectra_zero, set_energy_spectra_zero_single, &
           calc_energy_bins, init_energy_spectra_single, &
           free_energy_spectra_single, calc_energy_spectrum_single, &
           sum_spectra_over_mpi, save_particle_spectra, update_energy_spectrum, &
           init_maximum_energy, set_maximum_energy_zero, update_maximum_energy, &
           get_maximum_energy_global, save_maximum_energy, free_maximum_energy, &
           init_emax_pic_mpi, free_emax_pic_mpi, set_emax_pic_mpi_zero
    real(dp), allocatable, dimension(:) :: f, fsum, flog, flogsum
    real(dp), allocatable, dimension(:) :: ebins_lin, ebins_log
    real(fp), allocatable, dimension(:) :: emax_local, emax_global
    real(fp), allocatable, dimension(:, :, :) :: emax_pic_mpi
    real(fp) :: emax_tmp

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
    ! Initialize array of maximum energy.
    !---------------------------------------------------------------------------
    subroutine init_maximum_energy(nt)
        use mpi_module
        implicit none
        integer, intent(in) :: nt
        allocate(emax_local(nt))
        if (myid == master) then
            allocate(emax_global(nt))
        endif
        call set_maximum_energy_zero
    end subroutine init_maximum_energy

    !---------------------------------------------------------------------------
    ! Set the maximum energy to zeros.
    !---------------------------------------------------------------------------
    subroutine set_maximum_energy_zero
        use mpi_module
        implicit none
        emax_local = 0.0
        if (myid == master) then
            emax_global = 0.0
        endif
    end subroutine set_maximum_energy_zero

    !---------------------------------------------------------------------------
    ! Free the array of the maximum energy.
    !---------------------------------------------------------------------------
    subroutine free_maximum_energy
        use mpi_module
        implicit none
        deallocate(emax_local)
        if (myid == master) then
            deallocate(emax_global)
        endif
    end subroutine free_maximum_energy

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
    ! Initialize the maximum energy array for one PIC MPI process
    !---------------------------------------------------------------------------
    subroutine init_emax_pic_mpi
        use picinfo, only: domain
        implicit none
        allocate(emax_pic_mpi(domain%pic_nx, domain%pic_ny, domain%pic_nz))
        call set_emax_pic_mpi_zero
    end subroutine init_emax_pic_mpi

    !---------------------------------------------------------------------------
    ! Set the maximum energy array to zeros
    !---------------------------------------------------------------------------
    subroutine set_emax_pic_mpi_zero
        implicit none
        emax_pic_mpi = 0.0
    end subroutine set_emax_pic_mpi_zero

    !---------------------------------------------------------------------------
    ! Free the maximum energy array for one PIC MPI process
    !---------------------------------------------------------------------------
    subroutine free_emax_pic_mpi
        use picinfo, only: domain
        implicit none
        deallocate(emax_pic_mpi)
    end subroutine free_emax_pic_mpi

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
    !   ct: current time frame.
    !   species: 'e' for electron. 'h' for others.
    !---------------------------------------------------------------------------
    subroutine calc_energy_spectra(ct, species)
        use mpi_module
        use constants, only: fp
        use particle_frames, only: tinterval
        use particle_file, only: check_existence
        use commandline_arguments, only: is_emax_cell
        use particle_maximum_energy, only: write_emax
        implicit none
        integer, intent(in) :: ct
        character(len=1), intent(in) :: species
        integer :: tindex
        logical :: is_exist

        call calc_energy_bins

        tindex = ct * tinterval
        call set_energy_spectra_zero
        call check_existence(tindex, species, is_exist)
        emax_tmp = emax_local(ct)
        if (is_exist) then
            if (is_emax_cell) then
                call calc_energy_spectrum_mpi_con(tindex, species)
                call write_emax(ct, species)
            else
                call calc_energy_spectrum_mpi(tindex, species)
            endif
            emax_local(ct) = emax_tmp  ! emax_tmp has been updated
            call sum_spectra_over_mpi
            if (myid == master) then
                call save_particle_spectra(ct, species)
            endif
        endif
    end subroutine calc_energy_spectra

    !---------------------------------------------------------------------------
    ! Sum particle energy spectrum over all MPI process.
    !---------------------------------------------------------------------------
    subroutine sum_spectra_over_mpi
        use mpi_module
        use spectrum_config, only: nbins
        implicit none
        ! Sum over all nodes to get the total energy spectrum
        call MPI_REDUCE(f, fsum, nbins, MPI_DOUBLE_PRECISION, &
                MPI_SUM, 0, MPI_COMM_WORLD, ierr)
        call MPI_REDUCE(flog, flogsum, nbins, MPI_DOUBLE_PRECISION, &
                MPI_SUM, 0, MPI_COMM_WORLD, ierr)
    end subroutine sum_spectra_over_mpi

    !---------------------------------------------------------------------------
    ! Update the global array of maximum energy.
    !---------------------------------------------------------------------------
    subroutine get_maximum_energy_global(nt)
        use mpi_module
        implicit none
        integer, intent(in) :: nt
        call MPI_REDUCE(emax_local, emax_global, nt, MPI_FLOAT, MPI_MAX, &
            master, MPI_COMM_WORLD, ierr)
    end subroutine get_maximum_energy_global

    !---------------------------------------------------------------------------
    ! Save particle energy spectrum to file.
    !---------------------------------------------------------------------------
    subroutine save_particle_spectra(ct, species)
        use spectrum_config, only: nbins
        implicit none
        integer, intent(in) :: ct
        character(len=1), intent(in) :: species
        logical :: dir_e
        character(len=100) :: fname
        integer :: i
        inquire(file='./spectrum/.', exist=dir_e)
        if (.not. dir_e) then
            call system('mkdir spectrum')
        endif
        ! print *," *** Finished Creating Spectrum ***"
        write(fname, "(A,A1,A1,I0)") "spectrum/spectrum-", &
                                     species, ".", ct
        open(unit=10, file=trim(fname), status='unknown')
        do i=1, nbins
            write(10, "(4e12.4)") ebins_lin(i), fsum(i), &
                    ebins_log(i), flogsum(i)
        enddo
        close(10)
    end subroutine save_particle_spectra

    !---------------------------------------------------------------------------
    ! Save the array of the maximum energy.
    !---------------------------------------------------------------------------
    subroutine save_maximum_energy(nt, species)
        implicit none
        integer, intent(in) :: nt
        character(len=1), intent(in) :: species
        logical :: dir_e
        character(len=100) :: fname
        integer :: i
        inquire(file='./spectrum/.', exist=dir_e)
        if (.not. dir_e) then
            call system('mkdir spectrum')
        endif
        ! print *," *** Finished Creating Spectrum ***"
        write(fname, "(A,A1,A)") "spectrum/emax-", species, ".dat"
        open(unit=10, file=trim(fname), status='unknown')
        do i=1, nt
            write(10, "(e12.4)") emax_global(i)
        enddo
        close(10)
    end subroutine save_maximum_energy

    !---------------------------------------------------------------------------
    ! Read particle data and calculate the energy spectrum for one time frame.
    ! This subroutine is used in parallel procedures. This routine assigns not
    ! continuous jobs for each process. The job index jumps numprocs each time.
    ! Input:
    !   tindex: the time index, indicating the time step numbers in PIC simulation.
    !   species: 'e' for electron. 'h' for others.
    !---------------------------------------------------------------------------
    subroutine calc_energy_spectrum_mpi(tindex, species)
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

        ! Read particle data in parallel to generate distributions
        do np = myid, tot_pic_mpi-1, numprocs
            write(cid, "(I0)") pic_mpi_ranks(np+1)
            call open_particle_file(tindex, species, cid)
            isrange = check_particle_in_range(spatial_range)

            if (isrange) then
                ! Loop over particles
                do iptl = 1, pheader%dim, 1
                    IOstatus = single_particle_energy(fh)
                    if (IOstatus /= 0) exit
                enddo
            endif

            call close_particle_file
        enddo
    end subroutine calc_energy_spectrum_mpi

    !---------------------------------------------------------------------------
    ! This routine distribute jobs in a way that they are continuous.
    ! Input:
    !   tindex: the time index, indicating the time step numbers in PIC simulation.
    !   species: 'e' for electron. 'h' for others.
    !---------------------------------------------------------------------------
    subroutine calc_energy_spectrum_mpi_con(tindex, species)
        use mpi_module
        use picinfo, only: domain
        use file_header, only: pheader
        use spectrum_config, only: spatial_range, tot_pic_mpi, pic_mpi_ranks
        use particle_file, only: open_particle_file, check_particle_in_range, &
                close_particle_file, fh
        use particle_maximum_energy, only: txs, tys, tzs, txe, tye, tze, &
                update_emax_array
        use commandline_arguments, only: is_emax_cell
        implicit none
        character(len=1), intent(in) :: species
        integer, intent(in) :: tindex
        character(len=50) :: cid
        logical :: isrange
        integer :: np, iptl
        integer :: IOstatus, ix, iy, iz, otx, oty, otz

        ! Read particle data in parallel to generate distributions
        do iz = tzs, tze
            do iy = tys, tye
                do ix = txs, txe
                    np = ix + iy*domain%pic_tx + iz*domain%pic_tx*domain%pic_ty
                    write(cid, "(I0)") pic_mpi_ranks(np+1)
                    call open_particle_file(tindex, species, cid)
                    isrange = check_particle_in_range(spatial_range)

                    if (isrange) then
                        ! Loop over particles
                        do iptl = 1, pheader%dim, 1
                            IOstatus = single_particle_energy(fh)
                            if (IOstatus /= 0) exit
                        enddo
                        if (is_emax_cell) then
                            otx = ix - txs
                            oty = iy - tys
                            otz = iz - tzs
                            call update_emax_array(emax_pic_mpi, otx, oty, otz)
                        endif
                    endif
                    call close_particle_file
                    ! Set to zeros, so it is not accumulated
                    call set_emax_pic_mpi_zero
                enddo
            enddo
        enddo
    end subroutine calc_energy_spectrum_mpi_con

    !---------------------------------------------------------------------------
    ! Read particle data and calculate the energy spectrum for one time frame.
    ! This procedure is only use one CPU core.
    ! Input:
    !   tindex: the time index, indicating the time step numbers in PIC simulation.
    !   species: 'e' for electron. 'h' for others.
    !---------------------------------------------------------------------------
    subroutine calc_energy_spectrum_single(tindex, species)
        use picinfo, only: domain
        use file_header, only: pheader
        use spectrum_config, only: spatial_range, corners_mpi
        use particle_file, only: open_particle_file, check_particle_in_range, &
                close_particle_file, fh
        implicit none
        character(len=1), intent(in) :: species
        integer, intent(in) :: tindex
        character(len=50) :: cid
        integer :: np, iptl
        integer :: ix, iy, iz
        integer :: IOstatus

        ! Read particle data and update the spectra
        do iz = corners_mpi(1,3), corners_mpi(2,3)
            do iy = corners_mpi(1,2), corners_mpi(2,2)
                do ix = corners_mpi(1,1), corners_mpi(2,1)
                    np = ix + iy*domain%pic_tx + iz*domain%pic_tx*domain%pic_ty
                    write(cid, "(I0)") np
                    call open_particle_file(tindex, species, cid)

                    ! Loop over particles
                    do iptl = 1, pheader%dim, 1
                        IOstatus = single_particle_energy(fh)
                        if (IOstatus /= 0) exit
                    enddo

                    call close_particle_file

                enddo ! X
            enddo ! Y
        enddo ! Z
    end subroutine calc_energy_spectrum_single

    !---------------------------------------------------------------------------
    ! Read one single particle information, check if it is in the spatial range,
    ! calculate its energy and put it into the flux arrays.
    ! Input:
    !   fh: file handler.
    ! Returns:
    !   IOstatus: '0' is OK. 'negative' indicates the end of a file.
    !             'positive' indicates something is wrong.
    !---------------------------------------------------------------------------
    function single_particle_energy(fh) result(IOstatus)
        use particle_module, only: ptl, calc_particle_energy, px, py, pz, &
                                   calc_ptl_coord
        use spectrum_config, only: spatial_range
        use constants, only: fp
        use commandline_arguments, only: is_emax_cell
        implicit none
        integer, intent(in) :: fh
        integer :: IOstatus

        read(fh, IOSTAT=IOstatus) ptl
        if (IOstatus == 0) then
            call calc_ptl_coord

            if ((px >= spatial_range(1, 1)) .and. (px <= spatial_range(2, 1)) .and. &
                (py >= spatial_range(1, 2)) .and. (py <= spatial_range(2, 2)) .and. &
                (pz >= spatial_range(1, 3)) .and. (pz <= spatial_range(2, 3))) then

                call calc_particle_energy
                call update_energy_spectrum
                call update_maximum_energy
                if (is_emax_cell) then
                    call update_emax_pic_mpi
                endif
            endif
        endif

    end function single_particle_energy

    !---------------------------------------------------------------------------
    ! Update maximum energy array for each cell.
    !---------------------------------------------------------------------------
    subroutine update_emax_pic_mpi
        use particle_module, only: ke, ci, cj, ck
        implicit none
        if (ke > emax_pic_mpi(ci, cj, ck)) then
            emax_pic_mpi(ci, cj, ck) = ke
        endif
    end subroutine update_emax_pic_mpi

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
    
    !---------------------------------------------------------------------------
    ! Update particle maximum energy.
    !---------------------------------------------------------------------------
    subroutine update_maximum_energy
        use particle_module, only: ke
        implicit none
        if (ke > emax_tmp) then
            emax_tmp = ke
        endif
    end subroutine update_maximum_energy

end module particle_energy_spectrum
