!*******************************************************************************
! Module for velocity distribution.
!*******************************************************************************
module velocity_distribution
    use constants, only: fp
    use path_info, only: rootpath
    implicit none
    private

    real(fp), allocatable, dimension(:, :) :: fvel_2d, fvel_xy, fvel_xz, fvel_yz
    real(fp), allocatable, dimension(:) :: fvel_para, fvel_perp
    real(fp), allocatable, dimension(:, :) :: fvel_2d_sum, fvel_xy_sum
    real(fp), allocatable, dimension(:, :) :: fvel_xz_sum, fvel_yz_sum
    real(fp), allocatable, dimension(:) :: fvel_para_sum, fvel_perp_sum

    contains

    !---------------------------------------------------------------------------
    ! Initialize 2D velocity distribution for a parallel analysis using MPI.
    !---------------------------------------------------------------------------
    subroutine init_vdist_2d
        use mpi_module
        use spectrum_config, only: nbins_vdist
        implicit none
        call init_vdist_2d_single
        if (myid == master) then
            allocate(fvel_2d_sum(nbins_vdist*2, nbins_vdist))
            allocate(fvel_xy_sum(nbins_vdist*2, nbins_vdist*2))
            allocate(fvel_xz_sum(nbins_vdist*2, nbins_vdist*2))
            allocate(fvel_yz_sum(nbins_vdist*2, nbins_vdist*2))
        endif
        call set_vdist_2d_zero
    end subroutine init_vdist_2d

    !---------------------------------------------------------------------------
    ! Initialize 2D velocity distributions for a single process.
    !---------------------------------------------------------------------------
    subroutine init_vdist_2d_single
        use spectrum_config, only: nbins_vdist
        implicit none
        allocate(fvel_2d(nbins_vdist*2, nbins_vdist))
        allocate(fvel_xy(nbins_vdist*2, nbins_vdist*2))
        allocate(fvel_xz(nbins_vdist*2, nbins_vdist*2))
        allocate(fvel_yz(nbins_vdist*2, nbins_vdist*2))
        call set_vdist_2d_zero_single
    end subroutine init_vdist_2d_single

    !---------------------------------------------------------------------------
    ! Set 2D velocity distributions to zero for a single process.
    !---------------------------------------------------------------------------
    subroutine set_vdist_2d_zero_single
        implicit none
        fvel_2d = 0.0
        fvel_xy = 0.0
        fvel_xz = 0.0
        fvel_yz = 0.0
    end subroutine set_vdist_2d_zero_single

    !---------------------------------------------------------------------------
    ! Set the 2D velocity distributions to zeros for analysis using MPI.
    !---------------------------------------------------------------------------
    subroutine set_vdist_2d_zero
        use mpi_module
        implicit none
        call set_vdist_2d_zero_single
        if (myid == master) then
            fvel_2d_sum = 0.0
            fvel_xy_sum = 0.0
            fvel_xz_sum = 0.0
            fvel_yz_sum = 0.0
        endif
    end subroutine set_vdist_2d_zero

    !---------------------------------------------------------------------------
    ! Free 2D velocity distribution for a parallel analysis using MPI.
    !---------------------------------------------------------------------------
    subroutine free_vdist_2d
        use mpi_module
        implicit none
        call free_vdist_2d_single
        deallocate(fvel_2d_sum)
        if (myid == master) then
            deallocate(fvel_xy_sum, fvel_xz_sum, fvel_yz_sum)
        endif
    end subroutine free_vdist_2d

    !---------------------------------------------------------------------------
    ! Free 2D velocity distribution a single process.
    !---------------------------------------------------------------------------
    subroutine free_vdist_2d_single
        implicit none
        deallocate(fvel_2d)
        deallocate(fvel_xy, fvel_xz, fvel_yz)
    end subroutine free_vdist_2d_single

    !---------------------------------------------------------------------------
    ! Initialize 1D velocity distributions (parallel and perpendicular to the
    ! local magnetic field. This is for parallel analysis using MPI
    !---------------------------------------------------------------------------
    subroutine init_vdist_1d
        use mpi_module
        use spectrum_config, only: nbins_vdist
        implicit none
        call init_vdist_1d_single
        if (myid == master) then
            allocate(fvel_para_sum(nbins_vdist*2))
            allocate(fvel_perp_sum(nbins_vdist))
        endif
        call set_vdist_1d_zero
    end subroutine init_vdist_1d

    !---------------------------------------------------------------------------
    ! Initialize 1D velocity distributions (parallel and perpendicular to the
    ! local magnetic field. This is for a single process.
    !---------------------------------------------------------------------------
    subroutine init_vdist_1d_single
        use spectrum_config, only: nbins_vdist
        implicit none
        allocate(fvel_para(nbins_vdist*2))
        allocate(fvel_perp(nbins_vdist))
        call set_vdist_1d_zero_single
    end subroutine init_vdist_1d_single

    !---------------------------------------------------------------------------
    ! Set 1D velocity distributions to zero.
    !---------------------------------------------------------------------------
    subroutine set_vdist_1d_zero_single
        implicit none
        fvel_para = 0.0
        fvel_perp = 0.0
    end subroutine set_vdist_1d_zero_single

    !---------------------------------------------------------------------------
    ! Set 1D velocity distributions to zero for parallel analysis using MPI
    !---------------------------------------------------------------------------
    subroutine set_vdist_1d_zero
        use mpi_module
        implicit none
        call set_vdist_1d_zero_single
        if (myid == master) then
            fvel_para_sum = 0.0
            fvel_perp_sum = 0.0
        endif
    end subroutine set_vdist_1d_zero

    !---------------------------------------------------------------------------
    ! Free 1D velocity distributions for a single process.
    !---------------------------------------------------------------------------
    subroutine free_vdist_1d_single
        implicit none
        deallocate(fvel_para, fvel_perp)
    end subroutine free_vdist_1d_single

    !---------------------------------------------------------------------------
    ! Free 1D velocity distributions for parallel analysis using MPI.
    !---------------------------------------------------------------------------
    subroutine free_vdist_1d
        use mpi_module
        implicit none
        call free_vdist_1d_single
        if (myid == master) then
            deallocate(fvel_para_sum, fvel_perp_sum)
        endif
    end subroutine free_vdist_1d

    !---------------------------------------------------------------------------
    ! Get particle energy spectrum from individual particle information.
    ! Input:
    !   ct: current time frame.
    !   species: 'e' for electron. 'h' for others.
    !---------------------------------------------------------------------------
    subroutine calc_vdist_2d(ct, species)
        use mpi_module
        use constants, only: fp
        use particle_frames, only: tinterval
        use spectrum_config, only: nbins_vdist
        implicit none
        integer, intent(in) :: ct
        character(len=1), intent(in) :: species
        character(len=100) :: fname
        integer :: i, tindex
        logical :: is_exist, dir_e

        if (myid == master) then
            inquire(file='./spectrum/.', exist=dir_e)
            if (.not. dir_e) then
                call system('mkdir spectrum')
            endif
        endif

        call calc_energy_bins

        tindex = ct * tinterval
        call set_energy_spectra_zero
        call check_existence(tindex, species, is_exist)
        if (is_exist) then
            !call calc_vdist_2d_mpi(tindex, species)
            ! ! Sum over all nodes to get the total energy spectrum
            ! call MPI_REDUCE(f, fsum, nbins, MPI_DOUBLE_PRECISION, &
            !         MPI_SUM, 0, MPI_COMM_WORLD, ierr)
            ! call MPI_REDUCE(flog, flogsum, nbins, MPI_DOUBLE_PRECISION, &
            !         MPI_SUM, 0, MPI_COMM_WORLD, ierr)
        endif
    end subroutine calc_vdist_2d

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

!     !---------------------------------------------------------------------------
!     ! Read particle data and calculate the energy spectrum for one time frame.
!     ! This subroutine is used in parallel procedures.
!     ! Input:
!     !   tindex: the time index, indicating the time step numbers in PIC simulation.
!     !   species: 'e' for electron. 'h' for others.
!     !---------------------------------------------------------------------------
!     subroutine calc_energy_spectrum_mpi(tindex, species)
!         use mpi_module
!         use file_header, only: read_boilerplate, read_particle_header, &
!                                pheader, v0
!         use spectrum_config, only: spatial_range
!         use picinfo, only: domain
!         implicit none
!         character(len=1), intent(in) :: species
!         integer, intent(in) :: tindex
!         character(len=50) :: ctindex, cid
!         character(len=150) :: dataset, fname
!         real(fp) :: x0, y0, z0, x1, y1, z1
!         logical :: isrange1, isrange2
!         integer :: fh, np, iptl

!         fh = 10
!         write(ctindex, "(I0)") tindex
!         dataset = trim(adjustl(rootpath))//"particle/T."//trim(ctindex)
!         dataset = trim(adjustl(dataset))//"/"//species//"particle."
!         ! Read particle data in parallel to generate distributions
!         do np = 0, domain%nproc-numprocs, numprocs
!             write(cid, "(I0)") myid + np
!             fname = trim(dataset)//trim(ctindex)//"."//trim(cid)
!             open(unit=fh, file=trim(fname), status='unknown', &
!                  form='unformatted', access='stream', action='read')
!             write(*, '(A,A,A,I0,A,I0)') "Reading --> ", trim(fname), &
!                 " Physical rank = ", myid, " np = ", np

!             call read_boilerplate(fh)
!             call read_particle_header(fh)

!             ! Corners of this MPI process's domain
!             x0 = v0%x0
!             y0 = v0%y0
!             z0 = v0%z0
!             x1 = v0%x0 + domain%pic_nx * domain%dx
!             y1 = v0%y0 + domain%pic_ny * domain%dy
!             z1 = v0%z0 + domain%pic_nz * domain%dz

!             ! Only if the corners are within the box.
!             ! Shift one grid to cover boundary.
!             isrange1 = x1 >= (spatial_range(1,1) - domain%dx) &
!                  .and. x1 <= (spatial_range(2,1) + domain%dx) &
!                  .and. y1 >= (spatial_range(1,2) - domain%dy) &
!                  .and. y1 <= (spatial_range(2,2) + domain%dy) &
!                  .and. z1 >= (spatial_range(1,3) - domain%dz) &
!                  .and. z1 <= (spatial_range(2,3) + domain%dz)
!             isrange2 = x0 >= (spatial_range(1,1) - domain%dx) &
!                  .and. x0 <= (spatial_range(2,1) + domain%dx) &
!                  .and. y0 >= (spatial_range(1,2) - domain%dy) &
!                  .and. y0 <= (spatial_range(2,2) + domain%dy) &
!                  .and. z0 >= (spatial_range(1,3) - domain%dz) &
!                  .and. z0 <= (spatial_range(2,3) + domain%dz)

!             if (isrange1 .or. isrange2) then
!                 ! Loop over particles
!                 do iptl = 1, pheader%dim, 1
!                     call single_particle_energy(fh)
!                 enddo
!             endif

!             close(fh)
!         enddo
!     end subroutine calc_energy_spectrum_mpi

!     !---------------------------------------------------------------------------
!     ! Read particle data and calculate the energy spectrum for one time frame.
!     ! This procedure is only use one CPU core.
!     ! Input:
!     !   tindex: the time index, indicating the time step numbers in PIC simulation.
!     !   species: 'e' for electron. 'h' for others.
!     !---------------------------------------------------------------------------
!     subroutine calc_energy_spectrum_single(tindex, species)
!         use file_header, only: read_boilerplate, read_particle_header, &
!                                pheader, v0
!         use spectrum_config, only: spatial_range, corners_mpi
!         use picinfo, only: domain
!         implicit none
!         character(len=1), intent(in) :: species
!         integer, intent(in) :: tindex
!         character(len=50) :: ctindex, cid
!         character(len=150) :: dataset, fname
!         integer :: np, iptl, fh
!         integer :: ix, iy, iz

!         fh = 10
!         write(ctindex, "(I0)") tindex
!         dataset = trim(adjustl(rootpath))//"particle/T."//trim(ctindex)
!         dataset = trim(adjustl(dataset))//"/"//species//"particle."

!         ! Read particle data and update the spectra
!         do iz = corners_mpi(1,3), corners_mpi(2,3)
!             do iy = corners_mpi(1,2), corners_mpi(2,2)
!                 do ix = corners_mpi(1,1), corners_mpi(2,1)
!                     np = ix + iy*domain%pic_tx + iz*domain%pic_tx*domain%pic_ty
!                     write(cid, "(I0)") np
!                     fname = trim(dataset)//trim(ctindex)//"."//trim(cid)
!                     open(unit=fh, file=trim(fname), status='unknown', &
!                          form='unformatted', access='stream', action='read')
!                     write(*, '(A,A,A,I0)') "Reading --> ", trim(fname), &
!                         " np = ", np

!                     call read_boilerplate(fh)
!                     call read_particle_header(fh)

!                     ! Loop over particles
!                     do iptl = 1, pheader%dim, 1
!                         call single_particle_energy(fh)
!                     enddo

!                     close(fh)

!                 enddo ! Z
!             enddo ! Y
!         enddo ! X
!     end subroutine calc_energy_spectrum_single

!     !---------------------------------------------------------------------------
!     ! Read one single particle information, check if it is in the spatial range,
!     ! calculate its energy and put it into the flux arrays.
!     ! Input:
!     !   fh: file handler.
!     !---------------------------------------------------------------------------
!     subroutine single_particle_energy(fh)
!         use particle_module, only: ptl, calc_particle_energy, px, py, pz, &
!                                    calc_ptl_coord
!         use spectrum_config, only: spatial_range
!         use constants, only: fp
!         implicit none
!         integer, intent(in) :: fh

!         read(fh) ptl
!         call calc_ptl_coord

!         if ((px >= spatial_range(1, 1)) .and. (px <= spatial_range(2, 1)) .and. &
!             (py >= spatial_range(1, 2)) .and. (py <= spatial_range(2, 2)) .and. &
!             (pz >= spatial_range(1, 3)) .and. (pz <= spatial_range(2, 3))) then

!             call calc_particle_energy
!             call update_energy_spectrum
!         endif

!     end subroutine single_particle_energy

!     !---------------------------------------------------------------------------
!     ! Update particle energy spectrum.
!     !---------------------------------------------------------------------------
!     subroutine update_energy_spectrum
!         use particle_module, only: ke
!         use spectrum_config, only: dve, dlogve, emin, nbins
!         implicit none
!         real(fp) :: rbin, shift, dke
!         integer :: ibin, ibin1

!         rbin = ke / dve
!         ibin = int(rbin)
!         ibin1 = ibin + 1
        
!         if ((ibin >= 1) .and. (ibin1 <= nbins)) then 
!             shift = rbin - ibin
!             f(ibin)  = f(ibin) - shift + 1
!             f(ibin1) = f(ibin1) + shift
!         endif

!         ! Exceptions
!         if ( ibin .eq. 0) then
!             ! Add lower energies to the 1st band.
!             f(ibin1) = f(ibin1) + 1
!         endif
!         dke = ke * dlogve
     
!         ! Logarithmic scale
!         ibin = (log10(ke)-log10(emin))/dlogve + 1
!         if ((ibin >= 1) .and. (ibin1 <= nbins)) then 
!             flog(ibin)  = flog(ibin) + 1.0/dke
!         endif
!     end subroutine update_energy_spectrum

end module velocity_distribution
