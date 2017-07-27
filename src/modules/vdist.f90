!*******************************************************************************
! Module for velocity distribution. The velocity here is actually 4-velocity
! multiplied by the sqrt(mass of particle). So that it can be used for both
! electrons and ions without changing vmin, vmax.
!*******************************************************************************
module velocity_distribution
    use constants, only: fp, dp
    use path_info, only: rootpath
    implicit none
    private

    public fvel_2d, fvel_xy, fvel_xz, fvel_yz, fvel_para, fvel_perp, &
           fvel_2d_sum, fvel_xy_sum, fvel_xz_sum, fvel_yz_sum, &
           fvel_para_sum, fvel_perp_sum, vbins_short, vbins_long, vbins_log
    public init_velocity_bins, free_velocity_bins, init_vdist_2d, &
           set_vdist_2d_zero, free_vdist_2d, init_vdist_2d_single, &
           set_vdist_2d_zero_single, free_vdist_2d_single, init_vdist_1d, &
           set_vdist_1d_zero, free_vdist_1d, init_vdist_1d_single, &
           set_vdist_1d_zero_single, free_vdist_1d_single, &
           calc_vdist_2d, calc_vdist_1d, calc_vdist_2d_single, &
           calc_vdist_1d_single, sum_vdist_1d_over_mpi, &
           sum_vdist_2d_over_mpi, save_vdist_1d, save_vdist_2d, &
           update_vdist_1d, update_vdist_2d

    real(dp), allocatable, dimension(:, :) :: fvel_2d, fvel_xy, fvel_xz, fvel_yz
    real(dp), allocatable, dimension(:) :: fvel_para, fvel_perp
    real(dp), allocatable, dimension(:) :: fvel_para_log, fvel_perp_log
    real(dp), allocatable, dimension(:, :) :: fvel_2d_sum, fvel_xy_sum
    real(dp), allocatable, dimension(:, :) :: fvel_xz_sum, fvel_yz_sum
    real(dp), allocatable, dimension(:) :: fvel_para_sum, fvel_perp_sum
    real(dp), allocatable, dimension(:) :: fvel_para_log_sum, fvel_perp_log_sum
    real(dp), allocatable, dimension(:) :: vbins_short, vbins_long, vbins_log
    ! The index of the left corner of the bin that contains the particle.
    integer :: ibin_para, ibin_perp, ibinx, ibiny, ibinz
    ! The offset from the left corner. [0, 1)
    real(fp) :: offset_para, offset_perp, offsetx, offsety, offsetz
    real(fp) :: vmin_log

    contains

    !---------------------------------------------------------------------------
    ! Initialize short (nbins_vdist) and long (2*nbins_vdist) velocity bins.
    !---------------------------------------------------------------------------
    subroutine init_velocity_bins
        use spectrum_config, only: nbins_vdist, vmax, vmin_nonzero, dv, dv_log
        implicit none
        integer :: i

        vmin_log = log10(vmin_nonzero)

        allocate(vbins_short(nbins_vdist))
        allocate(vbins_long(nbins_vdist*2))
        allocate(vbins_log(nbins_vdist))

        do i = 1, nbins_vdist
            vbins_short(i) = (i - 0.5) * dv
        enddo

        do i = 1, nbins_vdist*2
            vbins_long(i) = (i - 0.5)*dv - vmax
        enddo

        do i = 1, nbins_vdist
            vbins_log(i) = vmin_nonzero * 10**((i-1)*dv_log)
        enddo
    end subroutine init_velocity_bins

    !---------------------------------------------------------------------------
    ! Free velocity bins.
    !---------------------------------------------------------------------------
    subroutine free_velocity_bins
        implicit none
        deallocate(vbins_short, vbins_long, vbins_log)
    end subroutine free_velocity_bins

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
        if (myid == master) then
            deallocate(fvel_2d_sum)
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
            allocate(fvel_para_log_sum(nbins_vdist))
            allocate(fvel_perp_log_sum(nbins_vdist))
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
        allocate(fvel_para_log(nbins_vdist))
        allocate(fvel_perp_log(nbins_vdist))
        call set_vdist_1d_zero_single
    end subroutine init_vdist_1d_single

    !---------------------------------------------------------------------------
    ! Set 1D velocity distributions to zero.
    !---------------------------------------------------------------------------
    subroutine set_vdist_1d_zero_single
        implicit none
        fvel_para = 0.0
        fvel_perp = 0.0
        fvel_para_log = 0.0
        fvel_perp_log = 0.0
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
            fvel_para_log_sum = 0.0
            fvel_perp_log_sum = 0.0
        endif
    end subroutine set_vdist_1d_zero

    !---------------------------------------------------------------------------
    ! Free 1D velocity distributions for a single process.
    !---------------------------------------------------------------------------
    subroutine free_vdist_1d_single
        implicit none
        deallocate(fvel_para, fvel_perp)
        deallocate(fvel_para_log, fvel_perp_log)
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
            deallocate(fvel_para_log_sum, fvel_perp_log_sum)
        endif
    end subroutine free_vdist_1d

    !---------------------------------------------------------------------------
    ! Calculate particle 2d velocity distribution from individual particle
    ! information.
    ! Input:
    !   ct: current time frame.
    !   species: 'e' for electron. 'h' for others.
    !---------------------------------------------------------------------------
    subroutine calc_vdist_2d(ct, species)
        use mpi_module
        use constants, only: fp
        use particle_frames, only: tinterval
        use particle_file, only: check_existence
        implicit none
        integer, intent(in) :: ct
        character(len=1), intent(in) :: species
        integer :: tindex
        logical :: is_exist

        tindex = ct * tinterval
        call check_existence(tindex, species, is_exist)
        if (is_exist) then
            call calc_vdist_2d_mpi(tindex, species)
            call sum_vdist_2d_over_mpi
            call save_vdist_2d(ct, species)
        endif
    end subroutine calc_vdist_2d

    !---------------------------------------------------------------------------
    ! Sum the 2D velocity distributions over all of the MPI processes.
    !---------------------------------------------------------------------------
    subroutine sum_vdist_2d_over_mpi
        use mpi_module
        use spectrum_config, only: nbins_vdist
        implicit none
        ! Sum over all MPI processes to get the total velocity distributions
        call MPI_REDUCE(fvel_2d, fvel_2d_sum, 2*nbins_vdist**2, &
                MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
        call MPI_REDUCE(fvel_xy, fvel_xy_sum, 4*nbins_vdist**2, &
                MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
        call MPI_REDUCE(fvel_xz, fvel_xz_sum, 4*nbins_vdist**2, &
                MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
        call MPI_REDUCE(fvel_yz, fvel_yz_sum, 4*nbins_vdist**2, &
                MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
    end subroutine sum_vdist_2d_over_mpi

    !---------------------------------------------------------------------------
    ! Save 2D velocity distributions to files.
    ! Input:
    !   ct: current time frame.
    !   species: 'e' for electron. 'h' for others.
    !---------------------------------------------------------------------------
    subroutine save_vdist_2d(ct, species)
        use mpi_module
        use spectrum_config, only: nbins_vdist, center, sizes, vmin, vmax
        implicit none
        integer, intent(in) :: ct
        character(len=1), intent(in) :: species
        character(len=100) :: fname
        logical :: dir_e
        
        ! Check if the folder exist.
        if (myid == master) then
            inquire(file='./vdistributions/.', exist=dir_e)
            if (.not. dir_e) then
                call system('mkdir vdistributions')
            endif
        endif

        ! Save the distributions to files.
        if (myid == master) then
            write(fname, "(A,A1,A1,I0)") "vdistributions/vdist_2d-", &
                                         species, ".", ct
            open(unit=10, file=trim(fname), access='stream', &
                 status='unknown', form='unformatted', action='write')
            write(10) center, sizes
            write(10) vmin, vmax, nbins_vdist
            write(10) vbins_short, vbins_long, vbins_log
            write(10) fvel_2d_sum, fvel_xy_sum, fvel_xz_sum, fvel_yz_sum
            close(10)
        endif
    end subroutine save_vdist_2d

    !---------------------------------------------------------------------------
    ! Read particle data and calculate the 2D velocity distribution for one
    ! time frame. This subroutine is used in parallel procedures.
    ! Input:
    !   tindex: the time index, indicating the time step numbers in PIC simulation.
    !   species: 'e' for electron. 'h' for others.
    !---------------------------------------------------------------------------
    subroutine calc_vdist_2d_mpi(tindex, species)
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

        ! Read particle data in parallel to generate distributions
        do np = myid, tot_pic_mpi-1, numprocs
            write(cid, "(I0)") pic_mpi_ranks(np+1)
            call open_particle_file(tindex, species, cid)
            isrange = check_particle_in_range(spatial_range)

            if (isrange) then
                ! Loop over particles
                do iptl = 1, pheader%dim, 1
                    call single_particle_vdist_2d(fh)
                enddo
            endif

            call close_particle_file
        enddo
    end subroutine calc_vdist_2d_mpi

    !---------------------------------------------------------------------------
    ! Read particle data and calculate the velocity distributions for one time
    ! frame. This procedure is only use one CPU core.
    ! Input:
    !   tindex: the time index, indicating the time step numbers in PIC simulation.
    !   species: 'e' for electron. 'h' for others.
    !---------------------------------------------------------------------------
    subroutine calc_vdist_2d_single(tindex, species)
        use picinfo, only: domain
        use file_header, only: pheader
        use spectrum_config, only: corners_mpi
        use particle_file, only: open_particle_file, check_particle_in_range, &
                close_particle_file, fh
        implicit none
        character(len=1), intent(in) :: species
        integer, intent(in) :: tindex
        character(len=50) :: cid
        integer :: np, iptl
        integer :: ix, iy, iz

        ! Read particle data and update the spectra
        do iz = corners_mpi(1,3), corners_mpi(2,3)
            do iy = corners_mpi(1,2), corners_mpi(2,2)
                do ix = corners_mpi(1,1), corners_mpi(2,1)

                    np = ix + iy*domain%pic_tx + iz*domain%pic_tx*domain%pic_ty
                    write(cid, "(I0)") np
                    call open_particle_file(tindex, species, cid)

                    ! Loop over particles
                    do iptl = 1, pheader%dim, 1
                        call single_particle_vdist_2d(fh)
                    enddo

                    call close_particle_file

                enddo ! X
            enddo ! Y
        enddo ! Z
    end subroutine calc_vdist_2d_single

    !---------------------------------------------------------------------------
    ! Read one single particle information, check if it is in the spatial range,
    ! calculate its parallel and perpendicular velocity and update the 2D
    ! velocity distributions.
    ! Input:
    !   fh: file handler.
    !---------------------------------------------------------------------------
    subroutine single_particle_vdist_2d(fh)
        use particle_module, only: ptl, calc_para_perp_velocity, px, py, pz, &
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

            call calc_para_perp_velocity
            call update_vdist_2d
        endif

    end subroutine single_particle_vdist_2d

    !---------------------------------------------------------------------------
    ! Calculate particle 1d velocity distribution from individual particle
    ! information.
    ! Input:
    !   ct: current time frame.
    !   species: 'e' for electron. 'h' for others.
    !---------------------------------------------------------------------------
    subroutine calc_vdist_1d(ct, species)
        use mpi_module
        use constants, only: fp
        use particle_frames, only: tinterval
        use particle_file, only: check_existence
        implicit none
        integer, intent(in) :: ct
        character(len=1), intent(in) :: species
        integer :: tindex
        logical :: is_exist

        tindex = ct * tinterval
        call check_existence(tindex, species, is_exist)
        if (is_exist) then
            call calc_vdist_1d_mpi(tindex, species)
            call sum_vdist_1d_over_mpi
            call save_vdist_1d(ct, species)
        endif
    end subroutine calc_vdist_1d

    !---------------------------------------------------------------------------
    ! Sum the 1D velocity distributions over all of the MPI processes.
    !---------------------------------------------------------------------------
    subroutine sum_vdist_1d_over_mpi
        use mpi_module
        use spectrum_config, only: nbins_vdist
        implicit none
        ! Sum over all MPI processes to get the total velocity distributions
        call MPI_REDUCE(fvel_para, fvel_para_sum, 2*nbins_vdist, &
                MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
        call MPI_REDUCE(fvel_perp, fvel_perp_sum, nbins_vdist, &
                MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
        call MPI_REDUCE(fvel_para_log, fvel_para_log_sum, nbins_vdist, &
                MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
        call MPI_REDUCE(fvel_perp_log, fvel_perp_log_sum, nbins_vdist, &
                MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
    end subroutine sum_vdist_1d_over_mpi

    !---------------------------------------------------------------------------
    ! Save 1D velocity distributions to files.
    ! Input:
    !   ct: current time frame.
    !   species: 'e' for electron. 'h' for others.
    !---------------------------------------------------------------------------
    subroutine save_vdist_1d(ct, species)
        use mpi_module
        use spectrum_config, only: nbins_vdist, center, sizes, vmin, vmax
        implicit none
        integer, intent(in) :: ct
        character(len=1), intent(in) :: species
        character(len=100) :: fname
        logical :: dir_e
        
        ! Check if the folder exist.
        if (myid == master) then
            inquire(file='./vdistributions/.', exist=dir_e)
            if (.not. dir_e) then
                call system('mkdir vdistributions')
            endif
        endif

        ! Save the distributions to files.
        if (myid == master) then
            write(fname, "(A,A1,A1,I0)") "vdistributions/vdist_1d-", &
                                         species, ".", ct
            open(unit=10, file=trim(fname), access='stream', &
                 status='unknown', form='unformatted', action='write')
            write(10) center, sizes
            write(10) vmin, vmax, nbins_vdist
            write(10) vbins_short, vbins_long, vbins_log
            write(10) fvel_para_sum, fvel_perp_sum
            write(10) fvel_para_log_sum, fvel_perp_log_sum
            close(10)
        endif
    end subroutine save_vdist_1d

    !---------------------------------------------------------------------------
    ! Read particle data and calculate the 1D velocity distribution for one
    ! time frame. This subroutine is used in parallel procedures.
    ! Input:
    !   tindex: the time index, indicating the time step numbers in PIC simulation.
    !   species: 'e' for electron. 'h' for others.
    !---------------------------------------------------------------------------
    subroutine calc_vdist_1d_mpi(tindex, species)
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

        ! Read particle data in parallel to generate distributions
        do np = myid, tot_pic_mpi-1, numprocs
            write(cid, "(I0)") pic_mpi_ranks(np+1)
            call open_particle_file(tindex, species, cid)
            isrange = check_particle_in_range(spatial_range)

            if (isrange) then
                ! Loop over particles
                do iptl = 1, pheader%dim, 1
                    call single_particle_vdist_1d(fh)
                enddo
            endif

            call close_particle_file
        enddo
    end subroutine calc_vdist_1d_mpi

    !---------------------------------------------------------------------------
    ! Read particle data and calculate the velocity distributions for one time
    ! frame. This procedure is only use one CPU core.
    ! Input:
    !   tindex: the time index, indicating the time step numbers in PIC simulation.
    !   species: 'e' for electron. 'h' for others.
    !---------------------------------------------------------------------------
    subroutine calc_vdist_1d_single(tindex, species)
        use picinfo, only: domain
        use file_header, only: pheader
        use spectrum_config, only: corners_mpi
        use particle_file, only: open_particle_file, check_particle_in_range, &
                close_particle_file, fh
        implicit none
        character(len=1), intent(in) :: species
        integer, intent(in) :: tindex
        character(len=50) :: cid
        integer :: np, iptl
        integer :: ix, iy, iz

        ! Read particle data and update the spectra
        do iz = corners_mpi(1,3), corners_mpi(2,3)
            do iy = corners_mpi(1,2), corners_mpi(2,2)
                do ix = corners_mpi(1,1), corners_mpi(2,1)

                    np = ix + iy*domain%pic_tx + iz*domain%pic_tx*domain%pic_ty
                    write(cid, "(I0)") np
                    call open_particle_file(tindex, species, cid)

                    ! Loop over particles
                    do iptl = 1, pheader%dim, 1
                        call single_particle_vdist_1d(fh)
                    enddo

                    call close_particle_file

                enddo ! X
            enddo ! Y
        enddo ! Z
    end subroutine calc_vdist_1d_single

    !---------------------------------------------------------------------------
    ! Read one single particle information, check if it is in the spatial range,
    ! calculate its parallel and perpendicular velocity and update the 1D
    ! velocity distributions.
    ! Input:
    !   fh: file handler.
    !---------------------------------------------------------------------------
    subroutine single_particle_vdist_1d(fh)
        use particle_module, only: ptl, calc_para_perp_velocity, px, py, pz, &
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

            call calc_para_perp_velocity
            call update_vdist_1d
            call update_vdist_1d_log
        endif

    end subroutine single_particle_vdist_1d

    !---------------------------------------------------------------------------
    ! Calculate which bin (e.g. i-(i+1)) to put the particle in. The offset from
    ! i is also calculated.
    ! Inputs:
    !   v: particle velocity (actually \gamma v).
    !   vmin: the minimum velocity.
    !   dv: the velocity interval.
    ! Outputs:
    !   ibin: the energy bin.
    !   offset: offset from the left corner [0, 1).
    !---------------------------------------------------------------------------
    subroutine calc_bin_offset(v, vmin, dv, ibin, offset)
        implicit none
        real(fp), intent(in) :: v, vmin, dv
        integer, intent(out) :: ibin
        real(fp), intent(out) :: offset
        real(fp) :: rbin

        rbin = (v - vmin) / dv + 1
        ibin = floor(rbin)
        offset = rbin - ibin
    end subroutine calc_bin_offset

    !---------------------------------------------------------------------------
    ! Calculate the bin index and the offset from the left corner for the
    ! parallel and perpendicular direction to the local magnetic field.
    !---------------------------------------------------------------------------
    subroutine calc_bin_offset_para_perp
        use particle_module, only: vpara, vperp
        use spectrum_config, only: vmax, vmin, dv
        use particle_info, only: sqrt_ptl_mass
        implicit none
        call calc_bin_offset(vpara*sqrt_ptl_mass, -vmax, dv, ibin_para, offset_para)
        call calc_bin_offset(vperp*sqrt_ptl_mass, vmin, dv, ibin_perp, offset_perp)
    end subroutine calc_bin_offset_para_perp

    !---------------------------------------------------------------------------
    ! Calculate the bin index and the offset from the left corner for the
    ! x, y and z directions.
    !---------------------------------------------------------------------------
    subroutine calc_bin_offset_xyz
        use particle_module, only: ptl
        use spectrum_config, only: vmax, dv
        use particle_info, only: sqrt_ptl_mass
        implicit none
        call calc_bin_offset(ptl%vx*sqrt_ptl_mass, -vmax, dv, ibinx, offsetx)
        call calc_bin_offset(ptl%vy*sqrt_ptl_mass, -vmax, dv, ibiny, offsety)
        call calc_bin_offset(ptl%vz*sqrt_ptl_mass, -vmax, dv, ibinz, offsetz)
    end subroutine calc_bin_offset_xyz

    !---------------------------------------------------------------------------
    ! Calculate the weight for the four corners of the box where the particle is.
    ! Inputs:
    !   offset1, offset2: the offsets from the bottom-left corner.
    !   v1, v2: the weights for the bottom-left and the bottom-right corner.
    !   v3, v4: the weights for the top-right and the top-left corner.
    !---------------------------------------------------------------------------
    subroutine calc_weights(offset1, offset2, v1, v2, v3, v4)
        implicit none
        real(fp), intent(in) :: offset1, offset2
        real(fp), intent(out) :: v1, v2, v3, v4

        v1 = (1.0 - offset1) * (1.0 - offset2)
        v2 = offset1 * (1.0 - offset2)
        v3 = offset1 * offset2
        v4 = (1.0 - offset1) * offset2
    end subroutine calc_weights

    !---------------------------------------------------------------------------
    ! Update the parallel and perpendicular 2D velocity distribution.
    !---------------------------------------------------------------------------
    subroutine update_vdist_para_perp
        use spectrum_config, only: nbins_vdist
        implicit none
        real(fp) :: v1, v2, v3, v4
        call calc_bin_offset_para_perp
        call calc_weights(offset_para, offset_perp, v1, v2, v3, v4)
        if ((ibin_para >= 1) .and. ((ibin_para + 1) <= nbins_vdist*2) .and. &
            (ibin_perp >= 1) .and. ((ibin_perp + 1) <= nbins_vdist)) then 

            fvel_2d(ibin_para, ibin_perp) = fvel_2d(ibin_para, ibin_perp) + v1
            fvel_2d(ibin_para+1, ibin_perp) = fvel_2d(ibin_para+1, ibin_perp) + v2
            fvel_2d(ibin_para+1, ibin_perp+1) = fvel_2d(ibin_para+1, ibin_perp+1) + v3
            fvel_2d(ibin_para, ibin_perp+1) = fvel_2d(ibin_para, ibin_perp+1) + v4

        endif
    end subroutine update_vdist_para_perp

    !---------------------------------------------------------------------------
    ! Update the 2D velocity direction in xy, xz and yz planes.
    !---------------------------------------------------------------------------
    subroutine update_vdist_xyz
        use spectrum_config, only: nbins_vdist
        use particle_module, only: ptl
        implicit none
        real(fp) :: v1, v2, v3, v4

        call calc_bin_offset_xyz

        ! xy
        call calc_weights(offsetx, offsety, v1, v2, v3, v4)
        if ((ibinx >= 1) .and. ((ibinx + 1) <= nbins_vdist*2) .and. &
            (ibiny >= 1) .and. ((ibiny + 1) <= nbins_vdist*2)) then 

            fvel_xy(ibinx, ibiny) = fvel_xy(ibinx, ibiny) + v1
            fvel_xy(ibinx+1, ibiny) = fvel_xy(ibinx+1, ibiny) + v2
            fvel_xy(ibinx+1, ibiny+1) = fvel_xy(ibinx+1, ibiny+1) + v3
            fvel_xy(ibinx, ibiny+1) = fvel_xy(ibinx, ibiny+1) + v4

        endif

        ! xz
        call calc_weights(offsetx, offsetz, v1, v2, v3, v4)
        if ((ibinx >= 1) .and. ((ibinx + 1) <= nbins_vdist*2) .and. &
            (ibinz >= 1) .and. ((ibinz + 1) <= nbins_vdist*2)) then 

            fvel_xz(ibinx, ibinz) = fvel_xz(ibinx, ibinz) + v1
            fvel_xz(ibinx+1, ibinz) = fvel_xz(ibinx+1, ibinz) + v2
            fvel_xz(ibinx+1, ibinz+1) = fvel_xz(ibinx+1, ibinz+1) + v3
            fvel_xz(ibinx, ibinz+1) = fvel_xz(ibinx, ibinz+1) + v4

        endif

        ! yz
        call calc_weights(offsety, offsetz, v1, v2, v3, v4)
        if ((ibiny >= 1) .and. ((ibiny + 1) <= nbins_vdist*2) .and. &
            (ibinz >= 1) .and. ((ibinz + 1) <= nbins_vdist*2)) then 

            fvel_yz(ibiny, ibinz) = fvel_yz(ibiny, ibinz) + v1
            fvel_yz(ibiny+1, ibinz) = fvel_yz(ibiny+1, ibinz) + v2
            fvel_yz(ibiny+1, ibinz+1) = fvel_yz(ibiny+1, ibinz+1) + v3
            fvel_yz(ibiny, ibinz+1) = fvel_yz(ibiny, ibinz+1) + v4

        endif
    end subroutine update_vdist_xyz

    !---------------------------------------------------------------------------
    ! Update 2D particle velocity distributions.
    !---------------------------------------------------------------------------
    subroutine update_vdist_2d
        implicit none
        call update_vdist_para_perp
        call update_vdist_xyz
    end subroutine update_vdist_2d

    !---------------------------------------------------------------------------
    ! Update 1D particle velocity distributions.
    !---------------------------------------------------------------------------
    subroutine update_vdist_1d
        use spectrum_config, only: nbins_vdist
        implicit none
        call calc_bin_offset_para_perp
        if ((ibin_para >= 1) .and. ((ibin_para + 1) <= nbins_vdist*2) .and. &
            (ibin_perp >= 1) .and. ((ibin_perp + 1) <= nbins_vdist)) then 

            fvel_para(ibin_para) = fvel_para(ibin_para) + 1.0 - offset_para
            fvel_para(ibin_para+1) = fvel_para(ibin_para+1) + offset_para

            fvel_perp(ibin_perp) = fvel_perp(ibin_perp) + 1.0 - offset_perp
            fvel_perp(ibin_perp+1) = fvel_perp(ibin_perp+1) + offset_perp

        endif
    end subroutine update_vdist_1d

    !---------------------------------------------------------------------------
    ! Update 1D particle velocity distributions in log scale.
    !---------------------------------------------------------------------------
    subroutine update_vdist_1d_log
        use spectrum_config, only: nbins_vdist
        use particle_module, only: vpara, vperp
        use spectrum_config, only: dv_log
        use particle_info, only: sqrt_ptl_mass
        implicit none
        integer :: ibin_para, ibin_perp

        ibin_para = floor((log10(abs(vpara * sqrt_ptl_mass)) - vmin_log) / dv_log) + 1
        ibin_perp = floor((log10(vperp * sqrt_ptl_mass) - vmin_log) / dv_log) + 1

        if (ibin_para >= 1 .and. ibin_para <= nbins_vdist) then
            fvel_para_log(ibin_para) = fvel_para_log(ibin_para) + 1
        endif
        if (ibin_perp >= 1 .and. ibin_perp <= nbins_vdist) then
            fvel_perp_log(ibin_perp) = fvel_perp_log(ibin_perp) + 1
        endif
    end subroutine update_vdist_1d_log

end module velocity_distribution
