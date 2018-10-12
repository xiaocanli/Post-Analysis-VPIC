!<******************************************************************************
!< This module include the methods to calculate energization due to different
!< fluid drifts, compression and shear
!<******************************************************************************
module fluid_energization_module
    use constants, only: fp, dp
    use picinfo, only: domain
    use pic_fields, only: bx, by, bz, absB, ex, ey, ez
    implicit none
    private
    save

    public init_tmp_data, free_tmp_data, init_neighbors, free_neighbors, &
        get_neighbors, curv_drift_energization, grad_drift_energization, &
        magnetization_energization, para_perp_energization, &
        fluid_accel_energization, compression_shear_energization, &
        para_drift_energization, magnetic_moment_energization

    integer :: nx, ny, nz
    real(fp), allocatable, dimension(:, :, :) :: tmpx, tmpy, tmpz, stmp
    real(fp), allocatable, dimension(:, :, :) :: tmpx1, tmpy1, tmpz1
    integer, allocatable, dimension(:) :: ixl, iyl, izl
    integer, allocatable, dimension(:) :: ixh, iyh, izh
    real(fp), allocatable, dimension(:) :: idx, idy, idz

    contains

    !<--------------------------------------------------------------------------
    !< Initialize temporary data
    !<--------------------------------------------------------------------------
    subroutine init_tmp_data
        use mpi_topology, only: ht
        implicit none

        nx = ht%nx
        ny = ht%ny
        nz = ht%nz

        allocate(tmpx(nx, ny, nz))
        allocate(tmpy(nx, ny, nz))
        allocate(tmpz(nx, ny, nz))
        allocate(stmp(nx, ny, nz))
        allocate(tmpx1(nx, ny, nz))
        allocate(tmpy1(nx, ny, nz))
        allocate(tmpz1(nx, ny, nz))

        tmpx = 0.0
        tmpy = 0.0
        tmpz = 0.0
        tmpx1 = 0.0
        tmpy1 = 0.0
        tmpz1 = 0.0
        stmp = 0.0
    end subroutine init_tmp_data

    !<--------------------------------------------------------------------------
    !< Free temporary data
    !<--------------------------------------------------------------------------
    subroutine free_tmp_data
        implicit none
        deallocate(tmpx, tmpy, tmpz)
        deallocate(tmpx1, tmpy1, tmpz1)
        deallocate(stmp)
    end subroutine free_tmp_data

    !---------------------------------------------------------------------------
    ! Initialize the indices of the neighbors and the inverse of the distance
    ! between them.
    !---------------------------------------------------------------------------
    subroutine init_neighbors
        implicit none
        allocate(ixl(nx))
        allocate(ixh(nx))
        allocate(idx(nx))
        allocate(iyl(ny))
        allocate(iyh(ny))
        allocate(idy(ny))
        allocate(izl(nz))
        allocate(izh(nz))
        allocate(idz(nz))
        ixl = 0; iyl = 0; izl = 0
        ixh = 0; iyh = 0; izh = 0
        idx = 0.0; idy = 0.0; idz = 0.0
    end subroutine init_neighbors

    !---------------------------------------------------------------------------
    ! Free the indices of the neighbors and the inverse of the distance between
    ! them.
    !---------------------------------------------------------------------------
    subroutine free_neighbors
        implicit none
        deallocate(ixl, iyl, izl)
        deallocate(ixh, iyh, izh)
        deallocate(idx, idy, idz)
    end subroutine free_neighbors

    !---------------------------------------------------------------------------
    ! Decide the indices of two neighbors.
    ! Input:
    !   ntot: total number of points in this dimension.
    !   indexc: index of current point.
    ! Output:
    !   indexl: index of the neighbor with lower index.
    !   indexh: index of the neighbor with higher index.
    !---------------------------------------------------------------------------
    subroutine neighbors(ntot, indexc, indexl, indexh)
        implicit none
        integer, intent(in) :: ntot, indexc
        integer, intent(out) :: indexl, indexh
        if (ntot == 1) then
            indexl = 1
            indexh = 1
        else if (indexc == 1) then
            indexl = 1
            indexh = 2
        else if (indexc == ntot) then
            indexl = ntot - 1
            indexh = ntot
        else
            indexl = indexc - 1
            indexh = indexc + 1
        endif
    end subroutine neighbors

    !---------------------------------------------------------------------------
    ! Get the indices of the neighbors and the inverse of the distance between
    ! them.
    !---------------------------------------------------------------------------
    subroutine get_neighbors
        implicit none
        integer :: ix, iy, iz

        do ix = 1, nx
            call neighbors(nx, ix, ixl(ix), ixh(ix))
            if ((ixh(ix) - ixl(ix)) /= 0) then
                idx(ix) = domain%idx / (ixh(ix) - ixl(ix))
            else
                idx(ix) = 0.0
            endif
        enddo

        do iy = 1, ny
            call neighbors(ny, iy, iyl(iy), iyh(iy))
            if ((iyh(iy) - iyl(iy)) /= 0) then
                idy(iy) = domain%idy / (iyh(iy) - iyl(iy))
            else
                idy(iy) = 0.0
            endif
        enddo

        do iz = 1, nz
            call neighbors(nz, iz, izl(iz), izh(iz))
            if ((izh(iz) - izl(iz)) /= 0) then
                idz(iz) = domain%idz / (izh(iz) - izl(iz))
            else
                idz(iz) = 0.0
            endif
        enddo
    end subroutine get_neighbors

    !<--------------------------------------------------------------------------
    !< Calculate energization due to curvature drift.
    !<--------------------------------------------------------------------------
    function curv_drift_energization result(jdote)
        use pic_fields, only: num_rho, vx, vy, vz, ux, uy, uz
        use para_perp_pressure, only: ppara
        use particle_info, only: ptl_mass
        implicit none
        integer :: ix, iy, iz
        real(fp), dimension(2) :: jdote

        do iz = 1, nz
            do iy = 1, ny
                tmpx(:, iy, iz) = bx(1:nx, iy, iz) * &
                    (bx(ixh, iy, iz) - bx(ixl, iy, iz)) * idx
                tmpy(:, iy, iz) = bx(1:nx, iy, iz) * &
                    (by(ixh, iy, iz) - by(ixl, iy, iz)) * idx
                tmpz(:, iy, iz) = bx(1:nx, iy, iz) * &
                    (bz(ixh, iy, iz) - bz(ixl, iy, iz)) * idx
            enddo
        enddo

        do iz = 1, nz
            do ix = 1, nx
                tmpx(ix, :, iz) = tmpx(ix, :, iz) + by(ix, 1:ny, iz) * &
                    (bx(ix, iyh, iz) - bx(ix, iyl, iz)) * idy
                tmpy(ix, :, iz) = tmpy(ix, :, iz) + by(ix, 1:ny, iz) * &
                    (by(ix, iyh, iz) - by(ix, iyl, iz)) * idy
                tmpz(ix, :, iz) = tmpz(ix, :, iz) + by(ix, 1:ny, iz) * &
                    (bz(ix, iyh, iz) - bz(ix, iyl, iz)) * idy
            enddo
        enddo

        do iy = 1, ny
            do ix = 1, nx
                tmpx(ix, iy, :) = tmpx(ix, iy, :) + bz(ix, iy, 1:nz) * &
                    (bx(ix, iy, izh) - bx(ix, iy, izl)) * idz
                tmpy(ix, iy, :) = tmpy(ix, iy, :) + bz(ix, iy, 1:nz) * &
                    (by(ix, iy, izh) - by(ix, iy, izl)) * idz
                tmpz(ix, iy, :) = tmpz(ix, iy, :) + bz(ix, iy, 1:nz) * &
                    (bz(ix, iy, izh) - bz(ix, iy, izl)) * idz
            enddo
        enddo

        stmp = (ex(1:nx, 1:ny, 1:nz) * (by(1:nx, 1:ny, 1:nz) * tmpz - &
                                        bz(1:nx, 1:ny, 1:nz) * tmpy) + &
                ey(1:nx, 1:ny, 1:nz) * (bz(1:nx, 1:ny, 1:nz) * tmpx - &
                                        bx(1:nx, 1:ny, 1:nz) * tmpz) + &
                ez(1:nx, 1:ny, 1:nz) * (bx(1:nx, 1:ny, 1:nz) * tmpy - &
                                        by(1:nx, 1:ny, 1:nz) * tmpx)) / &
                absB(1:nx, 1:ny, 1:nz)**4

        tmpx = ((vx(1:nx, 1:ny, 1:nz) * bx(1:nx, 1:ny, 1:nz) + &
                 vy(1:nx, 1:ny, 1:nz) * by(1:nx, 1:ny, 1:nz) + &
                 vz(1:nx, 1:ny, 1:nz) * bz(1:nx, 1:ny, 1:nz)) * &
                (ux(1:nx, 1:ny, 1:nz) * bx(1:nx, 1:ny, 1:nz) + &
                 uy(1:nx, 1:ny, 1:nz) * by(1:nx, 1:ny, 1:nz) + &
                 uz(1:nx, 1:ny, 1:nz) * bz(1:nx, 1:ny, 1:nz)) * &
                num_rho(1:nx, 1:ny, 1:nz) / absB(1:nx, 1:ny, 1:nz)**2) * &
                ptl_mass
        jdote(1) = sum(stmp * ppara(1:nx, 1:ny, 1:nz)) * &
                   domain%dx * domain%dy * domain%dz
        jdote(2) = sum(stmp * tmpx) * domain%dx * domain%dy * domain%dz
    end function curv_drift_energization

    !<--------------------------------------------------------------------------
    !< Calculate energization due to gradient drift.
    !<--------------------------------------------------------------------------
    function grad_drift_energization result(jdote)
        use para_perp_pressure, only: pperp
        implicit none
        integer :: ix, iy, iz
        real(fp) :: jdote

        do iz = 1, nz
            do iy = 1, ny
                tmpx(:, iy, iz) = (absB(ixh, iy, iz) - absB(ixl, iy, iz)) * idx
            enddo
        enddo

        do iz = 1, nz
            do ix = 1, nx
                tmpy(ix, :, iz) = (absB(ix, iyh, iz) - absB(ix, iyl, iz)) * idy
            enddo
        enddo

        do iy = 1, ny
            do ix = 1, nx
                tmpz(ix, iy, :) = (absB(ix, iy, izh) - absB(ix, iy, izl)) * idz
            enddo
        enddo

        jdote = sum((ex(1:nx, 1:ny, 1:nz) * (by(1:nx, 1:ny, 1:nz) * tmpz - &
                                             bz(1:nx, 1:ny, 1:nz) * tmpy) + &
                     ey(1:nx, 1:ny, 1:nz) * (bz(1:nx, 1:ny, 1:nz) * tmpx - &
                                             bx(1:nx, 1:ny, 1:nz) * tmpz) + &
                     ez(1:nx, 1:ny, 1:nz) * (bx(1:nx, 1:ny, 1:nz) * tmpy - &
                                             by(1:nx, 1:ny, 1:nz) * tmpx)) * &
                    pperp(1:nx, 1:ny, 1:nz) / absB(1:nx, 1:ny, 1:nz)**3) * &
                    domain%dx * domain%dy * domain%dz
    end function grad_drift_energization

    !<--------------------------------------------------------------------------
    !< Calculate energization due to magnetization
    !<--------------------------------------------------------------------------
    function magnetization_energization result(jdote)
        use para_perp_pressure, only: pperp
        implicit none
        integer :: ix, iy, iz
        real(fp) :: jdote

        stmp = pperp(1:nx, 1:ny, 1:nz) / absB(1:nx, 1:ny, 1:nz)**2

        do iz = 1, nz
            do iy = 1, ny
                tmpy(:, iy, iz) = -(stmp(ixh, iy, iz) * bz(ixh, iy, iz) - &
                                    stmp(ixl, iy, iz) * bz(ixl, iy, iz)) * idx
                tmpz(:, iy, iz) =  (stmp(ixh, iy, iz) * by(ixh, iy, iz) - &
                                    stmp(ixl, iy, iz) * by(ixl, iy, iz)) * idx
            enddo
        enddo

        do iz = 1, nz
            do ix = 1, nx
                tmpx(ix, :, iz) = (stmp(ix, iyh, iz) * bz(ix, iyh, iz) - &
                                   stmp(ix, iyl, iz) * bz(ix, iyl, iz)) * idy
                tmpz(ix, :, iz) = tmpz(ix, :, iz) - &
                                  (stmp(ix, iyh, iz) * bx(ix, iyh, iz) - &
                                   stmp(ix, iyl, iz) * bx(ix, iyl, iz)) * idy
            enddo
        enddo

        do iy = 1, ny
            do ix = 1, nx
                tmpx(ix, iy, :) = tmpx(ix, iy, :) - &
                                  (stmp(ix, iy, izh) * by(ix, iy, izh) - &
                                   stmp(ix, iy, izl) * by(ix, iy, izl)) * idz
                tmpy(ix, iy, :) = tmpy(ix, iy, :) + &
                                  (stmp(ix, iy, izh) * bx(ix, iy, izh) - &
                                   stmp(ix, iy, izl) * bx(ix, iy, izl)) * idz
            enddo
        enddo

        jdote = sum(((tmpy * bz(1:nx, 1:ny, 1:nz) - tmpz * by(1:nx, 1:ny, 1:nz)) * &
                     (ey(1:nx, 1:ny, 1:nz) * bz(1:nx, 1:ny, 1:nz) - &
                      ez(1:nx, 1:ny, 1:nz) * by(1:nx, 1:ny, 1:nz)) + &
                     (tmpz * bx(1:nx, 1:ny, 1:nz) - tmpx * bz(1:nx, 1:ny, 1:nz)) * &
                     (ez(1:nx, 1:ny, 1:nz) * bx(1:nx, 1:ny, 1:nz) - &
                      ex(1:nx, 1:ny, 1:nz) * bz(1:nx, 1:ny, 1:nz)) + &
                     (tmpx * by(1:nx, 1:ny, 1:nz) - tmpy * bx(1:nx, 1:ny, 1:nz)) * &
                     (ex(1:nx, 1:ny, 1:nz) * by(1:nx, 1:ny, 1:nz) - &
                      ey(1:nx, 1:ny, 1:nz) * bx(1:nx, 1:ny, 1:nz))) / &
                    absB(1:nx, 1:ny, 1:nz)**2) * domain%dx * domain%dy * domain%dz
        jdote = -jdote
    end function magnetization_energization

    !<--------------------------------------------------------------------------
    !< Calculate energization due to parallel and perpendicular electric field.
    !<--------------------------------------------------------------------------
    function para_perp_energization result(jdote)
        use pic_fields, only: vx, vy, vz, num_rho
        use particle_info, only: ptl_charge
        implicit none
        integer :: ix, iy, iz
        real(fp), dimension(2) :: jdote

        stmp = (vx(1:nx, 1:ny, 1:nz) * bx(1:nx, 1:ny, 1:nz) + &
                vy(1:nx, 1:ny, 1:nz) * by(1:nx, 1:ny, 1:nz) + &
                vz(1:nx, 1:ny, 1:nz) * bz(1:nx, 1:ny, 1:nz)) / &
                absB(1:nx, 1:ny, 1:nz)**2
        tmpx = stmp * bx(1:nx, 1:ny, 1:nz)
        tmpy = stmp * by(1:nx, 1:ny, 1:nz)
        tmpz = stmp * bz(1:nx, 1:ny, 1:nz)
        jdote(1) = sum((tmpx * ex(1:nx, 1:ny, 1:nz) + &
                        tmpy * ey(1:nx, 1:ny, 1:nz) + &
                        tmpz * ez(1:nx, 1:ny, 1:nz)) * num_rho(1:nx, 1:ny, 1:nz)) * &
                   domain%dx * domain%dy * domain%dz * ptl_charge
        jdote(2) = sum((vx(1:nx, 1:ny, 1:nz) * ex(1:nx, 1:ny, 1:nz) + &
                        vy(1:nx, 1:ny, 1:nz) * ey(1:nx, 1:ny, 1:nz) + &
                        vz(1:nx, 1:ny, 1:nz) * ez(1:nx, 1:ny, 1:nz)) * &
                       num_rho(1:nx, 1:ny, 1:nz)) * &
                   domain%dx * domain%dy * domain%dz * ptl_charge
        jdote(2) = jdote(2) - jdote(1)
    end function para_perp_energization

    !<--------------------------------------------------------------------------
    !< Calculate energization due to fluid acceleration
    !<--------------------------------------------------------------------------
    function fluid_accel_energization(dt_fields) result(jdote)
        use pic_fields, only: vx, vy, vz, ux, uy, uz, num_rho
        use pre_post_hydro, only: udx1, udx2, udy1, udy2, udz1, udz2
        use particle_info, only: ptl_mass
        implicit none
        real(fp), intent(in) :: dt_fields
        integer :: ix, iy, iz
        real(fp), dimension(4) :: jdote
        real(fp) :: idt

        idt = 1.0 / dt_fields

        ! Total
        tmpx = (udx2(1:nx, 1:ny, 1:nz) - udx1(1:nx, 1:ny, 1:nz)) * idt
        tmpy = (udy2(1:nx, 1:ny, 1:nz) - udy1(1:nx, 1:ny, 1:nz)) * idt
        tmpz = (udz2(1:nx, 1:ny, 1:nz) - udz1(1:nx, 1:ny, 1:nz)) * idt

        jdote(1) = sum((ex(1:nx, 1:ny, 1:nz) * (by(1:nx, 1:ny, 1:nz) * tmpz - &
                                                bz(1:nx, 1:ny, 1:nz) * tmpy) + &
                        ey(1:nx, 1:ny, 1:nz) * (bz(1:nx, 1:ny, 1:nz) * tmpx - &
                                                bx(1:nx, 1:ny, 1:nz) * tmpz) + &
                        ez(1:nx, 1:ny, 1:nz) * (bx(1:nx, 1:ny, 1:nz) * tmpy - &
                                                by(1:nx, 1:ny, 1:nz) * tmpx)) * &
                       num_rho(1:nx, 1:ny, 1:nz) / absB(1:nx, 1:ny, 1:nz)**2) * &
                       domain%dx * domain%dy * domain%dz * ptl_mass
        ! Parallel. We use B at current time for simplicity.
        stmp = (udx1(1:nx, 1:ny, 1:nz) * bx(1:nx, 1:ny, 1:nz) + &
                udy1(1:nx, 1:ny, 1:nz) * by(1:nx, 1:ny, 1:nz) + &
                udz1(1:nx, 1:ny, 1:nz) * bz(1:nx, 1:ny, 1:nz)) / absB(1:nx, 1:ny, 1:nz)**2
        tmpx = stmp * bx(1:nx, 1:ny, 1:nz)
        tmpy = stmp * by(1:nx, 1:ny, 1:nz)
        tmpz = stmp * bz(1:nx, 1:ny, 1:nz)
        stmp = (udx2(1:nx, 1:ny, 1:nz) * bx(1:nx, 1:ny, 1:nz) + &
                udy2(1:nx, 1:ny, 1:nz) * by(1:nx, 1:ny, 1:nz) + &
                udz2(1:nx, 1:ny, 1:nz) * bz(1:nx, 1:ny, 1:nz)) / absB(1:nx, 1:ny, 1:nz)**2
        tmpx1 = stmp * bx(1:nx, 1:ny, 1:nz)
        tmpy1 = stmp * by(1:nx, 1:ny, 1:nz)
        tmpz1 = stmp * bz(1:nx, 1:ny, 1:nz)
        tmpx = (tmpx1 - tmpx) * idt
        tmpy = (tmpy1 - tmpy) * idt
        tmpz = (tmpz1 - tmpz) * idt
        jdote(2) = sum((ex(1:nx, 1:ny, 1:nz) * (by(1:nx, 1:ny, 1:nz) * tmpz - &
                                                bz(1:nx, 1:ny, 1:nz) * tmpy) + &
                        ey(1:nx, 1:ny, 1:nz) * (bz(1:nx, 1:ny, 1:nz) * tmpx - &
                                                bx(1:nx, 1:ny, 1:nz) * tmpz) + &
                        ez(1:nx, 1:ny, 1:nz) * (bx(1:nx, 1:ny, 1:nz) * tmpy - &
                                                by(1:nx, 1:ny, 1:nz) * tmpx)) * &
                       num_rho(1:nx, 1:ny, 1:nz) / absB(1:nx, 1:ny, 1:nz)**2) * &
                       domain%dx * domain%dy * domain%dz * ptl_mass
        jdote(1) = jdote(1) - jdote(2) ! Perpendicular

        ! Total
        do iz = 1, nz
            do iy = 1, ny
                tmpx(:, iy, iz) = vx(1:nx, iy, iz) * &
                    (ux(ixh, iy, iz) - ux(ixl, iy, iz)) * idx
                tmpy(:, iy, iz) = vx(1:nx, iy, iz) * &
                    (uy(ixh, iy, iz) - uy(ixl, iy, iz)) * idx
                tmpz(:, iy, iz) = vx(1:nx, iy, iz) * &
                    (uz(ixh, iy, iz) - uz(ixl, iy, iz)) * idx
            enddo
        enddo

        do iz = 1, nz
            do ix = 1, nx
                tmpx(ix, :, iz) = tmpx(ix, :, iz) + vy(ix, 1:ny, iz) * &
                    (ux(ix, iyh, iz) - ux(ix, iyl, iz)) * idy
                tmpy(ix, :, iz) = tmpy(ix, :, iz) + vy(ix, 1:ny, iz) * &
                    (uy(ix, iyh, iz) - uy(ix, iyl, iz)) * idy
                tmpz(ix, :, iz) = tmpz(ix, :, iz) + vy(ix, 1:ny, iz) * &
                    (uz(ix, iyh, iz) - uz(ix, iyl, iz)) * idy
            enddo
        enddo

        do iy = 1, ny
            do ix = 1, nx
                tmpx(ix, iy, :) = tmpx(ix, iy, :) + vz(ix, iy, 1:nz) * &
                    (ux(ix, iy, izh) - ux(ix, iy, izl)) * idz
                tmpy(ix, iy, :) = tmpy(ix, iy, :) + vz(ix, iy, 1:nz) * &
                    (uy(ix, iy, izh) - uy(ix, iy, izl)) * idz
                tmpz(ix, iy, :) = tmpz(ix, iy, :) + vz(ix, iy, 1:nz) * &
                    (uz(ix, iy, izh) - uz(ix, iy, izl)) * idz
            enddo
        enddo

        jdote(3) = sum((ex(1:nx, 1:ny, 1:nz) * (by(1:nx, 1:ny, 1:nz) * tmpz - &
                                                bz(1:nx, 1:ny, 1:nz) * tmpy) + &
                        ey(1:nx, 1:ny, 1:nz) * (bz(1:nx, 1:ny, 1:nz) * tmpx - &
                                                bx(1:nx, 1:ny, 1:nz) * tmpz) + &
                        ez(1:nx, 1:ny, 1:nz) * (bx(1:nx, 1:ny, 1:nz) * tmpy - &
                                                by(1:nx, 1:ny, 1:nz) * tmpx)) * &
                       num_rho(1:nx, 1:ny, 1:nz) / absB(1:nx, 1:ny, 1:nz)**2) * &
                       domain%dx * domain%dy * domain%dz * ptl_mass

        ! Parallel
        stmp = (ux(1:nx, 1:ny, 1:nz) * bx(1:nx, 1:ny, 1:nz) + &
                uy(1:nx, 1:ny, 1:nz) * by(1:nx, 1:ny, 1:nz) + &
                uz(1:nx, 1:ny, 1:nz) * bz(1:nx, 1:ny, 1:nz)) / absB(1:nx, 1:ny, 1:nz)**2
        tmpx1 = stmp * bx(1:nx, 1:ny, 1:nz)
        tmpy1 = stmp * by(1:nx, 1:ny, 1:nz)
        tmpz1 = stmp * bz(1:nx, 1:ny, 1:nz)
        do iz = 1, nz
            do iy = 1, ny
                tmpx(:, iy, iz) = vx(1:nx, iy, iz) * &
                    (tmpx1(ixh, iy, iz) - tmpx1(ixl, iy, iz)) * idx
                tmpy(:, iy, iz) = vx(1:nx, iy, iz) * &
                    (tmpy1(ixh, iy, iz) - tmpy1(ixl, iy, iz)) * idx
                tmpz(:, iy, iz) = vx(1:nx, iy, iz) * &
                    (tmpz1(ixh, iy, iz) - tmpz1(ixl, iy, iz)) * idx
            enddo
        enddo

        do iz = 1, nz
            do ix = 1, nx
                tmpx(ix, :, iz) = tmpx(ix, :, iz) + vy(ix, 1:ny, iz) * &
                    (tmpx1(ix, iyh, iz) - tmpx1(ix, iyl, iz)) * idy
                tmpy(ix, :, iz) = tmpy(ix, :, iz) + vy(ix, 1:ny, iz) * &
                    (tmpy1(ix, iyh, iz) - tmpy1(ix, iyl, iz)) * idy
                tmpz(ix, :, iz) = tmpz(ix, :, iz) + vy(ix, 1:ny, iz) * &
                    (tmpz1(ix, iyh, iz) - tmpz1(ix, iyl, iz)) * idy
            enddo
        enddo

        do iy = 1, ny
            do ix = 1, nx
                tmpx(ix, iy, :) = tmpx(ix, iy, :) + vz(ix, iy, 1:nz) * &
                    (tmpx1(ix, iy, izh) - tmpx1(ix, iy, izl)) * idz
                tmpy(ix, iy, :) = tmpy(ix, iy, :) + vz(ix, iy, 1:nz) * &
                    (tmpy1(ix, iy, izh) - tmpy1(ix, iy, izl)) * idz
                tmpz(ix, iy, :) = tmpz(ix, iy, :) + vz(ix, iy, 1:nz) * &
                    (tmpz1(ix, iy, izh) - tmpz1(ix, iy, izl)) * idz
            enddo
        enddo

        jdote(4) = sum((ex(1:nx, 1:ny, 1:nz) * (by(1:nx, 1:ny, 1:nz) * tmpz - &
                                                bz(1:nx, 1:ny, 1:nz) * tmpy) + &
                        ey(1:nx, 1:ny, 1:nz) * (bz(1:nx, 1:ny, 1:nz) * tmpx - &
                                                bx(1:nx, 1:ny, 1:nz) * tmpz) + &
                        ez(1:nx, 1:ny, 1:nz) * (bx(1:nx, 1:ny, 1:nz) * tmpy - &
                                                by(1:nx, 1:ny, 1:nz) * tmpx)) * &
                       num_rho(1:nx, 1:ny, 1:nz) / absB(1:nx, 1:ny, 1:nz)**2) * &
                       domain%dx * domain%dy * domain%dz * ptl_mass
        jdote(3) = jdote(3) - jdote(4)
    end function fluid_accel_energization

    !<--------------------------------------------------------------------------
    !< Calculate energization due to compression and shear, full pressure
    !< tensor, and gyrotropic pressure tensor. The last part is very dangerous,
    !< because it modifies pressure tensor data.
    !<--------------------------------------------------------------------------
    function compression_shear_energization result(econv)
        use para_perp_pressure, only: ppara, pperp
        use parameters, only: tp1, is_rel
        use pic_fields, only: pxx, pyy, pzz, pxy, pxz, pyz, pyx, pzy, pzx
        implicit none
        integer :: ix, iy, iz
        real(fp), dimension(4) :: econv

        ! ExB drift velocity
        tmpx = (ey(1:nx, 1:ny, 1:nz) * bz(1:nx, 1:ny, 1:nz) - &
                ez(1:nx, 1:ny, 1:nz) * by(1:nx, 1:ny, 1:nz)) / absB(1:nx, 1:ny, 1:nz)**2
        tmpy = (ez(1:nx, 1:ny, 1:nz) * bx(1:nx, 1:ny, 1:nz) - &
                ex(1:nx, 1:ny, 1:nz) * bz(1:nx, 1:ny, 1:nz)) / absB(1:nx, 1:ny, 1:nz)**2
        tmpz = (ex(1:nx, 1:ny, 1:nz) * by(1:nx, 1:ny, 1:nz) - &
                ey(1:nx, 1:ny, 1:nz) * bx(1:nx, 1:ny, 1:nz)) / absB(1:nx, 1:ny, 1:nz)**2

        ! Divergence of ExB drift velocity
        do ix = 1, nx
            stmp(ix, :, :) = &
                (tmpx(ixh(ix), :, :) - tmpx(ixl(ix), :, :)) * idx(ix)
        enddo

        do iy = 1, ny
            stmp(:, iy, :) = stmp(:, iy, :) + &
                (tmpy(:, iyh(iy), :) - tmpy(:, iyl(iy), :)) * idy(iy)
        enddo

        do iz = 1, nz
            stmp(:, :, iz) = stmp(:, :, iz) + &
                (tmpz(:, :, izh(iz)) - tmpz(:, :, izl(iz))) * idz(iz)
        enddo

        econv = 0.0

        ! Compression energization
        econv(1) = sum(-(ppara(1:nx, 1:ny, 1:nz) + 2*pperp(1:nx, 1:ny, 1:nz)) * stmp / 3)
        econv(1) = econv(1) * domain%dx * domain%dy * domain%dz

        ! Shear energization
        do iz = 1, nz
            do iy = 1, ny
                econv(2) = econv(2) + sum((bx(1:nx, iy, iz)**2 * &
                    ((tmpx(ixh, iy, iz) - tmpx(ixl, iy, iz)) * idx - stmp(:, iy, iz)/3) + &
                    bx(1:nx, iy, iz) * by(1:nx, iy, iz) * (tmpy(ixh, iy, iz) - tmpy(ixl, iy, iz)) * idx + &
                    bx(1:nx, iy, iz) * bz(1:nx, iy, iz) * (tmpz(ixh, iy, iz) - tmpz(ixl, iy, iz)) * idx) * &
                    (pperp(1:nx, iy, iz) - ppara(1:nx, iy, iz)) / absB(1:nx, iy, iz)**2)
            enddo
        enddo

        do iz = 1, nz
            do ix = 1, nx
                econv(2) = econv(2) + sum((by(ix, 1:ny, iz)**2 * &
                    ((tmpy(ix, iyh, iz) - tmpy(ix, iyl, iz)) * idy - stmp(ix, :, iz)/3) + &
                    by(ix, 1:ny, iz) * bx(ix, 1:ny, iz) * (tmpx(ix, iyh, iz) - tmpx(ix, iyl, iz)) * idy + &
                    by(ix, 1:ny, iz) * bz(ix, 1:ny, iz) * (tmpz(ix, iyh, iz) - tmpz(ix, iyl, iz)) * idy) * &
                    (pperp(ix, 1:ny, iz) - ppara(ix, 1:ny, iz)) / absB(ix, 1:ny, iz)**2)
            enddo
        enddo

        do iy = 1, ny
            do ix = 1, nx
                econv(2) = econv(2) + sum((bz(ix, iy, 1:nz)**2 * &
                    ((tmpz(ix, iy, izh) - tmpz(ix, iy, izl)) * idz - stmp(ix, iy, :)/3) + &
                    bz(ix, iy, 1:nz) * bx(ix, iy, 1:nz) * (tmpx(ix, iy, izh) - tmpx(ix, iy, izl)) * idz + &
                    bz(ix, iy, 1:nz) * by(ix, iy, 1:nz) * (tmpy(ix, iy, izh) - tmpy(ix, iy, izl)) * idz) * &
                    (pperp(ix, iy, 1:nz) - ppara(ix, iy, 1:nz)) / absB(ix, iy, 1:nz)**2)
            enddo
        enddo

        ! Energization due to full pressure tensor
        do iz = 1, nz
            do iy = 1, ny
                tmpx(:, iy, iz) = (pxx(ixh, iy, iz) - pxx(ixl, iy, iz)) * idx
                tmpy(:, iy, iz) = (pxy(ixh, iy, iz) - pxy(ixl, iy, iz)) * idx
                tmpz(:, iy, iz) = (pxz(ixh, iy, iz) - pxz(ixl, iy, iz)) * idx
            enddo
        enddo

        do iz = 1, nz
            do ix = 1, nx
                if (is_rel) then
                    tmpx(ix, :, iz) = tmpx(ix, :, iz) + (pyx(ix, iyh, iz) - pyx(ix, iyl, iz)) * idy
                else
                    tmpx(ix, :, iz) = tmpx(ix, :, iz) + (pxy(ix, iyh, iz) - pxy(ix, iyl, iz)) * idy
                endif
                tmpy(ix, :, iz) = tmpy(ix, :, iz) + (pyy(ix, iyh, iz) - pyy(ix, iyl, iz)) * idy
                tmpz(ix, :, iz) = tmpz(ix, :, iz) + (pyz(ix, iyh, iz) - pyz(ix, iyl, iz)) * idy
            enddo
        enddo

        do iy = 1, ny
            do ix = 1, nx
                if (is_rel) then
                    tmpx(ix, iy, :) = tmpx(ix, iy, :) + (pzx(ix, iy, izh) - pzx(ix, iy, izl)) * idz
                    tmpy(ix, iy, :) = tmpy(ix, iy, :) + (pzy(ix, iy, izh) - pzy(ix, iy, izl)) * idz
                else
                    tmpx(ix, iy, :) = tmpx(ix, iy, :) + (pxz(ix, iy, izh) - pxz(ix, iy, izl)) * idz
                    tmpy(ix, iy, :) = tmpy(ix, iy, :) + (pyz(ix, iy, izh) - pyz(ix, iy, izl)) * idz
                endif
                tmpz(ix, iy, :) = tmpz(ix, iy, :) + (pzz(ix, iy, izh) - pzz(ix, iy, izl)) * idz
            enddo
        enddo

        econv(3) = sum((ex(1:nx, 1:ny, 1:nz) * (by(1:nx, 1:ny, 1:nz) * tmpz - &
                                                bz(1:nx, 1:ny, 1:nz) * tmpy) + &
                        ey(1:nx, 1:ny, 1:nz) * (bz(1:nx, 1:ny, 1:nz) * tmpx - &
                                                bx(1:nx, 1:ny, 1:nz) * tmpz) + &
                        ez(1:nx, 1:ny, 1:nz) * (bx(1:nx, 1:ny, 1:nz) * tmpy - &
                                                by(1:nx, 1:ny, 1:nz) * tmpx)) / &
                        absB(1:nx, 1:ny, 1:nz)**2)

        ! Energization due to gyrotropic pressure tensor
        ! This is very dangerous, because it modifies pressure tensor data.
        stmp = ppara(1:nx, 1:ny, 1:nz) - pperp(1:nx, 1:ny, 1:nz)
        tmpx = 1.0 / absB(1:nx, 1:ny, 1:nz)**2
        pxx(1:nx, 1:ny, 1:nz) = &
            pperp(1:nx, 1:ny, 1:nz) + stmp * bx(1:nx, 1:ny, 1:nz)**2 * tmpx
        pyy(1:nx, 1:ny, 1:nz) = &
            pperp(1:nx, 1:ny, 1:nz) + stmp * by(1:nx, 1:ny, 1:nz)**2 * tmpx
        pzz(1:nx, 1:ny, 1:nz) = &
            pperp(1:nx, 1:ny, 1:nz) + stmp * bz(1:nx, 1:ny, 1:nz)**2 * tmpx
        pxy(1:nx, 1:ny, 1:nz) = &
            stmp * bx(1:nx, 1:ny, 1:nz) * by(1:nx, 1:ny, 1:nz) * tmpx
        pxz(1:nx, 1:ny, 1:nz) = &
            stmp * bx(1:nx, 1:ny, 1:nz) * bz(1:nx, 1:ny, 1:nz) * tmpx
        pyz(1:nx, 1:ny, 1:nz) = &
            stmp * by(1:nx, 1:ny, 1:nz) * bz(1:nx, 1:ny, 1:nz) * tmpx
        do iz = 1, nz
            do iy = 1, ny
                tmpx(:, iy, iz) = (pxx(ixh, iy, iz) - pxx(ixl, iy, iz)) * idx
                tmpy(:, iy, iz) = (pxy(ixh, iy, iz) - pxy(ixl, iy, iz)) * idx
                tmpz(:, iy, iz) = (pxz(ixh, iy, iz) - pxz(ixl, iy, iz)) * idx
            enddo
        enddo

        do iz = 1, nz
            do ix = 1, nx
                tmpx(ix, :, iz) = tmpx(ix, :, iz) + (pxy(ix, iyh, iz) - pxy(ix, iyl, iz)) * idy
                tmpy(ix, :, iz) = tmpy(ix, :, iz) + (pyy(ix, iyh, iz) - pyy(ix, iyl, iz)) * idy
                tmpz(ix, :, iz) = tmpz(ix, :, iz) + (pyz(ix, iyh, iz) - pyz(ix, iyl, iz)) * idy
            enddo
        enddo

        do iy = 1, ny
            do ix = 1, nx
                tmpx(ix, iy, :) = tmpx(ix, iy, :) + (pxz(ix, iy, izh) - pxz(ix, iy, izl)) * idz
                tmpy(ix, iy, :) = tmpy(ix, iy, :) + (pyz(ix, iy, izh) - pyz(ix, iy, izl)) * idz
                tmpz(ix, iy, :) = tmpz(ix, iy, :) + (pzz(ix, iy, izh) - pzz(ix, iy, izl)) * idz
            enddo
        enddo

        econv(4) = sum((ex(1:nx, 1:ny, 1:nz) * (by(1:nx, 1:ny, 1:nz) * tmpz - &
                                                bz(1:nx, 1:ny, 1:nz) * tmpy) + &
                        ey(1:nx, 1:ny, 1:nz) * (bz(1:nx, 1:ny, 1:nz) * tmpx - &
                                                bx(1:nx, 1:ny, 1:nz) * tmpz) + &
                        ez(1:nx, 1:ny, 1:nz) * (bx(1:nx, 1:ny, 1:nz) * tmpy - &
                                                by(1:nx, 1:ny, 1:nz) * tmpx)) / &
                        absB(1:nx, 1:ny, 1:nz)**2)

        econv(2:4) = econv(2:4) * domain%dx * domain%dy * domain%dz
    end function compression_shear_energization

    !<--------------------------------------------------------------------------
    !< Calculate energization due to parallel drift.
    !<--------------------------------------------------------------------------
    function para_drift_energization result(jdote)
        use para_perp_pressure, only: pperp
        implicit none
        integer :: ix, iy, iz
        real(fp) :: jdote

        ! Curv b, where b is the unit vector along magnetic field direction
        stmp = 1.0 / absB(1:nx, 1:ny, 1:nz)
        tmpx = bx(1:nx, 1:ny, 1:nz) * stmp
        tmpy = by(1:nx, 1:ny, 1:nz) * stmp
        tmpz = bz(1:nx, 1:ny, 1:nz) * stmp

        do iz = 1, nz
            do iy = 1, ny
                tmpy1(:, iy, iz) = -(tmpz(ixh, iy, iz) - tmpz(ixl, iy, iz)) * idx
                tmpz1(:, iy, iz) =  (tmpy(ixh, iy, iz) - tmpy(ixl, iy, iz)) * idx
            enddo
        enddo

        do iz = 1, nz
            do ix = 1, nx
                tmpx1(ix, :, iz) = (tmpz(ix, iyh, iz) - tmpz(ix, iyl, iz)) * idy
                tmpz1(ix, :, iz) = tmpz1(ix, :, iz) - &
                    (tmpx(ix, iyh, iz) - tmpx(ix, iyl, iz)) * idy
            enddo
        enddo

        do iy = 1, ny
            do ix = 1, nx
                tmpx1(ix, iy, :) = tmpx1(ix, iy, :) - &
                    (tmpy(ix, iy, izh) - tmpy(ix, iy, izl)) * idz
                tmpy1(ix, iy, :) = tmpy1(ix, iy, :) + &
                    (tmpx(ix, iy, izh) - tmpx(ix, iy, izl)) * idz
            enddo
        enddo

        stmp = tmpx1 * tmpx + tmpy1 * tmpy + tmpz1 * tmpz
        tmpx1 = stmp * tmpx
        tmpy1 = stmp * tmpy
        tmpz1 = stmp * tmpz

        jdote = sum((tmpx1 * ex(1:nx, 1:ny, 1:nz) + &
                     tmpy1 * ey(1:nx, 1:ny, 1:nz) + &
                     tmpz1 * ez(1:nx, 1:ny, 1:nz)) * &
                    pperp(1:nx, 1:ny, 1:nz) / absB(1:nx, 1:ny, 1:nz)) * &
                domain%dx * domain%dy * domain%dz
    end function para_drift_energization

    !<--------------------------------------------------------------------------
    !< Calculate energization due to the conservation of magnetic moment
    !<--------------------------------------------------------------------------
    function magnetic_moment_energization result(jdote)
        use para_perp_pressure, only: pperp
        implicit none
        integer :: ix, iy, iz
        real(fp) :: jdote

        ! Calculate the curvature of the electric field
        do iz = 1, nz
            do iy = 1, ny
                tmpy(:, iy, iz) = -(ez(ixh, iy, iz) - ez(ixl, iy, iz)) * idx
                tmpz(:, iy, iz) =  (ey(ixh, iy, iz) - ey(ixl, iy, iz)) * idx
            enddo
        enddo

        do iz = 1, nz
            do ix = 1, nx
                tmpx(ix, :, iz) = (ez(ix, iyh, iz) - ez(ix, iyl, iz)) * idy
                tmpz(ix, :, iz) = tmpz(ix, :, iz) - &
                    (ex(ix, iyh, iz) - ex(ix, iyl, iz)) * idy
            enddo
        enddo

        do iy = 1, ny
            do ix = 1, nx
                tmpx(ix, iy, :) = tmpx(ix, iy, :) - &
                    (ey(ix, iy, izh) - ey(ix, iy, izl)) * idz
                tmpy(ix, iy, :) = tmpy(ix, iy, :) + &
                    (ex(ix, iy, izh) - ex(ix, iy, izl)) * idz
            enddo
        enddo

        jdote = -sum((tmpx * bx(1:nx, 1:ny, 1:nz) + &
                      tmpy * by(1:nx, 1:ny, 1:nz) + &
                      tmpz * bz(1:nx, 1:ny, 1:nz)) * &
                     pperp(1:nx, 1:ny, 1:nz) / absB(1:nx, 1:ny, 1:nz)**2) * &
                 domain%dx * domain%dy * domain%dz
    end function magnetic_moment_energization
end module fluid_energization_module
