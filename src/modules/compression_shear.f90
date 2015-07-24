!*******************************************************************************
! Module dealing with compressional heating.
!*******************************************************************************
module compression_shear
    use constants, only: fp
    use mpi_topology, only: htg
    implicit none
    private

    real(fp), allocatable, dimension(:, :, :) :: pdiv_u, pshear
    real(fp), allocatable, dimension(:, :, :) :: udot_div_ptensor
    real(fp), allocatable, dimension(:, :, :) :: div_u, bbsigma

    contains

    !---------------------------------------------------------------------------
    ! Initialize the data arrays.
    !---------------------------------------------------------------------------
    subroutine init_compression_shear
        implicit none
        call init_compression
        call init_shear
        call init_udot_div_ptensor
    end subroutine init_compression_shear

    !---------------------------------------------------------------------------
    ! Initialize div_u, pdiv_u. 
    !---------------------------------------------------------------------------
    subroutine init_compression
        implicit none
        allocate(div_u(htg%nx, htg%ny, htg%nz))
        allocate(pdiv_u(htg%nx, htg%ny, htg%nz))
        div_u = 0.0
        pdiv_u = 0.0
    end subroutine init_compression

    !---------------------------------------------------------------------------
    ! Initialize pshear, bbsigma.
    !---------------------------------------------------------------------------
    subroutine init_shear
        implicit none
        allocate(pshear(htg%nx, htg%ny, htg%nz))
        allocate(bbsigma(htg%nx, htg%ny, htg%nz))
        pshear = 0.0
        bbsigma = 0.0
    end subroutine init_shear

    !---------------------------------------------------------------------------
    ! Initialize udot_div_ptensor
    !---------------------------------------------------------------------------
    subroutine init_udot_div_ptensor
        implicit none
        allocate(udot_div_ptensor(htg%nx, htg%ny, htg%nz))
        udot_div_ptensor = 0.0
    end subroutine init_udot_div_ptensor

    !---------------------------------------------------------------------------
    ! Free the data arrays.
    !---------------------------------------------------------------------------
    subroutine free_compression_shear
        implicit none
        call free_compression
        call free_shear
        call free_udot_div_ptensor
    end subroutine free_compression_shear

    !---------------------------------------------------------------------------
    ! Free div_u, pdiv_u.
    !---------------------------------------------------------------------------
    subroutine free_compression
        implicit none
        deallocate(div_u, pdiv_u)
    end subroutine free_compression

    !---------------------------------------------------------------------------
    ! Free pshear, bbsigma.
    !---------------------------------------------------------------------------
    subroutine free_shear
        implicit none
        deallocate(pshear, bbsigma)
    end subroutine free_shear

    !---------------------------------------------------------------------------
    ! Free udot_div_ptensor.
    !---------------------------------------------------------------------------
    subroutine free_udot_div_ptensor
        implicit none
        deallocate(udot_div_ptensor)
    end subroutine free_udot_div_ptensor

    !---------------------------------------------------------------------------
    ! Calculate the divergence of u.
    !---------------------------------------------------------------------------
    subroutine calc_div_u
        use pic_fields, only: ux, uy, uz
        use neighbors_module, only: ixl, iyl, izl, ixh, iyh, izh, idx, idy, idz
        implicit none
        integer :: nx, ny, nz, ix, iy, iz
        nx = htg%nx
        ny = htg%ny
        nz = htg%nz

        do ix = 1, nx
            div_u(ix, :, :) = (ux(ixh(ix), :, :) - ux(ixl(ix), :, :)) * idx(ix)
        enddo

        do iy = 1, ny
            div_u(:, iy, :) = div_u(:, iy, :) + &
                (uy(:, iyh(iy), :) - ux(:, iyl(iy), :)) * idy(iy)
        enddo

        do iz = 1, nz
            div_u(:, :, iz) = div_u(:, :, iz) + &
                (uz(:, :, izh(iz)) - uz(:, :, izl(iz))) * idz(iz)
        enddo
    end subroutine calc_div_u

    !---------------------------------------------------------------------------
    ! Calculate p\nabla\cdot\vec{u}. Here, p is the scalar pressure.
    !---------------------------------------------------------------------------
    subroutine calc_pdiv_u
        use pressure_tensor, only: pscalar
        implicit none

        pdiv_u = pscalar * div_u
    end subroutine calc_pdiv_u

    !---------------------------------------------------------------------------
    ! Calculate the compression related variables.
    !---------------------------------------------------------------------------
    subroutine calc_compression
        implicit none
        call calc_div_u
        call calc_pdiv_u
    end subroutine calc_compression

    !---------------------------------------------------------------------------
    ! Calculate bbsigma = b_ib_j\sigma_{ij}.
    !---------------------------------------------------------------------------
    subroutine calc_bbsigma
        use pic_fields, only: bx, by, bz, ux, uy, uz, absB
        use neighbors_module, only: ixl, iyl, izl, ixh, iyh, izh, idx, idy, idz
        implicit none
        real(fp) :: sigma_xx, sigma_xy, sigma_xz, sigma_yy, sigma_yz, sigma_zz
        real(fp) :: bxc, byc, bzc
        integer :: nx, ny, nz, ix, iy, iz
        nx = htg%nx
        ny = htg%ny
        nz = htg%nz

        do iz = 1, nz
            do iy = 1, ny
                 do ix = 1, nx
                    sigma_xx = (ux(ixh(ix), iy, iz) - ux(ixl(ix), iy, iz)) * &
                               idx(ix) - div_u(ix, iy, iz) / 3.0
                    sigma_yy = (uy(ix, iyh(iy), iz) - ux(ix, iyl(iy), iz)) * &
                               idy(iy) - div_u(ix, iy, iz) / 3.0
                    sigma_zz = (uz(ix, iy, izh(iz)) - uz(ix, iy, izl(iz))) * &
                               idz(iz) - div_u(ix, iy, iz) / 3.0
                    sigma_xy = 0.5 * (ux(ix, iyh(iy), iz) - &
                                      ux(ix, iyl(iy), iz)) * idy(iy) + &
                               0.5 * (uy(ixh(ix), iy, iz) - &
                                      uy(ixl(ix), iy, iz)) * idx(ix)
                    sigma_xz = 0.5 * (ux(ix, iy, izh(iz)) - &
                                      ux(ix, iy, izl(iz))) * idz(iz) + &
                               0.5 * (uz(ixh(ix), iy, iz) - &
                                      uz(ixl(ix), iy, iz)) * idx(ix)
                    sigma_yz = 0.5 * (uy(ix, iy, izh(iz)) - &
                                      uy(ix, iy, izl(iz))) * idz(iz) + &
                               0.5 * (uz(ix, iyh(iy), iz) - &
                                      uz(ix, iyl(iy), iz)) * idy(iy)
                    bxc = bx(ix, iy, iz)
                    byc = by(ix, iy, iz)
                    bzc = bz(ix, iy, iz)
                    bbsigma(ix, iy, iz) = bxc**2 * sigma_xx + &
                        byc**2 * sigma_yy + bzc**2 * sigma_zz + &
                        bxc * byc * sigma_xy + bxc * bzc * sigma_xz + &
                        byc * bzc * sigma_yz
                 enddo
            enddo
        enddo
    end subroutine calc_bbsigma

    !---------------------------------------------------------------------------
    ! Calculate (p_\parallel - p_\perp)b_ib_j\sigma_{ij}, where \sigma_{ij}
    ! is the shear tensor.
    !---------------------------------------------------------------------------
    subroutine calc_pshear
        use para_perp_pressure, only: ppara, pperp
        implicit none
        pshear = (ppara - pperp) * bbsigma
    end subroutine calc_pshear

    !---------------------------------------------------------------------------
    ! Calculate shear related variables.
    !---------------------------------------------------------------------------
    subroutine calc_shear
        implicit none
        call calc_bbsigma
        call calc_pshear
    end subroutine calc_shear

    !---------------------------------------------------------------------------
    ! Calculate \vec{u}\cdot(\nalba\cdot\tensor(P))
    !---------------------------------------------------------------------------
    subroutine calc_udot_div_ptensor
        use pressure_tensor, only: divp_x, divp_y, divp_z
        use pic_fields, only: ux, uy, uz
        implicit none
        udot_div_ptensor = ux * divp_x + uy * divp_y + uz * divp_z
    end subroutine calc_udot_div_ptensor

    !---------------------------------------------------------------------------
    ! Save div_u.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_div_u(ct)
        use mpi_module
        use constants, only: fp
        use mpi_io_fields, only: save_field
        implicit none
        integer, intent(in) :: ct
        if (myid == master) then
            print*, 'Saving the divergence of the velocity field', ct
        endif

        call save_field(div_u, 'div_u', ct)
    end subroutine save_div_u

    !---------------------------------------------------------------------------
    ! Save pdiv_u.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_pdiv_u(ct)
        use mpi_module
        use constants, only: fp
        use mpi_io_fields, only: save_field
        implicit none
        integer, intent(in) :: ct
        if (myid == master) then
            print*, 'Saving pdiv_u', ct
        endif

        call save_field(pdiv_u, 'pdiv_u', ct)
    end subroutine save_pdiv_u

    !---------------------------------------------------------------------------
    ! Save compression related variables.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_compression(ct)
        implicit none
        integer, intent(in) :: ct
        call save_div_u(ct)
        call save_pdiv_u(ct)
    end subroutine save_compression

    !---------------------------------------------------------------------------
    ! Save bbsigma.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_bbsigma(ct)
        use mpi_module
        use constants, only: fp
        use mpi_io_fields, only: save_field
        implicit none
        integer, intent(in) :: ct
        if (myid == master) then
            print*, 'Saving b_ib_j\sigma_{ij}', ct
        endif

        call save_field(bbsigma, 'bbsigma', ct)
    end subroutine save_bbsigma

    !---------------------------------------------------------------------------
    ! Save pshear
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_pshear(ct)
        use mpi_module
        use constants, only: fp
        use mpi_io_fields, only: save_field
        implicit none
        integer, intent(in) :: ct
        if (myid == master) then
            print*, 'Saving pshear', ct
        endif

        call save_field(pshear, 'pshear', ct)
    end subroutine save_pshear

    !---------------------------------------------------------------------------
    ! Save shear related variables.
    !---------------------------------------------------------------------------
    subroutine save_shear(ct)
        implicit none
        integer, intent(in) :: ct
        call save_bbsigma(ct)
        call save_pshear(ct)
    end subroutine save_shear

    !---------------------------------------------------------------------------
    ! Save udot_div_ptensor
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_udot_div_ptensor(ct)
        use mpi_module
        use constants, only: fp
        use mpi_io_fields, only: save_field
        implicit none
        integer, intent(in) :: ct
        if (myid == master) then
            print*, 'Saving udot_div_ptensor', ct
        endif

        call save_field(udot_div_ptensor, 'udot_div_ptensor', ct)
    end subroutine save_udot_div_ptensor

end module compression_shear
