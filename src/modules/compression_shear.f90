!*******************************************************************************
! Module dealing with compressional heating.
!*******************************************************************************
module compression_shear
    use constants, only: fp
    use mpi_topology, only: htg
    implicit none
    private
    public pdiv_v, pshear, vdot_div_ptensor, div_v, bbsigma
    public init_compression_shear, free_compression_shear, &
           calc_compression_shear, save_compression_shear, &
           save_tot_compression_shear, init_div_vdot_ptensor, &
           free_div_vdot_ptensor, calc_div_vdot_ptensor, &
           save_div_vdot_ptensor, save_tot_div_vdot_ptensor, &
           init_div_v, free_div_v, calc_div_v, calc_div_v_single, &
           calc_bbsigma_single

    real(fp), allocatable, dimension(:, :, :) :: pdiv_v, pshear, &
        pdiv_vsingle, pshear_vsingle, pdiv_vpara_vsingle, pshear_para_vsingle
    real(fp), allocatable, dimension(:, :, :) :: vdot_div_ptensor
    real(fp), allocatable, dimension(:, :, :) :: div_v, bbsigma
    real(fp), allocatable, dimension(:, :, :) :: div_vsingle, bbsigma_single
    real(fp), allocatable, dimension(:, :, :) :: div_vpara_vsingle
    real(fp), allocatable, dimension(:, :, :) :: bbsigma_para_vsingle
    real(fp), allocatable, dimension(:, :, :) :: vdot_ptensor_x, &
            vdot_ptensor_y, vdot_ptensor_z, div_vdot_ptensor
    real(fp), allocatable, dimension(:,:,:) :: vpx, vpy, vpz, vdotb

    contains

    !---------------------------------------------------------------------------
    ! Initialize the data arrays.
    !---------------------------------------------------------------------------
    subroutine init_compression_shear
        implicit none
        call init_compression
        call init_shear
        call init_vdot_div_ptensor
        call init_div_vdot_ptensor
    end subroutine init_compression_shear

    !---------------------------------------------------------------------------
    ! Initialize div_v.
    !---------------------------------------------------------------------------
    subroutine init_div_v
        implicit none
        allocate(div_v(htg%nx, htg%ny, htg%nz))

        div_v = 0.0
    end subroutine init_div_v

    !---------------------------------------------------------------------------
    ! Initialize div_v, pdiv_v.
    !---------------------------------------------------------------------------
    subroutine init_compression
        implicit none
        allocate(div_v(htg%nx, htg%ny, htg%nz))
        allocate(pdiv_v(htg%nx, htg%ny, htg%nz))
        allocate(div_vsingle(htg%nx, htg%ny, htg%nz))
        allocate(div_vpara_vsingle(htg%nx, htg%ny, htg%nz))
        allocate(vpx(htg%nx, htg%ny, htg%nz))
        allocate(vpy(htg%nx, htg%ny, htg%nz))
        allocate(vpz(htg%nx, htg%ny, htg%nz))
        allocate(vdotb(htg%nx, htg%ny, htg%nz))
        allocate(pdiv_vsingle(htg%nx, htg%ny, htg%nz))
        allocate(pdiv_vpara_vsingle(htg%nx, htg%ny, htg%nz))
        div_v = 0.0
        pdiv_v = 0.0
        div_vsingle = 0.0
        div_vpara_vsingle = 0.0
        vpx = 0.0
        vpy = 0.0
        vpz = 0.0
        vdotb = 0.0
        pdiv_vsingle = 0.0
        pdiv_vpara_vsingle = 0.0
    end subroutine init_compression

    !---------------------------------------------------------------------------
    ! Initialize pshear, bbsigma.
    !---------------------------------------------------------------------------
    subroutine init_shear
        implicit none
        allocate(pshear(htg%nx, htg%ny, htg%nz))
        allocate(bbsigma(htg%nx, htg%ny, htg%nz))
        allocate(pshear_vsingle(htg%nx, htg%ny, htg%nz))
        allocate(pshear_para_vsingle(htg%nx, htg%ny, htg%nz))
        allocate(bbsigma_single(htg%nx, htg%ny, htg%nz))
        allocate(bbsigma_para_vsingle(htg%nx, htg%ny, htg%nz))
        pshear = 0.0
        bbsigma = 0.0
        pshear_vsingle = 0.0
        pshear_para_vsingle = 0.0
        bbsigma_single = 0.0
        bbsigma_para_vsingle = 0.0
    end subroutine init_shear

    !---------------------------------------------------------------------------
    ! Initialize vdot_div_ptensor
    !---------------------------------------------------------------------------
    subroutine init_vdot_div_ptensor
        implicit none
        allocate(vdot_div_ptensor(htg%nx, htg%ny, htg%nz))
        vdot_div_ptensor = 0.0
    end subroutine init_vdot_div_ptensor

    !---------------------------------------------------------------------------
    ! Initialize div_vdot_ptensor and the 3 components of vdot_ptensor.
    !---------------------------------------------------------------------------
    subroutine init_div_vdot_ptensor
        implicit none
        allocate(div_vdot_ptensor(htg%nx, htg%ny, htg%nz))
        allocate(vdot_ptensor_x(htg%nx, htg%ny, htg%nz))
        allocate(vdot_ptensor_y(htg%nx, htg%ny, htg%nz))
        allocate(vdot_ptensor_z(htg%nx, htg%ny, htg%nz))
        div_vdot_ptensor = 0.0
        vdot_ptensor_x = 0.0
        vdot_ptensor_y = 0.0
        vdot_ptensor_z = 0.0
    end subroutine init_div_vdot_ptensor

    !---------------------------------------------------------------------------
    ! Free the data arrays.
    !---------------------------------------------------------------------------
    subroutine free_compression_shear
        implicit none
        call free_compression
        call free_shear
        call free_vdot_div_ptensor
        call free_div_vdot_ptensor
    end subroutine free_compression_shear

    !---------------------------------------------------------------------------
    ! Free div_v.
    !---------------------------------------------------------------------------
    subroutine free_div_v
        implicit none
        deallocate(div_v)
    end subroutine free_div_v

    !---------------------------------------------------------------------------
    ! Free div_v, pdiv_v.
    !---------------------------------------------------------------------------
    subroutine free_compression
        implicit none
        deallocate(div_v, pdiv_v)
        deallocate(div_vsingle, div_vpara_vsingle)
        deallocate(vpx, vpy, vpz, vdotb)
        deallocate(pdiv_vsingle, pdiv_vpara_vsingle)
    end subroutine free_compression

    !---------------------------------------------------------------------------
    ! Free pshear, bbsigma.
    !---------------------------------------------------------------------------
    subroutine free_shear
        implicit none
        deallocate(pshear, bbsigma)
        deallocate(pshear_vsingle, pshear_para_vsingle)
        deallocate(bbsigma_single, bbsigma_para_vsingle)
    end subroutine free_shear

    !---------------------------------------------------------------------------
    ! Free vdot_div_ptensor.
    !---------------------------------------------------------------------------
    subroutine free_vdot_div_ptensor
        implicit none
        deallocate(vdot_div_ptensor)
    end subroutine free_vdot_div_ptensor

    !---------------------------------------------------------------------------
    ! Free div_vdot_ptensor and the 3 components of vdot_ptensor.
    !---------------------------------------------------------------------------
    subroutine free_div_vdot_ptensor
        implicit none
        deallocate(div_vdot_ptensor)
        deallocate(vdot_ptensor_x, vdot_ptensor_y, vdot_ptensor_z)
    end subroutine free_div_vdot_ptensor

    !---------------------------------------------------------------------------
    ! Calculate the divergence of v.
    !---------------------------------------------------------------------------
    subroutine calc_div_v
        use pic_fields, only: vx, vy, vz
        use neighbors_module, only: ixl, iyl, izl, ixh, iyh, izh, idx, idy, idz
        implicit none
        integer :: nx, ny, nz, ix, iy, iz
        nx = htg%nx
        ny = htg%ny
        nz = htg%nz

        do ix = 1, nx
            div_v(ix, :, :) = (vx(ixh(ix), :, :) - vx(ixl(ix), :, :)) * idx(ix)
        enddo

        do iy = 1, ny
            div_v(:, iy, :) = div_v(:, iy, :) + &
                (vy(:, iyh(iy), :) - vy(:, iyl(iy), :)) * idy(iy)
        enddo

        do iz = 1, nz
            div_v(:, :, iz) = div_v(:, :, iz) + &
                (vz(:, :, izh(iz)) - vz(:, :, izl(iz))) * idz(iz)
        enddo
    end subroutine calc_div_v

    !---------------------------------------------------------------------------
    ! Calculate the divergence of v of a single fluid
    !---------------------------------------------------------------------------
    subroutine calc_div_v_single
        use usingle, only: vsx, vsy, vsz
        use neighbors_module, only: ixl, iyl, izl, ixh, iyh, izh, idx, idy, idz
        use pic_fields, only: bx, by, bz, absB
        implicit none
        integer :: nx, ny, nz, ix, iy, iz
        nx = htg%nx
        ny = htg%ny
        nz = htg%nz

        vdotb = (vsx*bx + vsy*by + vsz*bz) / absB**2
        vpx = vdotb * bx
        vpy = vdotb * by
        vpz = vdotb * bz

        do ix = 1, nx
            div_vsingle(ix, :, :) = &
                (vsx(ixh(ix), :, :) - vsx(ixl(ix), :, :)) * idx(ix)
            div_vpara_vsingle(ix, :, :) = &
                (vpx(ixh(ix), :, :) - vpx(ixl(ix), :, :)) * idx(ix)
        enddo

        do iy = 1, ny
            div_vsingle(:, iy, :) = div_vsingle(:, iy, :) + &
                (vsy(:, iyh(iy), :) - vsy(:, iyl(iy), :)) * idy(iy)
            div_vpara_vsingle(:, iy, :) = div_vpara_vsingle(:, iy, :) + &
                (vpy(:, iyh(iy), :) - vpy(:, iyl(iy), :)) * idy(iy)
        enddo

        do iz = 1, nz
            div_vsingle(:, :, iz) = div_vsingle(:, :, iz) + &
                (vsz(:, :, izh(iz)) - vsz(:, :, izl(iz))) * idz(iz)
            div_vpara_vsingle(:, :, iz) = div_vpara_vsingle(:, :, iz) + &
                (vpz(:, :, izh(iz)) - vpz(:, :, izl(iz))) * idz(iz)
        enddo
    end subroutine calc_div_v_single

    !---------------------------------------------------------------------------
    ! Calculate p\nabla\cdot\vec{v}. Here, p is the scalar pressure.
    !---------------------------------------------------------------------------
    subroutine calc_pdiv_v
        use pressure_tensor, only: pscalar
        use para_perp_pressure, only: pperp
        implicit none

        pdiv_v = - pscalar * div_v
        pdiv_vsingle = - pscalar * div_vsingle
        pdiv_vpara_vsingle = - pscalar * div_vpara_vsingle
    end subroutine calc_pdiv_v

    !---------------------------------------------------------------------------
    ! Calculate the compression related variables.
    !---------------------------------------------------------------------------
    subroutine calc_compression
        implicit none
        call calc_div_v
        call calc_div_v_single
        call calc_pdiv_v
    end subroutine calc_compression

    !---------------------------------------------------------------------------
    ! Calculate bbsigma = b_ib_j\sigma_{ij} for a single fluid
    !---------------------------------------------------------------------------
    subroutine calc_bbsigma_single
        use pic_fields, only: bx, by, bz, absB
        use usingle, only: vsx, vsy, vsz
        use neighbors_module, only: ixl, iyl, izl, ixh, iyh, izh, idx, idy, idz
        implicit none
        real(fp) :: sigma_xx, sigma_xy, sigma_xz, sigma_yy, sigma_yz, sigma_zz
        real(fp) :: sigma_yx, sigma_zx, sigma_zy
        real(fp) :: bxc, byc, bzc
        integer :: nx, ny, nz, ix, iy, iz
        nx = htg%nx
        ny = htg%ny
        nz = htg%nz

        do iz = 1, nz
            do iy = 1, ny
                 do ix = 1, nx
                    sigma_xx = (vpx(ixh(ix), iy, iz) - vpx(ixl(ix), iy, iz)) * &
                               idx(ix) - div_vpara_vsingle(ix, iy, iz) / 3.0
                    sigma_yy = (vpy(ix, iyh(iy), iz) - vpy(ix, iyl(iy), iz)) * &
                               idy(iy) - div_vpara_vsingle(ix, iy, iz) / 3.0
                    sigma_zz = (vpz(ix, iy, izh(iz)) - vpz(ix, iy, izl(iz))) * &
                               idz(iz) - div_vpara_vsingle(ix, iy, iz) / 3.0
                    sigma_xy = 0.5 * (vpx(ix, iyh(iy), iz) - &
                                      vpx(ix, iyl(iy), iz)) * idy(iy) + &
                               0.5 * (vpy(ixh(ix), iy, iz) - &
                                      vpy(ixl(ix), iy, iz)) * idx(ix)
                    sigma_xz = 0.5 * (vpx(ix, iy, izh(iz)) - &
                                      vpx(ix, iy, izl(iz))) * idz(iz) + &
                               0.5 * (vpz(ixh(ix), iy, iz) - &
                                      vpz(ixl(ix), iy, iz)) * idx(ix)
                    sigma_yz = 0.5 * (vpy(ix, iy, izh(iz)) - &
                                      vpy(ix, iy, izl(iz))) * idz(iz) + &
                               0.5 * (vpz(ix, iyh(iy), iz) - &
                                      vpz(ix, iyl(iy), iz)) * idy(iy)
                    bxc = bx(ix, iy, iz)
                    byc = by(ix, iy, iz)
                    bzc = bz(ix, iy, iz)
                    bbsigma_para_vsingle(ix, iy, iz) = bxc**2 * sigma_xx + &
                        byc**2 * sigma_yy + bzc**2 * sigma_zz + &
                        2.0 * bxc * byc * sigma_xy + &
                        2.0 * bxc * bzc * sigma_xz + &
                        2.0 * byc * bzc * sigma_yz
                    sigma_xx = (vsx(ixh(ix), iy, iz) - vsx(ixl(ix), iy, iz)) * &
                               idx(ix) - div_vsingle(ix, iy, iz) / 3.0
                    sigma_yy = (vsy(ix, iyh(iy), iz) - vsy(ix, iyl(iy), iz)) * &
                               idy(iy) - div_vsingle(ix, iy, iz) / 3.0
                    sigma_zz = (vsz(ix, iy, izh(iz)) - vsz(ix, iy, izl(iz))) * &
                               idz(iz) - div_vsingle(ix, iy, iz) / 3.0
                    sigma_xy = 0.5 * (vsx(ix, iyh(iy), iz) - &
                                      vsx(ix, iyl(iy), iz)) * idy(iy) + &
                               0.5 * (vsy(ixh(ix), iy, iz) - &
                                      vsy(ixl(ix), iy, iz)) * idx(ix)
                    sigma_xz = 0.5 * (vsx(ix, iy, izh(iz)) - &
                                      vsx(ix, iy, izl(iz))) * idz(iz) + &
                               0.5 * (vsz(ixh(ix), iy, iz) - &
                                      vsz(ixl(ix), iy, iz)) * idx(ix)
                    sigma_yz = 0.5 * (vsy(ix, iy, izh(iz)) - &
                                      vsy(ix, iy, izl(iz))) * idz(iz) + &
                               0.5 * (vsz(ix, iyh(iy), iz) - &
                                      vsz(ix, iyl(iy), iz)) * idy(iy)
                    bbsigma_single(ix, iy, iz) = bxc**2 * sigma_xx + &
                        byc**2 * sigma_yy + bzc**2 * sigma_zz + &
                        2.0 * bxc * byc * sigma_xy + &
                        2.0 * bxc * bzc * sigma_xz + &
                        2.0 * byc * bzc * sigma_yz
                 enddo
            enddo
        enddo
        bbsigma_para_vsingle = bbsigma_para_vsingle / absB**2
        bbsigma_single = bbsigma_single / absB**2
    end subroutine calc_bbsigma_single

    !---------------------------------------------------------------------------
    ! Calculate bbsigma = b_ib_j\sigma_{ij}.
    !---------------------------------------------------------------------------
    subroutine calc_bbsigma
        use pic_fields, only: bx, by, bz, vx, vy, vz, absB
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
                    sigma_xx = (vx(ixh(ix), iy, iz) - vx(ixl(ix), iy, iz)) * &
                               idx(ix) - div_v(ix, iy, iz) / 3.0
                    sigma_yy = (vy(ix, iyh(iy), iz) - vy(ix, iyl(iy), iz)) * &
                               idy(iy) - div_v(ix, iy, iz) / 3.0
                    sigma_zz = (vz(ix, iy, izh(iz)) - vz(ix, iy, izl(iz))) * &
                               idz(iz) - div_v(ix, iy, iz) / 3.0
                    sigma_xy = 0.5 * (vx(ix, iyh(iy), iz) - &
                                      vx(ix, iyl(iy), iz)) * idy(iy) + &
                               0.5 * (vy(ixh(ix), iy, iz) - &
                                      vy(ixl(ix), iy, iz)) * idx(ix)
                    sigma_xz = 0.5 * (vx(ix, iy, izh(iz)) - &
                                      vx(ix, iy, izl(iz))) * idz(iz) + &
                               0.5 * (vz(ixh(ix), iy, iz) - &
                                      vz(ixl(ix), iy, iz)) * idx(ix)
                    sigma_yz = 0.5 * (vy(ix, iy, izh(iz)) - &
                                      vy(ix, iy, izl(iz))) * idz(iz) + &
                               0.5 * (vz(ix, iyh(iy), iz) - &
                                      vz(ix, iyl(iy), iz)) * idy(iy)
                    bxc = bx(ix, iy, iz)
                    byc = by(ix, iy, iz)
                    bzc = bz(ix, iy, iz)
                    bbsigma(ix, iy, iz) = bxc**2 * sigma_xx + &
                        byc**2 * sigma_yy + bzc**2 * sigma_zz + &
                        2.0 * bxc * byc * sigma_xy + &
                        2.0 * bxc * bzc * sigma_xz + &
                        2.0 * byc * bzc * sigma_yz
                 enddo
            enddo
        enddo
        bbsigma = bbsigma / absB**2
    end subroutine calc_bbsigma

    !---------------------------------------------------------------------------
    ! Calculate (p_\parallel - p_\perp)b_ib_j\sigma_{ij}, where \sigma_{ij}
    ! is the shear tensor.
    !---------------------------------------------------------------------------
    subroutine calc_pshear
        use para_perp_pressure, only: ppara, pperp
        implicit none
        pshear = (pperp - ppara) * bbsigma
        pshear_vsingle = (pperp - ppara) * bbsigma_single
        pshear_para_vsingle = (pperp - ppara) * bbsigma_para_vsingle
    end subroutine calc_pshear

    !---------------------------------------------------------------------------
    ! Calculate shear related variables.
    !---------------------------------------------------------------------------
    subroutine calc_shear
        implicit none
        call calc_bbsigma
        call calc_bbsigma_single
        call calc_pshear
    end subroutine calc_shear

    !---------------------------------------------------------------------------
    ! Calculate \vec{u}\cdot(\nalba\cdot\tensor(P))
    !---------------------------------------------------------------------------
    subroutine calc_vdot_div_ptensor
        use pressure_tensor, only: divp_x, divp_y, divp_z
        use pic_fields, only: vx, vy, vz
        implicit none
        vdot_div_ptensor = vx * divp_x + vy * divp_y + vz * divp_z
    end subroutine calc_vdot_div_ptensor

    !---------------------------------------------------------------------------
    ! Calculate the compressional and shear heating terms.
    !---------------------------------------------------------------------------
    subroutine calc_compression_shear
        implicit none
        call calc_compression
        call calc_shear
        call calc_vdot_div_ptensor
        call calc_div_vdot_ptensor
    end subroutine calc_compression_shear

    !---------------------------------------------------------------------------
    ! Calculate div_vdot_ptensor.
    !---------------------------------------------------------------------------
    subroutine calc_div_vdot_ptensor
        use pic_fields, only: vx, vy, vz, pxx, pxy, pxz, pyy, pyz, pzz
        use neighbors_module, only: ixl, iyl, izl, ixh, iyh, izh, idx, idy, idz
        implicit none
        integer :: nx, ny, nz, ix, iy, iz

        vdot_ptensor_x = vx * pxx + vy * pxy + vz * pxz
        vdot_ptensor_y = vx * pxy + vy * pyy + vz * pyz
        vdot_ptensor_z = vx * pxz + vy * pyz + vz * pzz

        nx = htg%nx
        ny = htg%ny
        nz = htg%nz

        do ix = 1, nx
            div_vdot_ptensor(ix, :, :) = (vdot_ptensor_x(ixh(ix), :, :) - &
                    vdot_ptensor_x(ixl(ix), :, :)) * idx(ix)
        enddo

        do iy = 1, ny
            div_vdot_ptensor(:, iy, :) = div_vdot_ptensor(:, iy, :) + &
                    (vdot_ptensor_y(:, iyh(iy), :) - &
                     vdot_ptensor_y(:, iyl(iy), :)) * idy(iy)
        enddo

        do iz = 1, nz
            div_vdot_ptensor(:, :, iz) = div_vdot_ptensor(:, :, iz) + &
                    (vdot_ptensor_z(:, :, izh(iz)) - &
                     vdot_ptensor_z(:, :, izl(iz))) * idz(iz)
        enddo
    end subroutine calc_div_vdot_ptensor

    !---------------------------------------------------------------------------
    ! Save div_v.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_div_v(ct)
        use mpi_module
        use constants, only: fp
        use mpi_io_fields, only: save_field
        implicit none
        integer, intent(in) :: ct
        if (myid == master) then
            print*, 'Saving the divergence of the velocity field', ct
        endif

        call save_field(div_v, 'div_v', ct)
    end subroutine save_div_v

    !---------------------------------------------------------------------------
    ! Save div_vsingle and div_vpara_vsingle
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_div_vsingle(ct)
        use mpi_module
        use constants, only: fp
        use mpi_io_fields, only: save_field
        implicit none
        integer, intent(in) :: ct
        if (myid == master) then
            print*, 'Saving div_vsingle and div_vpara_vsingle', ct
        endif

        call save_field(div_vsingle, 'div_vsingle', ct)
        call save_field(div_vpara_vsingle, 'div_vpara_vsingle', ct)
    end subroutine save_div_vsingle

    !---------------------------------------------------------------------------
    ! Save pdiv_v.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_pdiv_v(ct)
        use mpi_module
        use constants, only: fp
        use mpi_io_fields, only: save_field
        implicit none
        integer, intent(in) :: ct
        if (myid == master) then
            print*, 'Saving pdiv_v', ct
        endif

        call save_field(pdiv_v, 'pdiv_v', ct)
    end subroutine save_pdiv_v

    !---------------------------------------------------------------------------
    ! Save pdiv_v.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_pdiv_vsingle(ct)
        use mpi_module
        use constants, only: fp
        use mpi_io_fields, only: save_field
        implicit none
        integer, intent(in) :: ct
        if (myid == master) then
            print*, 'Saving pdiv_vsingle and pdiv_vpara_vsingle', ct
        endif

        call save_field(pdiv_vsingle, 'pdiv_vsingle', ct)
        call save_field(pdiv_vpara_vsingle, 'pdiv_vpara_vsingle', ct)
    end subroutine save_pdiv_vsingle

    !---------------------------------------------------------------------------
    ! Save compression related variables.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_compression(ct)
        implicit none
        integer, intent(in) :: ct
        call save_div_v(ct)
        call save_pdiv_v(ct)
        call save_div_vsingle(ct)
        call save_pdiv_vsingle(ct)
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
    ! Save bbsigma_single and bbsigma_para_vsingle.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_bbsigma_single(ct)
        use mpi_module
        use constants, only: fp
        use mpi_io_fields, only: save_field
        implicit none
        integer, intent(in) :: ct
        if (myid == master) then
            print*, 'Saving b_ib_j\sigma_{ij} for single fluid velocity', ct
        endif

        call save_field(bbsigma_single, 'bbsigma_single', ct)
        call save_field(bbsigma_para_vsingle, 'bbsigma_para_vsingle', ct)
    end subroutine save_bbsigma_single

    !---------------------------------------------------------------------------
    ! Save pshear_vsingle and pshear_para_vsingle
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_pshear_single(ct)
        use mpi_module
        use constants, only: fp
        use mpi_io_fields, only: save_field
        implicit none
        integer, intent(in) :: ct
        if (myid == master) then
            print*, 'Saving pshear_vsingle and pshear_para_vsingle', ct
        endif

        call save_field(pshear_vsingle, 'pshear_vsingle', ct)
        call save_field(pshear_para_vsingle, 'pshear_para_vsingle', ct)
    end subroutine save_pshear_single

    !---------------------------------------------------------------------------
    ! Save shear related variables.
    !---------------------------------------------------------------------------
    subroutine save_shear(ct)
        implicit none
        integer, intent(in) :: ct
        call save_bbsigma(ct)
        call save_pshear(ct)
        call save_bbsigma_single(ct)
        call save_pshear_single(ct)
    end subroutine save_shear

    !---------------------------------------------------------------------------
    ! Save vdot_div_ptensor
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_vdot_div_ptensor(ct)
        use mpi_module
        use constants, only: fp
        use mpi_io_fields, only: save_field
        implicit none
        integer, intent(in) :: ct
        if (myid == master) then
            print*, 'Saving vdot_div_ptensor', ct
        endif

        call save_field(vdot_div_ptensor, 'vdot_div_ptensor', ct)
    end subroutine save_vdot_div_ptensor

    !---------------------------------------------------------------------------
    ! Save compressional and shear heating terms.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_compression_shear(ct)
        implicit none
        integer, intent(in) :: ct
        call save_compression(ct)
        call save_shear(ct)
        call save_vdot_div_ptensor(ct)
        call save_div_vdot_ptensor(ct)
    end subroutine save_compression_shear

    !---------------------------------------------------------------------------
    ! Save div_vdot_ptensor.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_div_vdot_ptensor(ct)
        use mpi_module
        use constants, only: fp
        use mpi_io_fields, only: save_field
        implicit none
        integer, intent(in) :: ct
        if (myid == master) then
            print*, 'Saving div_vdot_ptensor', ct
        endif

        call save_field(div_vdot_ptensor, 'div_vdot_ptensor', ct)
    end subroutine save_div_vdot_ptensor

    !---------------------------------------------------------------------------
    ! Save the total of the compressional terms.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_tot_compression(ct)
        use mpi_module
        use constants, only: fp
        use particle_info, only: ibtag, species
        use statistics, only: get_average_and_total
        use parameters, only: tp1
        implicit none
        integer, intent(in) :: ct
        real(fp) :: div_v_tot, pdiv_v_tot, avg
        real(fp) :: div_vsingle_tot, div_vpara_vsingle_tot
        real(fp) :: pdiv_vsingle_tot, pdiv_vpara_vsingle_tot
        character(len=100) :: fname
        integer :: current_pos, output_record
        logical :: dir_e

        if (myid == master) then
            inquire(file='./data/.', exist=dir_e)
            if (.not. dir_e) then
                call system('mkdir ./data')
            endif
        endif

        call get_average_and_total(div_v, avg, div_v_tot)
        call get_average_and_total(pdiv_v, avg, pdiv_v_tot)
        call get_average_and_total(div_vsingle, avg, div_vsingle_tot)
        call get_average_and_total(div_vpara_vsingle, avg, div_vpara_vsingle_tot)
        call get_average_and_total(pdiv_vsingle, avg, pdiv_vsingle_tot)
        call get_average_and_total(pdiv_vpara_vsingle, avg, &
            pdiv_vpara_vsingle_tot)
        if (myid == master) then
            fname = 'data/compression'//ibtag//'_'//species//'.gda'
            open(unit=51, file=trim(adjustl(fname)), access='stream',&
                 status='unknown', form='unformatted', action='write')
            output_record = ct - tp1 + 1
            current_pos = 6 * sizeof(fp) * (output_record-1) + 1
            write(51, pos=current_pos) div_v_tot, pdiv_v_tot, &
                div_vsingle_tot, div_vpara_vsingle_tot, &
                pdiv_vsingle_tot, pdiv_vpara_vsingle_tot
            close(51)
        endif
    end subroutine save_tot_compression

    !---------------------------------------------------------------------------
    ! Save the total of the shear terms.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_tot_shear(ct)
        use mpi_module
        use constants, only: fp
        use particle_info, only: ibtag, species
        use statistics, only: get_average_and_total
        use parameters, only: tp1
        implicit none
        integer, intent(in) :: ct
        real(fp) :: bbsigma_tot, pshear_tot, avg
        real(fp) :: bbsigma_single_tot, bbsigma_para_vsingle_tot
        real(fp) :: pshear_vsingle_tot, pshear_para_vsingle_tot
        character(len=100) :: fname
        integer :: current_pos, output_record
        logical :: dir_e

        if (myid == master) then
            inquire(file='./data/.', exist=dir_e)
            if (.not. dir_e) then
                call system('mkdir ./data')
            endif
        endif

        call get_average_and_total(bbsigma, avg, bbsigma_tot)
        call get_average_and_total(pshear, avg, pshear_tot)
        call get_average_and_total(bbsigma_single, avg, bbsigma_single_tot)
        call get_average_and_total(bbsigma_para_vsingle, avg, &
            bbsigma_para_vsingle_tot)
        call get_average_and_total(pshear_vsingle, avg, pshear_vsingle_tot)
        call get_average_and_total(pshear_para_vsingle, avg, &
            pshear_para_vsingle_tot)
        if (myid == master) then
            fname = 'data/shear'//ibtag//'_'//species//'.gda'
            open(unit=51, file=trim(adjustl(fname)), access='stream',&
                 status='unknown', form='unformatted', action='write')
            output_record = ct - tp1 + 1
            current_pos = 6 * sizeof(fp) * (output_record-1) + 1
            write(51, pos=current_pos) bbsigma_tot, pshear_tot, &
                bbsigma_single_tot, bbsigma_para_vsingle_tot, &
                pshear_vsingle_tot, pshear_para_vsingle_tot
            close(51)
        endif
    end subroutine save_tot_shear

    !---------------------------------------------------------------------------
    ! Save the total of vdot_div_ptensor.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_tot_vdot_div_ptensor(ct)
        use mpi_module
        use constants, only: fp
        use particle_info, only: ibtag, species
        use statistics, only: get_average_and_total
        use parameters, only: tp1
        implicit none
        integer, intent(in) :: ct
        real(fp) :: vdot_div_ptensor_tot, avg
        character(len=100) :: fname
        integer :: current_pos, output_record
        logical :: dir_e

        if (myid == master) then
            inquire(file='./data/.', exist=dir_e)
            if (.not. dir_e) then
                call system('mkdir ./data')
            endif
        endif

        call get_average_and_total(vdot_div_ptensor, avg, vdot_div_ptensor_tot)
        if (myid == master) then
            fname = 'data/vdot_div_ptensor'//ibtag//'_'//species//'.gda'
            open(unit=51, file=trim(adjustl(fname)), access='stream',&
                 status='unknown', form='unformatted', action='write')
            output_record = ct - tp1 + 1
            current_pos = sizeof(fp) * (output_record-1) + 1
            write(51, pos=current_pos) vdot_div_ptensor_tot
            close(51)
        endif
    end subroutine save_tot_vdot_div_ptensor

    !---------------------------------------------------------------------------
    ! Save the total of the compressional and shear heating terms.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_tot_compression_shear(ct)
        implicit none
        integer, intent(in) :: ct
        call save_tot_compression(ct)
        call save_tot_shear(ct)
        call save_tot_vdot_div_ptensor(ct)
        call save_tot_div_vdot_ptensor(ct)
    end subroutine save_tot_compression_shear

    !---------------------------------------------------------------------------
    ! Save the total of div_vdot_ptensor.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_tot_div_vdot_ptensor(ct)
        use mpi_module
        use constants, only: fp
        use particle_info, only: ibtag, species
        use statistics, only: get_average_and_total
        use parameters, only: tp1
        implicit none
        integer, intent(in) :: ct
        real(fp) :: div_vdot_ptensor_tot, avg
        character(len=100) :: fname
        integer :: current_pos, output_record
        logical :: dir_e

        if (myid == master) then
            inquire(file='./data/.', exist=dir_e)
            if (.not. dir_e) then
                call system('mkdir ./data')
            endif
        endif

        call get_average_and_total(div_vdot_ptensor, avg, div_vdot_ptensor_tot)
        if (myid == master) then
            fname = 'data/div_vdot_ptensor'//ibtag//'_'//species//'.gda'
            open(unit=51, file=trim(adjustl(fname)), access='stream',&
                 status='unknown', form='unformatted', action='write')
            output_record = ct - tp1 + 1
            current_pos = sizeof(fp) * (output_record-1) + 1
            write(51, pos=current_pos) div_vdot_ptensor_tot
            close(51)
        endif
    end subroutine save_tot_div_vdot_ptensor

end module compression_shear
