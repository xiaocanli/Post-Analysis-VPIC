!<******************************************************************************
!< This module include the methods to calculate the perpendicular components of
!< velocity field and their derivatives.
!<******************************************************************************
module vperp_derivatives
    use constants, only: fp, dp
    use picinfo, only: domain
    use pic_fields, only: bx, by, bz, vx, vy, vz, absB
    implicit none
    private
    save
    
    public init_vperp_derivatives, init_vperp_components, &
        free_vperp_derivatives, free_vperp_components, calc_vperp_components, &
        calc_vperp_derivatives
    public vperpx, vperpy, vperpz
    public dvperpx_dx, dvperpx_dy, dvperpx_dz, dvperpy_dx, dvperpy_dy, &
        dvperpy_dz, dvperpz_dx, dvperpz_dy, dvperpz_dz

    real(fp), allocatable, dimension(:,:,:) :: vperpx, vperpy, vperpz
    real(fp), allocatable, dimension(:,:,:) :: dvperpx_dx, dvperpx_dy, dvperpx_dz
    real(fp), allocatable, dimension(:,:,:) :: dvperpy_dx, dvperpy_dy, dvperpy_dz
    real(fp), allocatable, dimension(:,:,:) :: dvperpz_dx, dvperpz_dy, dvperpz_dz

    contains

    !<--------------------------------------------------------------------------
    !< Initialize the perpendicular components of velocity field
    !<--------------------------------------------------------------------------
    subroutine init_vperp_components(nx, ny, nz)
        implicit none
        integer, intent(in) :: nx, ny, nz

        allocate(vperpx(nx, ny, nz))
        allocate(vperpy(nx, ny, nz))
        allocate(vperpz(nx, ny, nz))

        vperpx = 0.0; vperpy = 0.0; vperpz = 0.0
    end subroutine init_vperp_components

    !<--------------------------------------------------------------------------
    !< Initialize the derivatives of vperp components
    !<--------------------------------------------------------------------------
    subroutine init_vperp_derivatives(nx, ny, nz)
        implicit none
        integer, intent(in) :: nx, ny, nz

        allocate(dvperpx_dx(nx, ny, nz))
        allocate(dvperpx_dy(nx, ny, nz))
        allocate(dvperpx_dz(nx, ny, nz))
        allocate(dvperpy_dx(nx, ny, nz))
        allocate(dvperpy_dy(nx, ny, nz))
        allocate(dvperpy_dz(nx, ny, nz))
        allocate(dvperpz_dx(nx, ny, nz))
        allocate(dvperpz_dy(nx, ny, nz))
        allocate(dvperpz_dz(nx, ny, nz))

        dvperpx_dx = 0.0; dvperpx_dy = 0.0; dvperpx_dz = 0.0
        dvperpy_dx = 0.0; dvperpy_dy = 0.0; dvperpy_dz = 0.0
        dvperpz_dx = 0.0; dvperpz_dy = 0.0; dvperpz_dz = 0.0
    end subroutine init_vperp_derivatives

    !<--------------------------------------------------------------------------
    !< Free the perpendicular components of velocity field
    !<--------------------------------------------------------------------------
    subroutine free_vperp_components
        implicit none
        deallocate(vperpx, vperpy, vperpz)
    end subroutine free_vperp_components

    !<--------------------------------------------------------------------------
    !< Free the derivatives of vperp components
    !<--------------------------------------------------------------------------
    subroutine free_vperp_derivatives
        implicit none
        deallocate(dvperpx_dx, dvperpx_dy, dvperpx_dz)
        deallocate(dvperpy_dx, dvperpy_dy, dvperpy_dz)
        deallocate(dvperpz_dx, dvperpz_dy, dvperpz_dz)
    end subroutine free_vperp_derivatives

    !<--------------------------------------------------------------------------
    !< Calculate the perpendicular components of velocity field
    !<--------------------------------------------------------------------------
    subroutine calc_vperp_components
        implicit none
        vperpx = (vx * bx + vy * by + vz * bz) / absB**2
        vperpy = vy - vperpx * by
        vperpz = vz - vperpx * bz
        vperpx = vx - vperpx * bx
    end subroutine calc_vperp_components

    !<--------------------------------------------------------------------------
    !< Calculate the derivatives of vperp
    !<--------------------------------------------------------------------------
    subroutine calc_vperp_derivatives(nx, ny, nz)
        use neighbors_module, only: ixl, iyl, izl, ixh, iyh, izh, idx, idy, idz
        implicit none
        integer, intent(in) :: nx, ny, nz
        integer :: ix, iy, iz
        do ix = 1, nx
            dvperpx_dx(ix, :, :) = (vperpx(ixh(ix), :, :) - vperpx(ixl(ix), :, :)) * idx(ix)
            dvperpy_dx(ix, :, :) = (vperpy(ixh(ix), :, :) - vperpy(ixl(ix), :, :)) * idx(ix)
            dvperpz_dx(ix, :, :) = (vperpz(ixh(ix), :, :) - vperpz(ixl(ix), :, :)) * idx(ix)
        enddo

        do iy = 1, ny
            dvperpx_dy(:, iy, :) = (vperpx(:, iyh(iy), :) - vperpx(:, iyl(iy), :)) * idy(iy)
            dvperpy_dy(:, iy, :) = (vperpy(:, iyh(iy), :) - vperpy(:, iyl(iy), :)) * idy(iy)
            dvperpz_dy(:, iy, :) = (vperpz(:, iyh(iy), :) - vperpz(:, iyl(iy), :)) * idy(iy)
        enddo

        do iz = 1, nz
            dvperpx_dz(:, :, iz) = (vperpx(:, :, izh(iz)) - vperpx(:, :, izl(iz))) * idz(iz)
            dvperpy_dz(:, :, iz) = (vperpy(:, :, izh(iz)) - vperpy(:, :, izl(iz))) * idz(iz)
            dvperpz_dz(:, :, iz) = (vperpz(:, :, izh(iz)) - vperpz(:, :, izl(iz))) * idz(iz)
        enddo
    end subroutine calc_vperp_derivatives
end module vperp_derivatives
