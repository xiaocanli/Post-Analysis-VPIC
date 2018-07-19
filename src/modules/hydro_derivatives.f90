!<******************************************************************************
!< This module include the methods to calculate the derivatives of hydro fields.
!<******************************************************************************
module hydro_derivatives
    use constants, only: fp, dp
    use picinfo, only: domain
    use pic_fields, only: ux, uy, uz, vx, vy, vz
    implicit none
    private
    save

    public init_ufield_derivatives, free_ufield_derivatives, &
        calc_ufield_derivatives
    public init_vfield_derivatives, free_vfield_derivatives, &
        calc_vfield_derivatives
    public duxdx, duxdy, duxdz, duydx, duydy, duydz, duzdx, duzdy, duzdz
    public dvelx_dx, dvelx_dy, dvelx_dz, dvely_dx, dvely_dy, dvely_dz, &
        dvelz_dx, dvelz_dy, dvelz_dz

    real(fp), allocatable, dimension(:,:,:) :: duxdx, duxdy, duxdz
    real(fp), allocatable, dimension(:,:,:) :: duydx, duydy, duydz
    real(fp), allocatable, dimension(:,:,:) :: duzdx, duzdy, duzdz
    real(fp), allocatable, dimension(:,:,:) :: dvelx_dx, dvelx_dy, dvelx_dz
    real(fp), allocatable, dimension(:,:,:) :: dvely_dx, dvely_dy, dvely_dz
    real(fp), allocatable, dimension(:,:,:) :: dvelz_dx, dvelz_dy, dvelz_dz

    contains

    !<--------------------------------------------------------------------------
    !< Initialize the derivatives of momentum field
    !<--------------------------------------------------------------------------
    subroutine init_ufield_derivatives(nx, ny, nz)
        implicit none
        integer, intent(in) :: nx, ny, nz

        allocate(duxdx(nx, ny, nz))
        allocate(duxdy(nx, ny, nz))
        allocate(duxdz(nx, ny, nz))
        allocate(duydx(nx, ny, nz))
        allocate(duydy(nx, ny, nz))
        allocate(duydz(nx, ny, nz))
        allocate(duzdx(nx, ny, nz))
        allocate(duzdy(nx, ny, nz))
        allocate(duzdz(nx, ny, nz))

        duxdx = 0.0; duxdy = 0.0; duxdz = 0.0
        duydx = 0.0; duydy = 0.0; duydz = 0.0
        duzdx = 0.0; duzdy = 0.0; duzdz = 0.0
    end subroutine init_ufield_derivatives

    !<--------------------------------------------------------------------------
    !< Initialize the derivatives of velocity field
    !<--------------------------------------------------------------------------
    subroutine init_vfield_derivatives(nx, ny, nz)
        implicit none
        integer, intent(in) :: nx, ny, nz

        allocate(dvelx_dx(nx, ny, nz))
        allocate(dvelx_dy(nx, ny, nz))
        allocate(dvelx_dz(nx, ny, nz))
        allocate(dvely_dx(nx, ny, nz))
        allocate(dvely_dy(nx, ny, nz))
        allocate(dvely_dz(nx, ny, nz))
        allocate(dvelz_dx(nx, ny, nz))
        allocate(dvelz_dy(nx, ny, nz))
        allocate(dvelz_dz(nx, ny, nz))

        dvelx_dx = 0.0; dvelx_dy = 0.0; dvelx_dz = 0.0
        dvely_dx = 0.0; dvely_dy = 0.0; dvely_dz = 0.0
        dvelz_dx = 0.0; dvelz_dy = 0.0; dvelz_dz = 0.0
    end subroutine init_vfield_derivatives

    !<--------------------------------------------------------------------------
    !< Free the derivatives of momentum field
    !<--------------------------------------------------------------------------
    subroutine free_ufield_derivatives
        implicit none
        deallocate(duxdx, duxdy, duxdz)
        deallocate(duydx, duydy, duydz)
        deallocate(duzdx, duzdy, duzdz)
    end subroutine free_ufield_derivatives

    !<--------------------------------------------------------------------------
    !< Free the derivatives of velocity field
    !<--------------------------------------------------------------------------
    subroutine free_vfield_derivatives
        implicit none
        deallocate(dvelx_dx, dvelx_dy, dvelx_dz)
        deallocate(dvely_dx, dvely_dy, dvely_dz)
        deallocate(dvelz_dx, dvelz_dy, dvelz_dz)
    end subroutine free_vfield_derivatives

    !<--------------------------------------------------------------------------
    !< Calculate the derivatives of momentum field
    !<--------------------------------------------------------------------------
    subroutine calc_ufield_derivatives(nx, ny, nz)
        use neighbors_module, only: ixl, iyl, izl, ixh, iyh, izh, idx, idy, idz
        implicit none
        integer, intent(in) :: nx, ny, nz
        integer :: ix, iy, iz
        do ix = 1, nx
            duxdx(ix, :, :) = (ux(ixh(ix), :, :) - ux(ixl(ix), :, :)) * idx(ix)
            duydx(ix, :, :) = (uy(ixh(ix), :, :) - uy(ixl(ix), :, :)) * idx(ix)
            duzdx(ix, :, :) = (uz(ixh(ix), :, :) - uz(ixl(ix), :, :)) * idx(ix)
        enddo

        do iy = 1, ny
            duxdy(:, iy, :) = (ux(:, iyh(iy), :) - ux(:, iyl(iy), :)) * idy(iy)
            duydy(:, iy, :) = (uy(:, iyh(iy), :) - uy(:, iyl(iy), :)) * idy(iy)
            duzdy(:, iy, :) = (uz(:, iyh(iy), :) - uz(:, iyl(iy), :)) * idy(iy)
        enddo

        do iz = 1, nz
            duxdz(:, :, iz) = (ux(:, :, izh(iz)) - ux(:, :, izl(iz))) * idz(iz)
            duydz(:, :, iz) = (uy(:, :, izh(iz)) - uy(:, :, izl(iz))) * idz(iz)
            duzdz(:, :, iz) = (uz(:, :, izh(iz)) - uz(:, :, izl(iz))) * idz(iz)
        enddo
    end subroutine calc_ufield_derivatives

    !<--------------------------------------------------------------------------
    !< Calculate the derivatives of velocity field
    !<--------------------------------------------------------------------------
    subroutine calc_vfield_derivatives(nx, ny, nz)
        use neighbors_module, only: ixl, iyl, izl, ixh, iyh, izh, idx, idy, idz
        implicit none
        integer, intent(in) :: nx, ny, nz
        integer :: ix, iy, iz
        do ix = 1, nx
            dvelx_dx(ix, :, :) = (vx(ixh(ix), :, :) - vx(ixl(ix), :, :)) * idx(ix)
            dvely_dx(ix, :, :) = (vy(ixh(ix), :, :) - vy(ixl(ix), :, :)) * idx(ix)
            dvelz_dx(ix, :, :) = (vz(ixh(ix), :, :) - vz(ixl(ix), :, :)) * idx(ix)
        enddo

        do iy = 1, ny
            dvelx_dy(:, iy, :) = (vx(:, iyh(iy), :) - vx(:, iyl(iy), :)) * idy(iy)
            dvely_dy(:, iy, :) = (vy(:, iyh(iy), :) - vy(:, iyl(iy), :)) * idy(iy)
            dvelz_dy(:, iy, :) = (vz(:, iyh(iy), :) - vz(:, iyl(iy), :)) * idy(iy)
        enddo

        do iz = 1, nz
            dvelx_dz(:, :, iz) = (vx(:, :, izh(iz)) - vx(:, :, izl(iz))) * idz(iz)
            dvely_dz(:, :, iz) = (vy(:, :, izh(iz)) - vy(:, :, izl(iz))) * idz(iz)
            dvelz_dz(:, :, iz) = (vz(:, :, izh(iz)) - vz(:, :, izl(iz))) * idz(iz)
        enddo
    end subroutine calc_vfield_derivatives
end module hydro_derivatives
