!<******************************************************************************
!< This module include the methods to calculate the derivatives of hydro fields.
!<******************************************************************************
module hydro_derivatives
    use constants, only: fp, dp
    use picinfo, only: domain
    use pic_fields, only: ux, uy, uz
    implicit none
    private
    save

    public init_ufield_derivatives, free_ufield_derivatives, &
        calc_ufield_derivatives
    public duxdx, duxdy, duxdz, duydx, duydy, duydz, duzdx, duzdy, duzdz

    real(fp), allocatable, dimension(:,:,:) :: duxdx, duxdy, duxdz
    real(fp), allocatable, dimension(:,:,:) :: duydx, duydy, duydz
    real(fp), allocatable, dimension(:,:,:) :: duzdx, duzdy, duzdz

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
    !< Free the derivatives of momentum field 
    !<--------------------------------------------------------------------------
    subroutine free_ufield_derivatives
        implicit none
        deallocate(duxdx, duxdy, duxdz)
        deallocate(duydx, duydy, duydz)
        deallocate(duzdx, duzdy, duzdz)
    end subroutine free_ufield_derivatives

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
end module hydro_derivatives
