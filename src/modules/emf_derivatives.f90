!<******************************************************************************
!< This module include the methods to calculate the derivatives of electric and
!< magnetic fields.
!<******************************************************************************
module emf_derivatives
    use constants, only: fp, dp
    use picinfo, only: domain
    use pic_fields, only: bx, by, bz, absB
    implicit none
    private
    save

    public init_bfield_derivatives, free_bfield_derivatives, &
        calc_bfield_derivatives
    public init_absb_derivatives, free_absb_derivatives, calc_absb_derivatives
    public dbxdx, dbxdy, dbxdz, dbydx, dbydy, dbydz, dbzdx, dbzdy, dbzdz
    public dbdx, dbdy, dbdz

    real(fp), allocatable, dimension(:,:,:) :: dbxdx, dbxdy, dbxdz
    real(fp), allocatable, dimension(:,:,:) :: dbydx, dbydy, dbydz
    real(fp), allocatable, dimension(:,:,:) :: dbzdx, dbzdy, dbzdz
    real(fp), allocatable, dimension(:,:,:) :: dbdx, dbdy, dbdz

    contains

    !<--------------------------------------------------------------------------
    !< Initialize the derivatives of magnetic field
    !<--------------------------------------------------------------------------
    subroutine init_bfield_derivatives(nx, ny, nz)
        implicit none
        integer, intent(in) :: nx, ny, nz

        allocate(dbxdx(nx, ny, nz))
        allocate(dbxdy(nx, ny, nz))
        allocate(dbxdz(nx, ny, nz))
        allocate(dbydx(nx, ny, nz))
        allocate(dbydy(nx, ny, nz))
        allocate(dbydz(nx, ny, nz))
        allocate(dbzdx(nx, ny, nz))
        allocate(dbzdy(nx, ny, nz))
        allocate(dbzdz(nx, ny, nz))

        dbxdx = 0.0; dbxdy = 0.0; dbxdz = 0.0
        dbydx = 0.0; dbydy = 0.0; dbydz = 0.0
        dbzdx = 0.0; dbzdy = 0.0; dbzdz = 0.0
    end subroutine init_bfield_derivatives

    !<--------------------------------------------------------------------------
    !< Initialize the derivatives of the magnitude of the magnetic field
    !<--------------------------------------------------------------------------
    subroutine init_absb_derivatives(nx, ny, nz)
        implicit none
        integer, intent(in) :: nx, ny, nz

        allocate(dbdx(nx, ny, nz))
        allocate(dbdy(nx, ny, nz))
        allocate(dbdz(nx, ny, nz))

        dbdx = 0.0; dbdy = 0.0; dbdz = 0.0
    end subroutine init_absb_derivatives

    !<--------------------------------------------------------------------------
    !< Free the derivatives of magnetic field
    !<--------------------------------------------------------------------------
    subroutine free_bfield_derivatives
        implicit none
        deallocate(dbxdx, dbxdy, dbxdz)
        deallocate(dbydx, dbydy, dbydz)
        deallocate(dbzdx, dbzdy, dbzdz)
    end subroutine free_bfield_derivatives

    !<--------------------------------------------------------------------------
    !< Free the derivatives of the magnitude of the magnetic field
    !<--------------------------------------------------------------------------
    subroutine free_absb_derivatives
        implicit none
        deallocate(dbdx, dbdy, dbdz)
    end subroutine free_absb_derivatives

    !<--------------------------------------------------------------------------
    !< Calculate the derivatives of magnetic field
    !<--------------------------------------------------------------------------
    subroutine calc_bfield_derivatives(nx, ny, nz)
        use neighbors_module, only: ixl, iyl, izl, ixh, iyh, izh, idx, idy, idz
        implicit none
        integer, intent(in) :: nx, ny, nz
        integer :: ix, iy, iz
        do ix = 1, nx
            dbxdx(ix, :, :) = (bx(ixh(ix), :, :) - bx(ixl(ix), :, :)) * idx(ix)
            dbydx(ix, :, :) = (by(ixh(ix), :, :) - by(ixl(ix), :, :)) * idx(ix)
            dbzdx(ix, :, :) = (bz(ixh(ix), :, :) - bz(ixl(ix), :, :)) * idx(ix)
        enddo

        do iy = 1, ny
            dbxdy(:, iy, :) = (bx(:, iyh(iy), :) - bx(:, iyl(iy), :)) * idy(iy)
            dbydy(:, iy, :) = (by(:, iyh(iy), :) - by(:, iyl(iy), :)) * idy(iy)
            dbzdy(:, iy, :) = (bz(:, iyh(iy), :) - bz(:, iyl(iy), :)) * idy(iy)
        enddo

        do iz = 1, nz
            dbxdz(:, :, iz) = (bx(:, :, izh(iz)) - bx(:, :, izl(iz))) * idz(iz)
            dbydz(:, :, iz) = (by(:, :, izh(iz)) - by(:, :, izl(iz))) * idz(iz)
            dbzdz(:, :, iz) = (bz(:, :, izh(iz)) - bz(:, :, izl(iz))) * idz(iz)
        enddo
    end subroutine calc_bfield_derivatives

    !<--------------------------------------------------------------------------
    !< Calculate the derivatives of the magnitude of the magnetic field
    !<--------------------------------------------------------------------------
    subroutine calc_absb_derivatives(nx, ny, nz)
        use neighbors_module, only: ixl, iyl, izl, ixh, iyh, izh, idx, idy, idz
        implicit none
        integer, intent(in) :: nx, ny, nz
        integer :: ix, iy, iz
        do ix = 1, nx
            dbdx(ix, :, :) = (absB(ixh(ix), :, :) - absB(ixl(ix), :, :)) * idx(ix)
        enddo

        do iy = 1, ny
            dbdy(:, iy, :) = (absB(:, iyh(iy), :) - absB(:, iyl(iy), :)) * idy(iy)
        enddo

        do iz = 1, nz
            dbdz(:, :, iz) = (absB(:, :, izh(iz)) - absB(:, :, izl(iz))) * idz(iz)
        enddo
    end subroutine calc_absb_derivatives
end module emf_derivatives
