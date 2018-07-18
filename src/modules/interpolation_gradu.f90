!<******************************************************************************
!< Module of doing interpolation the gradients of momentum field
!<******************************************************************************
module interpolation_ufield_derivatives
    use constants, only: fp
    use interpolation_funs, only: bounding_indcies
    implicit none
    private
    save

    public init_ufields_derivatives_single, free_ufields_derivatives_single, &
           set_ufield_derivatives, trilinear_interp_ufield_derivatives
    public duxdx0, duxdy0, duxdz0, duydx0, duydy0, duydz0, duzdx0, duzdy0, duzdz0
    real(fp), allocatable, dimension(:,:,:) :: duxdx, duxdy, duxdz
    real(fp), allocatable, dimension(:,:,:) :: duydx, duydy, duydz
    real(fp), allocatable, dimension(:,:,:) :: duzdx, duzdy, duzdz
    real(fp) :: duxdx0, duxdy0, duxdz0
    real(fp) :: duydx0, duydy0, duydz0
    real(fp) :: duzdx0, duzdy0, duzdz0
    real(fp), dimension(2,2,2) :: weights  ! The weights for trilinear interpolation
    integer :: nx, ny, nz

    contains

    !<--------------------------------------------------------------------------
    !< Initialize the derivatives of the momentum field
    !<--------------------------------------------------------------------------
    subroutine init_ufields_derivatives_single
        use picinfo, only: domain
        implicit none

        nx = domain%pic_nx + 2  ! Including ghost cells
        ny = domain%pic_ny + 2
        nz = domain%pic_nz + 2

        allocate(duxdx(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(duxdy(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(duxdz(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(duydx(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(duydy(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(duydz(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(duzdx(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(duzdy(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(duzdz(0:nx-1, 0:ny-1, 0:nz-1))

        duxdx = 0.0; duxdy = 0.0; duxdz = 0.0
        duydx = 0.0; duydy = 0.0; duydz = 0.0
        duzdx = 0.0; duzdy = 0.0; duzdz = 0.0
    end subroutine init_ufields_derivatives_single

    !<--------------------------------------------------------------------------
    !< Free the derivatives of the momentum field
    !<--------------------------------------------------------------------------
    subroutine free_ufields_derivatives_single
        implicit none
        deallocate(duxdx, duxdy, duxdz)
        deallocate(duydx, duydy, duydz)
        deallocate(duzdx, duzdy, duzdz)
    end subroutine free_ufields_derivatives_single

    !<--------------------------------------------------------------------------
    !< Calculate the weights for trilinear interpolation.
    !<
    !< Input:
    !<   dx, dy, dz: the distance to the lower-left corner, [0, 1]
    !<--------------------------------------------------------------------------
    subroutine calc_interp_weights(dx, dy, dz)
        implicit none
        real(fp), intent(in) :: dx, dy, dz
        weights(1, 1, 1) = (1 - dx) * (1 - dy) * (1 - dz)
        weights(2, 1, 1) = dx * (1 - dy) * (1 - dz)
        weights(1, 2, 1) = (1 - dx) * dy * (1 - dz)
        weights(2, 2, 1) = dx * dy * (1 - dz)
        weights(1, 1, 2) = (1 - dx) * (1 - dy) * dz
        weights(2, 1, 2) = dx * (1 - dy) * dz
        weights(1, 2, 2) = (1 - dx) * dy * dz
        weights(2, 2, 2) = dx * dy * dz
    end subroutine calc_interp_weights

    !<--------------------------------------------------------------------------
    !< Trilinear interpolation
    !< Input:
    !<   ix0, iy0, iz0: the indices of the lower-left corner.
    !<   dx, dy, dz: the distance to the lower-left corner, [0, 1]
    !<--------------------------------------------------------------------------
    subroutine trilinear_interp_ufield_derivatives(ix0, iy0, iz0, dx, dy, dz)
        implicit none
        integer, intent(in) :: ix0, iy0, iz0
        real(fp), intent(in) :: dx, dy, dz
        call calc_interp_weights(dx, dy, dz)
        duxdx0 = sum(duxdx(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        duxdy0 = sum(duxdy(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        duxdz0 = sum(duxdz(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        duydx0 = sum(duydx(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        duydy0 = sum(duydy(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        duydz0 = sum(duydz(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        duzdx0 = sum(duzdx(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        duzdy0 = sum(duzdy(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        duzdz0 = sum(duzdz(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
    end subroutine trilinear_interp_ufield_derivatives

    !<--------------------------------------------------------------------------
    !< Set the derivatives of the momentum field, which is read from
    !< translated files rather than directly from the PIC simulations
    !<--------------------------------------------------------------------------
    subroutine set_ufield_derivatives(i, j, k, tx, ty, tz, sx, sy, sz)
        use hydro_derivatives, only: duxdxt => duxdx, duxdyt => duxdy, &
            duxdzt => duxdz, duydxt => duydx, duydyt => duydy, duydzt => duydz, &
            duzdxt => duzdx, duzdyt => duzdy, duzdzt => duzdz
        use picinfo, only: domain
        implicit none
        integer, intent(in) :: i, j, k, tx, ty, tz, sx, sy, sz
        integer :: ixs_lo, ixe_lo, ixs_gl, ixe_gl
        integer :: iys_lo, iye_lo, iys_gl, iye_gl
        integer :: izs_lo, ize_lo, izs_gl, ize_gl
        integer :: pnx, pny, pnz
        pnx = domain%pic_nx
        pny = domain%pic_ny
        pnz = domain%pic_nz
        call bounding_indcies(i, pnx, tx, sx, ixs_lo, ixe_lo, ixs_gl, ixe_gl)
        call bounding_indcies(j, pny, ty, sy, iys_lo, iye_lo, iys_gl, iye_gl)
        call bounding_indcies(k, pnz, tz, sz, izs_lo, ize_lo, izs_gl, ize_gl)
        duxdx = 0.0; duxdy = 0.0; duxdz = 0.0
        duydx = 0.0; duydy = 0.0; duydz = 0.0
        duzdx = 0.0; duzdy = 0.0; duzdz = 0.0
        duxdx(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            duxdxt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        duxdy(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            duxdyt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        duxdz(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            duxdzt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        duydx(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            duydxt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        duydy(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            duydyt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        duydz(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            duydzt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        duzdx(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            duzdxt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        duzdy(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            duzdyt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        duzdz(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            duzdzt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        if (ixs_lo == 1) then
            duxdx(0, :, :) = duxdx(1, :, :)
            duxdy(0, :, :) = duxdy(1, :, :)
            duxdz(0, :, :) = duxdz(1, :, :)
            duydx(0, :, :) = duydx(1, :, :)
            duydy(0, :, :) = duydy(1, :, :)
            duydz(0, :, :) = duydz(1, :, :)
            duzdx(0, :, :) = duzdx(1, :, :)
            duzdy(0, :, :) = duzdy(1, :, :)
            duzdz(0, :, :) = duzdz(1, :, :)
        endif
        if (ixe_lo == pnx) then
            duxdx(pnx+1, :, :) = duxdx(pnx, :, :)
            duxdy(pnx+1, :, :) = duxdy(pnx, :, :)
            duxdz(pnx+1, :, :) = duxdz(pnx, :, :)
            duydx(pnx+1, :, :) = duydx(pnx, :, :)
            duydy(pnx+1, :, :) = duydy(pnx, :, :)
            duydz(pnx+1, :, :) = duydz(pnx, :, :)
            duzdx(pnx+1, :, :) = duzdx(pnx, :, :)
            duzdy(pnx+1, :, :) = duzdy(pnx, :, :)
            duzdz(pnx+1, :, :) = duzdz(pnx, :, :)
        endif
        if (iys_lo == 1) then
            duxdx(:, 0, :) = duxdx(:, 1, :)
            duxdy(:, 0, :) = duxdy(:, 1, :)
            duxdz(:, 0, :) = duxdz(:, 1, :)
            duydx(:, 0, :) = duydx(:, 1, :)
            duydy(:, 0, :) = duydy(:, 1, :)
            duydz(:, 0, :) = duydz(:, 1, :)
            duzdx(:, 0, :) = duzdx(:, 1, :)
            duzdy(:, 0, :) = duzdy(:, 1, :)
            duzdz(:, 0, :) = duzdz(:, 1, :)
        endif
        if (iye_lo == pny) then
            duxdx(:, pny+1, :) = duxdx(:, pny, :)
            duxdy(:, pny+1, :) = duxdy(:, pny, :)
            duxdz(:, pny+1, :) = duxdz(:, pny, :)
            duydx(:, pny+1, :) = duydx(:, pny, :)
            duydy(:, pny+1, :) = duydy(:, pny, :)
            duydz(:, pny+1, :) = duydz(:, pny, :)
            duzdx(:, pny+1, :) = duzdx(:, pny, :)
            duzdy(:, pny+1, :) = duzdy(:, pny, :)
            duzdz(:, pny+1, :) = duzdz(:, pny, :)
        endif
        if (izs_lo == 1) then
            duxdx(:, :, 0) = duxdx(:, :, 1)
            duxdy(:, :, 0) = duxdy(:, :, 1)
            duxdz(:, :, 0) = duxdz(:, :, 1)
            duydx(:, :, 0) = duydx(:, :, 1)
            duydy(:, :, 0) = duydy(:, :, 1)
            duydz(:, :, 0) = duydz(:, :, 1)
            duzdx(:, :, 0) = duzdx(:, :, 1)
            duzdy(:, :, 0) = duzdy(:, :, 1)
            duzdz(:, :, 0) = duzdz(:, :, 1)
        endif
        if (ize_lo == pnz) then
            duxdx(:, :, pnz+1) = duxdx(:, :, pnz)
            duxdy(:, :, pnz+1) = duxdy(:, :, pnz)
            duxdz(:, :, pnz+1) = duxdz(:, :, pnz)
            duydx(:, :, pnz+1) = duydx(:, :, pnz)
            duydy(:, :, pnz+1) = duydy(:, :, pnz)
            duydz(:, :, pnz+1) = duydz(:, :, pnz)
            duzdx(:, :, pnz+1) = duzdx(:, :, pnz)
            duzdy(:, :, pnz+1) = duzdy(:, :, pnz)
            duzdz(:, :, pnz+1) = duzdz(:, :, pnz)
        endif
    end subroutine set_ufield_derivatives
end module interpolation_ufield_derivatives
