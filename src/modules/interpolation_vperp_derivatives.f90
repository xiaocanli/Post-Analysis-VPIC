!<******************************************************************************
!< Module of doing interpolation the gradients of the perpendicular components
!< of velocity field
!<******************************************************************************
module interpolation_vperp_derivatives
    use constants, only: fp
    use interpolation_funs, only: bounding_indcies
    implicit none
    private
    save

    public init_vperp_derivatives_single, free_vperp_derivatives_single, &
        set_vperp_derivatives, trilinear_interp_vperp_derivatives
    public dvperpx_dx0, dvperpx_dy0, dvperpx_dz0, dvperpy_dx0, dvperpy_dy0, &
        dvperpy_dz0, dvperpz_dx0, dvperpz_dy0, dvperpz_dz0
    real(fp), allocatable, dimension(:,:,:) :: dvperpx_dx, dvperpx_dy, dvperpx_dz
    real(fp), allocatable, dimension(:,:,:) :: dvperpy_dx, dvperpy_dy, dvperpy_dz
    real(fp), allocatable, dimension(:,:,:) :: dvperpz_dx, dvperpz_dy, dvperpz_dz
    real(fp) :: dvperpx_dx0, dvperpx_dy0, dvperpx_dz0
    real(fp) :: dvperpy_dx0, dvperpy_dy0, dvperpy_dz0
    real(fp) :: dvperpz_dx0, dvperpz_dy0, dvperpz_dz0
    real(fp), dimension(2,2,2) :: weights  ! The weights for trilinear interpolation
    integer :: nx, ny, nz

    contains

    !<--------------------------------------------------------------------------
    !< Initialize the derivatives of vperp field
    !<--------------------------------------------------------------------------
    subroutine init_vperp_derivatives_single
        use picinfo, only: domain
        implicit none

        nx = domain%pic_nx + 2  ! Including ghost cells
        ny = domain%pic_ny + 2
        nz = domain%pic_nz + 2

        allocate(dvperpx_dx(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(dvperpx_dy(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(dvperpx_dz(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(dvperpy_dx(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(dvperpy_dy(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(dvperpy_dz(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(dvperpz_dx(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(dvperpz_dy(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(dvperpz_dz(0:nx-1, 0:ny-1, 0:nz-1))

        dvperpx_dx = 0.0; dvperpx_dy = 0.0; dvperpx_dz = 0.0
        dvperpy_dx = 0.0; dvperpy_dy = 0.0; dvperpy_dz = 0.0
        dvperpz_dx = 0.0; dvperpz_dy = 0.0; dvperpz_dz = 0.0
    end subroutine init_vperp_derivatives_single

    !<--------------------------------------------------------------------------
    !< Free the derivatives of vperp field
    !<--------------------------------------------------------------------------
    subroutine free_vperp_derivatives_single
        implicit none
        deallocate(dvperpx_dx, dvperpx_dy, dvperpx_dz)
        deallocate(dvperpy_dx, dvperpy_dy, dvperpy_dz)
        deallocate(dvperpz_dx, dvperpz_dy, dvperpz_dz)
    end subroutine free_vperp_derivatives_single

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
    subroutine trilinear_interp_vperp_derivatives(ix0, iy0, iz0, dx, dy, dz)
        implicit none
        integer, intent(in) :: ix0, iy0, iz0
        real(fp), intent(in) :: dx, dy, dz
        call calc_interp_weights(dx, dy, dz)
        dvperpx_dx0 = sum(dvperpx_dx(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        dvperpx_dy0 = sum(dvperpx_dy(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        dvperpx_dz0 = sum(dvperpx_dz(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        dvperpy_dx0 = sum(dvperpy_dx(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        dvperpy_dy0 = sum(dvperpy_dy(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        dvperpy_dz0 = sum(dvperpy_dz(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        dvperpz_dx0 = sum(dvperpz_dx(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        dvperpz_dy0 = sum(dvperpz_dy(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        dvperpz_dz0 = sum(dvperpz_dz(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
    end subroutine trilinear_interp_vperp_derivatives

    !<--------------------------------------------------------------------------
    !< Set the derivatives of vperp field
    !<--------------------------------------------------------------------------
    subroutine set_vperp_derivatives(i, j, k, tx, ty, tz, sx, sy, sz)
        use vperp_derivatives, only: dvperpx_dxt => dvperpx_dx, &
            dvperpx_dyt => dvperpx_dy, dvperpx_dzt => dvperpx_dz, &
            dvperpy_dxt => dvperpy_dx, dvperpy_dyt => dvperpy_dy, &
            dvperpy_dzt => dvperpy_dz, dvperpz_dxt => dvperpz_dx, &
            dvperpz_dyt => dvperpz_dy, dvperpz_dzt => dvperpz_dz
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
        dvperpx_dx = 0.0; dvperpx_dy = 0.0; dvperpx_dz = 0.0
        dvperpy_dx = 0.0; dvperpy_dy = 0.0; dvperpy_dz = 0.0
        dvperpz_dx = 0.0; dvperpz_dy = 0.0; dvperpz_dz = 0.0
        dvperpx_dx(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            dvperpx_dxt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        dvperpx_dy(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            dvperpx_dyt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        dvperpx_dz(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            dvperpx_dzt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        dvperpy_dx(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            dvperpy_dxt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        dvperpy_dy(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            dvperpy_dyt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        dvperpy_dz(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            dvperpy_dzt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        dvperpz_dx(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            dvperpz_dxt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        dvperpz_dy(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            dvperpz_dyt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        dvperpz_dz(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            dvperpz_dzt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        if (ixs_lo == 1) then
            dvperpx_dx(0, :, :) = dvperpx_dx(1, :, :)
            dvperpx_dy(0, :, :) = dvperpx_dy(1, :, :)
            dvperpx_dz(0, :, :) = dvperpx_dz(1, :, :)
            dvperpy_dx(0, :, :) = dvperpy_dx(1, :, :)
            dvperpy_dy(0, :, :) = dvperpy_dy(1, :, :)
            dvperpy_dz(0, :, :) = dvperpy_dz(1, :, :)
            dvperpz_dx(0, :, :) = dvperpz_dx(1, :, :)
            dvperpz_dy(0, :, :) = dvperpz_dy(1, :, :)
            dvperpz_dz(0, :, :) = dvperpz_dz(1, :, :)
        endif
        if (ixe_lo == pnx) then
            dvperpx_dx(pnx+1, :, :) = dvperpx_dx(pnx, :, :)
            dvperpx_dy(pnx+1, :, :) = dvperpx_dy(pnx, :, :)
            dvperpx_dz(pnx+1, :, :) = dvperpx_dz(pnx, :, :)
            dvperpy_dx(pnx+1, :, :) = dvperpy_dx(pnx, :, :)
            dvperpy_dy(pnx+1, :, :) = dvperpy_dy(pnx, :, :)
            dvperpy_dz(pnx+1, :, :) = dvperpy_dz(pnx, :, :)
            dvperpz_dx(pnx+1, :, :) = dvperpz_dx(pnx, :, :)
            dvperpz_dy(pnx+1, :, :) = dvperpz_dy(pnx, :, :)
            dvperpz_dz(pnx+1, :, :) = dvperpz_dz(pnx, :, :)
        endif
        if (iys_lo == 1) then
            dvperpx_dx(:, 0, :) = dvperpx_dx(:, 1, :)
            dvperpx_dy(:, 0, :) = dvperpx_dy(:, 1, :)
            dvperpx_dz(:, 0, :) = dvperpx_dz(:, 1, :)
            dvperpy_dx(:, 0, :) = dvperpy_dx(:, 1, :)
            dvperpy_dy(:, 0, :) = dvperpy_dy(:, 1, :)
            dvperpy_dz(:, 0, :) = dvperpy_dz(:, 1, :)
            dvperpz_dx(:, 0, :) = dvperpz_dx(:, 1, :)
            dvperpz_dy(:, 0, :) = dvperpz_dy(:, 1, :)
            dvperpz_dz(:, 0, :) = dvperpz_dz(:, 1, :)
        endif
        if (iye_lo == pny) then
            dvperpx_dx(:, pny+1, :) = dvperpx_dx(:, pny, :)
            dvperpx_dy(:, pny+1, :) = dvperpx_dy(:, pny, :)
            dvperpx_dz(:, pny+1, :) = dvperpx_dz(:, pny, :)
            dvperpy_dx(:, pny+1, :) = dvperpy_dx(:, pny, :)
            dvperpy_dy(:, pny+1, :) = dvperpy_dy(:, pny, :)
            dvperpy_dz(:, pny+1, :) = dvperpy_dz(:, pny, :)
            dvperpz_dx(:, pny+1, :) = dvperpz_dx(:, pny, :)
            dvperpz_dy(:, pny+1, :) = dvperpz_dy(:, pny, :)
            dvperpz_dz(:, pny+1, :) = dvperpz_dz(:, pny, :)
        endif
        if (izs_lo == 1) then
            dvperpx_dx(:, :, 0) = dvperpx_dx(:, :, 1)
            dvperpx_dy(:, :, 0) = dvperpx_dy(:, :, 1)
            dvperpx_dz(:, :, 0) = dvperpx_dz(:, :, 1)
            dvperpy_dx(:, :, 0) = dvperpy_dx(:, :, 1)
            dvperpy_dy(:, :, 0) = dvperpy_dy(:, :, 1)
            dvperpy_dz(:, :, 0) = dvperpy_dz(:, :, 1)
            dvperpz_dx(:, :, 0) = dvperpz_dx(:, :, 1)
            dvperpz_dy(:, :, 0) = dvperpz_dy(:, :, 1)
            dvperpz_dz(:, :, 0) = dvperpz_dz(:, :, 1)
        endif
        if (ize_lo == pnz) then
            dvperpx_dx(:, :, pnz+1) = dvperpx_dx(:, :, pnz)
            dvperpx_dy(:, :, pnz+1) = dvperpx_dy(:, :, pnz)
            dvperpx_dz(:, :, pnz+1) = dvperpx_dz(:, :, pnz)
            dvperpy_dx(:, :, pnz+1) = dvperpy_dx(:, :, pnz)
            dvperpy_dy(:, :, pnz+1) = dvperpy_dy(:, :, pnz)
            dvperpy_dz(:, :, pnz+1) = dvperpy_dz(:, :, pnz)
            dvperpz_dx(:, :, pnz+1) = dvperpz_dx(:, :, pnz)
            dvperpz_dy(:, :, pnz+1) = dvperpz_dy(:, :, pnz)
            dvperpz_dz(:, :, pnz+1) = dvperpz_dz(:, :, pnz)
        endif
    end subroutine set_vperp_derivatives
end module interpolation_vperp_derivatives
