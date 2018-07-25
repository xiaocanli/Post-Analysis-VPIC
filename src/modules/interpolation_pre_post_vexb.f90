!<******************************************************************************
!< Module of doing interpolation of the ExB drift at previous and next time
!< steps
!*<*****************************************************************************
module interpolation_pre_post_vexb
    use constants, only: fp
    use interpolation_funs, only: bounding_indcies
    implicit none
    private
    public init_vexb_components, free_vexb_components, &
        set_vexb_components, trilinear_interp_vexb_components
    public vexbx1_0, vexby1_0, vexbz1_0, vexbx2_0, vexby2_0, vexbz2_0

    real(fp), allocatable, dimension(:,:,:) :: vexbx1_s, vexby1_s, vexbz1_s
    real(fp), allocatable, dimension(:,:,:) :: vexbx2_s, vexby2_s, vexbz2_s
    real(fp) :: vexbx1_0, vexby1_0, vexbz1_0, vexbx2_0, vexby2_0, vexbz2_0
    real(fp), dimension(2,2,2) :: weights  ! The weights for trilinear interpolation
    integer :: nx, ny, nz

    contains

    !<--------------------------------------------------------------------------
    !< Initialize ExB drift components at previous and next time steps
    !<--------------------------------------------------------------------------
    subroutine init_vexb_components
        use picinfo, only: domain
        implicit none
        nx = domain%pic_nx + 2  ! Including ghost cells
        ny = domain%pic_ny + 2
        nz = domain%pic_nz + 2

        allocate(vexbx1_s(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(vexby1_s(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(vexbz1_s(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(vexbx2_s(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(vexby2_s(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(vexbz2_s(0:nx-1, 0:ny-1, 0:nz-1))
        vexbx1_s = 0.0; vexby1_s = 0.0; vexbz1_s = 0.0
        vexbx2_s = 0.0; vexby2_s = 0.0; vexbz2_s = 0.0
    end subroutine init_vexb_components

    !<--------------------------------------------------------------------------
    !< Free ExB drift components at previous and next time steps
    !<--------------------------------------------------------------------------
    subroutine free_vexb_components
        implicit none
        deallocate(vexbx1_s, vexby1_s, vexbz1_s)
        deallocate(vexbx2_s, vexby2_s, vexbz2_s)
    end subroutine free_vexb_components

    !<--------------------------------------------------------------------------
    !< Set ExB drift components, which are read from translated files rather
    !< than directly from the PIC simulations
    !<--------------------------------------------------------------------------
    subroutine set_vexb_components(i, j, k, tx, ty, tz, sx, sy, sz)
        use pre_post_vexb, only: vexbx1, vexby1, vexbz1, vexbx2, vexby2, vexbz2
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
        vexbx1_s = 0.0; vexby1_s = 0.0; vexbz1_s = 0.0
        vexbx2_s = 0.0; vexby2_s = 0.0; vexbz2_s = 0.0
        vexbx1_s(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            vexbx1(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        vexby1_s(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            vexby1(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        vexbz1_s(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            vexbz1(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        vexbx2_s(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            vexbx2(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        vexby2_s(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            vexby2(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        vexbz2_s(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            vexbz2(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        if (ixs_lo == 1) then
            vexbx1_s(0, :, :) = vexbx1_s(1, :, :)
            vexby1_s(0, :, :) = vexby1_s(1, :, :)
            vexbz1_s(0, :, :) = vexbz1_s(1, :, :)
            vexbx2_s(0, :, :) = vexbx2_s(1, :, :)
            vexby2_s(0, :, :) = vexby2_s(1, :, :)
            vexbz2_s(0, :, :) = vexbz2_s(1, :, :)
        endif
        if (ixe_lo == pnx) then
            vexbx1_s(pnx+1, :, :) = vexbx1_s(pnx, :, :)
            vexby1_s(pnx+1, :, :) = vexby1_s(pnx, :, :)
            vexbz1_s(pnx+1, :, :) = vexbz1_s(pnx, :, :)
            vexbx2_s(pnx+1, :, :) = vexbx2_s(pnx, :, :)
            vexby2_s(pnx+1, :, :) = vexby2_s(pnx, :, :)
            vexbz2_s(pnx+1, :, :) = vexbz2_s(pnx, :, :)
        endif
        if (iys_lo == 1) then
            vexbx1_s(:, 0, :) = vexbx1_s(:, 1, :)
            vexby1_s(:, 0, :) = vexby1_s(:, 1, :)
            vexbz1_s(:, 0, :) = vexbz1_s(:, 1, :)
            vexbx2_s(:, 0, :) = vexbx2_s(:, 1, :)
            vexby2_s(:, 0, :) = vexby2_s(:, 1, :)
            vexbz2_s(:, 0, :) = vexbz2_s(:, 1, :)
        endif
        if (iye_lo == pny) then
            vexbx1_s(:, pny+1, :) = vexbx1_s(:, pny, :)
            vexby1_s(:, pny+1, :) = vexby1_s(:, pny, :)
            vexbz1_s(:, pny+1, :) = vexbz1_s(:, pny, :)
            vexbx2_s(:, pny+1, :) = vexbx2_s(:, pny, :)
            vexby2_s(:, pny+1, :) = vexby2_s(:, pny, :)
            vexbz2_s(:, pny+1, :) = vexbz2_s(:, pny, :)
        endif
        if (izs_lo == 1) then
            vexbx1_s(:, :, 0) = vexbx1_s(:, :, 1)
            vexby1_s(:, :, 0) = vexby1_s(:, :, 1)
            vexbz1_s(:, :, 0) = vexbz1_s(:, :, 1)
            vexbx2_s(:, :, 0) = vexbx2_s(:, :, 1)
            vexby2_s(:, :, 0) = vexby2_s(:, :, 1)
            vexbz2_s(:, :, 0) = vexbz2_s(:, :, 1)
        endif
        if (ize_lo == pnz) then
            vexbx1_s(:, :, pnz+1) = vexbx1_s(:, :, pnz)
            vexby1_s(:, :, pnz+1) = vexby1_s(:, :, pnz)
            vexbz1_s(:, :, pnz+1) = vexbz1_s(:, :, pnz)
            vexbx2_s(:, :, pnz+1) = vexbx2_s(:, :, pnz)
            vexby2_s(:, :, pnz+1) = vexby2_s(:, :, pnz)
            vexbz2_s(:, :, pnz+1) = vexbz2_s(:, :, pnz)
        endif
    end subroutine set_vexb_components

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
    !< Trilinear interpolation for ExB drift components
    !<
    !< Input:
    !<   ix0, iy0, iz0: the indices of the lower-left corner.
    !<   dx, dy, dz: the distance to the lower-left corner, [0, 1]
    !<--------------------------------------------------------------------------
    subroutine trilinear_interp_vexb_components(ix0, iy0, iz0, dx, dy, dz)
        implicit none
        integer, intent(in) :: ix0, iy0, iz0
        real(fp), intent(in) :: dx, dy, dz
        call calc_interp_weights(dx, dy, dz)
        vexbx1_0 = sum(vexbx1_s(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        vexby1_0 = sum(vexby1_s(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        vexbz1_0 = sum(vexbz1_s(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        vexbx2_0 = sum(vexbx2_s(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        vexby2_0 = sum(vexby2_s(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        vexbz2_0 = sum(vexbz2_s(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
    end subroutine trilinear_interp_vexb_components
end module interpolation_pre_post_vexb
