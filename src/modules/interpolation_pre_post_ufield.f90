!<******************************************************************************
!< Module of doing interpolation of the momentum field at previous and next time
!< steps
!*<*****************************************************************************
module interpolation_pre_post_ufield
    use constants, only: fp
    use interpolation_funs, only: bounding_indcies
    implicit none
    private
    public init_ufield_components, free_ufield_components, &
        set_ufield_components, trilinear_interp_ufield_components
    public ux1_0, uy1_0, uz1_0, ux2_0, uy2_0, uz2_0

    real(fp), allocatable, dimension(:,:,:) :: ux1, uy1, uz1
    real(fp), allocatable, dimension(:,:,:) :: ux2, uy2, uz2
    real(fp) :: ux1_0, uy1_0, uz1_0, ux2_0, uy2_0, uz2_0
    real(fp), dimension(2,2,2) :: weights  ! The weights for trilinear interpolation
    integer :: nx, ny, nz

    contains

    !<--------------------------------------------------------------------------
    !< Initialize momentum field components at previous and next time steps
    !<--------------------------------------------------------------------------
    subroutine init_ufield_components
        use picinfo, only: domain
        implicit none
        nx = domain%pic_nx + 2  ! Including ghost cells
        ny = domain%pic_ny + 2
        nz = domain%pic_nz + 2

        allocate(ux1(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(uy1(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(uz1(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(ux2(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(uy2(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(uz2(0:nx-1, 0:ny-1, 0:nz-1))
        ux1 = 0.0; uy1 = 0.0; uz1 = 0.0
        ux2 = 0.0; uy2 = 0.0; uz2 = 0.0
    end subroutine init_ufield_components

    !<--------------------------------------------------------------------------
    !< Free momentum field components at previous and next time steps
    !<--------------------------------------------------------------------------
    subroutine free_ufield_components
        implicit none
        deallocate(ux1, uy1, uz1)
        deallocate(ux2, uy2, uz2)
    end subroutine free_ufield_components

    !<--------------------------------------------------------------------------
    !< Set momentum field components, which are read from translated files rather
    !< than directly from the PIC simulations
    !<--------------------------------------------------------------------------
    subroutine set_ufield_components(i, j, k, tx, ty, tz, sx, sy, sz)
        use pre_post_hydro, only: udx1, udy1, udz1, udx2, udy2, udz2
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
        ux1 = 0.0; uy1 = 0.0; uz1 = 0.0
        ux2 = 0.0; uy2 = 0.0; uz2 = 0.0
        ux1(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            udx1(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        uy1(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            udy1(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        uz1(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            udz1(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        ux2(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            udx2(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        uy2(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            udy2(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        uz2(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            udz2(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        if (ixs_lo == 1) then
            ux1(0, :, :) = ux1(1, :, :)
            uy1(0, :, :) = uy1(1, :, :)
            uz1(0, :, :) = uz1(1, :, :)
            ux2(0, :, :) = ux2(1, :, :)
            uy2(0, :, :) = uy2(1, :, :)
            uz2(0, :, :) = uz2(1, :, :)
        endif
        if (ixe_lo == pnx) then
            ux1(pnx+1, :, :) = ux1(pnx, :, :)
            uy1(pnx+1, :, :) = uy1(pnx, :, :)
            uz1(pnx+1, :, :) = uz1(pnx, :, :)
            ux2(pnx+1, :, :) = ux2(pnx, :, :)
            uy2(pnx+1, :, :) = uy2(pnx, :, :)
            uz2(pnx+1, :, :) = uz2(pnx, :, :)
        endif
        if (iys_lo == 1) then
            ux1(:, 0, :) = ux1(:, 1, :)
            uy1(:, 0, :) = uy1(:, 1, :)
            uz1(:, 0, :) = uz1(:, 1, :)
            ux2(:, 0, :) = ux2(:, 1, :)
            uy2(:, 0, :) = uy2(:, 1, :)
            uz2(:, 0, :) = uz2(:, 1, :)
        endif
        if (iye_lo == pny) then
            ux1(:, pny+1, :) = ux1(:, pny, :)
            uy1(:, pny+1, :) = uy1(:, pny, :)
            uz1(:, pny+1, :) = uz1(:, pny, :)
            ux2(:, pny+1, :) = ux2(:, pny, :)
            uy2(:, pny+1, :) = uy2(:, pny, :)
            uz2(:, pny+1, :) = uz2(:, pny, :)
        endif
        if (izs_lo == 1) then
            ux1(:, :, 0) = ux1(:, :, 1)
            uy1(:, :, 0) = uy1(:, :, 1)
            uz1(:, :, 0) = uz1(:, :, 1)
            ux2(:, :, 0) = ux2(:, :, 1)
            uy2(:, :, 0) = uy2(:, :, 1)
            uz2(:, :, 0) = uz2(:, :, 1)
        endif
        if (ize_lo == pnz) then
            ux1(:, :, pnz+1) = ux1(:, :, pnz)
            uy1(:, :, pnz+1) = uy1(:, :, pnz)
            uz1(:, :, pnz+1) = uz1(:, :, pnz)
            ux2(:, :, pnz+1) = ux2(:, :, pnz)
            uy2(:, :, pnz+1) = uy2(:, :, pnz)
            uz2(:, :, pnz+1) = uz2(:, :, pnz)
        endif
    end subroutine set_ufield_components

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
    !< Trilinear interpolation for momentum field components
    !< 
    !< Input:
    !<   ix0, iy0, iz0: the indices of the lower-left corner.
    !<   dx, dy, dz: the distance to the lower-left corner, [0, 1]
    !<--------------------------------------------------------------------------
    subroutine trilinear_interp_ufield_components(ix0, iy0, iz0, dx, dy, dz)
        implicit none
        integer, intent(in) :: ix0, iy0, iz0
        real(fp), intent(in) :: dx, dy, dz
        call calc_interp_weights(dx, dy, dz)
        ux1_0 = sum(ux1(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        uy1_0 = sum(uy1(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        uz1_0 = sum(uz1(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        ux2_0 = sum(ux2(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        uy2_0 = sum(uy2(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        uz2_0 = sum(uz2(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
    end subroutine trilinear_interp_ufield_components
end module interpolation_pre_post_ufield
