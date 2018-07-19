!<******************************************************************************
!< Module of doing interpolation of the momentum field at previous and next time
!< steps
!*<*****************************************************************************
module interpolation_pre_post_vfield
    use constants, only: fp
    use interpolation_funs, only: bounding_indcies
    implicit none
    private
    public init_vfield_components, free_vfield_components, &
        set_vfield_components, trilinear_interp_vfield_components
    public vx1_0, vy1_0, vz1_0, vx2_0, vy2_0, vz2_0

    real(fp), allocatable, dimension(:,:,:) :: vx1, vy1, vz1
    real(fp), allocatable, dimension(:,:,:) :: vx2, vy2, vz2
    real(fp) :: vx1_0, vy1_0, vz1_0, vx2_0, vy2_0, vz2_0
    real(fp), dimension(2,2,2) :: weights  ! The weights for trilinear interpolation
    integer :: nx, ny, nz

    contains

    !<--------------------------------------------------------------------------
    !< Initialize momentum field components at previous and next time steps
    !<--------------------------------------------------------------------------
    subroutine init_vfield_components
        use picinfo, only: domain
        implicit none
        nx = domain%pic_nx + 2  ! Including ghost cells
        ny = domain%pic_ny + 2
        nz = domain%pic_nz + 2

        allocate(vx1(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(vy1(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(vz1(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(vx2(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(vy2(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(vz2(0:nx-1, 0:ny-1, 0:nz-1))
        vx1 = 0.0; vy1 = 0.0; vz1 = 0.0
        vx2 = 0.0; vy2 = 0.0; vz2 = 0.0
    end subroutine init_vfield_components

    !<--------------------------------------------------------------------------
    !< Free momentum field components at previous and next time steps
    !<--------------------------------------------------------------------------
    subroutine free_vfield_components
        implicit none
        deallocate(vx1, vy1, vz1)
        deallocate(vx2, vy2, vz2)
    end subroutine free_vfield_components

    !<--------------------------------------------------------------------------
    !< Set momentum field components, which are read from translated files rather
    !< than directly from the PIC simulations
    !<--------------------------------------------------------------------------
    subroutine set_vfield_components(i, j, k, tx, ty, tz, sx, sy, sz)
        use pre_post_hydro, only: vdx1, vdy1, vdz1, vdx2, vdy2, vdz2
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
        vx1 = 0.0; vy1 = 0.0; vz1 = 0.0
        vx2 = 0.0; vy2 = 0.0; vz2 = 0.0
        vx1(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            vdx1(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        vy1(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            vdy1(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        vz1(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            vdz1(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        vx2(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            vdx2(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        vy2(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            vdy2(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        vz2(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            vdz2(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        if (ixs_lo == 1) then
            vx1(0, :, :) = vx1(1, :, :)
            vy1(0, :, :) = vy1(1, :, :)
            vz1(0, :, :) = vz1(1, :, :)
            vx2(0, :, :) = vx2(1, :, :)
            vy2(0, :, :) = vy2(1, :, :)
            vz2(0, :, :) = vz2(1, :, :)
        endif
        if (ixe_lo == pnx) then
            vx1(pnx+1, :, :) = vx1(pnx, :, :)
            vy1(pnx+1, :, :) = vy1(pnx, :, :)
            vz1(pnx+1, :, :) = vz1(pnx, :, :)
            vx2(pnx+1, :, :) = vx2(pnx, :, :)
            vy2(pnx+1, :, :) = vy2(pnx, :, :)
            vz2(pnx+1, :, :) = vz2(pnx, :, :)
        endif
        if (iys_lo == 1) then
            vx1(:, 0, :) = vx1(:, 1, :)
            vy1(:, 0, :) = vy1(:, 1, :)
            vz1(:, 0, :) = vz1(:, 1, :)
            vx2(:, 0, :) = vx2(:, 1, :)
            vy2(:, 0, :) = vy2(:, 1, :)
            vz2(:, 0, :) = vz2(:, 1, :)
        endif
        if (iye_lo == pny) then
            vx1(:, pny+1, :) = vx1(:, pny, :)
            vy1(:, pny+1, :) = vy1(:, pny, :)
            vz1(:, pny+1, :) = vz1(:, pny, :)
            vx2(:, pny+1, :) = vx2(:, pny, :)
            vy2(:, pny+1, :) = vy2(:, pny, :)
            vz2(:, pny+1, :) = vz2(:, pny, :)
        endif
        if (izs_lo == 1) then
            vx1(:, :, 0) = vx1(:, :, 1)
            vy1(:, :, 0) = vy1(:, :, 1)
            vz1(:, :, 0) = vz1(:, :, 1)
            vx2(:, :, 0) = vx2(:, :, 1)
            vy2(:, :, 0) = vy2(:, :, 1)
            vz2(:, :, 0) = vz2(:, :, 1)
        endif
        if (ize_lo == pnz) then
            vx1(:, :, pnz+1) = vx1(:, :, pnz)
            vy1(:, :, pnz+1) = vy1(:, :, pnz)
            vz1(:, :, pnz+1) = vz1(:, :, pnz)
            vx2(:, :, pnz+1) = vx2(:, :, pnz)
            vy2(:, :, pnz+1) = vy2(:, :, pnz)
            vz2(:, :, pnz+1) = vz2(:, :, pnz)
        endif
    end subroutine set_vfield_components

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
    subroutine trilinear_interp_vfield_components(ix0, iy0, iz0, dx, dy, dz)
        implicit none
        integer, intent(in) :: ix0, iy0, iz0
        real(fp), intent(in) :: dx, dy, dz
        call calc_interp_weights(dx, dy, dz)
        vx1_0 = sum(vx1(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        vy1_0 = sum(vy1(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        vz1_0 = sum(vz1(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        vx2_0 = sum(vx2(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        vy2_0 = sum(vy2(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        vz2_0 = sum(vz2(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
    end subroutine trilinear_interp_vfield_components
end module interpolation_pre_post_vfield
