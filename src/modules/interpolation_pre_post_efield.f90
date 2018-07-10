!<******************************************************************************
!< Module of doing interpolation of the electric field at previous and next time
!< steps
!*<*****************************************************************************
module interpolation_pre_post_efield
    use constants, only: fp
    use interpolation_funs, only: bounding_indcies
    implicit none
    private
    public init_efield_components, init_efield_magnitude, init_pre_post_efields, &
        free_efield_components, free_efield_magnitude, free_pre_post_efields, &
        set_efield_components, set_efield_magnitude, set_pre_post_efields, &
        trilinear_interp_efield_components, trilinear_interp_efield_magnitude, &
        trilinear_interp_pre_post_efields
    public ex1_0, ey1_0, ez1_0, absE1_0, ex2_0, ey2_0, ez2_0, absE2_0

    real(fp), allocatable, dimension(:,:,:) :: ex1, ey1, ez1, absE1
    real(fp), allocatable, dimension(:,:,:) :: ex2, ey2, ez2, absE2
    real(fp) :: ex1_0, ey1_0, ez1_0, absE1_0, ex2_0, ey2_0, ez2_0, absE2_0
    real(fp), dimension(2,2,2) :: weights  ! The weights for trilinear interpolation
    integer :: nx, ny, nz

    contains

    !<--------------------------------------------------------------------------
    !< Initialize electric field components at previous and next time steps
    !<--------------------------------------------------------------------------
    subroutine init_efield_components
        use picinfo, only: domain
        implicit none
        nx = domain%pic_nx + 2  ! Including ghost cells
        ny = domain%pic_ny + 2
        nz = domain%pic_nz + 2

        allocate(ex1(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(ey1(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(ez1(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(ex2(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(ey2(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(ez2(0:nx-1, 0:ny-1, 0:nz-1))
        ex1 = 0.0; ey1 = 0.0; ez1 = 0.0
        ex2 = 0.0; ey2 = 0.0; ez2 = 0.0
    end subroutine init_efield_components

    !<--------------------------------------------------------------------------
    !< Initialize electric field magnitude at previous and next time steps
    !<--------------------------------------------------------------------------
    subroutine init_efield_magnitude
        use picinfo, only: domain
        implicit none
        nx = domain%pic_nx + 2  ! Including ghost cells
        ny = domain%pic_ny + 2
        nz = domain%pic_nz + 2

        allocate(absE1(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(absE2(0:nx-1, 0:ny-1, 0:nz-1))
        absE1 = 0.0
        absE2 = 0.0
    end subroutine init_efield_magnitude

    !<--------------------------------------------------------------------------
    !< Initialize electric field components and magnitude at previous and next
    !< time steps
    !<--------------------------------------------------------------------------
    subroutine init_pre_post_efields
        implicit none
        call init_efield_components
        call init_efield_magnitude
    end subroutine init_pre_post_efields

    !<--------------------------------------------------------------------------
    !< Free electric field components at previous and next time steps
    !<--------------------------------------------------------------------------
    subroutine free_efield_components
        implicit none
        deallocate(ex1, ey1, ez1)
        deallocate(ex2, ey2, ez2)
    end subroutine free_efield_components

    !<--------------------------------------------------------------------------
    !< Free electric field magnitude at previous and next time steps
    !<--------------------------------------------------------------------------
    subroutine free_efield_magnitude
        implicit none
        deallocate(absE1, absE2)
    end subroutine free_efield_magnitude

    !<--------------------------------------------------------------------------
    !< Free electric field components and magnitude at previous and next time steps
    !<--------------------------------------------------------------------------
    subroutine free_pre_post_efields
        implicit none
        call free_efield_components
        call free_efield_magnitude
    end subroutine free_pre_post_efields

    !<--------------------------------------------------------------------------
    !< Set electric field components, which are read from translated files rather
    !< than directly from the PIC simulations
    !<--------------------------------------------------------------------------
    subroutine set_efield_components(i, j, k, tx, ty, tz, sx, sy, sz)
        use pre_post_emf, only: ext1 => ex1, eyt1 => ey1, ezt1 => ez1, &
                                ext2 => ex2, eyt2 => ey2, ezt2 => ez2
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
        ex1 = 0.0; ey1 = 0.0; ez1 = 0.0
        ex2 = 0.0; ey2 = 0.0; ez2 = 0.0
        ex1(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            ext1(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        ey1(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            eyt1(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        ez1(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            ezt1(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        ex2(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            ext2(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        ey2(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            eyt2(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        ez2(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            ezt2(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        if (ixs_lo == 1) then
            ex1(0, :, :) = ex1(1, :, :)
            ey1(0, :, :) = ey1(1, :, :)
            ez1(0, :, :) = ez1(1, :, :)
            ex2(0, :, :) = ex2(1, :, :)
            ey2(0, :, :) = ey2(1, :, :)
            ez2(0, :, :) = ez2(1, :, :)
        endif
        if (ixe_lo == pnx) then
            ex1(pnx+1, :, :) = ex1(pnx, :, :)
            ey1(pnx+1, :, :) = ey1(pnx, :, :)
            ez1(pnx+1, :, :) = ez1(pnx, :, :)
            ex2(pnx+1, :, :) = ex2(pnx, :, :)
            ey2(pnx+1, :, :) = ey2(pnx, :, :)
            ez2(pnx+1, :, :) = ez2(pnx, :, :)
        endif
        if (iys_lo == 1) then
            ex1(:, 0, :) = ex1(:, 1, :)
            ey1(:, 0, :) = ey1(:, 1, :)
            ez1(:, 0, :) = ez1(:, 1, :)
            ex2(:, 0, :) = ex2(:, 1, :)
            ey2(:, 0, :) = ey2(:, 1, :)
            ez2(:, 0, :) = ez2(:, 1, :)
        endif
        if (iye_lo == pny) then
            ex1(:, pny+1, :) = ex1(:, pny, :)
            ey1(:, pny+1, :) = ey1(:, pny, :)
            ez1(:, pny+1, :) = ez1(:, pny, :)
            ex2(:, pny+1, :) = ex2(:, pny, :)
            ey2(:, pny+1, :) = ey2(:, pny, :)
            ez2(:, pny+1, :) = ez2(:, pny, :)
        endif
        if (izs_lo == 1) then
            ex1(:, :, 0) = ex1(:, :, 1)
            ey1(:, :, 0) = ey1(:, :, 1)
            ez1(:, :, 0) = ez1(:, :, 1)
            ex2(:, :, 0) = ex2(:, :, 1)
            ey2(:, :, 0) = ey2(:, :, 1)
            ez2(:, :, 0) = ez2(:, :, 1)
        endif
        if (ize_lo == pnz) then
            ex1(:, :, pnz+1) = ex1(:, :, pnz)
            ey1(:, :, pnz+1) = ey1(:, :, pnz)
            ez1(:, :, pnz+1) = ez1(:, :, pnz)
            ex2(:, :, pnz+1) = ex2(:, :, pnz)
            ey2(:, :, pnz+1) = ey2(:, :, pnz)
            ez2(:, :, pnz+1) = ez2(:, :, pnz)
        endif
    end subroutine set_efield_components

    !<--------------------------------------------------------------------------
    !< Set electric field magnitude, which are read from translated files rather
    !< than directly from the PIC simulations
    !<--------------------------------------------------------------------------
    subroutine set_efield_magnitude(i, j, k, tx, ty, tz, sx, sy, sz)
        use pre_post_emf, only: absEt1 => absE1, absEt2 => absE2
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
        absE1 = 0.0; absE2 = 0.0
        absE1(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            absEt1(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        absE2(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            absEt2(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        if (ixs_lo == 1) then
            absE1(0, :, :) = absE1(1, :, :)
            absE2(0, :, :) = absE2(1, :, :)
        endif
        if (ixe_lo == pnx) then
            absE1(pnx+1, :, :) = absE1(pnx, :, :)
            absE2(pnx+1, :, :) = absE2(pnx, :, :)
        endif
        if (iys_lo == 1) then
            absE1(:, 0, :) = absE1(:, 1, :)
            absE2(:, 0, :) = absE2(:, 1, :)
        endif
        if (iye_lo == pny) then
            absE1(:, pny+1, :) = absE1(:, pny, :)
            absE2(:, pny+1, :) = absE2(:, pny, :)
        endif
        if (izs_lo == 1) then
            absE1(:, :, 0) = absE1(:, :, 1)
            absE2(:, :, 0) = absE2(:, :, 1)
        endif
        if (ize_lo == pnz) then
            absE1(:, :, pnz+1) = absE1(:, :, pnz)
            absE2(:, :, pnz+1) = absE2(:, :, pnz)
        endif
    end subroutine set_efield_magnitude

    !<--------------------------------------------------------------------------
    !< Set electric field components and magnitude, which are read from translated
    !< files rather than directly from the PIC simulations
    !<--------------------------------------------------------------------------
    subroutine set_pre_post_efields(i, j, k, tx, ty, tz, sx, sy, sz)
        implicit none
        integer, intent(in) :: i, j, k, tx, ty, tz, sx, sy, sz
        call set_efield_components(i, j, k, tx, ty, tz, sx, sy, sz)
        call set_efield_magnitude(i, j, k, tx, ty, tz, sx, sy, sz)
    end subroutine set_pre_post_efields

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
    !< Trilinear interpolation for electric field components
    !< 
    !< Input:
    !<   ix0, iy0, iz0: the indices of the lower-left corner.
    !<   dx, dy, dz: the distance to the lower-left corner, [0, 1]
    !<--------------------------------------------------------------------------
    subroutine trilinear_interp_efield_components(ix0, iy0, iz0, dx, dy, dz)
        implicit none
        integer, intent(in) :: ix0, iy0, iz0
        real(fp), intent(in) :: dx, dy, dz
        call calc_interp_weights(dx, dy, dz)
        ex1_0 = sum(ex1(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        ey1_0 = sum(ey1(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        ez1_0 = sum(ez1(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        ex2_0 = sum(ex2(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        ey2_0 = sum(ey2(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        ez2_0 = sum(ez2(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
    end subroutine trilinear_interp_efield_components

    !<--------------------------------------------------------------------------
    !< Trilinear interpolation for electric field magnitude
    !< 
    !< Input:
    !<   ix0, iy0, iz0: the indices of the lower-left corner.
    !<   dx, dy, dz: the distance to the lower-left corner, [0, 1]
    !<--------------------------------------------------------------------------
    subroutine trilinear_interp_efield_magnitude(ix0, iy0, iz0, dx, dy, dz)
        implicit none
        integer, intent(in) :: ix0, iy0, iz0
        real(fp), intent(in) :: dx, dy, dz
        call calc_interp_weights(dx, dy, dz)
        absE1_0 = sum(absE1(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        absE2_0 = sum(absE2(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
    end subroutine trilinear_interp_efield_magnitude

    !<--------------------------------------------------------------------------
    !< Trilinear interpolation for electric field components and magnitude
    !< 
    !< Input:
    !<   ix0, iy0, iz0: the indices of the lower-left corner.
    !<   dx, dy, dz: the distance to the lower-left corner, [0, 1]
    !<--------------------------------------------------------------------------
    subroutine trilinear_interp_pre_post_efields(ix0, iy0, iz0, dx, dy, dz)
        implicit none
        integer, intent(in) :: ix0, iy0, iz0
        real(fp), intent(in) :: dx, dy, dz
        call trilinear_interp_efield_components(ix0, iy0, iz0, dx, dy, dz)
        call trilinear_interp_efield_magnitude(ix0, iy0, iz0, dx, dy, dz)
    end subroutine trilinear_interp_pre_post_efields

end module interpolation_pre_post_efield
