!<******************************************************************************
!< Module of doing interpolation of the magnetic field at previous and next time
!< steps
!*<*****************************************************************************
module interpolation_pre_post_bfield
    use constants, only: fp
    use interpolation_funs, only: bounding_indcies
    implicit none
    private
    public init_bfield_components, init_bfield_magnitude, init_pre_post_bfields, &
        free_bfield_components, free_bfield_magnitude, free_pre_post_bfields, &
        set_bfield_components, set_bfield_magnitude, set_pre_post_bfields, &
        trilinear_interp_bfield_components, trilinear_interp_bfield_magnitude, &
        trilinear_interp_pre_post_bfields
    public bx1_0, by1_0, bz1_0, absB1_0, bx2_0, by2_0, bz2_0, absB2_0

    real(fp), allocatable, dimension(:,:,:) :: bx1, by1, bz1, absB1
    real(fp), allocatable, dimension(:,:,:) :: bx2, by2, bz2, absB2
    real(fp) :: bx1_0, by1_0, bz1_0, absB1_0, bx2_0, by2_0, bz2_0, absB2_0
    real(fp), dimension(2,2,2) :: weights  ! The weights for trilinear interpolation
    integer :: nx, ny, nz

    contains

    !<--------------------------------------------------------------------------
    !< Initialize magnetic field components at previous and next time steps
    !<--------------------------------------------------------------------------
    subroutine init_bfield_components
        use picinfo, only: domain
        implicit none
        nx = domain%pic_nx + 2  ! Including ghost cells
        ny = domain%pic_ny + 2
        nz = domain%pic_nz + 2

        allocate(bx1(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(by1(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(bz1(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(bx2(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(by2(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(bz2(0:nx-1, 0:ny-1, 0:nz-1))
        bx1 = 0.0; by1 = 0.0; bz1 = 0.0
        bx2 = 0.0; by2 = 0.0; bz2 = 0.0
    end subroutine init_bfield_components

    !<--------------------------------------------------------------------------
    !< Initialize magnetic field magnitude at previous and next time steps
    !<--------------------------------------------------------------------------
    subroutine init_bfield_magnitude
        use picinfo, only: domain
        implicit none
        nx = domain%pic_nx + 2  ! Including ghost cells
        ny = domain%pic_ny + 2
        nz = domain%pic_nz + 2

        allocate(absB1(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(absB2(0:nx-1, 0:ny-1, 0:nz-1))
        absB1 = 0.0
        absB2 = 0.0
    end subroutine init_bfield_magnitude

    !<--------------------------------------------------------------------------
    !< Initialize magnetic field components and magnitude at previous and next
    !< time steps
    !<--------------------------------------------------------------------------
    subroutine init_pre_post_bfields
        implicit none
        call init_bfield_components
        call init_bfield_magnitude
    end subroutine init_pre_post_bfields

    !<--------------------------------------------------------------------------
    !< Free magnetic field components at previous and next time steps
    !<--------------------------------------------------------------------------
    subroutine free_bfield_components
        implicit none
        deallocate(bx1, by1, bz1)
        deallocate(bx2, by2, bz2)
    end subroutine free_bfield_components

    !<--------------------------------------------------------------------------
    !< Free magnetic field magnitude at previous and next time steps
    !<--------------------------------------------------------------------------
    subroutine free_bfield_magnitude
        implicit none
        deallocate(absB1, absB2)
    end subroutine free_bfield_magnitude

    !<--------------------------------------------------------------------------
    !< Free magnetic field components and magnitude at previous and next time steps
    !<--------------------------------------------------------------------------
    subroutine free_pre_post_bfields
        implicit none
        call free_bfield_components
        call free_bfield_magnitude
    end subroutine free_pre_post_bfields

    !<--------------------------------------------------------------------------
    !< Set magnetic field components, which are read from translated files rather
    !< than directly from the PIC simulations
    !<--------------------------------------------------------------------------
    subroutine set_bfield_components(i, j, k, tx, ty, tz, sx, sy, sz)
        use pre_post_emf, only: bxt1 => bx1, byt1 => by1, bzt1 => bz1, &
                                bxt2 => bx2, byt2 => by2, bzt2 => bz2
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
        bx1 = 0.0; by1 = 0.0; bz1 = 0.0
        bx2 = 0.0; by2 = 0.0; bz2 = 0.0
        bx1(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            bxt1(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        by1(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            byt1(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        bz1(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            bzt1(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        bx2(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            bxt2(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        by2(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            byt2(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        bz2(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            bzt2(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        if (ixs_lo == 1) then
            bx1(0, :, :) = bx1(1, :, :)
            by1(0, :, :) = by1(1, :, :)
            bz1(0, :, :) = bz1(1, :, :)
            bx2(0, :, :) = bx2(1, :, :)
            by2(0, :, :) = by2(1, :, :)
            bz2(0, :, :) = bz2(1, :, :)
        endif
        if (ixe_lo == pnx) then
            bx1(pnx+1, :, :) = bx1(pnx, :, :)
            by1(pnx+1, :, :) = by1(pnx, :, :)
            bz1(pnx+1, :, :) = bz1(pnx, :, :)
            bx2(pnx+1, :, :) = bx2(pnx, :, :)
            by2(pnx+1, :, :) = by2(pnx, :, :)
            bz2(pnx+1, :, :) = bz2(pnx, :, :)
        endif
        if (iys_lo == 1) then
            bx1(:, 0, :) = bx1(:, 1, :)
            by1(:, 0, :) = by1(:, 1, :)
            bz1(:, 0, :) = bz1(:, 1, :)
            bx2(:, 0, :) = bx2(:, 1, :)
            by2(:, 0, :) = by2(:, 1, :)
            bz2(:, 0, :) = bz2(:, 1, :)
        endif
        if (iye_lo == pny) then
            bx1(:, pny+1, :) = bx1(:, pny, :)
            by1(:, pny+1, :) = by1(:, pny, :)
            bz1(:, pny+1, :) = bz1(:, pny, :)
            bx2(:, pny+1, :) = bx2(:, pny, :)
            by2(:, pny+1, :) = by2(:, pny, :)
            bz2(:, pny+1, :) = bz2(:, pny, :)
        endif
        if (izs_lo == 1) then
            bx1(:, :, 0) = bx1(:, :, 1)
            by1(:, :, 0) = by1(:, :, 1)
            bz1(:, :, 0) = bz1(:, :, 1)
            bx2(:, :, 0) = bx2(:, :, 1)
            by2(:, :, 0) = by2(:, :, 1)
            bz2(:, :, 0) = bz2(:, :, 1)
        endif
        if (ize_lo == pnz) then
            bx1(:, :, pnz+1) = bx1(:, :, pnz)
            by1(:, :, pnz+1) = by1(:, :, pnz)
            bz1(:, :, pnz+1) = bz1(:, :, pnz)
            bx2(:, :, pnz+1) = bx2(:, :, pnz)
            by2(:, :, pnz+1) = by2(:, :, pnz)
            bz2(:, :, pnz+1) = bz2(:, :, pnz)
        endif
    end subroutine set_bfield_components

    !<--------------------------------------------------------------------------
    !< Set magnetic field magnitude, which are read from translated files rather
    !< than directly from the PIC simulations
    !<--------------------------------------------------------------------------
    subroutine set_bfield_magnitude(i, j, k, tx, ty, tz, sx, sy, sz)
        use pre_post_emf, only: absBt1 => absB1, absBt2 => absB2
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
        absB1 = 0.0; absB2 = 0.0
        absB1(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            absBt1(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        absB2(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            absBt2(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        if (ixs_lo == 1) then
            absB1(0, :, :) = absB1(1, :, :)
            absB2(0, :, :) = absB2(1, :, :)
        endif
        if (ixe_lo == pnx) then
            absB1(pnx+1, :, :) = absB1(pnx, :, :)
            absB2(pnx+1, :, :) = absB2(pnx, :, :)
        endif
        if (iys_lo == 1) then
            absB1(:, 0, :) = absB1(:, 1, :)
            absB2(:, 0, :) = absB2(:, 1, :)
        endif
        if (iye_lo == pny) then
            absB1(:, pny+1, :) = absB1(:, pny, :)
            absB2(:, pny+1, :) = absB2(:, pny, :)
        endif
        if (izs_lo == 1) then
            absB1(:, :, 0) = absB1(:, :, 1)
            absB2(:, :, 0) = absB2(:, :, 1)
        endif
        if (ize_lo == pnz) then
            absB1(:, :, pnz+1) = absB1(:, :, pnz)
            absB2(:, :, pnz+1) = absB2(:, :, pnz)
        endif
    end subroutine set_bfield_magnitude

    !<--------------------------------------------------------------------------
    !< Set magnetic field components and magnitude, which are read from translated
    !< files rather than directly from the PIC simulations
    !<--------------------------------------------------------------------------
    subroutine set_pre_post_bfields(i, j, k, tx, ty, tz, sx, sy, sz)
        implicit none
        integer, intent(in) :: i, j, k, tx, ty, tz, sx, sy, sz
        call set_bfield_components(i, j, k, tx, ty, tz, sx, sy, sz)
        call set_bfield_magnitude(i, j, k, tx, ty, tz, sx, sy, sz)
    end subroutine set_pre_post_bfields

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
    !< Trilinear interpolation for magnetic field components
    !< 
    !< Input:
    !<   ix0, iy0, iz0: the indices of the lower-left corner.
    !<   dx, dy, dz: the distance to the lower-left corner, [0, 1]
    !<--------------------------------------------------------------------------
    subroutine trilinear_interp_bfield_components(ix0, iy0, iz0, dx, dy, dz)
        implicit none
        integer, intent(in) :: ix0, iy0, iz0
        real(fp), intent(in) :: dx, dy, dz
        call calc_interp_weights(dx, dy, dz)
        bx1_0 = sum(bx1(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        by1_0 = sum(by1(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        bz1_0 = sum(bz1(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        bx2_0 = sum(bx2(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        by2_0 = sum(by2(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        bz2_0 = sum(bz2(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
    end subroutine trilinear_interp_bfield_components

    !<--------------------------------------------------------------------------
    !< Trilinear interpolation for magnetic field magnitude
    !< 
    !< Input:
    !<   ix0, iy0, iz0: the indices of the lower-left corner.
    !<   dx, dy, dz: the distance to the lower-left corner, [0, 1]
    !<--------------------------------------------------------------------------
    subroutine trilinear_interp_bfield_magnitude(ix0, iy0, iz0, dx, dy, dz)
        implicit none
        integer, intent(in) :: ix0, iy0, iz0
        real(fp), intent(in) :: dx, dy, dz
        call calc_interp_weights(dx, dy, dz)
        absB1_0 = sum(absB1(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        absB2_0 = sum(absB2(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
    end subroutine trilinear_interp_bfield_magnitude

    !<--------------------------------------------------------------------------
    !< Trilinear interpolation for magnetic field components and magnitude
    !< 
    !< Input:
    !<   ix0, iy0, iz0: the indices of the lower-left corner.
    !<   dx, dy, dz: the distance to the lower-left corner, [0, 1]
    !<--------------------------------------------------------------------------
    subroutine trilinear_interp_pre_post_bfields(ix0, iy0, iz0, dx, dy, dz)
        implicit none
        integer, intent(in) :: ix0, iy0, iz0
        real(fp), intent(in) :: dx, dy, dz
        call trilinear_interp_bfield_components(ix0, iy0, iz0, dx, dy, dz)
        call trilinear_interp_bfield_magnitude(ix0, iy0, iz0, dx, dy, dz)
    end subroutine trilinear_interp_pre_post_bfields

end module interpolation_pre_post_bfield
