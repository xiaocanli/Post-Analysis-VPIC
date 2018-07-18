!*******************************************************************************
! Module of doing interpolation in current density fields
!*******************************************************************************
module interpolation_current_density
    use constants, only: fp
    use interpolation_funs, only: bounding_indcies
    implicit none
    private
    public init_current_density_single, free_current_density_single, &
        set_current_density, trilinear_interp_current_density
    public jx0, jy0, jz0

    real(fp), allocatable, dimension(:,:,:) :: jx, jy, jz
    real(fp) :: jx0, jy0, jz0
    real(fp), dimension(2,2,2) :: weights  ! The weights for trilinear interpolation
    integer :: nx, ny, nz

    contains

    !---------------------------------------------------------------------------
    ! Initialize current density fields
    !---------------------------------------------------------------------------
    subroutine init_current_density_single
        use picinfo, only: domain
        implicit none
        nx = domain%pic_nx + 2  ! Including ghost cells
        ny = domain%pic_ny + 2
        nz = domain%pic_nz + 2

        allocate(jx(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(jy(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(jz(0:nx-1, 0:ny-1, 0:nz-1))
        jx = 0.0; jy = 0.0; jz = 0.0
    end subroutine init_current_density_single

    !---------------------------------------------------------------------------
    ! Free current density fields
    !---------------------------------------------------------------------------
    subroutine free_current_density_single
        implicit none
        deallocate(jx, jy, jz)
    end subroutine free_current_density_single

    !<--------------------------------------------------------------------------
    !< Set current density fields, which are read from translated files
    !< rather than directly from the PIC simulations
    !<--------------------------------------------------------------------------
    subroutine set_current_density(i, j, k, tx, ty, tz, sx, sy, sz)
        use pic_fields, only: jxt => jx, jyt => jy, jzt => jz
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
        jx(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            jxt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        jy(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            jyt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        jz(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            jzt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        if (ixs_lo == 1) then
            jx(0, :, :) = jx(1, :, :)
            jy(0, :, :) = jy(1, :, :)
            jz(0, :, :) = jz(1, :, :)
        endif
        if (ixe_lo == pnx) then
            jx(pnx+1, :, :) = jx(pnx, :, :)
            jy(pnx+1, :, :) = jy(pnx, :, :)
            jz(pnx+1, :, :) = jz(pnx, :, :)
        endif
        if (iys_lo == 1) then
            jx(:, 0, :) = jx(:, 1, :)
            jy(:, 0, :) = jy(:, 1, :)
            jz(:, 0, :) = jz(:, 1, :)
        endif
        if (iye_lo == pny) then
            jx(:, pny+1, :) = jx(:, pny, :)
            jy(:, pny+1, :) = jy(:, pny, :)
            jz(:, pny+1, :) = jz(:, pny, :)
        endif
        if (izs_lo == 1) then
            jx(:, :, 0) = jx(:, :, 1)
            jy(:, :, 0) = jy(:, :, 1)
            jz(:, :, 0) = jz(:, :, 1)
        endif
        if (ize_lo == pnz) then
            jx(:, :, pnz+1) = jx(:, :, pnz)
            jy(:, :, pnz+1) = jy(:, :, pnz)
            jz(:, :, pnz+1) = jz(:, :, pnz)
        endif
    end subroutine set_current_density

    !---------------------------------------------------------------------------
    ! Calculate the weights for trilinear interpolation.
    !
    ! Input:
    !   dx, dy, dz: the distance to the lower-left corner, [0, 1]
    !---------------------------------------------------------------------------
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

    !---------------------------------------------------------------------------
    ! Trilinear interpolation for current density
    ! 
    ! Input:
    !   ix0, iy0, iz0: the indices of the lower-left corner.
    !   dx, dy, dz: the distance to the lower-left corner, [0, 1]
    !---------------------------------------------------------------------------
    subroutine trilinear_interp_current_density(ix0, iy0, iz0, dx, dy, dz)
        implicit none
        integer, intent(in) :: ix0, iy0, iz0
        real(fp), intent(in) :: dx, dy, dz
        call calc_interp_weights(dx, dy, dz)
        jx0 = sum(jx(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        jy0 = sum(jy(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        jz0 = sum(jz(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
    end subroutine trilinear_interp_current_density

end module interpolation_current_density
