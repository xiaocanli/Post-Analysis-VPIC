!*******************************************************************************
! Module of doing interpolation in fluid velocity and momentum fields
!*******************************************************************************
module interpolation_vel_mom
    use constants, only: fp
    implicit none
    private
    public init_vel_mom, free_vel_mom, set_vel_mom, trilinear_interp_vel_mom
    public vx0, vy0, vz0, ux0, uy0, uz0

    real(fp), allocatable, dimension(:,:,:) :: vx, vy, vz, ux, uy, uz
    real(fp) :: vx0, vy0, vz0, ux0, uy0, uz0
    real(fp), dimension(2,2,2) :: weights  ! The weights for trilinear interpolation
    integer :: nx, ny, nz

    contains

    !---------------------------------------------------------------------------
    ! Initialize velocity and momentum fields
    !---------------------------------------------------------------------------
    subroutine init_vel_mom
        use picinfo, only: domain
        implicit none
        nx = domain%pic_nx + 2  ! Including ghost cells
        ny = domain%pic_ny + 2
        nz = domain%pic_nz + 2

        allocate(vx(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(vy(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(vz(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(ux(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(uy(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(uz(0:nx-1, 0:ny-1, 0:nz-1))
        vx = 0.0; vy = 0.0; vz = 0.0
        ux = 0.0; uy = 0.0; uz = 0.0
    end subroutine init_vel_mom

    !---------------------------------------------------------------------------
    ! Free velocity and momentum fields
    !---------------------------------------------------------------------------
    subroutine free_vel_mom
        implicit none
        deallocate(vx, vy, vz)
        deallocate(ux, uy, uz)
    end subroutine free_vel_mom

    !<--------------------------------------------------------------------------
    !< Decide the starting and ending indices
    !<--------------------------------------------------------------------------
    subroutine bounding_indcies(ix, pic_nx, tx, sx, ixs_local, ixe_local, &
            ixs_global, ixe_global)
        implicit none
        integer, intent(in) :: ix, pic_nx, tx, sx
        integer, intent(out) :: ixs_local, ixe_local, ixs_global, ixe_global
        if (tx == 1) then
            ixs_local = 1
            ixe_local = pic_nx
            ixs_global = 1
            ixe_global = pic_nx
        else if (ix == 0 .and. ix < tx - 1) then
            ixs_local = 1
            ixe_local = pic_nx + 1
            ixs_global = 1
            ixe_global = pic_nx + 1
        else if (ix == tx - 1 .and. ix > 0) then
            ixs_local = 0
            ixe_local = pic_nx
            ixs_global = pic_nx * (ix - sx) + 1
            ixe_global = pic_nx * (ix - sx + 1) + 1
        else
            ixs_local = 0
            ixe_local = pic_nx + 1
            if (sx /= 0) then
                ixs_global = pic_nx * (ix - sx) + 1
                ixe_global = pic_nx * (ix - sx + 1) + 2
            else
                ixs_global = pic_nx * (ix - sx)
                ixe_global = pic_nx * (ix - sx + 1) + 1
            endif
        endif
    end subroutine bounding_indcies

    !<--------------------------------------------------------------------------
    !< Set velocity and momentum fields, which are read from translated files
    !< rather than directly from the PIC simulations
    !<--------------------------------------------------------------------------
    subroutine set_vel_mom(i, j, k, tx, ty, tz, sx, sy, sz)
        use pic_fields, only: vxt => vx, vyt => vy, vzt => vz, &
                              uxt => ux, uyt => uy, uzt => uz
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
        vx(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            vxt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        vy(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            vyt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        vz(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            vzt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        ux(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            uxt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        uy(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            uyt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        uz(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            uzt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        if (ixs_lo == 1) then
            vx(0, :, :) = vx(1, :, :)
            vy(0, :, :) = vy(1, :, :)
            vz(0, :, :) = vz(1, :, :)
            ux(0, :, :) = ux(1, :, :)
            uy(0, :, :) = uy(1, :, :)
            uz(0, :, :) = uz(1, :, :)
        endif
        if (ixe_lo == pnx) then
            vx(pnx+1, :, :) = vx(pnx, :, :)
            vy(pnx+1, :, :) = vy(pnx, :, :)
            vz(pnx+1, :, :) = vz(pnx, :, :)
            ux(pnx+1, :, :) = ux(pnx, :, :)
            uy(pnx+1, :, :) = uy(pnx, :, :)
            uz(pnx+1, :, :) = uz(pnx, :, :)
        endif
        if (iys_lo == 1) then
            vx(:, 0, :) = vx(:, 1, :)
            vy(:, 0, :) = vy(:, 1, :)
            vz(:, 0, :) = vz(:, 1, :)
            ux(:, 0, :) = ux(:, 1, :)
            uy(:, 0, :) = uy(:, 1, :)
            uz(:, 0, :) = uz(:, 1, :)
        endif
        if (iye_lo == pny) then
            vx(:, pny+1, :) = vx(:, pny, :)
            vy(:, pny+1, :) = vy(:, pny, :)
            vz(:, pny+1, :) = vz(:, pny, :)
            ux(:, pny+1, :) = ux(:, pny, :)
            uy(:, pny+1, :) = uy(:, pny, :)
            uz(:, pny+1, :) = uz(:, pny, :)
        endif
        if (izs_lo == 1) then
            vx(:, :, 0) = vx(:, :, 1)
            vy(:, :, 0) = vy(:, :, 1)
            vz(:, :, 0) = vz(:, :, 1)
            ux(:, :, 0) = ux(:, :, 1)
            uy(:, :, 0) = uy(:, :, 1)
            uz(:, :, 0) = uz(:, :, 1)
        endif
        if (ize_lo == pnz) then
            vx(:, :, pnz+1) = vx(:, :, pnz)
            vy(:, :, pnz+1) = vy(:, :, pnz)
            vz(:, :, pnz+1) = vz(:, :, pnz)
            ux(:, :, pnz+1) = ux(:, :, pnz)
            uy(:, :, pnz+1) = uy(:, :, pnz)
            uz(:, :, pnz+1) = uz(:, :, pnz)
        endif
    end subroutine set_vel_mom

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
    ! Trilinear interpolation for velocity and momentum
    ! 
    ! Input:
    !   ix0, iy0, iz0: the indices of the lower-left corner.
    !   dx, dy, dz: the distance to the lower-left corner, [0, 1]
    !---------------------------------------------------------------------------
    subroutine trilinear_interp_vel_mom(ix0, iy0, iz0, dx, dy, dz)
        implicit none
        integer, intent(in) :: ix0, iy0, iz0
        real(fp), intent(in) :: dx, dy, dz
        call calc_interp_weights(dx, dy, dz)
        vx0 = sum(vx(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        vy0 = sum(vy(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        vz0 = sum(vz(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        ux0 = sum(ux(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        uy0 = sum(uy(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        uz0 = sum(uz(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
    end subroutine trilinear_interp_vel_mom

end module interpolation_vel_mom
