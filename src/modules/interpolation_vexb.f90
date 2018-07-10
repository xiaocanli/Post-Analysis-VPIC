!<******************************************************************************
!< Module of doing interpolation of exb drift velocity and its derivatives
!<******************************************************************************
module interpolation_vexb
    use mpi_module
    use constants, only: fp
    use parameters, only: tp1, is_rel
    use picinfo, only: domain
    use mpi_topology, only: htg
    use mpi_io_module, only: open_data_mpi_io, read_data_mpi_io
    use mpi_datatype_fields, only: filetype_ghost, subsizes_ghost
    use path_info, only: filepath
    use mpi_info_module, only: fileinfo
    use interpolation_funs, only: bounding_indcies
    implicit none
    private
    public init_exb_drift, free_exb_drift, calc_exb_drift
    public init_exb_derivatives, free_exb_derivatives, calc_exb_derivatives
    public init_exb_derivatives_single, free_exb_derivatives_single, &
        set_exb_derivatives, trilinear_interp_exb_derivatives
    public dvxdx0, dvxdy0, dvxdz0, dvydx0, dvydy0, dvydz0, dvzdx0, dvzdy0, dvzdz0

    ! Fields for the whole local domain
    real(fp), allocatable, dimension(:,:,:) :: vexbx, vexby, vexbz
    real(fp), allocatable, dimension(:,:,:) :: dvxdx, dvxdy, dvxdz
    real(fp), allocatable, dimension(:,:,:) :: dvydx, dvydy, dvydz
    real(fp), allocatable, dimension(:,:,:) :: dvzdx, dvzdy, dvzdz
    ! Fields for the one MPI rank
    real(fp), allocatable, dimension(:,:,:) :: dvxdx_s, dvxdy_s, dvxdz_s
    real(fp), allocatable, dimension(:,:,:) :: dvydx_s, dvydy_s, dvydz_s
    real(fp), allocatable, dimension(:,:,:) :: dvzdx_s, dvzdy_s, dvzdz_s
    ! Fields at particle positions
    real(fp) :: dvxdx0, dvxdy0, dvxdz0
    real(fp) :: dvydx0, dvydy0, dvydz0
    real(fp) :: dvzdx0, dvzdy0, dvzdz0
    real(fp), dimension(2,2,2) :: weights  ! The weights for trilinear interpolation
    integer :: nx, ny, nz

    contains

    !<--------------------------------------------------------------------------
    !< Initialize the exb drift velocity
    !<--------------------------------------------------------------------------
    subroutine init_exb_drift(nx, ny, nz)
        implicit none
        integer, intent(in) :: nx, ny, nz
        allocate(vexbx(nx,ny,nz))
        allocate(vexby(nx,ny,nz))
        allocate(vexbz(nx,ny,nz))
        vexbx = 0.0; vexby = 0.0; vexbz = 0.0
    end subroutine init_exb_drift

    !<--------------------------------------------------------------------------
    !< Initialize the derivatives of exb drift velocity
    !<--------------------------------------------------------------------------
    subroutine init_exb_derivatives(nx, ny, nz)
        implicit none
        integer, intent(in) :: nx, ny, nz
        allocate(dvxdx(nx,ny,nz))
        allocate(dvxdy(nx,ny,nz))
        allocate(dvxdz(nx,ny,nz))
        allocate(dvydx(nx,ny,nz))
        allocate(dvydy(nx,ny,nz))
        allocate(dvydz(nx,ny,nz))
        allocate(dvzdx(nx,ny,nz))
        allocate(dvzdy(nx,ny,nz))
        allocate(dvzdz(nx,ny,nz))
        dvxdx = 0.0; dvxdy = 0.0; dvxdz = 0.0
        dvydx = 0.0; dvydy = 0.0; dvydz = 0.0
        dvzdx = 0.0; dvzdy = 0.0; dvzdz = 0.0
    end subroutine init_exb_derivatives

    !<--------------------------------------------------------------------------
    !< Free the exb drift velocity
    !<--------------------------------------------------------------------------
    subroutine free_exb_drift
        implicit none
        deallocate(vexbx, vexby, vexbz)
    end subroutine free_exb_drift

    !<--------------------------------------------------------------------------
    !< Free the derivatives of exb drift velocity
    !<--------------------------------------------------------------------------
    subroutine free_exb_derivatives
        implicit none
        deallocate(dvxdx, dvxdy, dvxdz)
        deallocate(dvydx, dvydy, dvydz)
        deallocate(dvzdx, dvzdy, dvzdz)
    end subroutine free_exb_derivatives

    !<--------------------------------------------------------------------------
    !< Calculate ExB drift velocity
    !<--------------------------------------------------------------------------
    subroutine calc_exb_drift
        use pic_fields, only: ex, ey, ez, bx, by, bz, absB
        implicit none
        vexbx = (ey * bz - ez * by) / absB**2
        vexby = (ez * bx - ex * bz) / absB**2
        vexbz = (ex * by - ey * bx) / absB**2
    end subroutine calc_exb_drift

    !<--------------------------------------------------------------------------
    !< Calculate the derivatives of exb drift velocity
    !<--------------------------------------------------------------------------
    subroutine calc_exb_derivatives(nx, ny, nz)
        use neighbors_module, only: ixl, iyl, izl, ixh, iyh, izh, idx, idy, idz
        implicit none
        integer, intent(in) :: nx, ny, nz
        integer :: ix, iy, iz
        do ix = 1, nx
            dvxdx(ix, :, :) = (vexbx(ixh(ix), :, :) - vexbx(ixl(ix), :, :)) * idx(ix)
            dvydx(ix, :, :) = (vexby(ixh(ix), :, :) - vexby(ixl(ix), :, :)) * idx(ix)
            dvzdx(ix, :, :) = (vexbz(ixh(ix), :, :) - vexbz(ixl(ix), :, :)) * idx(ix)
        enddo

        do iy = 1, ny
            dvxdy(:, iy, :) = (vexbx(:, iyh(iy), :) - vexbx(:, iyl(iy), :)) * idy(iy)
            dvydy(:, iy, :) = (vexby(:, iyh(iy), :) - vexby(:, iyl(iy), :)) * idy(iy)
            dvzdy(:, iy, :) = (vexbz(:, iyh(iy), :) - vexbz(:, iyl(iy), :)) * idy(iy)
        enddo

        do iz = 1, nz
            dvxdz(:, :, iz) = (vexbx(:, :, izh(iz)) - vexbx(:, :, izl(iz))) * idz(iz)
            dvydz(:, :, iz) = (vexby(:, :, izh(iz)) - vexby(:, :, izl(iz))) * idz(iz)
            dvzdz(:, :, iz) = (vexbz(:, :, izh(iz)) - vexbz(:, :, izl(iz))) * idz(iz)
        enddo
    end subroutine calc_exb_derivatives

    !<--------------------------------------------------------------------------
    !< Initialize the derivatives of exb drift velocity for a single PIC MPI rank
    !<--------------------------------------------------------------------------
    subroutine init_exb_derivatives_single
        use picinfo, only: domain
        implicit none
        nx = domain%pic_nx + 2  ! Including ghost cells
        ny = domain%pic_ny + 2
        nz = domain%pic_nz + 2

        allocate(dvxdx_s(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(dvxdy_s(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(dvxdz_s(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(dvydx_s(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(dvydy_s(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(dvydz_s(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(dvzdx_s(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(dvzdy_s(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(dvzdz_s(0:nx-1, 0:ny-1, 0:nz-1))
        dvxdx_s = 0.0; dvxdy_s = 0.0; dvxdz_s = 0.0
        dvydx_s = 0.0; dvydy_s = 0.0; dvydz_s = 0.0
        dvzdx_s = 0.0; dvzdy_s = 0.0; dvzdz_s = 0.0
    end subroutine init_exb_derivatives_single

    !<--------------------------------------------------------------------------
    !< Free the derivatives of exb drift velocity for a single PIC MPI rank
    !<--------------------------------------------------------------------------
    subroutine free_exb_derivatives_single
        implicit none
        deallocate(dvxdx_s, dvxdy_s, dvxdz_s)
        deallocate(dvydx_s, dvydy_s, dvydz_s)
        deallocate(dvzdx_s, dvzdy_s, dvzdz_s)
    end subroutine free_exb_derivatives_single

    !<--------------------------------------------------------------------------
    !< Set electromagnetic fields, which is read from translated files rather
    !< than directly from the PIC simulations
    !<--------------------------------------------------------------------------
    subroutine set_exb_derivatives(i, j, k, tx, ty, tz, sx, sy, sz)
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
        dvxdx_s(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            dvxdx(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        dvxdy_s(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            dvxdy(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        dvxdz_s(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            dvxdz(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        dvydx_s(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            dvydx(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        dvydy_s(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            dvydy(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        dvydz_s(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            dvydz(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        dvzdx_s(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            dvzdx(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        dvzdy_s(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            dvzdy(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        dvzdz_s(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            dvzdz(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        if (ixs_lo == 1) then
            dvxdx_s(0, :, :) = dvxdx_s(1, :, :)
            dvxdy_s(0, :, :) = dvxdy_s(1, :, :)
            dvxdz_s(0, :, :) = dvxdz_s(1, :, :)
            dvydx_s(0, :, :) = dvydx_s(1, :, :)
            dvydy_s(0, :, :) = dvydy_s(1, :, :)
            dvydz_s(0, :, :) = dvydz_s(1, :, :)
            dvzdx_s(0, :, :) = dvzdx_s(1, :, :)
            dvzdy_s(0, :, :) = dvzdy_s(1, :, :)
            dvzdz_s(0, :, :) = dvzdz_s(1, :, :)
        endif
        if (ixe_lo == pnx) then
            dvxdx_s(pnx+1, :, :) = dvxdx_s(pnx, :, :)
            dvxdy_s(pnx+1, :, :) = dvxdy_s(pnx, :, :)
            dvxdz_s(pnx+1, :, :) = dvxdz_s(pnx, :, :)
            dvydx_s(pnx+1, :, :) = dvydx_s(pnx, :, :)
            dvydy_s(pnx+1, :, :) = dvydy_s(pnx, :, :)
            dvydz_s(pnx+1, :, :) = dvydz_s(pnx, :, :)
            dvzdx_s(pnx+1, :, :) = dvzdx_s(pnx, :, :)
            dvzdy_s(pnx+1, :, :) = dvzdy_s(pnx, :, :)
            dvzdz_s(pnx+1, :, :) = dvzdz_s(pnx, :, :)
        endif
        if (iys_lo == 1) then
            dvxdx_s(:, 0, :) = dvxdx_s(:, 1, :)
            dvxdy_s(:, 0, :) = dvxdy_s(:, 1, :)
            dvxdz_s(:, 0, :) = dvxdz_s(:, 1, :)
            dvydx_s(:, 0, :) = dvydx_s(:, 1, :)
            dvydy_s(:, 0, :) = dvydy_s(:, 1, :)
            dvydz_s(:, 0, :) = dvydz_s(:, 1, :)
            dvzdx_s(:, 0, :) = dvzdx_s(:, 1, :)
            dvzdy_s(:, 0, :) = dvzdy_s(:, 1, :)
            dvzdz_s(:, 0, :) = dvzdz_s(:, 1, :)
        endif
        if (iye_lo == pny) then
            dvxdx_s(:, pny+1, :) = dvxdx_s(:, pny, :)
            dvxdy_s(:, pny+1, :) = dvxdy_s(:, pny, :)
            dvxdz_s(:, pny+1, :) = dvxdz_s(:, pny, :)
            dvydx_s(:, pny+1, :) = dvydx_s(:, pny, :)
            dvydy_s(:, pny+1, :) = dvydy_s(:, pny, :)
            dvydz_s(:, pny+1, :) = dvydz_s(:, pny, :)
            dvzdx_s(:, pny+1, :) = dvzdx_s(:, pny, :)
            dvzdy_s(:, pny+1, :) = dvzdy_s(:, pny, :)
            dvzdz_s(:, pny+1, :) = dvzdz_s(:, pny, :)
        endif
        if (izs_lo == 1) then
            dvxdx_s(:, :, 0) = dvxdx_s(:, :, 1)
            dvxdy_s(:, :, 0) = dvxdy_s(:, :, 1)
            dvxdz_s(:, :, 0) = dvxdz_s(:, :, 1)
            dvydx_s(:, :, 0) = dvydx_s(:, :, 1)
            dvydy_s(:, :, 0) = dvydy_s(:, :, 1)
            dvydz_s(:, :, 0) = dvydz_s(:, :, 1)
            dvzdx_s(:, :, 0) = dvzdx_s(:, :, 1)
            dvzdy_s(:, :, 0) = dvzdy_s(:, :, 1)
            dvzdz_s(:, :, 0) = dvzdz_s(:, :, 1)
        endif
        if (ize_lo == pnz) then
            dvxdx_s(:, :, pnz+1) = dvxdx_s(:, :, pnz)
            dvxdy_s(:, :, pnz+1) = dvxdy_s(:, :, pnz)
            dvxdz_s(:, :, pnz+1) = dvxdz_s(:, :, pnz)
            dvydx_s(:, :, pnz+1) = dvydx_s(:, :, pnz)
            dvydy_s(:, :, pnz+1) = dvydy_s(:, :, pnz)
            dvydz_s(:, :, pnz+1) = dvydz_s(:, :, pnz)
            dvzdx_s(:, :, pnz+1) = dvzdx_s(:, :, pnz)
            dvzdy_s(:, :, pnz+1) = dvzdy_s(:, :, pnz)
            dvzdz_s(:, :, pnz+1) = dvzdz_s(:, :, pnz)
        endif
    end subroutine set_exb_derivatives

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
    !< Trilinear interpolation for compression and shear tensor
    !<
    !< Input:
    !<   ix0, iy0, iz0: the indices of the lower-left corner.
    !<   dx, dy, dz: the distance to the lower-left corner, [0, 1]
    !<--------------------------------------------------------------------------
    subroutine trilinear_interp_exb_derivatives(ix0, iy0, iz0, dx, dy, dz)
        implicit none
        integer, intent(in) :: ix0, iy0, iz0
        real(fp), intent(in) :: dx, dy, dz
        call calc_interp_weights(dx, dy, dz)
        dvxdx0 = sum(dvxdx_s(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        dvxdy0 = sum(dvxdy_s(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        dvxdz0 = sum(dvxdz_s(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        dvydx0 = sum(dvydx_s(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        dvydy0 = sum(dvydy_s(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        dvydz0 = sum(dvydz_s(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        dvzdx0 = sum(dvzdx_s(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        dvzdy0 = sum(dvzdy_s(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        dvzdz0 = sum(dvzdz_s(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
    end subroutine trilinear_interp_exb_derivatives

end module interpolation_vexb
