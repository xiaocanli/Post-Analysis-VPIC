!!******************************************************************************
!! Module of doing interpolation of compression and shear
!!******************************************************************************
module interpolation_comp_shear
    use mpi_module
    use constants, only: fp
    use parameters, only: tp1, is_rel
    use picinfo, only: domain
    use mpi_topology, only: htg
    use mpi_io_module, only: open_data_mpi_io, read_data_mpi_io
    use mpi_datatype_fields, only: filetype_ghost, subsizes_ghost
    use path_info, only: filepath
    use mpi_info_module, only: fileinfo
    implicit none
    private
    public init_exb_drift, free_exb_drift, read_exb_drift
    public init_comp_shear, free_comp_shear, calc_comp_shear
    public init_comp_shear_single, free_comp_shear_single, set_comp_shear_single, &
        trilinear_interp_comp_shear
    public open_exb_drift_files, close_exb_drift_files
    public divv0, sigmaxx0, sigmayy0, sigmazz0, sigmaxy0, sigmaxz0, sigmayz0
    public vexb_fh

    ! Fields for the whole local domain
    real(fp), allocatable, dimension(:,:,:) :: vexb_x, vexb_y, vexb_z
    real(fp), allocatable, dimension(:,:,:) :: divv
    real(fp), allocatable, dimension(:,:,:) :: sigmaxx, sigmaxy, sigmaxz
    real(fp), allocatable, dimension(:,:,:) :: sigmayy, sigmayz, sigmazz
    ! Fields for a single PIC MPI domain
    real(fp), allocatable, dimension(:,:,:) :: divv_s
    real(fp), allocatable, dimension(:,:,:) :: sigmaxx_s, sigmaxy_s, sigmaxz_s
    real(fp), allocatable, dimension(:,:,:) :: sigmayy_s, sigmayz_s, sigmazz_s
    ! Fields at particle positions
    real(fp) :: divv0, sigmaxx0, sigmayy0, sigmazz0
    real(fp) :: sigmaxy0, sigmaxz0, sigmayz0
    real(fp), dimension(2,2,2) :: weights  ! The weights for trilinear interpolation
    integer :: nx, ny, nz
    integer, dimension(3) :: vexb_fh

    interface open_exb_drift_files
        module procedure &
            open_exb_drift_files_single, open_exb_drift_files_multiple
    end interface open_exb_drift_files

    contains

    !---------------------------------------------------------------------------
    ! Initialize the exb drift velocity
    !---------------------------------------------------------------------------
    subroutine init_exb_drift(nx, ny, nz)
        implicit none
        integer, intent(in) :: nx, ny, nz
        allocate(vexb_x(nx,ny,nz))
        allocate(vexb_y(nx,ny,nz))
        allocate(vexb_z(nx,ny,nz))
        vexb_x = 0.0; vexb_y = 0.0; vexb_z = 0.0
    end subroutine init_exb_drift

    !---------------------------------------------------------------------------
    ! Free the exb drift velocity
    !---------------------------------------------------------------------------
    subroutine free_exb_drift
        implicit none
        deallocate(vexb_x, vexb_y, vexb_z)
    end subroutine free_exb_drift

    !---------------------------------------------------------------------------
    ! Open exb drift velocity files when each field is saved in a single file.
    !---------------------------------------------------------------------------
    subroutine open_exb_drift_files_single
        implicit none
        character(len=256) :: fname
        vexb_fh = 0
        fname = trim(adjustl(filepath))//'exb_x.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, vexb_fh(1))
        fname = trim(adjustl(filepath))//'exb_y.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, vexb_fh(2))
        fname = trim(adjustl(filepath))//'exb_z.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, vexb_fh(3))
    end subroutine open_exb_drift_files_single

    !---------------------------------------------------------------------------
    ! Open exb drift velocity files if each time frame is saved separately.
    ! Inputs:
    !   tindex: the time index.
    !---------------------------------------------------------------------------
    subroutine open_exb_drift_files_multiple(tindex)
        implicit none
        integer, intent(in) :: tindex
        character(len=256) :: fname
        character(len=16) :: cfname
        write(cfname, "(I0)") tindex
        vexb_fh = 0
        fname = trim(adjustl(filepath))//'exb_x_'//trim(cfname)//'.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, vexb_fh(1))
        fname = trim(adjustl(filepath))//'exb_y_'//trim(cfname)//'.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, vexb_fh(2))
        fname = trim(adjustl(filepath))//'exb_z_'//trim(cfname)//'.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, vexb_fh(3))
    end subroutine open_exb_drift_files_multiple

    !---------------------------------------------------------------------------
    ! Close exb drift velocity files.
    !---------------------------------------------------------------------------
    subroutine close_exb_drift_files
        implicit none
        logical :: is_opened
        inquire(vexb_fh(1), opened=is_opened)
        if (is_opened) then
            call MPI_FILE_CLOSE(vexb_fh(1), ierror)
            call MPI_FILE_CLOSE(vexb_fh(2), ierror)
            call MPI_FILE_CLOSE(vexb_fh(3), ierror)
        endif
    end subroutine close_exb_drift_files

    !---------------------------------------------------------------------------
    ! Read exb drift velocity
    !---------------------------------------------------------------------------
    subroutine read_exb_drift(ct)
        implicit none
        integer, intent(in) :: ct
        integer(kind=MPI_OFFSET_KIND) :: disp, offset
        disp = domain%nx * domain%ny * domain%nz * sizeof(MPI_REAL) * (ct-tp1)
        offset = 0 
        call read_data_mpi_io(vexb_fh(1), filetype_ghost, &
            subsizes_ghost, disp, offset, vexb_x)
        call read_data_mpi_io(vexb_fh(2), filetype_ghost, &
            subsizes_ghost, disp, offset, vexb_y)
        call read_data_mpi_io(vexb_fh(3), filetype_ghost, &
            subsizes_ghost, disp, offset, vexb_z)
    end subroutine read_exb_drift

    !---------------------------------------------------------------------------
    ! Initialize compression ans shear tensor
    !---------------------------------------------------------------------------
    subroutine init_comp_shear(nx, ny, nz)
        implicit none
        integer, intent(in) :: nx, ny, nz
        allocate(divv(nx,ny,nz))
        allocate(sigmaxx(nx,ny,nz))
        allocate(sigmaxy(nx,ny,nz))
        allocate(sigmaxz(nx,ny,nz))
        allocate(sigmayy(nx,ny,nz))
        allocate(sigmayz(nx,ny,nz))
        allocate(sigmazz(nx,ny,nz))
        divv = 0.0
        sigmaxx = 0.0
        sigmaxy = 0.0
        sigmaxz = 0.0
        sigmayy = 0.0
        sigmayz = 0.0
        sigmazz = 0.0
    end subroutine init_comp_shear

    !---------------------------------------------------------------------------
    ! Free compression ans shear tensor
    !---------------------------------------------------------------------------
    subroutine free_comp_shear
        implicit none
        deallocate(divv)
        deallocate(sigmaxx, sigmaxy, sigmaxz)
        deallocate(sigmayy, sigmayz, sigmazz)
    end subroutine free_comp_shear

    !---------------------------------------------------------------------------
    ! Calculate compression ans shear tensor
    !---------------------------------------------------------------------------
    subroutine calc_comp_shear(nx, ny, nz)
        use neighbors_module, only: ixl, iyl, izl, ixh, iyh, izh, idx, idy, idz
        implicit none
        integer, intent(in) :: nx, ny, nz
        integer :: ix, iy, iz
        ! Part
        do ix = 1, nx
            sigmaxx(ix, :, :) = (vexb_x(ixh(ix), :, :) - vexb_x(ixl(ix), :, :)) * idx(ix)
        enddo

        ! Part
        do iy = 1, ny
            sigmaxy(:, iy, :) = (vexb_x(:, iyh(iy), :) - vexb_x(:, iyl(iy), :)) * idy(iy)
        enddo

        ! Part
        do iz = 1, nz
            sigmaxz(:, :, iz) = (vexb_x(:, :, izh(iz)) - vexb_x(:, :, izl(iz))) * idz(iz)
        enddo

        do ix = 1, nx
            sigmaxy(ix, :, :) = sigmaxy(ix, :, :) + &
                (vexb_y(ixh(ix), :, :) - vexb_y(ixl(ix), :, :)) * idx(ix)
        enddo

        do ix = 1, nx
            sigmaxz(ix, :, :) = sigmaxz(ix, :, :) + &
                (vexb_z(ixh(ix), :, :) - vexb_z(ixl(ix), :, :)) * idx(ix)
        enddo

        ! Part
        do iy = 1, ny
            sigmayy(:, iy, :) = (vexb_y(:, iyh(iy), :) - vexb_y(:, iyl(iy), :)) * idy(iy)
        enddo

        ! Part
        do iz = 1, nz
            sigmayz(:, :, iz) = (vexb_y(:, :, izh(iz)) - vexb_y(:, :, izl(iz))) * idz(iz)
        enddo

        do iy = 1, ny
            sigmayz(:, iy, :) = sigmayz(:, iy, :) + &
                (vexb_z(:, iyh(iy), :) - vexb_z(:, iyl(iy), :)) * idy(iy)
        enddo

        do iz = 1, nz
            sigmazz(:, :, iz) = (vexb_z(:, :, izh(iz)) - vexb_z(:, :, izl(iz))) * idz(iz)
        enddo
        
        divv = sigmaxx + sigmayy + sigmazz
        sigmaxx = sigmaxx - divv / 3
        sigmayy = sigmayy - divv / 3
        sigmazz = sigmazz - divv / 3
        sigmaxy = 0.5 * sigmaxy
        sigmaxz = 0.5 * sigmaxz
        sigmayz = 0.5 * sigmayz
    end subroutine calc_comp_shear

    !---------------------------------------------------------------------------
    ! Initialize compression and shear tensor for a single PIC MPI rank
    !---------------------------------------------------------------------------
    subroutine init_comp_shear_single
        use picinfo, only: domain
        implicit none
        nx = domain%pic_nx + 2  ! Including ghost cells
        ny = domain%pic_ny + 2
        nz = domain%pic_nz + 2

        allocate(divv_s(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(sigmaxx_s(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(sigmaxy_s(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(sigmaxz_s(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(sigmayy_s(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(sigmayz_s(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(sigmazz_s(0:nx-1, 0:ny-1, 0:nz-1))
        divv_s = 0.0
        sigmaxx_s = 0.0
        sigmaxy_s = 0.0
        sigmaxz_s = 0.0
        sigmayy_s = 0.0
        sigmayz_s = 0.0
        sigmazz_s = 0.0
    end subroutine init_comp_shear_single

    !---------------------------------------------------------------------------
    ! Free compression and shear tensor for a single PIC MPI rank
    !---------------------------------------------------------------------------
    subroutine free_comp_shear_single
        implicit none
        deallocate(divv_s)
        deallocate(sigmaxx_s, sigmaxy_s, sigmaxz_s)
        deallocate(sigmayy_s, sigmayz_s, sigmazz_s)
    end subroutine free_comp_shear_single

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
    !< Set electromagnetic fields, which is read from translated files rather
    !< than directly from the PIC simulations
    !<--------------------------------------------------------------------------
    subroutine set_comp_shear_single(i, j, k, tx, ty, tz, sx, sy, sz)
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
        divv_s(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            divv(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        sigmaxx_s(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            sigmaxx(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        sigmaxy_s(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            sigmaxy(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        sigmaxz_s(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            sigmaxz(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        sigmayy_s(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            sigmayy(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        sigmayz_s(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            sigmayz(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        sigmazz_s(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            sigmazz(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl) 
        if (ixs_lo == 1) then
            divv_s(0, :, :) = divv_s(1, :, :)
            sigmaxx_s(0, :, :) = sigmaxx_s(1, :, :)
            sigmaxy_s(0, :, :) = sigmaxy_s(1, :, :)
            sigmaxz_s(0, :, :) = sigmaxz_s(1, :, :)
            sigmayy_s(0, :, :) = sigmayy_s(1, :, :)
            sigmayz_s(0, :, :) = sigmayz_s(1, :, :)
            sigmazz_s(0, :, :) = sigmazz_s(1, :, :)
        endif
        if (ixe_lo == pnx) then
            divv_s(pnx+1, :, :) = divv_s(pnx, :, :)
            sigmaxx_s(pnx+1, :, :) = sigmaxx_s(pnx, :, :)
            sigmaxy_s(pnx+1, :, :) = sigmaxy_s(pnx, :, :)
            sigmaxz_s(pnx+1, :, :) = sigmaxz_s(pnx, :, :)
            sigmayy_s(pnx+1, :, :) = sigmayy_s(pnx, :, :)
            sigmayz_s(pnx+1, :, :) = sigmayz_s(pnx, :, :)
            sigmazz_s(pnx+1, :, :) = sigmazz_s(pnx, :, :)
        endif
        if (iys_lo == 1) then
            divv_s(:, 0, :) = divv_s(:, 1, :)
            sigmaxx_s(:, 0, :) = sigmaxx_s(:, 1, :)
            sigmaxy_s(:, 0, :) = sigmaxy_s(:, 1, :)
            sigmaxz_s(:, 0, :) = sigmaxz_s(:, 1, :)
            sigmayy_s(:, 0, :) = sigmayy_s(:, 1, :)
            sigmayz_s(:, 0, :) = sigmayz_s(:, 1, :)
            sigmazz_s(:, 0, :) = sigmazz_s(:, 1, :)
        endif
        if (iye_lo == pny) then
            divv_s(:, pny+1, :) = divv_s(:, pny, :)
            sigmaxx_s(:, pny+1, :) = sigmaxx_s(:, pny, :)
            sigmaxy_s(:, pny+1, :) = sigmaxy_s(:, pny, :)
            sigmaxz_s(:, pny+1, :) = sigmaxz_s(:, pny, :)
            sigmayy_s(:, pny+1, :) = sigmayy_s(:, pny, :)
            sigmayz_s(:, pny+1, :) = sigmayz_s(:, pny, :)
            sigmazz_s(:, pny+1, :) = sigmazz_s(:, pny, :)
        endif
        if (izs_lo == 1) then
            divv_s(:, :, 0) = divv_s(:, :, 1)
            sigmaxx_s(:, :, 0) = sigmaxx_s(:, :, 1)
            sigmaxy_s(:, :, 0) = sigmaxy_s(:, :, 1)
            sigmaxz_s(:, :, 0) = sigmaxz_s(:, :, 1)
            sigmayy_s(:, :, 0) = sigmayy_s(:, :, 1)
            sigmayz_s(:, :, 0) = sigmayz_s(:, :, 1)
            sigmazz_s(:, :, 0) = sigmazz_s(:, :, 1)
        endif
        if (ize_lo == pnz) then
            divv_s(:, :, pnz+1) = divv_s(:, :, pnz)
            sigmaxx_s(:, :, pnz+1) = sigmaxx_s(:, :, pnz)
            sigmaxy_s(:, :, pnz+1) = sigmaxy_s(:, :, pnz)
            sigmaxz_s(:, :, pnz+1) = sigmaxz_s(:, :, pnz)
            sigmayy_s(:, :, pnz+1) = sigmayy_s(:, :, pnz)
            sigmayz_s(:, :, pnz+1) = sigmayz_s(:, :, pnz)
            sigmazz_s(:, :, pnz+1) = sigmazz_s(:, :, pnz)
        endif
    end subroutine set_comp_shear_single

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
    ! Trilinear interpolation for compression and shear tensor
    ! 
    ! Input:
    !   ix0, iy0, iz0: the indices of the lower-left corner.
    !   dx, dy, dz: the distance to the lower-left corner, [0, 1]
    !---------------------------------------------------------------------------
    subroutine trilinear_interp_comp_shear(ix0, iy0, iz0, dx, dy, dz)
        implicit none
        integer, intent(in) :: ix0, iy0, iz0
        real(fp), intent(in) :: dx, dy, dz
        call calc_interp_weights(dx, dy, dz)
        divv0 = sum(divv_s(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        sigmaxx0 = sum(sigmaxx_s(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        sigmaxy0 = sum(sigmaxy_s(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        sigmaxz0 = sum(sigmaxz_s(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        sigmayy0 = sum(sigmayy_s(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        sigmayz0 = sum(sigmayz_s(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        sigmazz0 = sum(sigmazz_s(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
    end subroutine trilinear_interp_comp_shear

end module interpolation_comp_shear
