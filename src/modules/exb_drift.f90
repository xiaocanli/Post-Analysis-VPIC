!<******************************************************************************
!< Module of ExB drift velocity and its derivatives
!<******************************************************************************
module exb_drift
    use mpi_module
    use constants, only: fp
    use parameters, only: tp1
    use picinfo, only: domain
    use mpi_io_module, only: open_data_mpi_io, read_data_mpi_io
    use mpi_datatype_fields, only: filetype_ghost, subsizes_ghost
    use path_info, only: outputpath
    use mpi_info_module, only: fileinfo
    implicit none
    private
    public init_exb_drift, free_exb_drift, calc_exb_drift
    public init_exb_derivatives, free_exb_derivatives, calc_exb_derivatives
    public open_exb_drift_files, read_exb_drift, close_exb_drift_files
    public vexbx, vexby, vexbz, dvxdx, dvxdy, dvxdz, &
        dvydx, dvydy, dvydz, dvzdx, dvzdy, dvzdz
    public vexb_fh

    ! Fields for the whole local domain
    real(fp), allocatable, dimension(:,:,:) :: vexbx, vexby, vexbz
    real(fp), allocatable, dimension(:,:,:) :: dvxdx, dvxdy, dvxdz
    real(fp), allocatable, dimension(:,:,:) :: dvydx, dvydy, dvydz
    real(fp), allocatable, dimension(:,:,:) :: dvzdx, dvzdy, dvzdz
    integer, dimension(3) :: vexb_fh

    interface open_exb_drift_files
        module procedure &
            open_exb_drift_files_single, open_exb_drift_files_multiple
    end interface open_exb_drift_files

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
    !< Open exb drift velocity files when each field is saved in a single file.
    !<--------------------------------------------------------------------------
    subroutine open_exb_drift_files_single
        implicit none
        character(len=256) :: fname
        vexb_fh = 0
        fname = trim(adjustl(outputpath))//'vexb_x.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, vexb_fh(1))
        fname = trim(adjustl(outputpath))//'vexb_y.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, vexb_fh(2))
        fname = trim(adjustl(outputpath))//'vexb_z.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, vexb_fh(3))
    end subroutine open_exb_drift_files_single

    !<--------------------------------------------------------------------------
    !< Open exb drift velocity files if each time frame is saved separately.
    !< Inputs:
    !<   tindex: the time index.
    !<--------------------------------------------------------------------------
    subroutine open_exb_drift_files_multiple(tindex)
        implicit none
        integer, intent(in) :: tindex
        character(len=256) :: fname
        character(len=16) :: cfname
        write(cfname, "(I0)") tindex
        vexb_fh = 0
        fname = trim(adjustl(outputpath))//'exb_x_'//trim(cfname)//'.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, vexb_fh(1))
        fname = trim(adjustl(outputpath))//'exb_y_'//trim(cfname)//'.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, vexb_fh(2))
        fname = trim(adjustl(outputpath))//'exb_z_'//trim(cfname)//'.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, vexb_fh(3))
    end subroutine open_exb_drift_files_multiple

    !---------------------------------------------------------------------------
    ! Close exb drift velocity files.
    !---------------------------------------------------------------------------
    subroutine close_exb_drift_files
        implicit none
        call MPI_FILE_CLOSE(vexb_fh(1), ierror)
        call MPI_FILE_CLOSE(vexb_fh(2), ierror)
        call MPI_FILE_CLOSE(vexb_fh(3), ierror)
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
            subsizes_ghost, disp, offset, vexbx)
        call read_data_mpi_io(vexb_fh(2), filetype_ghost, &
            subsizes_ghost, disp, offset, vexby)
        call read_data_mpi_io(vexb_fh(3), filetype_ghost, &
            subsizes_ghost, disp, offset, vexbz)
    end subroutine read_exb_drift

end module exb_drift
