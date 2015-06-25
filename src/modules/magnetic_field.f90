!*******************************************************************************
! This module includes the magnetic field data, methods to read, free the
! magnetic field and get the magnetic field at one point.
!*******************************************************************************
module magnetic_field
    use constants, only: fp
    implicit none
    private
    public init_magnetic_fields, read_magnetic_fields, &
           free_magnetic_fields, get_magnetic_field_at_point
    public bx0, by0, bz0

    real(fp), allocatable, dimension(:,:) :: Bx, By, Bz
    integer :: nx, nz, ix1, ix2, iz1, iz2
    real(fp):: shiftx, shiftz   ! The offset of one point from the corner.
    real(fp) :: bx0, by0, bz0   ! Magnetic field at one point.

    contains

    !---------------------------------------------------------------------------
    ! Initialize the magnetic field.
    !---------------------------------------------------------------------------
    subroutine init_magnetic_fields
        use picinfo, only: domain
        implicit none
        nx = domain%nx
        nz = domain%nz
        allocate(Bx(nx, nz))
        allocate(By(nx, nz))
        allocate(Bz(nx, nz))
        Bx = 0.0
        Bz = 0.0
        Bz = 0.0
    end subroutine init_magnetic_fields

    !---------------------------------------------------------------------------
    ! Free the magnetic field.
    !---------------------------------------------------------------------------
    subroutine free_magnetic_fields
        implicit none
        deallocate(Bx, By, Bz)
    end subroutine free_magnetic_fields

    !---------------------------------------------------------------------------
    ! Read the magnetic field using the master MPI process and broadcast to
    ! other MPI processes.
    !---------------------------------------------------------------------------
    subroutine read_magnetic_fields(ct)
        use mpi_module
        use constants, only: fp, dp
        use path_info, only: rootpath
        use parameters, only: tp1
        implicit none
        integer, intent(in) :: ct
        character(len=150) :: filename
        integer(dp) :: pos1
        integer :: fh

        fh = 101
        if (myid == master) then
            ! Bx
            filename = trim(adjustl(rootpath))//'data/bx.gda'
            open(unit=fh, file=trim(filename), access='stream',&
                status='unknown', form='unformatted', action='read')
            pos1 = nx * nz * sizeof(fp) * (ct-tp1) + 1
            read(fh, pos=pos1) Bx
            close(fh)

            ! By
            filename = trim(adjustl(rootpath))//'data/by.gda'
            open(unit=fh, file=trim(filename), access='stream',&
                status='unknown', form='unformatted', action='read')
            read(fh, pos=pos1) By
            close(fh)

            ! Bz
            filename = trim(adjustl(rootpath))//'data/bz.gda'
            open(unit=fh, file=trim(filename), access='stream',&
                status='unknown', form='unformatted', action='read')
            read(fh, pos=pos1) Bz
            close(fh)
        endif
        call MPI_BCAST(Bx, nx*nz, MPI_REAL, master, MPI_COMM_WORLD, ierr)
        call MPI_BCAST(By, nx*nz, MPI_REAL, master, MPI_COMM_WORLD, ierr)
        call MPI_BCAST(Bz, nx*nz, MPI_REAL, master, MPI_COMM_WORLD, ierr)
    end subroutine read_magnetic_fields

    !---------------------------------------------------------------------------
    ! Calculate the 2D grid indices for one point.
    ! Input:
    !   x, z: the coordinates of the point.
    !   dx, dz: the grid sizes.
    ! Updates:
    !   ix1, iz1: grid indices for the bottom left corner.
    !   ix2, iz2: grid indices for the top right corner.
    !   shiftx, shifty: the offsets from the bottom left corner.
    ! Note:
    !   Make sure that x, z, dx, dz are in the same unit (di or de).
    !---------------------------------------------------------------------------
    subroutine calc_grid_indices(x, z, dx, dz)
        implicit none
        real(fp), intent(in) :: x, z, dx, dz

        ix1 = floor(x / dx)
        iz1 = floor(z / dz)
        ix2 = ix1 + 1
        iz2 = iz1 + 1
        shiftx = x/dx - ix1
        shiftz = z/dz - iz1
    end subroutine calc_grid_indices

    !---------------------------------------------------------------------------
    ! Get magnetic field at one point.
    ! Inputs:
    !   x, z: the coordinates of the point.
    !   dx, dz: the grid sizes.
    !---------------------------------------------------------------------------
    subroutine get_magnetic_field_at_point(x, z, dx, dz)
        implicit none
        real(fp), intent(in) :: x, z, dx, dz
        real(fp) :: v1, v2, v3, v4

        call calc_grid_indices(x, z, dx, dz)

        if (ix1 >=1 .and. ix1 <= nx .and. ix2 >=0 .and. ix2 <= nx .and. &
                iz1 >= 1 .and. iz1 <= nz .and. iz2 >= 1 .and. iz2 <= nz) then

            v1 = (1.0-shiftx) * (1.0-shiftz)
            v2 = shiftx * (1.0-shiftz)
            v3 = shiftx * shiftz
            v4 = (1.0-shiftx) * shiftz

            bx0 = Bx(ix1,iz1)*v1 + Bx(ix1,iz2)*v2 + Bx(ix2,iz2)*v3 + Bx(ix2,iz1)*v4
            by0 = By(ix1,iz1)*v1 + By(ix1,iz2)*v2 + By(ix2,iz2)*v3 + By(ix2,iz1)*v4
            bz0 = Bz(ix1,iz1)*v1 + Bz(ix1,iz2)*v2 + Bz(ix2,iz2)*v3 + Bz(ix2,iz1)*v4
        endif
    end subroutine get_magnetic_field_at_point

end module magnetic_field
