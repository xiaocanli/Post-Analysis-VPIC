!*******************************************************************************
! This module includes the electric field data, methods to read, free the
! electric field and get the electric field at one point.
!*******************************************************************************
module electric_field
    use constants, only: fp
    implicit none
    private
    public init_electric_fields, read_electric_fields, &
           free_electric_fields, get_electric_field_at_point
    public ex0, ey0, ez0

    real(fp), allocatable, dimension(:,:) :: Ex, Ey, Ez
    integer :: nx, nz, ix1, ix2, iz1, iz2
    real(fp):: shiftx, shiftz   ! The offset of one point from the corner.
    real(fp) :: ex0, ey0, ez0   ! electric field at one point.

    contains

    !---------------------------------------------------------------------------
    ! Initialize the electric field.
    !---------------------------------------------------------------------------
    subroutine init_electric_fields
        use picinfo, only: domain
        implicit none
        nx = domain%nx
        nz = domain%nz
        allocate(Ex(nx, nz))
        allocate(Ey(nx, nz))
        allocate(Ez(nx, nz))
        Ex = 0.0
        Ez = 0.0
        Ez = 0.0
    end subroutine init_electric_fields

    !---------------------------------------------------------------------------
    ! Free the electric field.
    !---------------------------------------------------------------------------
    subroutine free_electric_fields
        implicit none
        deallocate(ex, ey, ez)
    end subroutine free_electric_fields

    !---------------------------------------------------------------------------
    ! Read the electric field using the master MPI process and broadcast to
    ! other MPI processes.
    !---------------------------------------------------------------------------
    subroutine read_electric_fields(ct)
        use mpi_module
        use constants, only: fp, dp
        use path_info, only: rootpath
        use parameters, only: it1
        implicit none
        integer, intent(in) :: ct
        character(len=150) :: filename
        integer(dp) :: pos1
        integer :: fh

        fh = 101
        if (myid == master) then
            ! Ex
            filename = trim(adjustl(rootpath))//'data/ex.gda'
            open(unit=fh, file=trim(filename), access='stream',&
                status='unknown', form='unformatted', action='read')
            pos1 = nx * nz * sizeof(fp) * (ct-it1) + 1
            read(fh, pos=pos1) Ex
            close(fh)

            ! Ey
            filename = trim(adjustl(rootpath))//'data/ey.gda'
            open(unit=fh, file=trim(filename), access='stream',&
                status='unknown', form='unformatted', action='read')
            read(fh, pos=pos1) Ey
            close(fh)

            ! Ez
            filename = trim(adjustl(rootpath))//'data/ez.gda'
            open(unit=fh, file=trim(filename), access='stream',&
                status='unknown', form='unformatted', action='read')
            read(fh, pos=pos1) Ez
            close(fh)
        endif
        call MPI_BCAST(Ex, nx*nz, MPI_REAL, master, MPI_COMM_WORLD, ierr)
        call MPI_BCAST(Ey, nx*nz, MPI_REAL, master, MPI_COMM_WORLD, ierr)
        call MPI_BCAST(Ez, nx*nz, MPI_REAL, master, MPI_COMM_WORLD, ierr)
    end subroutine read_electric_fields

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
    ! Get electric field at one point.
    ! Inputs:
    !   x, z: the coordinates of the point.
    !   dx, dz: the grid sizes.
    !---------------------------------------------------------------------------
    subroutine get_electric_field_at_point(x, z, dx, dz)
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

            ex0 = Ex(ix1,iz1)*v1 + Ex(ix1,iz2)*v2 + Ex(ix2,iz2)*v3 + Ex(ix2,iz1)*v4
            ey0 = Ey(ix1,iz1)*v1 + Ey(ix1,iz2)*v2 + Ey(ix2,iz2)*v3 + Ey(ix2,iz1)*v4
            ez0 = Ez(ix1,iz1)*v1 + Ez(ix1,iz2)*v2 + Ez(ix2,iz2)*v3 + Ez(ix2,iz1)*v4
        endif
    end subroutine get_electric_field_at_point

end module electric_field
