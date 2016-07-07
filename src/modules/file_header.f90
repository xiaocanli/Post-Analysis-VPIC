!*******************************************************************************
! Module of the header of a file.
!*******************************************************************************
module file_header
    implicit none
    private
    public v0, pheader, fheader, read_boilerplate, read_fields_header, &
           read_particle_header, set_v0header
    integer :: itype, ndim
    ! Define structure for V0 header
    type v0header
        integer(kind=4) :: version, type, nt, nx, ny, nz
        real(kind=4) :: dt, dx, dy, dz
        real(kind=4) :: x0, y0, z0
        real(kind=4) :: cvac, eps0, damp
        integer(kind=4) :: rank, ndom, spid, spqm
    end type v0header

    type header_particle
        integer :: size, ndim, dim
    end type header_particle

    type header_fields
        integer :: itype, ndim
        integer, dimension(3) :: nc
    end type header_fields

    ! Declare the headers
    type(v0header) :: v0
    type(header_particle) :: pheader
    type(header_fields) :: fheader

    contains

    !---------------------------------------------------------------------------
    ! Read boilerplate of a file.
    !---------------------------------------------------------------------------
    subroutine read_boilerplate(fh)
        implicit none
        integer, intent(in) :: fh
        integer(kind=1) sizearr(5)
        integer(kind=2) cafevar 
        integer(kind=4) deadbeefvar
        real(kind=4) realone
        real(kind=8) doubleone

        read(fh) sizearr
        read(fh) cafevar
        read(fh) deadbeefvar
        read(fh) realone
        read(fh) doubleone
        !  print *, sizearr,cafevar, deadbeefvar, realone, doubleone
        return
    end subroutine read_boilerplate

    !---------------------------------------------------------------------------
    ! Read fields header.
    !---------------------------------------------------------------------------
    subroutine read_fields_header(fh)
        implicit none
        integer, intent(in) :: fh
        read(fh) v0
        read(fh) fheader
    end subroutine read_fields_header

    !---------------------------------------------------------------------------
    ! Read particle header.
    !---------------------------------------------------------------------------
    subroutine read_particle_header(fh)
        implicit none
        integer, intent(in) :: fh
        read(fh) v0
        read(fh) pheader
    end subroutine read_particle_header

    !<--------------------------------------------------------------------------
    !< Set v0 header. Currently, it includes only spatial information.
    !<--------------------------------------------------------------------------
    subroutine set_v0header(nx, ny, nz, x0, y0, z0, dx, dy, dz)
        implicit none
        integer(kind=4), intent(in) :: nx, ny, nz
        real(kind=4), intent(in) :: x0, y0, z0, dx, dy, dz
        v0%nx = nx
        v0%ny = ny
        v0%nz = nz
        v0%x0 = x0
        v0%y0 = y0
        v0%z0 = z0
        v0%dx = dx
        v0%dy = dy
        v0%dz = dz
    end subroutine set_v0header

end module file_header
