!*******************************************************************************
! Module of particle information.
!*******************************************************************************
module particle_module
    use constants, only: fp
    implicit none
    private
    public ptl, ke, px, py, pz, vpara, vperp, gama
    public calc_particle_energy, calc_ptl_coord, calc_para_perp_velocity

    type particle
        real(fp) :: dx, dy, dz    ! Particle relative position in a cell [-1,1]
        integer :: icell          ! Index of cell containing the particle
        real(fp) :: vx, vy, vz    ! Particle normalized momentum
        real(fp) :: q
    end type particle

    type(particle) :: ptl
    real(fp) :: ke                ! Kinetic energy
    real(fp) :: gama              ! Lorentz factor
    real(fp) :: px, py, pz        ! Particle position
    real(fp) :: vpara, vperp      ! Parallel and perpendicular momentum

    contains

    !---------------------------------------------------------------------------
    ! Read one single particle information, calculate its energy and put it into
    ! the right place of the flux arrays.
    ! Input:
    !   fh: file handler.
    !---------------------------------------------------------------------------
    subroutine calc_particle_energy
        implicit none

        gama = sqrt(1.0 + ptl%vx**2 + ptl%vy**2 + ptl%vz**2)        
        ke = gama - 1.0
    end subroutine calc_particle_energy

    !---------------------------------------------------------------------------
    ! Calculate the particle's coordinates. [0 - lx, -ly/2 - ly/2, -lz/2 - lz/2]
    ! They are in electron skin length (de).
    !---------------------------------------------------------------------------
    subroutine calc_ptl_coord
        use file_header, only: v0
        implicit none
        integer :: nx, ny, nz
        integer :: icell, ix, iy, iz

        ! Including ghost cells
        nx = v0%nx + 2
        ny = v0%ny + 2
        nz = v0%nz + 2

        icell = ptl%icell

        iz = icell / (nx*ny)            ! [1,nz-2]
        iy = (icell-iz*nx*ny) / nx      ! [1,ny-2]
        ix = icell - iz*nx*ny - iy*nx   ! [1,nx-2]

        ! nx, ny, nz include ghost cells. The actual cells start at 1.
        pz = v0%z0 + ((iz-1)+(ptl%dz+1)*0.5) * v0%dz
        py = v0%y0 + ((iy-1)+(ptl%dy+1)*0.5) * v0%dy
        px = v0%x0 + ((ix-1)+(ptl%dx+1)*0.5) * v0%dx
    end subroutine calc_ptl_coord

    !---------------------------------------------------------------------------
    ! Calculate the particle's parallel and perpendicular momentum to the local
    ! magnetic field.
    !---------------------------------------------------------------------------
    subroutine calc_para_perp_velocity
        use picinfo, only: domain
        use magnetic_field, only: bx0, by0, bz0, get_magnetic_field_at_point
        implicit none
        real(fp) :: absB
        real(fp) :: dx, dz
        dx = real(domain%dx, kind=4)
        dz = real(domain%dz, kind=4)
        call get_magnetic_field_at_point(px, pz, dx, dz)
        absB = sqrt(bx0**2 + by0**2 + bz0**2)
        vpara = (ptl%vx*bx0 + ptl%vy*by0 + ptl%vz*bz0) / absB
        vperp = sqrt(ptl%vx**2 + ptl%vy**2 + ptl%vz**2 - vpara**2)
    end subroutine calc_para_perp_velocity

end module particle_module
