!*******************************************************************************
! Module of particle information.
!*******************************************************************************
module particle_module
    use constants, only: fp
    implicit none
    private
    public ptl, ke, px, py, pz
    public calc_particle_energy, calc_ptl_coord

    type particle
        real(fp) :: dx, dy, dz    ! Particle relative position in a cell [-1,1]
        integer :: icell          ! Index of cell containing the particle
        real(fp) :: ux, uy, uz    !  Particle normalized momentum
        real(fp) :: q
    end type particle

    type(particle) :: ptl
    real(fp) :: ke  ! Kinetic energy
    real(fp) :: px, py, pz

    contains

    !---------------------------------------------------------------------------
    ! Read one single particle information, calculate its energy and put it into
    ! the right place of the flux arrays.
    ! Input:
    !   fh: file handler.
    !---------------------------------------------------------------------------
    subroutine calc_particle_energy
        implicit none
        real(fp) :: gama

        gama = sqrt(1.0 + ptl%ux**2 + ptl%uy**2 + ptl%uz**2)        
        ke = gama - 1.0
    end subroutine calc_particle_energy

    !---------------------------------------------------------------------------
    ! Calculate the particle's coordinates. [0 - lx, -ly/2 - ly/2, -lz/2 - lz/2]
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

end module particle_module
