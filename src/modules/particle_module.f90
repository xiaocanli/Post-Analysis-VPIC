!*******************************************************************************
! Module of particle information.
!*******************************************************************************
module particle_module
    use constants, only: fp
    implicit none
    private
    public ptl, ke, px, py, pz, vpara, vperp, gama, igama, vparax, vparay, &
           vparaz, vperpx, vperpy, vperpz, gyrof, vgx, vgy, vgz, vcx, vcy, &
           vcz, ci, cj, ck, iex, jex, kex, iey, jey, key, iez, jez, kez, &
           ibx, jbx, kbx, iby, jby, kby, ibz, jbz, kbz, dx_ex, dy_ex, dz_ex, &
           dx_ey, dy_ey, dz_ey, dx_ez, dy_ez, dz_ez, dx_bx, dx_by, dx_bz, &
           dy_bx, dy_by, dy_bz, dz_bx, dz_by, dz_bz, ino, jno, kno, dnx, dny, &
           dnz
    public calc_particle_energy, calc_ptl_coord, calc_para_perp_velocity, &
           calc_para_perp_velocity_3d, calc_gyrofrequency, &
           calc_gradient_drift_velocity, calc_curvature_drift_velocity, &
           calc_interp_param
    public particle

    type particle
        real(fp) :: dx, dy, dz    ! Particle relative position in a cell [-1,1]
        integer :: icell          ! Index of cell containing the particle
        real(fp) :: vx, vy, vz    ! Particle normalized momentum
        real(fp) :: q
    end type particle

    type(particle) :: ptl
    real(fp) :: ke                ! Kinetic energy
    real(fp) :: gama, igama       ! Lorentz factor and its inverse
    real(fp) :: px, py, pz        ! Particle position
    real(fp) :: vpara, vperp      ! Parallel and perpendicular momentum
    real(fp) :: vparax, vparay, vparaz  ! Parallel velocity
    real(fp) :: vperpx, vperpy, vperpz  ! Perpendicular velocity
    real(fp) :: gyrof          ! Gyrofrequency
    real(fp) :: vgx, vgy, vgz  ! Gradient drift velocity
    real(fp) :: vcx, vcy, vcz  ! Curvature drift velocity
    integer :: ci, cj, ck      ! Cell indices where the particle is in
    integer :: iex, jex, kex   ! The indices of the corner for Ex interpolation
    integer :: iey, jey, key, iez, jez, kez
    integer :: ibx, iby, ibz, jbx, jby, jbz, kbx, kby, kbz
    integer :: ino, jno, kno   ! w.r.t the node
    real(fp) :: dx_ex, dy_ex, dz_ex  ! The offsets for Ex interpolation
    real(fp) :: dx_ey, dy_ey, dz_ey, dx_ez, dy_ez, dz_ez
    real(fp) :: dx_bx, dx_by, dx_bz, dy_bx, dy_by, dy_bz
    real(fp) :: dz_bx, dz_by, dz_bz
    real(fp) :: dnx, dny, dnz

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
        igama = 1.0 / gama
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
        integer :: icell

        ! Including ghost cells
        nx = v0%nx + 2
        ny = v0%ny + 2
        nz = v0%nz + 2

        icell = ptl%icell

        ck = icell / (nx*ny)            ! [1,nz-2]
        cj = (icell-ck*nx*ny) / nx      ! [1,ny-2]
        ci = icell - ck*nx*ny - cj*nx   ! [1,nx-2]

        ! nx, ny, nz include ghost cells. The actual cells start at 1.
        pz = v0%z0 + ((ck-1)+(ptl%dz+1)*0.5) * v0%dz
        py = v0%y0 + ((cj-1)+(ptl%dy+1)*0.5) * v0%dy
        px = v0%x0 + ((ci-1)+(ptl%dx+1)*0.5) * v0%dx
    end subroutine calc_ptl_coord

    !---------------------------------------------------------------------------
    ! Get the parameters for fields interpolation. See the notes for details.
    !---------------------------------------------------------------------------
    subroutine calc_interp_param
        use file_header, only: v0
        implicit none
        integer :: nx, ny, nz, icell
        real(fp) :: dx, dy, dz

        ! Including ghost cells
        nx = v0%nx + 2
        ny = v0%ny + 2
        nz = v0%nz + 2

        icell = ptl%icell

        ck = icell / (nx*ny)            ! [1,nz-2]
        cj = (icell-ck*nx*ny) / nx      ! [1,ny-2]
        ci = icell - ck*nx*ny - cj*nx   ! [1,nx-2]

        ! Particle offset from the center of a cell. [-1, 1]
        dx = ptl%dx
        dy = ptl%dy
        dz = ptl%dz

        ! Ex
        jex = cj
        kex = ck
        dy_ex = 0.5 * (1 + dy)
        dz_ex = 0.5 * (1 + dz)
        ! Ey
        iey = ci
        key = ck
        dx_ey = 0.5 * (1 + dx)
        dz_ey = 0.5 * (1 + dz)
        ! Ez
        iez = ci
        jez = cj
        dx_ez = 0.5 * (1 + dx)
        dy_ez = 0.5 * (1 + dy)
        ! Bx
        ibx = ci
        dx_bx = 0.5 * (1 + dx)
        ! By
        jby = cj
        dy_by = 0.5 * (1 + dy)
        ! Bz
        kbz = ck
        dz_bz = 0.5 * (1 + dz)

        if (dx < 0) then
            iex = ci - 1
            dx_ex = 0.5*dx + 1
            iby = ci - 1
            dx_by = 0.5*dx + 1
            ibz = ci - 1
            dx_bz = 0.5*dx + 1
        else
            iex = ci
            dx_ex = 0.5*dx
            iby = ci
            dx_by = 0.5*dx
            ibz = ci
            dx_bz = 0.5*dx
        endif

        if (dy < 0) then
            jey = cj - 1
            dy_ey = 0.5*dy + 1
            jbx = cj - 1
            dy_bx = 0.5*dy + 1
            jbz = cj - 1
            dy_bz = 0.5*dy + 1
        else
            jey = cj
            dy_ey = 0.5*dy
            jbx = cj
            dy_bx = 0.5*dy
            jbz = cj
            dy_bz = 0.5*dy
        endif

        if (dz < 0) then
            kez = ck - 1
            dz_ez = 0.5*dz + 1
            kbx = ck - 1
            dz_bx = 0.5*dz + 1
            kby = ck - 1
            dz_by = 0.5*dz + 1
        else
            kez = ck
            dz_ez = 0.5*dz
            kbx = ck
            dz_bx = 0.5*dz
            kby = ck
            dz_by = 0.5*dz
        endif

        ! When the fields are on the node
        ino = ci
        jno = cj
        kno = ck
        dnx = (1 + dx) * 0.5
        dny = (1 + dy) * 0.5
        dnz = (1 + dz) * 0.5
    end subroutine calc_interp_param

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

    !---------------------------------------------------------------------------
    ! Calculate the gyrofrequency. The Lorentz factor should be calculated.
    ! The gyrofrequency is charge-dependent, not actually gyrofrequency.
    !---------------------------------------------------------------------------
    subroutine calc_gyrofrequency
        use interpolation_emf, only: absB0
        use particle_info, only: ptl_mass, ptl_charge
        implicit none
        gyrof = ptl_charge * absB0 / (gama * ptl_mass)
    end subroutine calc_gyrofrequency

    !---------------------------------------------------------------------------
    ! Calculate the particle's parallel and perpendicular momentum to the local
    ! magnetic field for 3D simulation.
    !---------------------------------------------------------------------------
    subroutine calc_para_perp_velocity_3d
        use interpolation_emf, only: bxn, byn, bzn
        implicit none
        real(fp) :: vx, vy, vz, vdotb  ! 3-velocity here
        vx = ptl%vx * igama  ! vx in ptl is actually four-velocity
        vy = ptl%vy * igama
        vz = ptl%vz * igama
        vdotb = vx * bxn + vy * byn + vz * bzn
        vparax = vdotb * bxn
        vparay = vdotb * byn
        vparaz = vdotb * bzn
        vpara = vdotb
        vperp = sqrt(vx**2 + vy**2 + vz**2 - vpara**2)
        vperpx = vx - vparax
        vperpy = vy - vparay
        vperpz = vz - vparaz
    end subroutine calc_para_perp_velocity_3d

    !---------------------------------------------------------------------------
    ! Calculate the gradient drift velocity.
    !---------------------------------------------------------------------------
    subroutine calc_gradient_drift_velocity
        use interpolation_emf, only: bxn, byn, bzn, dBdx, dBdy, dBdz, absB0
        implicit none
        real(fp) :: param
        param = vperp * vperp / (2 * gyrof * absB0)
        vgx = param * (byn*dBdz - bzn*dBdy)
        vgy = param * (bzn*dBdx - bxn*dBdz)
        vgz = param * (bxn*dBdy - byn*dBdx)
    end subroutine calc_gradient_drift_velocity

    !---------------------------------------------------------------------------
    ! Calculate the curvature drift velocity.
    !---------------------------------------------------------------------------
    subroutine calc_curvature_drift_velocity
        use interpolation_emf, only: bxn, byn, bzn, kappax, kappay, kappaz
        implicit none
        real(fp) :: param
        param = vpara * vpara / gyrof
        vcx = param * (byn*kappaz - bzn*kappay)
        vcy = param * (bzn*kappax - bxn*kappaz)
        vcz = param * (bxn*kappay - byn*kappax)
    end subroutine calc_curvature_drift_velocity
end module particle_module
