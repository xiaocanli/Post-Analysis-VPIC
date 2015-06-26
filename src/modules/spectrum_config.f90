!*******************************************************************************
! Module of the initial setup, including the energy bins, the maximum and
! minimum of the particle energy, the energy interval (both linear and
! logarithmic), the center of the box (de), the sizes of the box (cells).
!*******************************************************************************
module spectrum_config
    use constants, only: fp
    implicit none
    private
    public nbins, emax, emin, dve, dlogve, spatial_range, center, sizes
    public read_spectrum_config, set_spatial_range_de, calc_pic_mpi_ids, &
           calc_energy_interval
    public corners_mpi, umax, umin, du, nbins_vdist, tframe
    public calc_velocity_interval
    integer :: nbins
    real(fp) :: emax, emin, dve, dlogve
    real(fp) :: umax, umin, du                  ! For velocity distribution.
    integer :: nbins_vdist
    integer :: tframe                           ! Time frame. 
    real(fp), dimension(3) :: center            ! In electron skin length (de).
    real(fp), dimension(3) :: sizes             ! In number of cells.
    real(fp), dimension(2,3) :: spatial_range   ! In electron skin length (de).
    integer, dimension(2,3) :: corners_mpi      ! MPI IDs of the corners.

    contains

    !---------------------------------------------------------------------------
    ! Read the setup information from file.
    !---------------------------------------------------------------------------
    subroutine read_spectrum_config
        use read_config, only: get_variable
        implicit none
        integer :: fh
        real(fp) :: temp
        character(len=32) :: fname
        fh = 10
        if (iargc() > 0) then
            call getarg(1, fname)
        else
            fname = 'config_files/spectrum_config.dat'
        endif
        open(unit=fh, file=fname, status='old')
        temp = get_variable(fh, 'nbins', '=')   ! Number of energy bins
        nbins = int(temp)
        emax = get_variable(fh, 'emax', '=')    ! Maximum energy
        emin = get_variable(fh, 'emin', '=')    ! Minimum energy
        center(1) = get_variable(fh, 'xc/de', '=') ! x-coord of the box center
        center(2) = get_variable(fh, 'yc/de', '=') ! y-coord
        center(3) = get_variable(fh, 'zc/de', '=') ! z-coord
        sizes(1) = get_variable(fh, 'xsize', '=')  ! Number of cells along x
        sizes(2) = get_variable(fh, 'ysize', '=')
        sizes(3) = get_variable(fh, 'zsize', '=')
        temp = get_variable(fh, 'nbins_vdist', '=')
        nbins_vdist = int(temp)
        umax = get_variable(fh, 'umax', '=')
        umin = get_variable(fh, 'umin', '=')
        temp = get_variable(fh, 'tframe', '=')
        tframe = int(temp)
        close(fh)

        call calc_energy_interval
    end subroutine read_spectrum_config

    !---------------------------------------------------------------------------
    ! Calculate the energy interval for each energy bin.
    !---------------------------------------------------------------------------
    subroutine calc_energy_interval
        implicit none
        dve = emax/real(nbins)  ! Linear-scale interval
        dlogve = (log10(emax)-log10(emin))/real(nbins)  ! Logarithmic-scale.
    end subroutine calc_energy_interval

    !---------------------------------------------------------------------------
    ! Calculate velocity integral.
    !---------------------------------------------------------------------------
    subroutine calc_velocity_interval
        implicit none
        du = (umax - umin) / nbins_vdist
    end subroutine calc_velocity_interval

    !---------------------------------------------------------------------------
    ! As the xsize, ysize, zsize are in number of cell, we shall set the spatial
    ! range in electron skin length (de).
    !---------------------------------------------------------------------------
    subroutine set_spatial_range_de
        use picinfo, only: domain
        implicit none
        real(fp) :: dx, dy, dz, lx, ly, lz
        dx = domain%dx
        dy = domain%dy
        dz = domain%dz
        lx = domain%lx_de
        ly = domain%ly_de
        lz = domain%lz_de
        ! x
        spatial_range(1, 1) = center(1) - 0.5*sizes(1)*dx
        spatial_range(2, 1) = center(1) + 0.5*sizes(1)*dx
        if (spatial_range(1, 1) < 0.0) spatial_range(1, 1) = 0.0
        if (spatial_range(2, 1) > lx) spatial_range(2, 1) = lx
        ! y
        spatial_range(1, 2) = center(2) - 0.5*sizes(2)*dy
        spatial_range(2, 2) = center(2) + 0.5*sizes(2)*dy
        if (spatial_range(1, 2) < -ly/2) spatial_range(1, 2) = -ly/2
        if (spatial_range(2, 2) > ly/2) spatial_range(2, 2) = ly/2
        ! z
        spatial_range(1, 3) = center(3) - 0.5*sizes(3)*dz
        spatial_range(2, 3) = center(3) + 0.5*sizes(3)*dz
        if (spatial_range(1, 3) < -lz/2) spatial_range(1, 3) = -lz/2
        if (spatial_range(2, 3) > lz/2) spatial_range(2, 3) = lz/2
    end subroutine set_spatial_range_de

    !---------------------------------------------------------------------------
    ! Calculate the IDs of the MPI processes which contains the bottom-left
    ! and top-right corners of the box. The MPI processes are in the MPI
    ! topology of the PIC simulation.
    !---------------------------------------------------------------------------
    subroutine calc_pic_mpi_ids
        use picinfo, only: domain
        implicit none
        real(fp) :: cx, cy, cz
        cx = center(1) * domain%idx
        cy = center(2) * domain%idy
        cz = center(3) * domain%idz
        corners_mpi(1, 1) = floor((cx-sizes(1))/domain%pic_nx)
        corners_mpi(1, 2) = floor((cy-sizes(2))/domain%pic_ny)
        corners_mpi(1, 3) = floor((cz-sizes(3))/domain%pic_nz)
        if (corners_mpi(1, 1) < 0) corners_mpi(1, 1) = 0
        if (corners_mpi(1, 2) < 0) corners_mpi(1, 2) = 0
        if (corners_mpi(1, 3) < 0) corners_mpi(1, 3) = 0

        corners_mpi(2, 1) = ceiling((cx+sizes(1))/domain%pic_nx)
        corners_mpi(2, 2) = ceiling((cy+sizes(2))/domain%pic_ny)
        corners_mpi(2, 3) = ceiling((cz+sizes(3))/domain%pic_nz)
        if (corners_mpi(2, 1) > domain%pic_tx-1) corners_mpi(2, 1) = domain%pic_tx-1
        if (corners_mpi(2, 2) > domain%pic_ty-1) corners_mpi(2, 2) = domain%pic_ty-1
        if (corners_mpi(2, 3) > domain%pic_tz-1) corners_mpi(2, 3) = domain%pic_tz-1
    end subroutine calc_pic_mpi_ids

end module spectrum_config
