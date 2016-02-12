!*******************************************************************************
! Module of jdote for different energy band.
!*******************************************************************************
module jdote_energy_band
    use constants, only: dp, fp
    implicit none
    private
    public read_config_jdote_eband

    integer :: nbands
    real(fp) :: emin, emax, dloge, emin_log, emax_log
    real(fp), allocatable, dimension(:, :, :, :) :: jdote_eband_para
    real(fp), allocatable, dimension(:, :, :, :) :: jdote_eband_perp
    integer, dimension(2, 3) :: mpi_ranges

    contains

    !---------------------------------------------------------------------------
    ! Read the configuration file for jdote for different energy band.
    ! And calculate the logarithmic energy interval.
    !---------------------------------------------------------------------------
    subroutine read_config_jdote_eband
        use read_config, only: get_variable
        implicit none
        integer :: fh
        real(fp) :: temp
        character(len=100) :: fname
        fname = 'config_files/jdote_eband_config.dat'
        fh = 10
        open(unit=fh, file=fname, status='old')
        temp = get_variable(fh, 'nbands', '=')  ! Number of energy bands
        nbands = int(temp)
        emax = get_variable(fh, 'emax', '=')    ! Maximum energy
        emin = get_variable(fh, 'emin', '=')    ! Minimum energy
        close(fh)

        ! Calculate the logarithmic energy interval
        emin_log = log10(emin)
        emax_log = log10(emax)
        dloge = (emax_log - emin_log) / nbands
    end subroutine read_config_jdote_eband

    !---------------------------------------------------------------------------
    ! Initialize the jdote_ebands_para and  jdote_ebands_perp
    !---------------------------------------------------------------------------
    subroutine init_jdote_ebands
        use mpi_topology, only: htg
        implicit none
        integer :: nx, ny, nz
        nx = htg%nx
        ny = htg%ny
        nz = htg%nz
        allocate(jdote_eband_para(nx, ny, nz, nbands))
        allocate(jdote_eband_perp(nx, ny, nz, nbands))
        jdote_eband_para = 0.0
        jdote_eband_perp = 0.0
    end subroutine init_jdote_ebands

    !---------------------------------------------------------------------------
    ! Free the jdote_ebands_para and  jdote_ebands_perp
    !---------------------------------------------------------------------------
    subroutine free_jdote_ebands
        implicit none
        deallocate(jdote_eband_para)
        deallocate(jdote_eband_perp)
    end subroutine free_jdote_ebands

    !---------------------------------------------------------------------------
    ! Calculate the range of VPIC MPI processes for current MPI process.
    !---------------------------------------------------------------------------
    subroutine calc_mpi_ranges
        use picinfo, only: domain
        use mpi_topology, only: ht
        implicit none
        integer :: mx, my, mz
        mpi_ranges = 0
        mx = domain%pic_tx / ht%tx
        my = domain%pic_ty / ht%ty
        mz = domain%pic_tz / ht%tz
        mpi_ranges(1, 1) = mx * ht%ix
        mpi_ranges(2, 1) = mx * (ht%ix + 1) - 1
        mpi_ranges(1, 2) = my * ht%iy
        mpi_ranges(2, 2) = my * (ht%iy + 1) - 1
        mpi_ranges(1, 3) = mz * ht%iz
        mpi_ranges(2, 3) = mz * (ht%iz + 1) - 1
    end subroutine calc_mpi_ranges

    !---------------------------------------------------------------------------
    ! Read particle files and calculate jdote for different energy band.
    !---------------------------------------------------------------------------
    subroutine calc_jdote_eband
        use picinfo, only: domain
        implicit none
        integer :: i, j, k, tx, ty, tz, cid
        tx = domain%pic_tx
        ty = domain%pic_ty
        tz = domain%pic_tz
        do k = mpi_ranges(1, 3), mpi_ranges(2, 3)
            do j = mpi_ranges(1, 2), mpi_ranges(2, 2)
                do i = mpi_ranges(1, 1), mpi_ranges(2, 1)
                enddo
            enddo
        enddo
    end subroutine calc_jdote_eband

end module jdote_energy_band
