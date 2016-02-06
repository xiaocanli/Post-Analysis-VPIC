!*******************************************************************************
! Get the total number of particles in each energy band. 
!*******************************************************************************
program particle_number_eband
    use mpi_module
    use constants, only: fp, dp
    use picinfo, only: domain
    use mpi_topology, only: htg
    use particle_info, only: species, ibtag, get_ptl_mass_charge
    use analysis_management, only: init_analysis, end_analysis
    use pic_fields, only: open_fraction_eband_file, init_fraction_eband, &
            free_fraction_eband, close_fraction_eband_file, &
            read_fraction_eband, eb
    use statistics, only: get_average_and_total
    use parameters, only: tp1, tp2
    implicit none
    character(len=150) :: fname
    integer, parameter :: nbands = 5
    real(dp), allocatable, dimension(:, :) :: rho_band_sum
    real(fp) :: rho_band_tot, rho_band_avg
    character(len=20) :: format1
    integer :: iband, ct

    call init_analysis
    species = 'e'
    call init_fraction_eband(htg%nx, htg%ny, htg%nz)

    if (myid == master) then
        allocate(rho_band_sum(tp2-tp1+1, nbands))
        rho_band_sum = 0.0
    endif

    do iband = 1, nbands
        if (myid == master) print*, 'Energy band: ', iband
        call open_fraction_eband_file(species, iband)
        do ct = tp1, tp2
            call read_fraction_eband(ct)
            call get_average_and_total(eb, rho_band_avg, rho_band_tot)
            rho_band_avg = rho_band_avg / (domain%dx*domain%dy*domain%dx)
            if (myid == master) then
                rho_band_sum(ct-tp1+1, iband) = rho_band_avg
            endif
        end do
        call close_fraction_eband_file
    end do

    call MPI_BARRIER(MPI_COMM_WORLD, ierror)
    if (myid == master) then
        open(unit=62, file='data/rho_eband_'//species//'.dat', &
            action="write", status="replace")
        do ct = tp1, tp2
            do iband = 1, nbands
                write(62, '(F14.6)', advance="no") rho_band_sum(ct-tp1+1, iband)
            enddo
            write(62, *)
        enddo
        close(62)
    endif

    if (myid == master) then
        deallocate(rho_band_sum)
    endif

    call free_fraction_eband
    call end_analysis

end program particle_number_eband
