!*******************************************************************************
! Program to calculate the bulk flow energy.
!*******************************************************************************
program bulk_flow_energy
    use mpi_module
    use constants, only: fp
    use mpi_topology, only: htg
    use pic_fields, only: ux, uy, uz, open_velocity_field_files, &
            init_velocity_fields, read_velocity_fields, free_velocity_fields, &
            close_velocity_field_files
    use particle_info, only: species, ibtag, get_ptl_mass_charge
    use parameters, only: tp1, tp2
    use analysis_management, only: init_analysis, end_analysis
    use mpi_io_fields, only: save_field
    use statistics, only: get_average_and_total
    implicit none
    real(fp), allocatable, dimension(:, :) :: bulk_energy
    real(fp) :: bene_tot, bene_avg
    integer :: ct

    call init_analysis

    if (myid == master) then
        allocate(bulk_energy(3, tp2-tp1+1))
        bulk_energy = 0.0
    endif

    species = 'e'
    call open_velocity_field_files(species)
    call init_velocity_fields(htg%nx, htg%ny, htg%nz)

    do ct = tp1, tp2
        call read_velocity_fields(ct)
        call get_average_and_total(0.5*ux*ux, bene_avg, bene_tot)
        if (myid == master) then
            bulk_energy(1, ct-tp1+1) = bene_tot
        endif
        call get_average_and_total(0.5*uy*uy, bene_avg, bene_tot)
        if (myid == master) then
            bulk_energy(1, ct-tp1+1) = bene_tot
        endif
        call get_average_and_total(0.5*uz*uz, bene_avg, bene_tot)
        if (myid == master) then
            bulk_energy(1, ct-tp1+1) = bene_tot
        endif
    enddo

    call close_velocity_field_files
    call free_velocity_fields

    if (myid == master) then
        open(unit=62, file='data/bulk_energy_'//species//'.dat', &
            action="write", status="replace")
        do ct = tp1, tp2
            write(62, "(3F14.6)") bulk_energy(:, ct)
        enddo
        close(62)
    endif

    if (myid == master) then
        deallocate(bulk_energy)
    endif

    call end_analysis
end program bulk_flow_energy
