!*******************************************************************************
! Program to calculate the bulk flow energy and internal energy.
!*******************************************************************************
program bulk_flow_energy
    use particle_info, only: species, get_ptl_mass_charge
    use analysis_management, only: init_analysis, end_analysis
    implicit none
    call init_analysis

    species = 'e'
    call get_ptl_mass_charge(species)
    call commit_analysis
    species = 'i'
    call get_ptl_mass_charge(species)
    call commit_analysis

    call end_analysis

    contains

    !---------------------------------------------------------------------------
    ! Commit analysis.
    !---------------------------------------------------------------------------
    subroutine commit_analysis
        use mpi_module
        use constants, only: fp
        use mpi_topology, only: htg
        use pic_fields, only: ux, uy, uz, pxx, pyy, pzz, num_rho, &
                open_velocity_field_files, open_pressure_tensor_files, &
                init_velocity_fields, init_pressure_tensor, &
                read_velocity_fields, read_pressure_tensor, &
                free_velocity_fields, free_pressure_tensor, &
                close_velocity_field_files, close_pressure_tensor_files, &
                open_number_density_file, init_number_density, &
                read_number_density, free_number_density, &
                close_number_density_file
        use particle_info, only: species, ibtag, ptl_mass
        use parameters, only: tp1, tp2
        use mpi_io_fields, only: save_field
        use statistics, only: get_average_and_total
        implicit none
        real(fp), allocatable, dimension(:, :) :: bulk_energy, internal_energy
        real(fp) :: bene_tot, bene_avg, iene_tot, iene_avg
        integer :: ct

        if (myid == master) then
            allocate(bulk_energy(3, tp2-tp1+1))
            allocate(internal_energy(3, tp2-tp1+1))
            bulk_energy = 0.0
            internal_energy = 0.0
        endif

        call open_velocity_field_files(species)
        call open_pressure_tensor_files(species)
        call open_number_density_file(species)
        call init_velocity_fields(htg%nx, htg%ny, htg%nz)
        call init_pressure_tensor(htg%nx, htg%ny, htg%nz)
        call init_number_density(htg%nx, htg%ny, htg%nz)

        do ct = tp1, tp2
            if (myid == master) then
                print*, ct
            endif
            call read_velocity_fields(ct)
            call read_pressure_tensor(ct)
            call read_number_density(ct)
            call get_average_and_total(0.5*ux*ux*ptl_mass*num_rho, &
                    bene_avg, bene_tot)
            call get_average_and_total(0.5*pxx, iene_avg, iene_tot)
            if (myid == master) then
                bulk_energy(1, ct-tp1+1) = bene_tot
                internal_energy(1, ct-tp1+1) = iene_tot
            endif
            call get_average_and_total(0.5*uy*uy*ptl_mass*num_rho, &
                    bene_avg, bene_tot)
            call get_average_and_total(0.5*pyy, iene_avg, iene_tot)
            if (myid == master) then
                bulk_energy(2, ct-tp1+1) = bene_tot
                internal_energy(2, ct-tp1+1) = iene_tot
            endif
            call get_average_and_total(0.5*uz*uz*ptl_mass*num_rho, &
                    bene_avg, bene_tot)
            call get_average_and_total(0.5*pzz, iene_avg, iene_tot)
            if (myid == master) then
                bulk_energy(3, ct-tp1+1) = bene_tot
                internal_energy(3, ct-tp1+1) = iene_tot
            endif
        enddo

        call close_number_density_file
        call close_pressure_tensor_files
        call close_velocity_field_files
        call free_number_density
        call free_velocity_fields
        call free_pressure_tensor

        if (myid == master) then
            open(unit=62, file='data/bulk_internal_energy_'//species//'.dat', &
                action="write", status="replace")
            do ct = tp1, tp2
                write(62, "(6F14.6)") bulk_energy(:, ct), internal_energy(:, ct)
            enddo
            close(62)
        endif

        if (myid == master) then
            deallocate(bulk_energy)
            deallocate(internal_energy)
        endif
    end subroutine commit_analysis

end program bulk_flow_energy
