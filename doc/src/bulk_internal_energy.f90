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
        use pic_fields, only: vx, vy, vz, ux, uy, uz, pxx, pyy, pzz, num_rho, &
                open_velocity_field_files, open_pressure_tensor_files, &
                init_velocity_fields, init_pressure_tensor, &
                read_velocity_fields, read_pressure_tensor, &
                free_velocity_fields, free_pressure_tensor, &
                close_velocity_field_files, close_pressure_tensor_files, &
                open_number_density_file, init_number_density, &
                read_number_density, free_number_density, &
                close_number_density_file
        use particle_info, only: species, ibtag, ptl_mass
        use parameters, only: tp1, tp2, is_rel
        use mpi_io_fields, only: save_field
        use statistics, only: get_average_and_total
        implicit none
        real(fp), allocatable, dimension(:, :) :: bulk_energy, internal_energy
        real(fp), dimension(4) :: bene_tot, iene_tot
        real(fp) :: avg
        logical :: dir_e
        integer :: ct

        bene_tot = 0.0  ! Bulk energy
        iene_tot = 0.0  ! Internal energy
        if (myid == master) then
            allocate(bulk_energy(4, tp2-tp1+1))
            allocate(internal_energy(4, tp2-tp1+1))
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
            if (is_rel == 0) then
                call get_average_and_total(0.5*vx*vx*ptl_mass*num_rho, &
                        avg, bene_tot(1))
                call get_average_and_total(0.5*vy*vy*ptl_mass*num_rho, &
                        avg, bene_tot(2))
                call get_average_and_total(0.5*vz*vz*ptl_mass*num_rho, &
                        avg, bene_tot(3))
                bene_tot(4) = sum(bene_tot(1:3))
            else
                call get_average_and_total(ux*ux*ptl_mass**2*num_rho**2, &
                        avg, bene_tot(1))
                call get_average_and_total(uy*uy*ptl_mass**2*num_rho**2, &
                        avg, bene_tot(2))
                call get_average_and_total(uz*uz*ptl_mass**2*num_rho**2, &
                        avg, bene_tot(3))
                call get_average_and_total(&
                    (sqrt(1.0+ux*ux+uy*uy+uz*uz) - 1.0)*ptl_mass*num_rho, &
                    avg, bene_tot(4))
            endif
            call get_average_and_total(0.5*pxx, avg, iene_tot(1))
            call get_average_and_total(0.5*pyy, avg, iene_tot(2))
            call get_average_and_total(0.5*pzz, avg, iene_tot(3))
            iene_tot(4) = sum(iene_tot(1:3))
            if (myid == master) then
                bulk_energy(1:4, ct-tp1+1) = bene_tot
                internal_energy(1:4, ct-tp1+1) = iene_tot
            endif
        enddo

        call close_number_density_file
        call close_pressure_tensor_files
        call close_velocity_field_files
        call free_number_density
        call free_velocity_fields
        call free_pressure_tensor

        if (myid == master) then
            dir_e = .false.
            inquire(file='./data/.', exist=dir_e)
            if (.not. dir_e) then
                call system('mkdir ./data')
            endif

            open(unit=62, file='data/bulk_internal_energy_'//species//'.dat', &
                action="write", status="replace")
            do ct = tp1, tp2
                write(62, "(8e20.6)") bulk_energy(:, ct), internal_energy(:, ct)
            enddo
            close(62)
        endif

        if (myid == master) then
            deallocate(bulk_energy)
            deallocate(internal_energy)
        endif
    end subroutine commit_analysis

end program bulk_flow_energy
