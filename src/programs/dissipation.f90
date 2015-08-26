!*******************************************************************************
! The main procedure. 
!*******************************************************************************
program dissipation
    use mpi_module
    use particle_info, only: species, ibtag, get_ptl_mass_charge
    use analysis_management, only: init_analysis, end_analysis
    implicit none
    integer :: ct

    ibtag = '00'
    ct = 1

    call init_analysis

    species = 'e'
    call commit_analysis

    call MPI_BARRIER(MPI_COMM_WORLD, ierror)

    species = 'i'
    call commit_analysis

    call end_analysis

    contains

    !---------------------------------------------------------------------------
    ! This subroutine does the analysis.
    !---------------------------------------------------------------------------
    subroutine commit_analysis
        use particle_info, only: species
        use para_perp_pressure, only: init_para_perp_pressure, &
                free_para_perp_pressure, save_averaged_para_perp_pressure
        use pic_fields, only: open_pic_fields, init_pic_fields, &
                free_pic_fields, close_pic_fields_file
        use saving_flags, only: get_saving_flags
        use neighbors_module, only: init_neighbors, free_neighbors, get_neighbors
        use compression_shear, only: init_div_v, free_div_v
        implicit none

        call get_ptl_mass_charge(species)
        call init_pic_fields
        call init_para_perp_pressure
        call init_div_v  ! For compression related current density.
        call get_saving_flags

        call open_pic_fields(species)

        call init_neighbors
        call get_neighbors

        call energy_conversion_from_current

        call free_neighbors
        call free_para_perp_pressure
        call free_pic_fields
        call close_pic_fields_file
        call free_div_v
    end subroutine commit_analysis

    !---------------------------------------------------------------------------
    ! This subroutine calculates the energy conversion through electric current.
    !---------------------------------------------------------------------------
    subroutine energy_conversion_from_current
        use mpi_module
        use parameters, only: tp1, tp2, inductive, is_rel
        use particle_info, only: ibtag, species
        use pic_fields, only: open_pic_fields, read_pic_fields, &
                close_pic_fields_file
        use pic_fields, only: vfields_fh, ufields_fh
        use inductive_electric_field, only: calc_inductive_e, &
                init_inductive, free_inductive
        use previous_post_velocities, only: init_pre_post_velocities, &
                read_pre_post_velocities, free_pre_post_velocities
        use current_densities, only: init_current_densities, &
                free_current_densities, set_current_densities_to_zero, &
                init_ava_current_densities, free_avg_current_densities, &
                save_averaged_current
        use para_perp_pressure, only: save_averaged_para_perp_pressure
        use jdote_module, only: init_jdote, free_jdote, &
                init_jdote_total, free_jdote_total, save_jdote_total

        implicit none

        integer :: input_record, output_record

        if (inductive == 1) then
            call init_inductive(species)
        endif

        ! Calculate electric current due to all kinds of terms.
        ! And calculate energy conversion due to j.E.
        call init_current_densities
        call init_ava_current_densities
        call init_pre_post_velocities
        call init_jdote
        call init_jdote_total
        do input_record = tp1, tp2
            if (myid==master) print*, input_record
            output_record = input_record - tp1 + 1
            call read_pic_fields(input_record)
            if (inductive == 1) then
                call calc_inductive_e(input_record, species)
            endif
            if (is_rel == 1) then
                call read_pre_post_velocities(input_record, ufields_fh)
            else
                call read_pre_post_velocities(input_record, vfields_fh)
            endif
            call calc_energy_conversion(input_record)
            call set_current_densities_to_zero
        enddo

        if (myid == master) then
            call save_averaged_current
            call save_jdote_total
            call save_averaged_para_perp_pressure
        endif

        call free_jdote_total
        call free_jdote
        call free_pre_post_velocities
        call free_avg_current_densities
        call free_current_densities

        if (inductive == 1) then
            call free_inductive(species)
        endif
    end subroutine energy_conversion_from_current

    !---------------------------------------------------------------------------
    ! This subroutine calculates energy conversion for one time frame.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine calc_energy_conversion(ct)
        use para_perp_pressure, only: calc_para_perp_pressure, &
                calc_real_para_perp_pressure
        use current_densities, only: calc_current_densities
        implicit none
        integer, intent(in) :: ct
        call calc_real_para_perp_pressure(ct)
        ! call calc_para_perp_pressure(ct)
        call calc_current_densities(ct)
    end subroutine calc_energy_conversion

end program dissipation
