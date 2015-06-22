!*******************************************************************************
! The main procedure. 
!*******************************************************************************
program dissipation
    use mpi_module
    use particle_info, only: species, ibtag, get_ptl_mass_charge
    use para_perp_pressure, only: init_para_perp_pressure, &
                                  free_para_perp_pressure
    use analysis_management, only: init_analysis, end_analysis
    use pic_fields, only: open_pic_fields, init_pic_fields, &
                          free_pic_fields, close_pic_fields_file
    implicit none
    integer :: ct

    species = 'e'
    ibtag = '00'
    ct = 1

    call get_ptl_mass_charge(species)
    call init_analysis
    call init_pic_fields
    call init_para_perp_pressure

    call open_pic_fields(species)
    call energy_conversion_from_current

    call free_para_perp_pressure
    call free_pic_fields
    call close_pic_fields_file
    call end_analysis

    contains

    !---------------------------------------------------------------------------
    ! This subroutine calculates the energy conversion through electric current.
    !---------------------------------------------------------------------------
    subroutine energy_conversion_from_current
        use mpi_module
        use parameters, only: icurrent, it1, it2, inductive
        use particle_info, only: ibtag, species
        use pic_fields, only: open_pic_fields, read_pic_fields, &
                close_pic_fields_file
        use pic_fields, only: fields_fh
        use inductive_electric_field, only: open_velocity_field, &
                close_velocity_field, calc_indective_e, &
                init_inductive_electric_field, init_single_fluid_velocity, &
                free_inductive_electric_field, free_single_fluid_velocity
        use previous_post_velocities, only: init_pre_post_velocities, &
                read_pre_post_velocities, free_pre_post_velocities
        use current_densities, only: init_current_densities, &
                free_current_densities, set_current_densities_to_zero
        use jdote_module, only: init_jdote, free_jdote

        implicit none

        integer :: input_record, output_record

        if (inductive == 1) then
            call open_velocity_field(species)
            call init_single_fluid_velocity
            call init_inductive_electric_field
        endif

        if (icurrent == 1) then
            ! Calculate electric current due to all kinds of terms.
            ! And calculate energy conversion due to j.E.
            call init_current_densities
            call init_pre_post_velocities
            call init_jdote
            do input_record = it1, it2
                if (myid==master) print*, input_record
                output_record = input_record - it1 + 1
                call read_pic_fields(input_record)
                if (inductive == 1) then
                    call calc_indective_e(input_record, species)
                endif
                call read_pre_post_velocities(input_record, fields_fh(17:19))
                call calc_energy_conversion(input_record)
                call set_current_densities_to_zero
            enddo
            call free_jdote
            call free_pre_post_velocities
            call free_current_densities
        endif

        if (inductive == 1) then
            call free_inductive_electric_field
            call free_single_fluid_velocity
            call close_velocity_field
        endif
    end subroutine energy_conversion_from_current

    !---------------------------------------------------------------------------
    ! This subroutine calculates energy conversion for one time frame.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine calc_energy_conversion(ct)
        use para_perp_pressure, only: calc_para_perp_pressure
        use current_densities, only: calc_current_densities
        implicit none
        integer, intent(in) :: ct
        !call calc_real_para_perp_pressure(it)
        call calc_para_perp_pressure(ct)
        call calc_current_densities(ct)
    end subroutine calc_energy_conversion

end program dissipation
