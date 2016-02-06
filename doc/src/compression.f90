!*******************************************************************************
! The main procedure to calculate the compressional and shear heating terms.
!*******************************************************************************
program compression
    use particle_info, only: species, ibtag
    use analysis_management, only: init_analysis, end_analysis
    implicit none

    call init_analysis

    ibtag = '00'
    species = 'e'
    call commit_analysis
    species = 'i'
    call commit_analysis

    call end_analysis

    contains

    !---------------------------------------------------------------------------
    ! Doing the analysis for one species.
    !---------------------------------------------------------------------------
    subroutine commit_analysis
        use mpi_module
        use mpi_topology, only: htg
        use particle_info, only: species, ibtag, get_ptl_mass_charge
        use para_perp_pressure, only: init_para_perp_pressure, &
                free_para_perp_pressure, calc_para_perp_pressure
        use pic_fields, only: open_pic_fields, init_pic_fields, &
                free_pic_fields, close_pic_fields_file, &
                read_pic_fields
        use saving_flags, only: get_saving_flags
        use neighbors_module, only: init_neighbors, free_neighbors, get_neighbors
        use compression_shear, only: init_compression_shear, &
                free_compression_shear, calc_compression_shear, &
                save_compression_shear, save_tot_compression_shear
        use pressure_tensor, only: init_scalar_pressure, init_div_ptensor, &
                free_scalar_pressure, free_div_ptensor, calc_scalar_pressure, &
                calc_div_ptensor
        use parameters, only: tp1, tp2
        implicit none
        integer :: input_record, output_record

        call get_ptl_mass_charge(species)

        call init_pic_fields
        call init_para_perp_pressure
        call get_saving_flags

        call open_pic_fields(species)

        call init_neighbors(htg%nx, htg%ny, htg%nz)
        call get_neighbors

        call init_scalar_pressure
        call init_div_ptensor
        call init_compression_shear
        do input_record = tp1, tp2
            if (myid==master) print*, input_record
            output_record = input_record - tp1 + 1
            call read_pic_fields(input_record)
            call calc_para_perp_pressure(input_record)
            call calc_scalar_pressure
            call calc_div_ptensor
            call calc_compression_shear
            call save_compression_shear(input_record)
            call save_tot_compression_shear(input_record)
        enddo
        call free_compression_shear
        call free_div_ptensor
        call free_scalar_pressure

        call free_neighbors
        call free_para_perp_pressure
        call free_pic_fields
        call close_pic_fields_file
    end subroutine commit_analysis

end program compression
