!*******************************************************************************
! Parser for the commandline arguments.
!*******************************************************************************
module commandline_arguments
    use mpi_module
    use kinds
    use cla
    implicit none
    private
    public is_species, is_config_dist, is_emax_cell
    public get_cmdline_arguments
    logical :: is_species, is_config_dist, is_emax_cell
    
    contains

    !---------------------------------------------------------------------------
    ! Get the commandline arguments and update corresponding variables.
    !---------------------------------------------------------------------------
    subroutine get_cmdline_arguments
        implicit none
        character(len=STRLEN) :: config_dist
        call cla_init
        call get_particle_species
        call get_config_dist
        call get_emax_cell_flag
    end subroutine get_cmdline_arguments

    !---------------------------------------------------------------------------
    ! Get particle species.
    ! Note: the species_local should be 'e' for electron and 'i' for ion.
    !---------------------------------------------------------------------------
    subroutine get_particle_species
        use particle_info, only: species
        implicit none
        character(len=STRLEN) :: species_local
        logical :: log1, log2
        species_local = 'e'
        call cla_register('-s', '--species', 'character', cla_char, &
            species_local)
        log1 = cla_key_present('-s')
        log2 = cla_key_present('--species')
        is_species = log1 .or. log2
        call cla_get('-s', species_local)
        if (is_species) then
            species = species_local
            if (myid == master) then
                print *,"---------------------------------------------------"
                if (species == 'e') then
                    write(*, '(A)') ' Particle species is electron'
                else
                    write(*, '(A)') ' Particle species is ion'
                endif
            endif
        else
            if (myid == master) then
                print *,"---------------------------------------------------"
                write(*, '(A)') ' Doing analysis for both species'
            endif
        endif
    end subroutine get_particle_species

    !---------------------------------------------------------------------------
    ! Get configuration filename for particle energy distribution or velocity
    ! distributions, since they share the same configuration file.
    !---------------------------------------------------------------------------
    subroutine get_config_dist
        use spectrum_config, only: config_name
        implicit none
        character(len=STRLEN) :: config_name_arg
        logical :: log1, log2
        config_name_arg = 'config_files/spectrum_config.dat'
        config_name = config_name_arg
        call cla_register('-c', '--config_dist', 'character', cla_char, &
            config_name_arg)
        log1 = cla_key_present('-c')
        log2 = cla_key_present('--config_dist')
        is_config_dist = log1 .or. log2
        call cla_get('-c', config_name_arg)
        if (is_config_dist) then
            config_name = config_name_arg
        endif
        if (myid == master) then
            write(*, '(A)') ' The configuration file for spectrum and velocity'
            write(*, '(A,A)') ' distributions is ', config_name
        endif
    end subroutine get_config_dist

    !---------------------------------------------------------------------------
    ! Get flag on whether to get the maximum energy for each cell.
    !---------------------------------------------------------------------------
    subroutine get_emax_cell_flag
        implicit none
        logical :: log1, log2
        is_emax_cell = .False.
        call cla_register('-e', '--emax', 'character', cla_flag, 'f')
        log1 = cla_key_present('-e')
        log2 = cla_key_present('--emax')
        is_emax_cell = log1 .or. log2
        if (is_emax_cell) then
            if (myid == master) then
                print *,"---------------------------------------------------"
                write(*, '(A)') ' Get the maximum energy for each cell'
            endif
        endif
    end subroutine get_emax_cell_flag

end module commandline_arguments
