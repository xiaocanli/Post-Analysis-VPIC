!*******************************************************************************
! User defined parameters for analysis.
!*******************************************************************************
module parameters
    implicit none
    private
    public tp1, tp2, inductive, is_rel, is_only_emf
    public get_start_end_time_points, get_inductive_flag, &
        get_relativistic_flag, get_emf_flag

    integer :: tp1, tp2  ! Starting and ending time points for analysis.
    integer :: inductive, is_rel, is_only_emf


    contains


    !---------------------------------------------------------------------------
    ! Read starting and ending time points from the configuration file.
    !---------------------------------------------------------------------------
    subroutine get_start_end_time_points
        use mpi_module
        use constants, only: fp
        use read_config, only: get_variable
        implicit none
        integer :: fh
        real(fp) :: temp

        fh = 10
        ! Read the configuration file
        if (myid==master) then
            open(unit=fh, file='config_files/analysis_config.dat', &
                 form='formatted', status='old')
            temp = get_variable(fh, 'tp1', '=')
            tp1 = int(temp)
            temp = get_variable(fh, 'tp2', '=')
            tp2 = int(temp)
            close(fh)
            write(*, "(A,I0,A,I0)") &
                " Starting and ending time point for this analysis: ", &
                tp1, ', ', tp2
        endif

        call MPI_BCAST(tp1, 1, MPI_INTEGER, master, MPI_COMM_WORLD, ierr)
        call MPI_BCAST(tp2, 1, MPI_INTEGER, master, MPI_COMM_WORLD, ierr)
        
    end subroutine get_start_end_time_points

    !---------------------------------------------------------------------------
    ! Read the variable 'inductive' to decide whether inductive electric field
    ! is used. '1' for yes, '0' for no.
    !---------------------------------------------------------------------------
    subroutine get_inductive_flag
        use mpi_module
        use constants, only: fp
        use read_config, only: get_variable
        implicit none
        integer :: fh
        real(fp) :: temp

        fh = 10
        ! Read the configuration file
        if (myid==master) then
            open(unit=fh, file='config_files/analysis_config.dat', &
                 form='formatted', status='old')
            temp = get_variable(fh, 'inductive', '=')
            inductive = int(temp)
            if (inductive == 1) then
                write(*, "(A)") ' Using motional electric field.'
            else
                write(*, "(A)") ' Using total electric field.'
            endif
            close(fh)
        endif

        call MPI_BCAST(inductive, 1, MPI_INTEGER, master, MPI_COMM_WORLD, ierr)

    end subroutine get_inductive_flag

    !---------------------------------------------------------------------------
    ! Read from the configuration file whether to use relativistic forms of
    ! calculation.
    !---------------------------------------------------------------------------
    subroutine get_relativistic_flag
        use mpi_module
        use constants, only: fp
        use read_config, only: get_variable
        implicit none
        integer :: fh
        real(fp) :: temp

        fh = 10
        ! Read the configuration file
        if (myid==master) then
            open(unit=fh, file='config_files/analysis_config.dat', &
                 form='formatted', status='old')
            temp = get_variable(fh, 'is_rel', '=')
            is_rel = int(temp)
            if (is_rel == 1) then
                write(*, "(A)") ' Using relativistic fields to do analysis.'
            else
                write(*, "(A)") ' Using nonrelativistic fields to do analysis.'
            endif
            close(fh)
        endif

        call MPI_BCAST(is_rel, 1, MPI_INTEGER, master, MPI_COMM_WORLD, ierr)

    end subroutine get_relativistic_flag

    !---------------------------------------------------------------------------
    ! Read from the configuration file whether only EMF are saved. The others
    ! include, for example, div_e_err and div_b_err. Including additional
    ! fields will change the dump files structures.
    !---------------------------------------------------------------------------
    subroutine get_emf_flag
        use mpi_module
        use constants, only: fp
        use read_config, only: get_variable
        implicit none
        integer :: fh
        real(fp) :: temp

        fh = 10
        ! Read the configuration file
        if (myid==master) then
            open(unit=fh, file='config_files/analysis_config.dat', &
                 form='formatted', status='old')
            temp = get_variable(fh, 'is_only_emf', '=')
            is_only_emf = int(temp)
            if (is_only_emf == 1) then
                write(*, "(A)") ' Only electromagnetic fiels are dumped.'
            else
                write(*, "(A)") ' Additional fields other than EMF are dumped.'
            endif
            close(fh)
        endif

        call MPI_BCAST(is_only_emf, 1, MPI_INTEGER, master, MPI_COMM_WORLD, ierr)

    end subroutine get_emf_flag

end module parameters
