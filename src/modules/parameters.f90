!*******************************************************************************
! User defined parameters for analysis.
!*******************************************************************************
module parameters
    implicit none
    private
    public tp1, tp2, inductive
    public get_start_end_time_points, get_inductive_flag

    integer :: tp1, tp2  ! Starting and ending time points for analysis.
    integer :: inductive


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
            close(fh)
        endif

        call MPI_BCAST(inductive, 1, MPI_INTEGER, master, MPI_COMM_WORLD, ierr)

    end subroutine get_inductive_flag

end module parameters
