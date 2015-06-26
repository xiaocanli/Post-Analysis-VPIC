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
    ! Function to the value of one variable with name var_name.
    ! Inputs:
    !   fh: file handler.
    !   var_name: the variable name.
    !   delimiter: the delimiter. The value of the variable is after the delimiter.
    ! Returns:
    !   var_value: the variable value.
    !---------------------------------------------------------------------------
    function get_variable(fh, var_name, delimiter) result(var_value)
        use constants, only: fp
        implicit none
        integer, intent(in) :: fh
        character(*), intent(in) :: var_name, delimiter
        real(fp) :: var_value
        character(len=150) :: single_line
        do while (index(single_line, var_name) == 0)
            read(10, '(A)') single_line
        enddo
        read(single_line(index(single_line, delimiter)+1:), *) var_value
    end function

    !---------------------------------------------------------------------------
    ! Read starting and ending time points from the configuration file.
    !---------------------------------------------------------------------------
    subroutine get_start_end_time_points
        use mpi_module
        use constants, only: fp
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
