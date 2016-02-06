!*******************************************************************************
! This module gives the parameters and routines for analysis dealing with
! particle information along a field line, e.g. particle spectrum along a
! field line or particle velocity distribution along a field line.
!*******************************************************************************
module particle_fieldline
    use mpi_module
    use constants, only: fp
    implicit none
    private
    public init_analysis, end_analysis, get_fieldline_points
    public nptot, np, startp, endp

    integer :: nptot    ! The actual number of points along the field line.
    integer :: np       ! Number of points for current MPI process.
    integer :: startp, endp  ! Starting and ending points.

    contains

    !---------------------------------------------------------------------------
    ! Initialize this analysis
    ! Input:
    !   ct: current time frame for fields.
    !---------------------------------------------------------------------------
    subroutine init_analysis(ct)
        use path_info, only: get_file_paths
        use picinfo, only: read_domain, broadcast_pic_info
        use particle_frames, only: get_particle_frames, tinterval
        use fieldline_tracing, only: init_fieldline_tracing, &
                Dormand_Prince_parameters, init_fieldline_points
        use magnetic_field, only: read_magnetic_fields
        use parameters, only: get_start_end_time_points, get_inductive_flag, &
                get_relativistic_flag
        implicit none
        integer, intent(in) :: ct

        call MPI_INIT(ierr)
        call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
        call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)

        call get_file_paths
        call validate_time_frame(ct)

        ! The PIC simulation information.
        if (myid==master) then
            call read_domain
            call get_particle_frames
        endif
        call MPI_BCAST(tinterval, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
        call broadcast_pic_info
        call get_start_end_time_points
        call get_inductive_flag
        call get_relativistic_flag

        ! Get the information for field line tracing.
        call init_fieldline_tracing
        call init_fieldline_points
        call Dormand_Prince_parameters
        call read_magnetic_fields(ct)

    end subroutine init_analysis

    !---------------------------------------------------------------------------
    ! Trace a field line starting at one point, save the points along the field
    ! line, and distribute the coordinates of the points for further analysis.
    ! Inputs:
    !   x0, z0: the coordinates of the starting point.
    !---------------------------------------------------------------------------
    subroutine get_fieldline_points(x0, z0)
        use fieldline_tracing, only: npoints, trace_field_line
        use mpi_topology, only: distribute_tasks
        implicit none
        real(fp), intent(in) :: x0, z0
        ! Set the tasks for each MPI process.
        call trace_field_line(x0, z0)  ! Recored npoints at the same time.
        nptot = npoints
        call distribute_tasks(nptot, numprocs, myid, np, startp, endp)
    end subroutine get_fieldline_points

    !---------------------------------------------------------------------------
    ! End the analysis.
    !---------------------------------------------------------------------------
    subroutine end_analysis
        use fieldline_tracing, only: end_fieldline_tracing, free_fieldline_points
        implicit none
        call free_fieldline_points
        call end_fieldline_tracing
        call MPI_FINALIZE(ierr)
    end subroutine end_analysis

    !---------------------------------------------------------------------------
    ! Check if both particle and fields exist at current time frame. If not,
    ! the analysis is ended and an error message is given.
    !---------------------------------------------------------------------------
    subroutine validate_time_frame(ct)
        use particle_file, only: check_both_particle_fields_exist, &
                get_ratio_interval, ratio_interval
        implicit none
        integer, intent(in) :: ct
        logical :: is_time_valid
        ! Get the ratio of the particle output and field output.
        if (myid == master) then
            call get_ratio_interval
        endif
        call MPI_BCAST(ratio_interval, 1, MPI_INTEGER, master, &
                MPI_COMM_WORLD, ierr)

        ! Check whether the time frame is valid for both fields and particles.
        is_time_valid = check_both_particle_fields_exist(ct)
        if (.not. is_time_valid) then
            if (myid == master) then
                write(*, '(A,I0,A)') 'ct = ', ct, ' is invalid.'
                write(*, '(A,I0)') 'Choose a time that is a multiple of ', &
                        ratio_interval
            endif
            call MPI_FINALIZE(ierr)
            stop
        endif
    end subroutine validate_time_frame

end module particle_fieldline
