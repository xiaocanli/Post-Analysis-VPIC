!*******************************************************************************
! Module of bulk flow velocity for single fluid.
!*******************************************************************************
module usingle
    use constants, only: fp
    implicit none
    private
    public usx, usy, usz    ! s indicates 'single fluid'
    public init_usingle, free_usingle, calc_usingle
    real(fp), allocatable, dimension(:, :, :) :: usx, usy, usz

    contains

    !---------------------------------------------------------------------------
    ! Initialize the bulk flow velocity for single fluid.
    !---------------------------------------------------------------------------
    subroutine init_usingle
        use mpi_topology, only: htg
        implicit none
        allocate(usx(htg%nx, htg%ny, htg%nz))
        allocate(usy(htg%nx, htg%ny, htg%nz))
        allocate(usz(htg%nx, htg%ny, htg%nz))
    end subroutine init_usingle

    !---------------------------------------------------------------------------
    ! Free the bulk flow velocity for single fluid.
    !---------------------------------------------------------------------------
    subroutine free_usingle
        implicit none
        deallocate(usx, usy, usz)
    end subroutine free_usingle

    !---------------------------------------------------------------------------
    ! Calculate the bulk flow velocity for single fluid.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine calc_usingle(ct)
        use picinfo, only: mime
        use pic_fields, only: open_velocity_field_files, read_velocity_fields, &
                close_velocity_field_files, ux, uy, uz 
        implicit none
        integer, intent(in) :: ct
        call open_velocity_field_files('e')
        call read_velocity_fields(ct)
        call close_velocity_field_files
        usx = ux
        usy = uy
        usz = uz

        call open_velocity_field_files('i')
        call read_velocity_fields(ct)
        call close_velocity_field_files
        usx = (usx + ux*mime) / (mime+1)
        usy = (usy + uy*mime) / (mime+1)
        usz = (usz + uz*mime) / (mime+1)
    end subroutine calc_usingle

end module usingle
