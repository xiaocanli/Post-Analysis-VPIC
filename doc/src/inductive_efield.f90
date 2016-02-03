!*******************************************************************************
! This module includes the inductive electric field, and the method to
! calculate it.
!*******************************************************************************
module inductive_electric_field
    use mpi_module
    use constants, only: fp
    use picinfo, only: domain
    implicit none
    private
    public exin, eyin, ezin
    public init_inductive, free_inductive, calc_inductive_e
    public init_inductive_electric_field, free_inductive_electric_field
    real(fp), allocatable, dimension(:,:,:) :: exin, eyin, ezin
    ! Single fluid velocity.
    integer, dimension(3) :: fh_vel     ! File handler for velocity field.
    integer, dimension(3) :: fh_nrho    ! File handler for number density.

    contains

    !---------------------------------------------------------------------------
    ! Initialize the inductive electric field.
    !---------------------------------------------------------------------------
    subroutine init_inductive_electric_field
        use mpi_topology, only: htg
        implicit none
        integer :: nx, ny, nz
        nx = htg%nx
        ny = htg%ny
        nz = htg%nz
        allocate(exin(nx,ny,nz))
        allocate(eyin(nx,ny,nz))
        allocate(ezin(nx,ny,nz))
        exin = 0.0
        eyin = 0.0
        ezin = 0.0
    end subroutine init_inductive_electric_field

    !---------------------------------------------------------------------------
    ! Free the inductive electric field.
    !---------------------------------------------------------------------------
    subroutine free_inductive_electric_field
        implicit none
        deallocate(exin, eyin, ezin)
    end subroutine free_inductive_electric_field

    !---------------------------------------------------------------------------
    ! Initialize the calculation of the inductive electric field.
    ! Input:
    !   species: current particle species. 'e' for electron. 'i' for ion.
    !---------------------------------------------------------------------------
    subroutine init_inductive(species)
        use usingle, only: open_velocity_density_files, init_usingle
        implicit none
        character(len=1), intent(in) :: species
        call init_inductive_electric_field
        call init_usingle(species)
        call open_velocity_density_files(species)
    end subroutine init_inductive

    !---------------------------------------------------------------------------
    ! Finalize the calculation of the inductive electric field.
    ! Input:
    !   species: current particle species. 'e' for electron. 'i' for ion.
    !---------------------------------------------------------------------------
    subroutine free_inductive(species)
        use usingle, only: close_velocity_density_files, free_usingle
        implicit none
        character(len=1), intent(in) :: species
        call free_inductive_electric_field
        call free_usingle(species)
        call close_velocity_density_files(species)
    end subroutine free_inductive

    !---------------------------------------------------------------------------
    ! Calculate the inductive electric field -v*B.
    ! Input:
    !   ct: current time point. It is from tp1 to tp2.
    !   species: particle species. 'e' for electron. 'i' for ion.
    !---------------------------------------------------------------------------
    subroutine calc_inductive_e(ct, species)
        use mpi_module
        use parameters, only: tp1
        use pic_fields, only: bx, by, bz
        use picinfo, only: domain, mime
        use mpi_datatype_fields, only: filetype_ghost, subsizes_ghost
        use mpi_io_module, only: read_data_mpi_io
        use usingle, only: calc_usingle, vsx, vsy, vsz
        implicit none
        integer, intent(in) :: ct
        character(*), intent(in) :: species

        call calc_usingle(species)

        exin = by*vsz - bz*vsy
        eyin = bz*vsx - bx*vsz
        ezin = bx*vsy - by*vsx
    end subroutine calc_inductive_e

end module inductive_electric_field
