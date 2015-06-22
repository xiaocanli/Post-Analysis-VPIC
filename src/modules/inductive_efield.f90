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
    public exin, eyin, ezin, utotx, utoty, utotz
    public open_velocity_field, close_velocity_field, calc_indective_e
    public init_inductive_electric_field, free_inductive_electric_field
    public init_single_fluid_velocity, free_single_fluid_velocity
    real(fp), allocatable, dimension(:,:,:) :: exin, eyin, ezin
    ! Single fluid velocity.
    real(fp), allocatable, dimension(:,:,:) :: utotx, utoty, utotz
    integer, dimension(3) :: fh_vel     ! File handler for velocity field.

    contains

    !---------------------------------------------------------------------------
    ! Initialize the inductive electric field.
    !---------------------------------------------------------------------------
    subroutine init_inductive_electric_field
        implicit none
        integer :: nx, ny, nz
        nx = domain%nx
        ny = domain%ny
        nz = domain%nz
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
    ! Initialize the velocity field for single fluid.
    !---------------------------------------------------------------------------
    subroutine init_single_fluid_velocity
        implicit none
        integer :: nx, ny, nz
        nx = domain%nx
        ny = domain%ny
        nz = domain%nz
        allocate(utotx(nx,ny,nz))
        allocate(utoty(nx,ny,nz))
        allocate(utotz(nx,ny,nz))
        utotx = 0.0
        utoty = 0.0
        utotz = 0.0
    end subroutine init_single_fluid_velocity

    !---------------------------------------------------------------------------
    ! Free the velocity field for single fluid.
    !---------------------------------------------------------------------------
    subroutine free_single_fluid_velocity
        implicit none
        deallocate(utotx, utoty, utotz)
    end subroutine free_single_fluid_velocity

    !---------------------------------------------------------------------------
    ! Open the data files of velocity fields for the other species.
    ! e.g. when the current species is electron, this procedure will open
    ! velocity files for ions.
    ! Inputs:
    !   species: particle species.
    !---------------------------------------------------------------------------
    subroutine open_velocity_field(species)
        use path_info, only: filepath
        use mpi_info_object, only: fileinfo
        use mpi_io_module, only: open_data_mpi_io
        implicit none
        character(*), intent(in) :: species
        character(len=100) :: fname
        character(len=1) :: species_other
        if (species == 'e') then
            species_other = 'i'
        else
            species_other = 'e'
        endif
        fname = trim(adjustl(filepath))//'u'//species_other//'x'//'.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, fh_vel(1))
        fname = trim(adjustl(filepath))//'u'//species_other//'y'//'.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, fh_vel(2))
        fname = trim(adjustl(filepath))//'u'//species_other//'z'//'.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, fh_vel(3))
    end subroutine open_velocity_field

    !---------------------------------------------------------------------------
    ! Close the data files of velocity fields for the other species.
    !---------------------------------------------------------------------------
    subroutine close_velocity_field
        implicit none
        integer :: i
        do i = 1, 3
            call MPI_FILE_CLOSE(fh_vel(i), ierror)
        enddo
    end subroutine close_velocity_field

    !---------------------------------------------------------------------------
    ! Calculate the inductive electric field -v*B.
    ! Input:
    !   ct: current time point. It is from it1 to it2.
    !   species: particle species. 'e' for electron. 'i' for ion.
    !---------------------------------------------------------------------------
    subroutine calc_indective_e(ct, species)
        use mpi_module
        use parameters, only: it1
        use pic_fields, only: ux, uy, uz, bx, by, bz
        use picinfo, only: domain, mime
        use mpi_datatype, only: filetype_ghost, subsizes_ghost
        use mpi_io_module, only: read_data_mpi_io
        implicit none
        integer, intent(in) :: ct
        character(*), intent(in) :: species
        integer(kind=MPI_OFFSET_KIND) :: disp, offset

        disp = domain%nx * domain%ny * domain%nz * sizeof(MPI_REAL) * (ct-it1)
        offset = 0 
        call read_data_mpi_io(fh_vel(1), filetype_ghost, subsizes_ghost, &
            disp, offset, utotx)
        call read_data_mpi_io(fh_vel(2), filetype_ghost, subsizes_ghost, &
            disp, offset, utoty)
        call read_data_mpi_io(fh_vel(3), filetype_ghost, subsizes_ghost, &
            disp, offset, utotz)
        if (species == 'e') then
            utotx = (utotx*mime + ux) / (mime + 1)
            utoty = (utoty*mime + uy) / (mime + 1)
            utotz = (utotz*mime + uz) / (mime + 1)
        else
            utotx = (utotx + ux*mime) / (mime + 1)
            utoty = (utoty + uy*mime) / (mime + 1)
            utotz = (utotz + uz*mime) / (mime + 1)
        endif

        exin = by*utotz - bz*utoty
        eyin = bz*utotx - bx*utotz
        ezin = bx*utoty - by*utotx
    end subroutine calc_indective_e

end module inductive_electric_field
