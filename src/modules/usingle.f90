!*******************************************************************************
! Module of bulk flow velocity for single fluid.
!*******************************************************************************
module usingle
    use constants, only: fp
    implicit none
    private
    public vsx, vsy, vsz    ! s indicates 'single fluid'
    public init_usingle, free_usingle, calc_usingle, &
           open_velocity_density_files, close_velocity_density_files
    real(fp), allocatable, dimension(:, :, :) :: vsx, vsy, vsz
    real(fp), allocatable, dimension(:, :, :) :: vx, vy, vz
    real(fp), allocatable, dimension(:, :, :) :: nrho_a, nrho_b
    integer, dimension(3) :: fh_vel     ! File handler for velocity field.
    integer, dimension(3) :: fh_vel_b   ! For the other species.
    integer :: fh_nrho    ! File handler for number density.
    integer :: fh_nrho_b  ! For the other species.

    interface init_usingle
        module procedure &
            init_usingle_s, init_usingle_b
    end interface init_usingle

    interface free_usingle
        module procedure &
            free_usingle_s, free_usingle_b
    end interface free_usingle

    interface open_velocity_density_files
        module procedure &
            open_velocity_density_files_s, open_velocity_density_files_b
    end interface open_velocity_density_files

    interface close_velocity_density_files
        module procedure &
            close_velocity_density_files_s, close_velocity_density_files_b
    end interface close_velocity_density_files

    interface calc_usingle
        module procedure &
            calc_usingle_s, calc_usingle_b
    end interface calc_usingle

    contains

    !---------------------------------------------------------------------------
    ! Initialize the bulk flow velocity for single fluid.
    !---------------------------------------------------------------------------
    subroutine init_usingle_s
        use mpi_topology, only: htg
        implicit none
        allocate(vsx(htg%nx, htg%ny, htg%nz))
        allocate(vsy(htg%nx, htg%ny, htg%nz))
        allocate(vsz(htg%nx, htg%ny, htg%nz))
        allocate(nrho_a(htg%nx, htg%ny, htg%nz))
    end subroutine init_usingle_s

    subroutine init_usingle_b(species)
        use mpi_topology, only: htg
        implicit none
        character(*), intent(in) :: species
        allocate(vsx(htg%nx, htg%ny, htg%nz))
        allocate(vsy(htg%nx, htg%ny, htg%nz))
        allocate(vsz(htg%nx, htg%ny, htg%nz))
        allocate(vx(htg%nx, htg%ny, htg%nz))
        allocate(vy(htg%nx, htg%ny, htg%nz))
        allocate(vz(htg%nx, htg%ny, htg%nz))
        allocate(nrho_a(htg%nx, htg%ny, htg%nz))
        allocate(nrho_b(htg%nx, htg%ny, htg%nz))
    end subroutine init_usingle_b

    !---------------------------------------------------------------------------
    ! Free the bulk flow velocity for single fluid.
    !---------------------------------------------------------------------------
    subroutine free_usingle_s
        implicit none
        deallocate(vsx, vsy, vsz)
        deallocate(nrho_a)
    end subroutine free_usingle_s

    subroutine free_usingle_b(species)
        implicit none
        character(*), intent(in) :: species
        deallocate(vsx, vsy, vsz)
        deallocate(vx, vy, vz)
        deallocate(nrho_a, nrho_b)
    end subroutine free_usingle_b

    !---------------------------------------------------------------------------
    ! Open the data files of velocity fields and number density for the other
    ! species. e.g. when the current species is electron, this procedure will
    ! open velocity files for ions.
    ! Inputs:
    !   species: particle species. 'e' for electron. 'i' for ion.
    !---------------------------------------------------------------------------
    subroutine open_velocity_density_files_s(species)
        use path_info, only: filepath
        use mpi_info_module, only: fileinfo
        use pic_fields, only: open_velocity_field_files, ufields_fh, &
                open_number_density_file, nrho_fh
        implicit none
        character(*), intent(in) :: species
        character(len=100) :: fname
        character(len=1) :: species_other
        if (species == 'e') then
            species_other = 'i'
        else
            species_other = 'e'
        endif
        fh_vel = 0
        fh_nrho = 0
        call open_velocity_field_files(species_other)
        call open_number_density_file(species_other)
        fh_vel = ufields_fh
        fh_nrho = nrho_fh
    end subroutine open_velocity_density_files_s

    !---------------------------------------------------------------------------
    ! Open the data files of velocity fields and number density for both
    ! species. 
    !---------------------------------------------------------------------------
    subroutine open_velocity_density_files_b
        use path_info, only: filepath
        use mpi_info_module, only: fileinfo
        use pic_fields, only: open_velocity_field_files, ufields_fh, &
                open_number_density_file, nrho_fh
        implicit none
        character(len=100) :: fname
        fh_vel = 0
        fh_nrho = 0
        fh_vel_b = 0
        fh_nrho_b = 0
        ! Electron
        call open_velocity_field_files('e')
        call open_number_density_file('e')
        fh_vel = ufields_fh
        fh_nrho = nrho_fh

        ! Ion
        call open_velocity_field_files('i')
        call open_number_density_file('i')
        fh_vel_b = ufields_fh
        fh_nrho_b = nrho_fh
    end subroutine open_velocity_density_files_b

    !---------------------------------------------------------------------------
    ! Close the data file of velocity field and number density for one species.
    ! Input:
    !   species: particle species. 'e' for electron. 'i' for ion.
    !---------------------------------------------------------------------------
    subroutine close_velocity_density_files_s(species)
        use mpi_module
        implicit none
        character(*), intent(in) :: species
        integer :: i
        do i = 1, 3
            call MPI_FILE_CLOSE(fh_vel(i), ierror)
        enddo
        call MPI_FILE_CLOSE(fh_nrho)
    end subroutine close_velocity_density_files_s

    !---------------------------------------------------------------------------
    ! Close the data file of velocity field and number density for both species.
    !---------------------------------------------------------------------------
    subroutine close_velocity_density_files_b
        use mpi_module
        implicit none
        integer :: i
        do i = 1, 3
            call MPI_FILE_CLOSE(fh_vel(i), ierror)
            call MPI_FILE_CLOSE(fh_vel_b(i), ierror)
        enddo
        call MPI_FILE_CLOSE(fh_nrho)
        call MPI_FILE_CLOSE(fh_nrho_b)
    end subroutine close_velocity_density_files_b

    !---------------------------------------------------------------------------
    ! Read the velocity and density for the other species.
    ! Input:
    !   ct: current time frame.
    !   species: particle species. 'e' for electron. 'i' for ion.
    !---------------------------------------------------------------------------
    subroutine read_velocity_density_s(ct, species)
        use mpi_module
        use parameters, only: tp1
        use picinfo, only: domain, mime
        use mpi_io_module, only: read_data_mpi_io
        use mpi_datatype_fields, only: filetype_ghost, subsizes_ghost
        implicit none
        integer, intent(in) :: ct
        character(*), intent(in) :: species
        integer(kind=MPI_OFFSET_KIND) :: disp, offset

        disp = domain%nx * domain%ny * domain%nz * sizeof(MPI_REAL) * (ct-tp1)
        offset = 0 
        call read_data_mpi_io(fh_vel(1), filetype_ghost, subsizes_ghost, &
            disp, offset, vsx)
        call read_data_mpi_io(fh_vel(2), filetype_ghost, subsizes_ghost, &
            disp, offset, vsy)
        call read_data_mpi_io(fh_vel(3), filetype_ghost, subsizes_ghost, &
            disp, offset, vsz)
        call read_data_mpi_io(fh_nrho, filetype_ghost, subsizes_ghost, &
            disp, offset, nrho_a)
    end subroutine read_velocity_density_s

    !---------------------------------------------------------------------------
    ! Read the velocity and density for both species.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine read_velocity_density_b(ct)
        use mpi_module
        use parameters, only: tp1
        use picinfo, only: domain, mime
        use mpi_io_module, only: read_data_mpi_io
        use mpi_datatype_fields, only: filetype_ghost, subsizes_ghost
        implicit none
        integer, intent(in) :: ct
        integer(kind=MPI_OFFSET_KIND) :: disp, offset

        disp = domain%nx * domain%ny * domain%nz * sizeof(MPI_REAL) * (ct-tp1)
        offset = 0 
        ! Electron
        call read_data_mpi_io(fh_vel(1), filetype_ghost, subsizes_ghost, &
            disp, offset, vsx)
        call read_data_mpi_io(fh_vel(2), filetype_ghost, subsizes_ghost, &
            disp, offset, vsy)
        call read_data_mpi_io(fh_vel(3), filetype_ghost, subsizes_ghost, &
            disp, offset, vsz)
        call read_data_mpi_io(fh_nrho, filetype_ghost, subsizes_ghost, &
            disp, offset, nrho_a)
        ! Ion
        call read_data_mpi_io(fh_vel_b(1), filetype_ghost, subsizes_ghost, &
            disp, offset, vx)
        call read_data_mpi_io(fh_vel_b(2), filetype_ghost, subsizes_ghost, &
            disp, offset, vy)
        call read_data_mpi_io(fh_vel_b(3), filetype_ghost, subsizes_ghost, &
            disp, offset, vz)
        call read_data_mpi_io(fh_nrho, filetype_ghost, subsizes_ghost, &
            disp, offset, nrho_b)
    end subroutine read_velocity_density_b

    !---------------------------------------------------------------------------
    ! Calculate the bulk flow velocity of single fluid, when the velocity is
    ! known for one species.
    !---------------------------------------------------------------------------
    subroutine calc_usingle_s(species)
        use picinfo, only: mime
        use pic_fields, only: vx, vy, vz, num_rho
        implicit none
        character(*), intent(in) :: species
        if (species == 'e') then
            vsx = (vsx*mime*nrho_a + vx*num_rho) / (mime*nrho_a + num_rho)
            vsy = (vsy*mime*nrho_a + vy*num_rho) / (mime*nrho_a + num_rho)
            vsz = (vsz*mime*nrho_a + vz*num_rho) / (mime*nrho_a + num_rho)
        else
            vsx = (vsx*nrho_a + vx*mime*num_rho) / (mime*num_rho + nrho_a)
            vsy = (vsy*nrho_a + vy*mime*num_rho) / (mime*num_rho + nrho_a)
            vsz = (vsz*nrho_a + vz*mime*num_rho) / (mime*num_rho + nrho_a)
        endif
    end subroutine calc_usingle_s

    !---------------------------------------------------------------------------
    ! Calculate the bulk flow velocity of single fluid, when the velocities are
    ! known for both species.
    !---------------------------------------------------------------------------
    subroutine calc_usingle_b
        use picinfo, only: mime
        use pic_fields, only: vx, vy, vz, num_rho
        implicit none
        vsx = (vsx*nrho_a + vx*mime*nrho_b) / (mime*nrho_b + nrho_a)
        vsy = (vsy*nrho_a + vy*mime*nrho_b) / (mime*nrho_b + nrho_a)
        vsz = (vsz*nrho_a + vz*mime*nrho_b) / (mime*nrho_b + nrho_a)
    end subroutine calc_usingle_b

end module usingle
