!*******************************************************************************
! Module of fields from particle-in-cell simulations. This module include the
! subroutines for opening and closing the files, initialisation, reading and
! free the field data.
!*******************************************************************************
module pic_fields
    use mpi_module
    use constants, only: fp
    use parameters, only: tp1
    use picinfo, only: domain
    use mpi_topology, only: htg
    use mpi_io_module, only: open_data_mpi_io, read_data_mpi_io
    use mpi_datatype_fields, only: filetype_ghost, subsizes_ghost
    use path_info, only: filepath
    use mpi_info_module, only: fileinfo
    implicit none
    private
    public init_pic_fields, open_pic_fields, read_pic_fields, &
           free_pic_fields, close_pic_fields_file
    public init_magnetic_fields, init_electric_fields, init_current_densities, &
           init_pressure_tensor, init_velocity_fields, init_number_density, &
           init_fraction_eband
    public open_magnetic_field_files, open_electric_field_files, &
           open_current_density_files, open_pressure_tensor_files, &
           open_velocity_field_files, open_number_density_file, &
           open_fraction_eband_file
    public read_mangeitc_fields, read_electric_fields, read_current_desities, &
           read_pressure_tensor, read_velocity_fields, read_number_density, &
           read_fraction_eband
    public close_magnetic_field_files, close_electric_field_files, &
           close_current_density_files, close_pressure_tensor_files, &
           close_velocity_field_files, close_number_density_file, &
           close_fraction_eband_file
    public free_magnetic_fields, free_electric_fields, free_current_densities, &
           free_pressure_tensor, free_velocity_fields, free_number_density, &
           free_fraction_eband

    public bx, by, bz, ex, ey, ez, absB  ! Electromagnetic fields
    public pxx, pxy, pxz, pyy, pyz, pzz  ! Pressure tensor
    public vx, vy, vz, num_rho           ! Bulk flow velocity and number density
    public jx, jy, jz                    ! Current density for single fluid
    public eb
    ! File handlers
    public bfields_fh, efields_fh, pre_fh, vfields_fh, jfields_fh, nrho_fh
    public eband_fh

    real(fp), allocatable, dimension(:,:,:) :: bx, by, bz, ex, ey, ez, absB
    real(fp), allocatable, dimension(:,:,:) :: pxx, pxy, pxz, pyy, pyz, pzz
    real(fp), allocatable, dimension(:,:,:) :: vx, vy, vz, num_rho
    real(fp), allocatable, dimension(:,:,:) :: jx, jy, jz
    real(fp), allocatable, dimension(:,:,:) :: eb
    integer, dimension(4) :: bfields_fh
    integer, dimension(3) :: efields_fh, vfields_fh, jfields_fh
    integer, dimension(6) :: pre_fh
    integer :: nrho_fh, eband_fh

    contains

    !---------------------------------------------------------------------------
    ! Initialize the current densities
    !---------------------------------------------------------------------------
    subroutine init_current_densities(nx, ny, nz)
        implicit none
        integer, intent(in) :: nx, ny, nz
        allocate(jx(nx,ny,nz))
        allocate(jy(nx,ny,nz))
        allocate(jz(nx,ny,nz))
        jx = 0.0; jy = 0.0; jz = 0.0
    end subroutine init_current_densities

    !---------------------------------------------------------------------------
    ! Initialize the electric fields.
    !---------------------------------------------------------------------------
    subroutine init_electric_fields(nx, ny, nz)
        implicit none
        integer, intent(in) :: nx, ny, nz
        allocate(ex(nx,ny,nz))
        allocate(ey(nx,ny,nz))
        allocate(ez(nx,ny,nz))
        ex = 0.0; ey = 0.0; ez = 0.0
    end subroutine init_electric_fields

    !---------------------------------------------------------------------------
    ! Initialize the magnetic fields.
    !---------------------------------------------------------------------------
    subroutine init_magnetic_fields(nx, ny, nz)
        implicit none
        integer, intent(in) :: nx, ny, nz
        allocate(bx(nx,ny,nz))
        allocate(by(nx,ny,nz))
        allocate(bz(nx,ny,nz))
        allocate(absB(nx,ny,nz))
        bx = 0.0; by = 0.0; bz = 0.0
        absB = 0.0
    end subroutine init_magnetic_fields

    !---------------------------------------------------------------------------
    ! Initialize the magnetic fields.
    !---------------------------------------------------------------------------
    subroutine init_pressure_tensor(nx, ny, nz)
        implicit none
        integer, intent(in) :: nx, ny, nz
        allocate(pxx(nx,ny,nz))
        allocate(pxy(nx,ny,nz))
        allocate(pxz(nx,ny,nz))
        allocate(pyy(nx,ny,nz))
        allocate(pyz(nx,ny,nz))
        allocate(pzz(nx,ny,nz))
        pxx = 0.0; pyy = 0.0; pzz = 0.0
        pxy = 0.0; pxz = 0.0; pyz = 0.0
    end subroutine init_pressure_tensor

    !---------------------------------------------------------------------------
    ! Initialize the velocity fields.
    !---------------------------------------------------------------------------
    subroutine init_velocity_fields(nx, ny, nz)
        implicit none
        integer, intent(in) :: nx, ny, nz
        allocate(vx(nx,ny,nz))
        allocate(vy(nx,ny,nz))
        allocate(vz(nx,ny,nz))
        vx = 0.0; vy = 0.0; vz = 0.0
    end subroutine init_velocity_fields

    !---------------------------------------------------------------------------
    ! Initialize the number density.
    !---------------------------------------------------------------------------
    subroutine init_number_density(nx, ny, nz)
        implicit none
        integer, intent(in) :: nx, ny, nz
        allocate(num_rho(nx,ny,nz))
        num_rho = 0.0
    end subroutine init_number_density

    !---------------------------------------------------------------------------
    ! Initialize the fraction of particle in one energy band.
    !---------------------------------------------------------------------------
    subroutine init_fraction_eband(nx, ny, nz)
        implicit none
        integer, intent(in) :: nx, ny, nz
        allocate(eb(nx,ny,nz))
        eb = 0.0
    end subroutine init_fraction_eband

    !---------------------------------------------------------------------------
    ! Initialization of the fields arrays from PIC simulation outputs.
    !---------------------------------------------------------------------------
    subroutine init_pic_fields
        use mpi_topology, only: htg
        implicit none
        integer :: nx, ny, nz
        nx = htg%nx
        ny = htg%ny
        nz = htg%nz
        call init_current_densities(nx, ny, nz)
        call init_electric_fields(nx, ny, nz)
        call init_magnetic_fields(nx, ny, nz)
        call init_velocity_fields(nx, ny, nz)
        call init_pressure_tensor(nx, ny, nz)
        call init_number_density(nx, ny, nz)
    end subroutine init_pic_fields

    !---------------------------------------------------------------------------
    ! Read magnetic field.
    !---------------------------------------------------------------------------
    subroutine read_mangeitc_fields(ct)
        implicit none
        integer, intent(in) :: ct
        integer(kind=MPI_OFFSET_KIND) :: disp, offset
        disp = domain%nx * domain%ny * domain%nz * sizeof(MPI_REAL) * (ct-tp1)
        offset = 0 
        call read_data_mpi_io(bfields_fh(1), filetype_ghost, &
            subsizes_ghost, disp, offset, bx)
        call read_data_mpi_io(bfields_fh(2), filetype_ghost, &
            subsizes_ghost, disp, offset, by)
        call read_data_mpi_io(bfields_fh(3), filetype_ghost, &
            subsizes_ghost, disp, offset, bz)
        call read_data_mpi_io(bfields_fh(4), filetype_ghost, &
            subsizes_ghost, disp, offset, absB)
    end subroutine read_mangeitc_fields
    
    !---------------------------------------------------------------------------
    ! Read electric field.
    !---------------------------------------------------------------------------
    subroutine read_electric_fields(ct)
        implicit none
        integer, intent(in) :: ct
        integer(kind=MPI_OFFSET_KIND) :: disp, offset
        disp = domain%nx * domain%ny * domain%nz * sizeof(MPI_REAL) * (ct-tp1)
        offset = 0 
        call read_data_mpi_io(efields_fh(1), filetype_ghost, &
            subsizes_ghost, disp, offset, ex)
        call read_data_mpi_io(efields_fh(2), filetype_ghost, &
            subsizes_ghost, disp, offset, ey)
        call read_data_mpi_io(efields_fh(3), filetype_ghost, &
            subsizes_ghost, disp, offset, ez)
    end subroutine read_electric_fields

    !---------------------------------------------------------------------------
    ! Read current densities.
    !---------------------------------------------------------------------------
    subroutine read_current_desities(ct)
        implicit none
        integer, intent(in) :: ct
        integer(kind=MPI_OFFSET_KIND) :: disp, offset
        disp = domain%nx * domain%ny * domain%nz * sizeof(MPI_REAL) * (ct-tp1)
        offset = 0 
        call read_data_mpi_io(jfields_fh(1), filetype_ghost, &
            subsizes_ghost, disp, offset, jx)
        call read_data_mpi_io(jfields_fh(2), filetype_ghost, &
            subsizes_ghost, disp, offset, jy)
        call read_data_mpi_io(jfields_fh(3), filetype_ghost, &
            subsizes_ghost, disp, offset, jz)
    end subroutine read_current_desities

    !---------------------------------------------------------------------------
    ! Read pressure tensor.
    !---------------------------------------------------------------------------
    subroutine read_pressure_tensor(ct)
        implicit none
        integer, intent(in) :: ct
        integer(kind=MPI_OFFSET_KIND) :: disp, offset
        disp = domain%nx * domain%ny * domain%nz * sizeof(MPI_REAL) * (ct-tp1)
        offset = 0 
        call read_data_mpi_io(pre_fh(1), filetype_ghost, &
            subsizes_ghost, disp, offset, pxx)
        call read_data_mpi_io(pre_fh(2), filetype_ghost, &
            subsizes_ghost, disp, offset, pxy)
        call read_data_mpi_io(pre_fh(3), filetype_ghost, &
            subsizes_ghost, disp, offset, pxz)
        call read_data_mpi_io(pre_fh(4), filetype_ghost, &
            subsizes_ghost, disp, offset, pyy)
        call read_data_mpi_io(pre_fh(5), filetype_ghost, &
            subsizes_ghost, disp, offset, pyz)
        call read_data_mpi_io(pre_fh(6), filetype_ghost, &
            subsizes_ghost, disp, offset, pzz)
    end subroutine read_pressure_tensor

    !---------------------------------------------------------------------------
    ! Read velocity field.
    !---------------------------------------------------------------------------
    subroutine read_velocity_fields(ct)
        implicit none
        integer, intent(in) :: ct
        integer(kind=MPI_OFFSET_KIND) :: disp, offset
        disp = domain%nx * domain%ny * domain%nz * sizeof(MPI_REAL) * (ct-tp1)
        offset = 0 
        call read_data_mpi_io(vfields_fh(1), filetype_ghost, &
            subsizes_ghost, disp, offset, vx)
        call read_data_mpi_io(vfields_fh(2), filetype_ghost, &
            subsizes_ghost, disp, offset, vy)
        call read_data_mpi_io(vfields_fh(3), filetype_ghost, &
            subsizes_ghost, disp, offset, vz)
    end subroutine read_velocity_fields

    !---------------------------------------------------------------------------
    ! Read number density.
    !---------------------------------------------------------------------------
    subroutine read_number_density(ct)
        implicit none
        integer, intent(in) :: ct
        integer(kind=MPI_OFFSET_KIND) :: disp, offset
        disp = domain%nx * domain%ny * domain%nz * sizeof(MPI_REAL) * (ct-tp1)
        offset = 0 
        call read_data_mpi_io(nrho_fh, filetype_ghost, &
            subsizes_ghost, disp, offset, num_rho)
    end subroutine read_number_density

    !---------------------------------------------------------------------------
    ! Read the fraction of particles in each energy band.
    !---------------------------------------------------------------------------
    subroutine read_fraction_eband(ct)
        implicit none
        integer, intent(in) :: ct
        integer(kind=MPI_OFFSET_KIND) :: disp, offset
        disp = domain%nx * domain%ny * domain%nz * sizeof(MPI_REAL) * (ct-tp1)
        offset = 0 
        call read_data_mpi_io(eband_fh, filetype_ghost, &
            subsizes_ghost, disp, offset, eb)
    end subroutine read_fraction_eband

    !---------------------------------------------------------------------------
    ! Read PIC simulation fields.
    ! Input:
    !   ct: current time point.
    !---------------------------------------------------------------------------
    subroutine read_pic_fields(ct)
        implicit none
        integer, intent(in) :: ct
        call read_mangeitc_fields(ct)
        call read_electric_fields(ct)
        call read_current_desities(ct)
        call read_pressure_tensor(ct)
        call read_velocity_fields(ct)
        call read_number_density(ct)
    end subroutine read_pic_fields

    !---------------------------------------------------------------------------
    ! Open magnetic field files.
    !---------------------------------------------------------------------------
    subroutine open_magnetic_field_files
        implicit none
        character(len=100) :: fname
        bfields_fh = 0
        fname = trim(adjustl(filepath))//'bx.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, bfields_fh(1))
        fname = trim(adjustl(filepath))//'by.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, bfields_fh(2))
        fname = trim(adjustl(filepath))//'bz.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, bfields_fh(3))
        fname = trim(adjustl(filepath))//'absB.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, bfields_fh(4))
    end subroutine open_magnetic_field_files

    !---------------------------------------------------------------------------
    ! Open electric field files.
    !---------------------------------------------------------------------------
    subroutine open_electric_field_files
        implicit none
        character(len=100) :: fname
        efields_fh = 0
        fname = trim(adjustl(filepath))//'ex.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, efields_fh(1))
        fname = trim(adjustl(filepath))//'ey.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, efields_fh(2))
        fname = trim(adjustl(filepath))//'ez.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, efields_fh(3))
    end subroutine open_electric_field_files

    !---------------------------------------------------------------------------
    ! Open current density files.
    !---------------------------------------------------------------------------
    subroutine open_current_density_files
        implicit none
        character(len=100) :: fname
        jfields_fh = 0
        fname = trim(adjustl(filepath))//'jx.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, jfields_fh(1))
        fname = trim(adjustl(filepath))//'jy.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, jfields_fh(2))
        fname = trim(adjustl(filepath))//'jz.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, jfields_fh(3))
    end subroutine open_current_density_files

    !---------------------------------------------------------------------------
    ! Open press tensor files.
    !---------------------------------------------------------------------------
    subroutine open_pressure_tensor_files(species)
        implicit none
        character(*), intent(in) :: species
        character(len=100) :: fname
        pre_fh = 0
        fname = trim(adjustl(filepath))//'p'//species//'-xx.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, pre_fh(1))
        fname = trim(adjustl(filepath))//'p'//species//'-xy.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, pre_fh(2))
        fname = trim(adjustl(filepath))//'p'//species//'-xz.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, pre_fh(3))
        fname = trim(adjustl(filepath))//'p'//species//'-yy.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, pre_fh(4))
        fname = trim(adjustl(filepath))//'p'//species//'-yz.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, pre_fh(5))
        fname = trim(adjustl(filepath))//'p'//species//'-zz.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, pre_fh(6))
    end subroutine open_pressure_tensor_files

    !---------------------------------------------------------------------------
    ! Open velocity field files.
    !---------------------------------------------------------------------------
    subroutine open_velocity_field_files(species)
        implicit none
        character(*), intent(in) :: species
        character(len=100) :: fname
        vfields_fh = 0
        fname = trim(adjustl(filepath))//'u'//species//'x.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, vfields_fh(1))
        fname = trim(adjustl(filepath))//'u'//species//'y.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, vfields_fh(2))
        fname = trim(adjustl(filepath))//'u'//species//'z.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, vfields_fh(3))
    end subroutine open_velocity_field_files

    !---------------------------------------------------------------------------
    ! Open number density file.
    !---------------------------------------------------------------------------
    subroutine open_number_density_file(species)
        implicit none
        character(*), intent(in) :: species
        character(len=100) :: fname
        nrho_fh = 0
        fname = trim(adjustl(filepath))//'n'//species//'.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, nrho_fh)
    end subroutine open_number_density_file

    !---------------------------------------------------------------------------
    ! Open the file of the fraction of particle in one energy band.
    !---------------------------------------------------------------------------
    subroutine open_fraction_eband_file(species, iband)
        implicit none
        character(*), intent(in) :: species
        integer, intent(in) :: iband
        character(len=100) :: fname
        character(len=2) :: tag_band
        eband_fh = 0
        write(tag_band, '(I2.2)') iband
        fname = trim(adjustl(filepath))//species//'EB'//tag_band//'.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, eband_fh)
    end subroutine open_fraction_eband_file

    !---------------------------------------------------------------------------
    ! Open PIC fields file collectively using MPI procedures.
    !---------------------------------------------------------------------------
    subroutine open_pic_fields(species)
        implicit none
        character(*), intent(in) :: species

        call open_magnetic_field_files
        call open_electric_field_files
        call open_current_density_files
        call open_pressure_tensor_files(species)
        call open_velocity_field_files(species)
        call open_number_density_file(species)
    end subroutine open_pic_fields

    !---------------------------------------------------------------------------
    ! Free magnetic fields.
    !---------------------------------------------------------------------------
    subroutine free_magnetic_fields
        implicit none
        deallocate(bx, by, bz, absB)
    end subroutine free_magnetic_fields

    !---------------------------------------------------------------------------
    ! Free electric fields.
    !---------------------------------------------------------------------------
    subroutine free_electric_fields
        implicit none
        deallocate(ex, ey, ez)
    end subroutine free_electric_fields

    !---------------------------------------------------------------------------
    ! Free current densities.
    !---------------------------------------------------------------------------
    subroutine free_current_densities
        implicit none
        deallocate(jx, jy, jz)
    end subroutine free_current_densities

    !---------------------------------------------------------------------------
    ! Free pressure tensor.
    !---------------------------------------------------------------------------
    subroutine free_pressure_tensor
        implicit none
        deallocate(pxx, pxy, pxz, pyy, pyz, pzz)
    end subroutine free_pressure_tensor

    !---------------------------------------------------------------------------
    ! Free velocity fields.
    !---------------------------------------------------------------------------
    subroutine free_velocity_fields
        implicit none
        deallocate(vx, vy, vz)
    end subroutine free_velocity_fields

    !---------------------------------------------------------------------------
    ! Free number density.
    !---------------------------------------------------------------------------
    subroutine free_number_density
        implicit none
        deallocate(num_rho)
    end subroutine free_number_density

    !---------------------------------------------------------------------------
    ! Free the fraction of particle for each energy band.
    !---------------------------------------------------------------------------
    subroutine free_fraction_eband
        implicit none
        deallocate(eb)
    end subroutine free_fraction_eband

    !---------------------------------------------------------------------------
    ! Free the memory used by the PIC fields.
    !---------------------------------------------------------------------------
    subroutine free_pic_fields
        implicit none
        call free_magnetic_fields
        call free_electric_fields
        call free_current_densities
        call free_pressure_tensor
        call free_velocity_fields
        call free_number_density
    end subroutine free_pic_fields

    !---------------------------------------------------------------------------
    ! Close magnetic field files.
    !---------------------------------------------------------------------------
    subroutine close_magnetic_field_files
        implicit none
        call MPI_FILE_CLOSE(bfields_fh(1), ierror)
        call MPI_FILE_CLOSE(bfields_fh(2), ierror)
        call MPI_FILE_CLOSE(bfields_fh(3), ierror)
        call MPI_FILE_CLOSE(bfields_fh(4), ierror)
    end subroutine close_magnetic_field_files

    !---------------------------------------------------------------------------
    ! Close electric field files.
    !---------------------------------------------------------------------------
    subroutine close_electric_field_files
        implicit none
        call MPI_FILE_CLOSE(efields_fh(1), ierror)
        call MPI_FILE_CLOSE(efields_fh(2), ierror)
        call MPI_FILE_CLOSE(efields_fh(3), ierror)
    end subroutine close_electric_field_files

    !---------------------------------------------------------------------------
    ! Close current density files.
    !---------------------------------------------------------------------------
    subroutine close_current_density_files
        implicit none
        call MPI_FILE_CLOSE(jfields_fh(1), ierror)
        call MPI_FILE_CLOSE(jfields_fh(2), ierror)
        call MPI_FILE_CLOSE(jfields_fh(3), ierror)
    end subroutine close_current_density_files

    !---------------------------------------------------------------------------
    ! Close pressure tensor files.
    !---------------------------------------------------------------------------
    subroutine close_pressure_tensor_files
        implicit none
        integer :: i
        do i = 1, 6
            call MPI_FILE_CLOSE(pre_fh(i), ierror)
        end do
    end subroutine close_pressure_tensor_files

    !---------------------------------------------------------------------------
    ! Close velocity field files.
    !---------------------------------------------------------------------------
    subroutine close_velocity_field_files
        implicit none
        call MPI_FILE_CLOSE(vfields_fh(1), ierror)
        call MPI_FILE_CLOSE(vfields_fh(2), ierror)
        call MPI_FILE_CLOSE(vfields_fh(3), ierror)
    end subroutine close_velocity_field_files

    !---------------------------------------------------------------------------
    ! Close number density file.
    !---------------------------------------------------------------------------
    subroutine close_number_density_file
        implicit none
        call MPI_FILE_CLOSE(nrho_fh, ierror)
    end subroutine close_number_density_file

    !---------------------------------------------------------------------------
    ! Close the file of the fraction of particles in one energy band.
    !---------------------------------------------------------------------------
    subroutine close_fraction_eband_file
        implicit none
        call MPI_FILE_CLOSE(eband_fh, ierror)
    end subroutine close_fraction_eband_file

    !---------------------------------------------------------------------------
    ! Close PIC fields file collectively using MPI procedures.
    !---------------------------------------------------------------------------
    subroutine close_pic_fields_file
        implicit none
        call close_magnetic_field_files
        call close_electric_field_files
        call close_current_density_files
        call close_pressure_tensor_files
        call close_velocity_field_files
        call close_number_density_file
    end subroutine close_pic_fields_file

end module pic_fields
