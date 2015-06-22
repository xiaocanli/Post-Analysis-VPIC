!*******************************************************************************
! Calculate the Agyrotropy.
! References:
!   Scudder, Jack, and William Daughton. "“Illuminating” electron diffusion
!   regions of collisionless magnetic reconnection using electron agyrotropy."
!   Journal of Geophysical Research: Space Physics (1978–2012) 113.A6 (2008).
!*******************************************************************************
program calc_agyrotropy
    use mpi_module
    use constants, only: fp
    use pic_fields, only: bx, by, bz, absB, pxx, pxy, pxz, pyy, pyz, pzz
    use particle_info, only: species, ibtag, get_ptl_mass_charge
    use parameters, only: it1, it2
    use analysis_management, only: init_analysis, end_analysis
    use mpi_io_module, only: save_field
    implicit none
    real(fp), allocatable, dimension(:, :, :) :: Nxx, Nxy, Nxz, Nyy, Nyz, Nzz
    real(fp), allocatable, dimension(:, :, :) :: bxn, byn, bzn
    real(fp), allocatable, dimension(:, :, :) :: alpha, beta, agyrotropy
    integer :: input_record, output_record

    species = 'e'
    ibtag = '00'

    call get_ptl_mass_charge(species)
    call init_analysis
    call init_pic_fields
    call open_pic_fields(species)
    call init_data

    if (myid == master) then
        print*, 'Calculating agyrotropy for ', species
    endif

    do input_record = it1, 10
        if (myid==master) print*, input_record
        output_record = input_record - it1 + 1
        call read_pic_fields(input_record)

        bxn = bx / absB
        byn = by / absB
        bzn = bz / absB

        Nxx =  byn*byn*pzz - 2.0*byn*bzn*pyz + bzn*bzn*pyy
        Nxy = -byn*bxn*pzz + byn*bzn*pxz + bzn*bxn*pyz - bzn*bzn*pxy
        Nxz =  byn*bxn*pyz - byn*byn*pxz - bzn*bxn*pyy + bzn*byn*pxy
        Nyy =  bxn*bxn*pzz - 2.0*bxn*bzn*pxz + bzn*bzn*pxx
        Nyz = -bxn*bxn*pyz + bxn*byn*pxz + bzn*bxn*pxy - bzn*byn*pxx
        Nzz =  bxn*bxn*pyy - 2.0*bxn*byn*pxy + byn*byn*pxx

        alpha = Nxx + Nyy + Nzz
        beta = -(Nxy**2 + Nxz**2 + Nyz**2 - Nxx*Nyy - Nxx*Nzz - Nyy*Nzz)
        !agyrotropy = 2.0*sqrt(alpha**2-4.0*beta)/alpha
        agyrotropy = 4.0*(alpha**2-4.0*beta) / alpha**2
        agyrotropy = sqrt(agyrotropy)
        where (ISNAN(agyrotropy))
            agyrotropy = 0.0
        end where

        call save_field(agyrotropy, 'agyrotropy', output_record)
    enddo

    call free_data
    call free_pic_fields
    call close_pic_fields_file
    call end_analysis

    contains

    !---------------------------------------------------------------------------
    ! Initializing the data array for this analysis.
    !---------------------------------------------------------------------------
    subroutine init_data
        use mpi_topology, only: htg
        implicit none
        integer :: nx, ny, nz
        nx = htg%nx
        ny = htg%ny
        nz = htg%nz

        allocate(Nxx(nx, ny, nz))
        allocate(Nxy(nx, ny, nz))
        allocate(Nxz(nx, ny, nz))
        allocate(Nyy(nx, ny, nz))
        allocate(Nyz(nx, ny, nz))
        allocate(Nzz(nx, ny, nz))
        allocate(bxn(nx, ny, nz))
        allocate(byn(nx, ny, nz))
        allocate(bzn(nx, ny, nz))
        allocate(alpha(nx, ny, nz))
        allocate(beta(nx, ny, nz))
        allocate(agyrotropy(nx, ny, nz))

        Nxx = 0.0; Nxy = 0.0; Nxz = 0.0
        Nyy = 0.0; Nyz = 0.0; Nzz = 0.0
        bxn = 0.0; byn = 0.0; bzn = 0.0
        alpha = 0.0; beta = 0.0; agyrotropy = 0.0
    end subroutine init_data

    !---------------------------------------------------------------------------
    ! Free the data array for this analysis.
    !---------------------------------------------------------------------------
    subroutine free_data
        implicit none
        deallocate(Nxx, Nxy, Nxz, Nyy, Nyz, Nzz)
        deallocate(bxn, byn, bzn, alpha, beta, agyrotropy)
    end subroutine free_data

    !---------------------------------------------------------------------------
    ! Initialize PIC fields. Not all PIC fields are required for this analysis.
    !---------------------------------------------------------------------------
    subroutine init_pic_fields
        use mpi_topology, only: htg
        use pic_fields, only: init_magnetic_fields, init_pressure_tensor
        implicit none
        integer :: nx, ny, nz
        nx = htg%nx
        ny = htg%ny
        nz = htg%nz
        call init_magnetic_fields(nx, ny, nz)
        call init_pressure_tensor(nx, ny, nz)
    end subroutine init_pic_fields

    !---------------------------------------------------------------------------
    ! Open PIC fields file collectively using MPI procedures.
    !---------------------------------------------------------------------------
    subroutine open_pic_fields(species)
        use pic_fields, only: fields_fh, open_magnetic_field_files, &
                              open_pressure_tensor_files
        implicit none
        character(*), intent(in) :: species

        fields_fh = 0
        call open_magnetic_field_files
        call open_pressure_tensor_files(species)
    end subroutine open_pic_fields

    !---------------------------------------------------------------------------
    ! Read PIC simulation fields.
    ! Input:
    !   ct: current time point.
    !---------------------------------------------------------------------------
    subroutine read_pic_fields(ct)
        use pic_fields, only: read_mangeitc_fields, read_pressure_tensor
        implicit none
        integer, intent(in) :: ct
        call read_mangeitc_fields(ct)
        call read_pressure_tensor(ct)
    end subroutine read_pic_fields

    !---------------------------------------------------------------------------
    ! Close PIC fields file collectively using MPI procedures.
    !---------------------------------------------------------------------------
    subroutine close_pic_fields_file
        use pic_fields, only: close_magnetic_field_files, &
                              close_pressure_tensor_files
        implicit none
        call close_magnetic_field_files
        call close_pressure_tensor_files
    end subroutine close_pic_fields_file

    !---------------------------------------------------------------------------
    ! Free the memory used by the PIC fields.
    !---------------------------------------------------------------------------
    subroutine free_pic_fields
        use pic_fields, only: free_magnetic_fields, free_pressure_tensor
        implicit none
        call free_magnetic_fields
        call free_pressure_tensor
    end subroutine free_pic_fields

end program calc_agyrotropy
