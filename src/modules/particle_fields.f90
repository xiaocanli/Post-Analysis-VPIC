!*******************************************************************************
! Module of particle related fields, including bulk flow velocity, particle
! number density, particle fraction in each energy band (eb), particle stress
! tensor and current density.
!*******************************************************************************
module particle_fields
    use constants, only: fp
    use path_info, only: rootpath
    use picinfo, only: nbands
    use parameters, only: is_rel  ! Whether they are relativistic fields.
    implicit none
    private
    public init_particle_fields, init_density_fields, init_velocity_fields, &
           init_current_densities, init_pressure_tensor, read_particle_fields, &
           read_density_fields_single, read_velocity_fields_single, &
           read_pressure_tensor_single, free_particle_fields, &
           free_velocity_fields, free_density_fields, free_current_densities, &
           free_pressure_tensor, set_current_density_zero, adjust_particle_fields, &
           write_particle_fields, calc_current_density, calc_absJ, &
           write_current_densities
    public nrho, eb
    real(fp), allocatable, dimension(:,:,:) :: vx, vy, vz, nrho
    real(fp), allocatable, dimension(:,:,:) :: pxx, pxy, pxz, pyy, pyz, pzz
    real(fp), allocatable, dimension(:,:,:) :: jx, jy, jz, absJ
    real(fp), allocatable, dimension(:,:,:) :: pyx, pzx, pzy, ux, uy, uz
    real(fp), allocatable, dimension(:,:,:) :: ke_density
    real(fp), allocatable, dimension(:,:,:,:) :: eb

    contains

    !---------------------------------------------------------------------------
    ! Initialize particle related fields.
    !---------------------------------------------------------------------------
    subroutine init_particle_fields
        call init_density_fields
        call init_velocity_fields
        call init_pressure_tensor
        call init_current_densities
        call init_kinetic_energy_density
    end subroutine init_particle_fields

    !---------------------------------------------------------------------------
    ! Initialize density and particle fraction for different energy band.
    !---------------------------------------------------------------------------
    subroutine init_density_fields
        use topology_translate, only: ht
        implicit none
        allocate(nrho(ht%nx, ht%ny, ht%nz))
        if (nbands > 0) then
            allocate(eb(ht%nx, ht%ny, ht%nz, nbands))
            eb = 0.0
        endif
        nrho = 0.0
    end subroutine init_density_fields

    !---------------------------------------------------------------------------
    ! Initialize velocity fields.
    !---------------------------------------------------------------------------
    subroutine init_velocity_fields
        use topology_translate, only: ht
        implicit none
        allocate(vx(ht%nx, ht%ny, ht%nz))
        allocate(vy(ht%nx, ht%ny, ht%nz))
        allocate(vz(ht%nx, ht%ny, ht%nz))
        vx = 0.0; vy = 0.0; vz = 0.0
        if (is_rel == 1) then
            ! Relativistic fields
            allocate(ux(ht%nx, ht%ny, ht%nz))
            allocate(uy(ht%nx, ht%ny, ht%nz))
            allocate(uz(ht%nx, ht%ny, ht%nz))
            ux = 0.0; uy = 0.0; uz = 0.0
        endif
    end subroutine init_velocity_fields

    !---------------------------------------------------------------------------
    ! Initialize pressure tensor fields.
    !---------------------------------------------------------------------------
    subroutine init_pressure_tensor
        use topology_translate, only: ht
        implicit none
        allocate(pxx(ht%nx, ht%ny, ht%nz))
        allocate(pxy(ht%nx, ht%ny, ht%nz))
        allocate(pxz(ht%nx, ht%ny, ht%nz))
        allocate(pyy(ht%nx, ht%ny, ht%nz))
        allocate(pyz(ht%nx, ht%ny, ht%nz))
        allocate(pzz(ht%nx, ht%ny, ht%nz))
        pxx = 0.0; pxy = 0.0; pxz = 0.0
        pyy = 0.0; pyz = 0.0; pzz = 0.0
        if (is_rel == 1) then
            ! Relativistic fields
            allocate(pyx(ht%nx, ht%ny, ht%nz))
            allocate(pzx(ht%nx, ht%ny, ht%nz))
            allocate(pzy(ht%nx, ht%ny, ht%nz))
            pyx = 0.0; pzx = 0.0; pzy = 0.0
        endif
    end subroutine init_pressure_tensor

    !---------------------------------------------------------------------------
    ! Initialize current densities.
    !---------------------------------------------------------------------------
    subroutine init_current_densities
        use topology_translate, only: ht
        implicit none
        allocate(jx(ht%nx, ht%ny, ht%nz))
        allocate(jy(ht%nx, ht%ny, ht%nz))
        allocate(jz(ht%nx, ht%ny, ht%nz))
        allocate(absJ(ht%nx, ht%ny, ht%nz))
        call set_current_density_zero
    end subroutine init_current_densities

    !---------------------------------------------------------------------------
    ! Initialize kinetic energy density
    !---------------------------------------------------------------------------
    subroutine init_kinetic_energy_density
        use topology_translate, only: ht
        implicit none
        if (is_rel == 1) then
            allocate(ke_density(ht%nx, ht%ny, ht%nz))
            ke_density = 0.0
        endif
    end subroutine init_kinetic_energy_density

    !---------------------------------------------------------------------------
    ! Set current densities to zero to avoid accumulation.
    !---------------------------------------------------------------------------
    subroutine set_current_density_zero
        implicit none
        jx = 0.0; jy = 0.0; jz = 0.0
        absJ = 0.0
    end subroutine set_current_density_zero

    !---------------------------------------------------------------------------
    ! Free particle related fields.
    !---------------------------------------------------------------------------
    subroutine free_particle_fields
        implicit none
        call free_density_fields
        call free_velocity_fields
        call free_pressure_tensor
        call free_current_densities
        call free_kinetic_energy_density
    end subroutine free_particle_fields

    !---------------------------------------------------------------------------
    ! Free density and particle fraction for different energy band.
    !---------------------------------------------------------------------------
    subroutine free_density_fields
        implicit none
        deallocate(nrho)
        if (nbands > 0) then
            deallocate(eb)
        endif
    end subroutine free_density_fields

    !---------------------------------------------------------------------------
    ! Free velocity fields.
    !---------------------------------------------------------------------------
    subroutine free_velocity_fields
        implicit none
        deallocate(vx, vy, vz)
        if (is_rel == 1) then
            deallocate(ux, uy, uz)
        endif
    end subroutine free_velocity_fields

    !---------------------------------------------------------------------------
    ! Free pressure tensor fields.
    !---------------------------------------------------------------------------
    subroutine free_pressure_tensor
        implicit none
        deallocate(pxx, pxy, pxz, pyy, pyz, pzz)
        if (is_rel == 1) then
            deallocate(pyx, pzx, pzy)
        endif
    end subroutine free_pressure_tensor

    !---------------------------------------------------------------------------
    ! Free current densities.
    !---------------------------------------------------------------------------
    subroutine free_current_densities
        implicit none
        deallocate(jx, jy, jz, absJ)
    end subroutine free_current_densities

    !---------------------------------------------------------------------------
    ! Free kinetic energy density.
    !---------------------------------------------------------------------------
    subroutine free_kinetic_energy_density
        implicit none
        if (is_rel == 1) then
            deallocate(ke_density)
        endif
    end subroutine free_kinetic_energy_density

    !---------------------------------------------------------------------------
    ! Read electromagnetic fields from file.
    ! Inputs:
    !   tindex0: the time step index.
    !   species: 'e' for electron, 'H' for ion.
    !---------------------------------------------------------------------------
    subroutine read_particle_fields(tindex0, species)
        use rank_index_mapping, only: index_to_rank
        use picinfo, only: domain
        use topology_translate, only: ht
        implicit none
        integer, intent(in) :: tindex0
        character(len=1), intent(in) :: species
        integer :: dom_x, dom_y, dom_z, n
        do dom_x = ht%start_x, ht%stop_x
            do dom_y = ht%start_y, ht%stop_y
                do dom_z = ht%start_z, ht%stop_z
                    call index_to_rank(dom_x, dom_y, dom_z, domain%pic_tx, &
                                       domain%pic_ty, domain%pic_tz, n)
                    call read_particle_fields_single(tindex0, n-1, species)
                enddo ! x
            enddo ! y
        enddo ! z
    end subroutine read_particle_fields

    !---------------------------------------------------------------------------
    ! Open one particle fields file.
    ! Input:
    !   fh: file handler
    !   tindex0: the time step index.
    !   pic_mpi_id: MPI id for the PIC simulation to identify the file.
    !   species: 'e' for electron, 'H' for ion.
    !---------------------------------------------------------------------------
    subroutine open_particle_file(fh, tindex0, pic_mpi_id, species)
        use file_header, only: read_boilerplate, read_fields_header
        implicit none
        integer, intent(in) :: fh   ! File handler
        integer, intent(in) :: tindex0, pic_mpi_id
        character(len=1), intent(in) :: species
        character(len=256) :: fname
        integer :: tindex
        logical :: is_exist
        tindex = tindex0
        !! Index 0 does not have proper current, so use index 1 if it exists
        if (tindex == 0) then
            write(fname, "(A,I0,A1,A1,A6,I0,A1,I0)") &
                trim(adjustl(rootpath))//"hydro/T.", 1, "/", species, &
                "hydro.", 1, ".", pic_mpi_id
            is_exist = .false.
            inquire(file=trim(fname), exist=is_exist)
            if (is_exist) tindex = 1
        endif
        write(fname, "(A,I0,A1,A1,A6,I0,A1,I0)") &
            trim(adjustl(rootpath))//"hydro/T.", tindex, "/", species, &
            "hydro.", tindex, ".", pic_mpi_id
        is_exist = .false.
        inquire(file=trim(fname), exist=is_exist)
      
        if (is_exist) then 
            open(unit=fh, file=trim(fname), access='stream', status='unknown', &
                 form='unformatted', action='read')
        else
            print *, "Can't find file:", fname
            print *
            print *, " ***  Terminating ***"
            stop
        endif
        call read_boilerplate(fh)
        call read_fields_header(fh)
    end subroutine open_particle_file

    !---------------------------------------------------------------------------
    ! Set array indices.
    ! Input:
    !   pic_mpi_id: MPI id for the PIC simulation to identify the file.
    !---------------------------------------------------------------------------
    subroutine set_array_indices(pic_mpi_id, ixl, ixh, iyl, iyh, izl, izh, &
            nc1, nc2, nc3)
        use file_header, only: fheader
        use topology_translate, only: idxstart, idxstop
        implicit none
        integer, intent(in) :: pic_mpi_id
        integer, intent(out) :: ixl, ixh, iyl, iyh, izl, izh, nc1, nc2, nc3
        integer :: n
        n = pic_mpi_id + 1  ! MPI ID starts at 0. The 1D rank starts at 1.
        ixl = idxstart(n, 1)
        iyl = idxstart(n, 2)
        izl = idxstart(n, 3)
        ixh = idxstop(n, 1)
        iyh = idxstop(n, 2)
        izh = idxstop(n, 3)
        nc1 = fheader%nc(1) - 1
        nc2 = fheader%nc(2) - 1
        nc3 = fheader%nc(3) - 1
    end subroutine set_array_indices

    !---------------------------------------------------------------------------
    ! Read the particle related fields for a single MPI process of PIC
    ! simulation.
    ! Inputs:
    !   tindex0: the time step index.
    !   pic_mpi_id: MPI id for the PIC simulation to identify the file.
    !   species: 'e' for electron, 'H' for ion.
    !---------------------------------------------------------------------------
    subroutine read_particle_fields_single(tindex0, pic_mpi_id, species)
        use constants, only: fp
        use file_header, only: fheader
        use topology_translate, only: idxstart, idxstop
        implicit none
        integer, intent(in) :: tindex0, pic_mpi_id
        character(len=1), intent(in) :: species
        real(fp), allocatable, dimension(:,:,:) :: buffer
        integer :: ixl, iyl, izl, ixh, iyh, izh
        integer :: nc1, nc2, nc3
        integer :: i, fh

        fh = 10

        call open_particle_file(fh, tindex0, pic_mpi_id, species)

        allocate(buffer(fheader%nc(1), fheader%nc(2), fheader%nc(3)))
        call set_array_indices(pic_mpi_id, ixl, ixh, iyl, iyh, &
                izl, izh, nc1, nc2, nc3)
        
        read(fh) buffer
        ! vx(ixl:ixh, iyl:iyh, izl:izh) = (buffer(2:nc1, 2:nc2, 2:nc3) + &
        !     buffer(3:nc1+1, 2:nc2, 2:nc3) + buffer(2:nc1, 3:nc2+1, 2:nc3) + &
        !     buffer(3:nc1+1, 3:nc2+1, 2:nc3) + buffer(2:nc1, 2:nc2, 3:nc3+1) + &
        !     buffer(3:nc1+1, 2:nc2, 3:nc3+1) + buffer(2:nc1, 3:nc2+1, 3:nc3+1) + &
        !     buffer(3:nc1+1, 3:nc2+1, 3:nc3+1)) * 0.125
        vx(ixl:ixh, iyl:iyh, izl:izh) = buffer(2:nc1, 2:nc2, 2:nc3)
        read(fh) buffer
        ! vy(ixl:ixh, iyl:iyh, izl:izh) = (buffer(2:nc1, 2:nc2, 2:nc3) + &
        !     buffer(3:nc1+1, 2:nc2, 2:nc3) + buffer(2:nc1, 3:nc2+1, 2:nc3) + &
        !     buffer(3:nc1+1, 3:nc2+1, 2:nc3) + buffer(2:nc1, 2:nc2, 3:nc3+1) + &
        !     buffer(3:nc1+1, 2:nc2, 3:nc3+1) + buffer(2:nc1, 3:nc2+1, 3:nc3+1) + &
        !     buffer(3:nc1+1, 3:nc2+1, 3:nc3+1)) * 0.125
        vy(ixl:ixh, iyl:iyh, izl:izh) = buffer(2:nc1, 2:nc2, 2:nc3)
        read(fh) buffer
        ! vz(ixl:ixh, iyl:iyh, izl:izh) = (buffer(2:nc1, 2:nc2, 2:nc3) + &
        !     buffer(3:nc1+1, 2:nc2, 2:nc3) + buffer(2:nc1, 3:nc2+1, 2:nc3) + &
        !     buffer(3:nc1+1, 3:nc2+1, 2:nc3) + buffer(2:nc1, 2:nc2, 3:nc3+1) + &
        !     buffer(3:nc1+1, 2:nc2, 3:nc3+1) + buffer(2:nc1, 3:nc2+1, 3:nc3+1) + &
        !     buffer(3:nc1+1, 3:nc2+1, 3:nc3+1)) * 0.125
        vz(ixl:ixh, iyl:iyh, izl:izh) = buffer(2:nc1, 2:nc2, 2:nc3)
        read(fh) buffer
        ! nrho(ixl:ixh, iyl:iyh, izl:izh) = (buffer(2:nc1, 2:nc2, 2:nc3) + &
        !     buffer(3:nc1+1, 2:nc2, 2:nc3) + buffer(2:nc1, 3:nc2+1, 2:nc3) + &
        !     buffer(3:nc1+1, 3:nc2+1, 2:nc3) + buffer(2:nc1, 2:nc2, 3:nc3+1) + &
        !     buffer(3:nc1+1, 2:nc2, 3:nc3+1) + buffer(2:nc1, 3:nc2+1, 3:nc3+1) + &
        !     buffer(3:nc1+1, 3:nc2+1, 3:nc3+1)) * 0.125
        nrho(ixl:ixh, iyl:iyh, izl:izh) = buffer(2:nc1, 2:nc2, 2:nc3)

        if (is_rel == 1) then
            read(fh) buffer
            ux(ixl:ixh, iyl:iyh, izl:izh) = buffer(2:nc1, 2:nc2, 2:nc3)
            read(fh) buffer
            uy(ixl:ixh, iyl:iyh, izl:izh) = buffer(2:nc1, 2:nc2, 2:nc3)
            read(fh) buffer
            uz(ixl:ixh, iyl:iyh, izl:izh) = buffer(2:nc1, 2:nc2, 2:nc3)
            read(fh) buffer
            ke_density(ixl:ixh, iyl:iyh, izl:izh) = buffer(2:nc1, 2:nc2, 2:nc3)
        endif

        read(fh) buffer
        ! pxx(ixl:ixh, iyl:iyh, izl:izh) = (buffer(2:nc1, 2:nc2, 2:nc3) + &
        !     buffer(3:nc1+1, 2:nc2, 2:nc3) + buffer(2:nc1, 3:nc2+1, 2:nc3) + &
        !     buffer(3:nc1+1, 3:nc2+1, 2:nc3) + buffer(2:nc1, 2:nc2, 3:nc3+1) + &
        !     buffer(3:nc1+1, 2:nc2, 3:nc3+1) + buffer(2:nc1, 3:nc2+1, 3:nc3+1) + &
        !     buffer(3:nc1+1, 3:nc2+1, 3:nc3+1)) * 0.125
        pxx(ixl:ixh, iyl:iyh, izl:izh) = buffer(2:nc1, 2:nc2, 2:nc3)
        read(fh) buffer
        ! pyy(ixl:ixh, iyl:iyh, izl:izh) = (buffer(2:nc1, 2:nc2, 2:nc3) + &
        !     buffer(3:nc1+1, 2:nc2, 2:nc3) + buffer(2:nc1, 3:nc2+1, 2:nc3) + &
        !     buffer(3:nc1+1, 3:nc2+1, 2:nc3) + buffer(2:nc1, 2:nc2, 3:nc3+1) + &
        !     buffer(3:nc1+1, 2:nc2, 3:nc3+1) + buffer(2:nc1, 3:nc2+1, 3:nc3+1) + &
        !     buffer(3:nc1+1, 3:nc2+1, 3:nc3+1)) * 0.125
        pyy(ixl:ixh, iyl:iyh, izl:izh) = buffer(2:nc1, 2:nc2, 2:nc3)
        read(fh) buffer
        ! pzz(ixl:ixh, iyl:iyh, izl:izh) = (buffer(2:nc1, 2:nc2, 2:nc3) + &
        !     buffer(3:nc1+1, 2:nc2, 2:nc3) + buffer(2:nc1, 3:nc2+1, 2:nc3) + &
        !     buffer(3:nc1+1, 3:nc2+1, 2:nc3) + buffer(2:nc1, 2:nc2, 3:nc3+1) + &
        !     buffer(3:nc1+1, 2:nc2, 3:nc3+1) + buffer(2:nc1, 3:nc2+1, 3:nc3+1) + &
        !     buffer(3:nc1+1, 3:nc2+1, 3:nc3+1)) * 0.125
        pzz(ixl:ixh, iyl:iyh, izl:izh) = buffer(2:nc1, 2:nc2, 2:nc3)
        read(fh) buffer
        ! pyz(ixl:ixh, iyl:iyh, izl:izh) = (buffer(2:nc1, 2:nc2, 2:nc3) + &
        !     buffer(3:nc1+1, 2:nc2, 2:nc3) + buffer(2:nc1, 3:nc2+1, 2:nc3) + &
        !     buffer(3:nc1+1, 3:nc2+1, 2:nc3) + buffer(2:nc1, 2:nc2, 3:nc3+1) + &
        !     buffer(3:nc1+1, 2:nc2, 3:nc3+1) + buffer(2:nc1, 3:nc2+1, 3:nc3+1) + &
        !     buffer(3:nc1+1, 3:nc2+1, 3:nc3+1)) * 0.125
        pyz(ixl:ixh, iyl:iyh, izl:izh) = buffer(2:nc1, 2:nc2, 2:nc3)
        read(fh) buffer
        ! pxz(ixl:ixh, iyl:iyh, izl:izh) = (buffer(2:nc1, 2:nc2, 2:nc3) + &
        !     buffer(3:nc1+1, 2:nc2, 2:nc3) + buffer(2:nc1, 3:nc2+1, 2:nc3) + &
        !     buffer(3:nc1+1, 3:nc2+1, 2:nc3) + buffer(2:nc1, 2:nc2, 3:nc3+1) + &
        !     buffer(3:nc1+1, 2:nc2, 3:nc3+1) + buffer(2:nc1, 3:nc2+1, 3:nc3+1) + &
        !     buffer(3:nc1+1, 3:nc2+1, 3:nc3+1)) * 0.125
        pxz(ixl:ixh, iyl:iyh, izl:izh) = buffer(2:nc1, 2:nc2, 2:nc3)
        read(fh) buffer
        ! pxy(ixl:ixh, iyl:iyh, izl:izh) = (buffer(2:nc1, 2:nc2, 2:nc3) + &
        !     buffer(3:nc1+1, 2:nc2, 2:nc3) + buffer(2:nc1, 3:nc2+1, 2:nc3) + &
        !     buffer(3:nc1+1, 3:nc2+1, 2:nc3) + buffer(2:nc1, 2:nc2, 3:nc3+1) + &
        !     buffer(3:nc1+1, 2:nc2, 3:nc3+1) + buffer(2:nc1, 3:nc2+1, 3:nc3+1) + &
        !     buffer(3:nc1+1, 3:nc2+1, 3:nc3+1)) * 0.125
        pxy(ixl:ixh, iyl:iyh, izl:izh) = buffer(2:nc1, 2:nc2, 2:nc3)
        
        ! Particle fraction in each energy band.
        if (nbands > 0) then
            do i = 1, nbands
                read(fh) buffer
                eb(ixl:ixh, iyl:iyh, izl:izh, i) = buffer(2:nc1, 2:nc2, 2:nc3)
            end do
        endif

        deallocate(buffer)
        close(fh)
    end subroutine read_particle_fields_single

    !---------------------------------------------------------------------------
    ! Read the particle velocity fields only.
    ! Inputs:
    !   tindex0: the time step index.
    !   pic_mpi_id: MPI id for the PIC simulation to identify the file.
    !   species: 'e' for electron, 'H' for ion.
    !---------------------------------------------------------------------------
    subroutine read_velocity_fields_single(tindex0, pic_mpi_id, species)
        use constants, only: fp
        use file_header, only: fheader
        use topology_translate, only: idxstart, idxstop
        implicit none
        integer, intent(in) :: tindex0, pic_mpi_id
        character(len=1), intent(in) :: species
        real(fp), allocatable, dimension(:,:,:) :: buffer
        integer :: ixl, iyl, izl, ixh, iyh, izh
        integer :: nc1, nc2, nc3
        integer :: i, fh

        fh = 10

        call open_particle_file(fh, tindex0, pic_mpi_id, species)

        allocate(buffer(fheader%nc(1), fheader%nc(2), fheader%nc(3)))
        call set_array_indices(pic_mpi_id, ixl, ixh, iyl, iyh, &
                izl, izh, nc1, nc2, nc3)
        
        read(fh) buffer
        ! vx(ixl:ixh, iyl:iyh, izl:izh) = (buffer(2:nc1, 2:nc2, 2:nc3) + &
        !     buffer(3:nc1+1, 2:nc2, 2:nc3) + buffer(2:nc1, 3:nc2+1, 2:nc3) + &
        !     buffer(3:nc1+1, 3:nc2+1, 2:nc3) + buffer(2:nc1, 2:nc2, 3:nc3+1) + &
        !     buffer(3:nc1+1, 2:nc2, 3:nc3+1) + buffer(2:nc1, 3:nc2+1, 3:nc3+1) + &
        !     buffer(3:nc1+1, 3:nc2+1, 3:nc3+1)) * 0.125
        vx(ixl:ixh, iyl:iyh, izl:izh) = buffer(2:nc1, 2:nc2, 2:nc3)
        read(fh) buffer
        ! vy(ixl:ixh, iyl:iyh, izl:izh) = (buffer(2:nc1, 2:nc2, 2:nc3) + &
        !     buffer(3:nc1+1, 2:nc2, 2:nc3) + buffer(2:nc1, 3:nc2+1, 2:nc3) + &
        !     buffer(3:nc1+1, 3:nc2+1, 2:nc3) + buffer(2:nc1, 2:nc2, 3:nc3+1) + &
        !     buffer(3:nc1+1, 2:nc2, 3:nc3+1) + buffer(2:nc1, 3:nc2+1, 3:nc3+1) + &
        !     buffer(3:nc1+1, 3:nc2+1, 3:nc3+1)) * 0.125
        vy(ixl:ixh, iyl:iyh, izl:izh) = buffer(2:nc1, 2:nc2, 2:nc3)
        read(fh) buffer
        ! vz(ixl:ixh, iyl:iyh, izl:izh) = (buffer(2:nc1, 2:nc2, 2:nc3) + &
        !     buffer(3:nc1+1, 2:nc2, 2:nc3) + buffer(2:nc1, 3:nc2+1, 2:nc3) + &
        !     buffer(3:nc1+1, 3:nc2+1, 2:nc3) + buffer(2:nc1, 2:nc2, 3:nc3+1) + &
        !     buffer(3:nc1+1, 2:nc2, 3:nc3+1) + buffer(2:nc1, 3:nc2+1, 3:nc3+1) + &
        !     buffer(3:nc1+1, 3:nc2+1, 3:nc3+1)) * 0.125
        vz(ixl:ixh, iyl:iyh, izl:izh) = buffer(2:nc1, 2:nc2, 2:nc3)
        read(fh) buffer

        if (is_rel == 1) then
            read(fh) buffer
            ux(ixl:ixh, iyl:iyh, izl:izh) = buffer(2:nc1, 2:nc2, 2:nc3)
            read(fh) buffer
            uy(ixl:ixh, iyl:iyh, izl:izh) = buffer(2:nc1, 2:nc2, 2:nc3)
            read(fh) buffer
            uz(ixl:ixh, iyl:iyh, izl:izh) = buffer(2:nc1, 2:nc2, 2:nc3)
            read(fh) buffer
        endif

        deallocate(buffer)
        close(fh)
    end subroutine read_velocity_fields_single

    !---------------------------------------------------------------------------
    ! Read pressure tensor only.
    ! Inputs:
    !   tindex0: the time step index.
    !   pic_mpi_id: MPI id for the PIC simulation to identify the file.
    !   species: 'e' for electron, 'H' for ion.
    !---------------------------------------------------------------------------
    subroutine read_pressure_tensor_single(tindex0, pic_mpi_id, species)
        use constants, only: fp
        use file_header, only: fheader, v0
        use topology_translate, only: idxstart, idxstop
        implicit none
        integer, intent(in) :: tindex0, pic_mpi_id
        character(len=1), intent(in) :: species
        real(fp), allocatable, dimension(:,:,:) :: buffer
        integer :: ixl, iyl, izl, ixh, iyh, izh
        integer :: nc1, nc2, nc3
        integer :: i, fh
        integer :: offset, buffer_size

        fh = 10

        call open_particle_file(fh, tindex0, pic_mpi_id, species)
        offset = 23 + sizeof(fheader) + sizeof(v0)  ! 23 is the size of boilerplate

        allocate(buffer(fheader%nc(1), fheader%nc(2), fheader%nc(3)))
        call set_array_indices(pic_mpi_id, ixl, ixh, iyl, iyh, &
                izl, izh, nc1, nc2, nc3)
        
        buffer_size = fheader%nc(1) * fheader%nc(2) * fheader%nc(3) * 4
        offset = offset + 4 * buffer_size

        if (is_rel == 1) then
            offset = offset + 4 * buffer_size
        endif

        read(fh, pos=offset+1) buffer
        pxx(ixl:ixh, iyl:iyh, izl:izh) = buffer(2:nc1, 2:nc2, 2:nc3)
        read(fh) buffer
        pyy(ixl:ixh, iyl:iyh, izl:izh) = buffer(2:nc1, 2:nc2, 2:nc3)
        read(fh) buffer
        pzz(ixl:ixh, iyl:iyh, izl:izh) = buffer(2:nc1, 2:nc2, 2:nc3)
        read(fh) buffer
        pyz(ixl:ixh, iyl:iyh, izl:izh) = buffer(2:nc1, 2:nc2, 2:nc3)
        read(fh) buffer
        pxz(ixl:ixh, iyl:iyh, izl:izh) = buffer(2:nc1, 2:nc2, 2:nc3)
        read(fh) buffer
        pxy(ixl:ixh, iyl:iyh, izl:izh) = buffer(2:nc1, 2:nc2, 2:nc3)
        
        deallocate(buffer)
        close(fh)
    end subroutine read_pressure_tensor_single

    !---------------------------------------------------------------------------
    ! Read density fields only.
    ! Inputs:
    !   tindex0: the time step index.
    !   pic_mpi_id: MPI id for the PIC simulation to identify the file.
    !   species: 'e' for electron, 'H' for ion.
    !---------------------------------------------------------------------------
    subroutine read_density_fields_single(tindex0, pic_mpi_id, species)
        use constants, only: fp
        use file_header, only: fheader, v0
        use topology_translate, only: idxstart, idxstop
        implicit none
        integer, intent(in) :: tindex0, pic_mpi_id
        character(len=1), intent(in) :: species
        real(fp), allocatable, dimension(:,:,:) :: buffer
        integer :: ixl, iyl, izl, ixh, iyh, izh
        integer :: nc1, nc2, nc3
        integer :: i, fh
        integer :: offset, buffer_size

        fh = 10

        call open_particle_file(fh, tindex0, pic_mpi_id, species)
        offset = 23 + sizeof(fheader) + sizeof(v0)  ! 23 is the size of boilerplate

        allocate(buffer(fheader%nc(1), fheader%nc(2), fheader%nc(3)))
        call set_array_indices(pic_mpi_id, ixl, ixh, iyl, iyh, &
                izl, izh, nc1, nc2, nc3)
        
        buffer_size = fheader%nc(1) * fheader%nc(2) * fheader%nc(3) * 4
        offset = offset + 3 * buffer_size
        read(fh, pos=offset+1) buffer
        nrho(ixl:ixh, iyl:iyh, izl:izh) = buffer(2:nc1, 2:nc2, 2:nc3)
        offset = offset + buffer_size

        if (is_rel == 1) then
            offset = offset + 4 * buffer_size
        endif

        offset = offset + 6 * buffer_size

        ! Particle fraction in each energy band.
        if (nbands > 0) then
            do i = 1, nbands
                read(fh, pos=offset+1) buffer
                eb(ixl:ixh, iyl:iyh, izl:izh, i) = buffer(2:nc1, 2:nc2, 2:nc3)
                offset = offset + buffer_size
            end do
        endif

        deallocate(buffer)
        close(fh)
    end subroutine read_density_fields_single

    !---------------------------------------------------------------------------
    ! Calculate the 3 components of the current density. This subroutine has
    ! to be executed before adjust_particle_fields, which will change vx, vy,
    ! vz to real bulk velocity.
    !---------------------------------------------------------------------------
    subroutine calc_current_density
        implicit none
        jx = jx + vx
        jy = jy + vy
        jz = jz + vz
    end subroutine calc_current_density

    !---------------------------------------------------------------------------
    ! Calculate the magnitude of the current density.
    !---------------------------------------------------------------------------
    subroutine calc_absJ
        implicit none
        absJ = sqrt(jx**2 + jy**2 + jz**2)
    end subroutine calc_absJ

    !---------------------------------------------------------------------------
    ! Adjust particle fields. The changes include
    !   1. vx, vy, vz are actually current densities. So we need to change them
    !      to be revl bulk velocities.
    !   2. pxx ... vzz are stress tensor. We'll convert it to pressure tensor.
    !---------------------------------------------------------------------------
    subroutine adjust_particle_fields(species)
        use picinfo, only: mime
        use constants, only: fp
        implicit none
        character(len=1), intent(in) :: species
        real(fp) :: ptl_mass, ptl_charge
        if (species == 'e') then
            ptl_mass = 1.0
            ptl_charge = -1.0
        else
            ptl_mass = real(mime, kind=fp)
            ptl_charge = 1.0
        endif

        nrho = abs(nrho / ptl_charge)
        if (is_rel == 1) then
            where (nrho > 0.0) 
               vx = (vx/nrho) * ptl_charge
               vy = (vy/nrho) * ptl_charge
               vz = (vz/nrho) * ptl_charge
               ux = (ux/nrho) / ptl_mass
               uy = (uy/nrho) / ptl_mass
               uz = (uz/nrho) / ptl_mass
               pxx = (pxx - ptl_mass*nrho*vx*ux)
               pyy = (pyy - ptl_mass*nrho*vy*uy)
               pzz = (pzz - ptl_mass*nrho*vz*uz)
               pyx = (pxy - ptl_mass*nrho*vy*ux)
               pzx = (pxz - ptl_mass*nrho*vz*ux)
               pzy = (pyz - ptl_mass*nrho*vz*uy)
               pxy = (pxy - ptl_mass*nrho*vx*uy)
               pxz = (pxz - ptl_mass*nrho*vx*uz)
               pyz = (pyz - ptl_mass*nrho*vy*uz)
            elsewhere
               pxx = 0.0
               pyy = 0.0
               pzz = 0.0
               pyx = 0.0
               pzx = 0.0
               pzy = 0.0
               pxy = 0.0
               pxz = 0.0
               pyz = 0.0
               vx = 0.0
               vy = 0.0
               vz = 0.0
               ux = 0.0
               uy = 0.0
               uz = 0.0
            endwhere
        else
            where (nrho > 0)
                pxx = (pxx - ptl_mass*vx*vx/nrho)
                pyy = (pyy - ptl_mass*vy*vy/nrho)
                pzz = (pzz - ptl_mass*vz*vz/nrho)
                pxy = (pxy - ptl_mass*vx*vy/nrho)
                pxz = (pxz - ptl_mass*vx*vz/nrho)
                pyz = (pyz - ptl_mass*vy*vz/nrho)
                vx = ptl_charge * (vx/nrho)
                vy = ptl_charge * (vy/nrho)
                vz = ptl_charge * (vz/nrho)
            elsewhere
                pxx = 0.0
                pyy = 0.0
                pzz = 0.0
                pxy = 0.0
                pxz = 0.0
                pyz = 0.0
                vx = 0.0
                vy = 0.0
                vz = 0.0
            endwhere
        endif
    end subroutine adjust_particle_fields

    !---------------------------------------------------------------------------
    ! Save particle fields to file.
    !   tindex: the time step index.
    !   output_record: it decides the offset from the file head.
    !   species: 'e' for electron. 'i' for ion.
    !   with_suffix: whether files will have suffix
    !   suffix: indicates the kind of data
    !---------------------------------------------------------------------------
    subroutine write_particle_fields(tindex, output_record, species, &
                                     with_suffix, suffix)
        use mpi_io_translate, only: write_data
        implicit none
        integer, intent(in) :: tindex, output_record
        character(len=1), intent(in) :: species
        character(*), intent(in) :: suffix
        logical, intent(in) :: with_suffix
        character(len=256) :: fname
        integer :: ib
        if (with_suffix) then
            fname = trim(adjustl(rootpath))//'data/v'//species//'x'//suffix
            call write_data(fname, vx, tindex, output_record)              
            fname = trim(adjustl(rootpath))//'data/v'//species//'y'//suffix
            call write_data(fname, vy, tindex, output_record)              
            fname = trim(adjustl(rootpath))//'data/v'//species//'z'//suffix
            call write_data(fname, vz, tindex, output_record)
            fname = trim(adjustl(rootpath))//'data/n'//species//suffix
            call write_data(fname, nrho, tindex, output_record)
            fname = trim(adjustl(rootpath))//'data/p'//species//'-xx'//suffix
            call write_data(fname, pxx, tindex, output_record)       
            fname = trim(adjustl(rootpath))//'data/p'//species//'-yy'//suffix
            call write_data(fname, pyy, tindex, output_record)       
            fname = trim(adjustl(rootpath))//'data/p'//species//'-zz'//suffix
            call write_data(fname, pzz, tindex, output_record)       
            fname = trim(adjustl(rootpath))//'data/p'//species//'-yz'//suffix
            call write_data(fname, pyz, tindex, output_record)       
            fname = trim(adjustl(rootpath))//'data/p'//species//'-xz'//suffix
            call write_data(fname, pxz, tindex, output_record)       
            fname = trim(adjustl(rootpath))//'data/p'//species//'-xy'//suffix
            call write_data(fname, pxy, tindex, output_record)
        else
            fname = trim(adjustl(rootpath))//'data/v'//species//'x'
            call write_data(fname, vx, tindex, output_record)      
            fname = trim(adjustl(rootpath))//'data/v'//species//'y'
            call write_data(fname, vy, tindex, output_record)      
            fname = trim(adjustl(rootpath))//'data/v'//species//'z'
            call write_data(fname, vz, tindex, output_record)
            fname = trim(adjustl(rootpath))//'data/n'//species
            call write_data(fname, nrho, tindex, output_record)
            fname = trim(adjustl(rootpath))//'data/p'//species//'-xx'
            call write_data(fname, pxx, tindex, output_record)       
            fname = trim(adjustl(rootpath))//'data/p'//species//'-yy'
            call write_data(fname, pyy, tindex, output_record)       
            fname = trim(adjustl(rootpath))//'data/p'//species//'-zz'
            call write_data(fname, pzz, tindex, output_record)       
            fname = trim(adjustl(rootpath))//'data/p'//species//'-yz'
            call write_data(fname, pyz, tindex, output_record)       
            fname = trim(adjustl(rootpath))//'data/p'//species//'-xz'
            call write_data(fname, pxz, tindex, output_record)       
            fname = trim(adjustl(rootpath))//'data/p'//species//'-xy'
            call write_data(fname, pxy, tindex, output_record)
        endif

        if (is_rel == 1) then
            if (with_suffix) then
                fname = trim(adjustl(rootpath))//'data/u'//species//'x'//suffix
                call write_data(fname, ux, tindex, output_record)              
                fname = trim(adjustl(rootpath))//'data/u'//species//'y'//suffix
                call write_data(fname, uy, tindex, output_record)              
                fname = trim(adjustl(rootpath))//'data/u'//species//'z'//suffix
                call write_data(fname, uz, tindex, output_record)
                fname = trim(adjustl(rootpath))//'data/p'//species//'-yx'//suffix
                call write_data(fname, pyx, tindex, output_record)       
                fname = trim(adjustl(rootpath))//'data/p'//species//'-zx'//suffix
                call write_data(fname, pzx, tindex, output_record)       
                fname = trim(adjustl(rootpath))//'data/p'//species//'-zy'//suffix
                call write_data(fname, pzy, tindex, output_record)
                fname = trim(adjustl(rootpath))//'data/ke-'//species//suffix
                call write_data(fname, ke_density, tindex, output_record)              
            else
                fname = trim(adjustl(rootpath))//'data/u'//species//'x'
                call write_data(fname, ux, tindex, output_record)
                fname = trim(adjustl(rootpath))//'data/u'//species//'y'
                call write_data(fname, uy, tindex, output_record)
                fname = trim(adjustl(rootpath))//'data/u'//species//'z'
                call write_data(fname, uz, tindex, output_record)
                fname = trim(adjustl(rootpath))//'data/p'//species//'-yx'
                call write_data(fname, pyx, tindex, output_record)
                fname = trim(adjustl(rootpath))//'data/p'//species//'-zx'
                call write_data(fname, pzx, tindex, output_record)
                fname = trim(adjustl(rootpath))//'data/p'//species//'-zy'
                call write_data(fname, pzy, tindex, output_record)
                fname = trim(adjustl(rootpath))//'data/ke-'//species
                call write_data(fname, ke_density, tindex, output_record)              
            endif
        endif
                    
        if (nbands > 0) then
            do ib = 1, nbands
                if (with_suffix) then
                    write(fname, '(A,A,A,I2.2,A)') &
                        trim(adjustl(rootpath))//'data/', species, 'EB', ib, suffix
                else
                    write(fname, '(A,A,A,I2.2)') &
                        trim(adjustl(rootpath))//'data/', species, 'EB', ib
                endif
                call write_data(fname, reshape(eb(:, :, :, ib), shape(nrho)), &
                                tindex, output_record)
            enddo
        endif
    end subroutine write_particle_fields

    !---------------------------------------------------------------------------
    ! Save current densities to file.
    !   with_suffix: whether files will have suffix
    !   suffix: indicates the kind of data
    !---------------------------------------------------------------------------
    subroutine write_current_densities(tindex, output_record, with_suffix, suffix)
        use mpi_io_translate, only: write_data
        implicit none
        integer, intent(in) :: tindex, output_record
        logical, intent(in) :: with_suffix
        character(*), intent(in) :: suffix
        if (with_suffix) then
            call write_data(trim(adjustl(rootpath))//'data/jx'//suffix, &
                            jx, tindex, output_record)
            call write_data(trim(adjustl(rootpath))//'data/jy'//suffix, &
                            jy, tindex, output_record)
            call write_data(trim(adjustl(rootpath))//'data/jz'//suffix, &
                            jz, tindex, output_record)
            call write_data(trim(adjustl(rootpath))//'data/absJ'//suffix, &
                            absJ, tindex, output_record)
        else
            call write_data(trim(adjustl(rootpath))//'data/jx', &
                            jx, tindex, output_record)
            call write_data(trim(adjustl(rootpath))//'data/jy', &
                            jy, tindex, output_record)
            call write_data(trim(adjustl(rootpath))//'data/jz', &
                            jz, tindex, output_record)
            call write_data(trim(adjustl(rootpath))//'data/absJ', &
                            absJ, tindex, output_record)
        endif
    end subroutine write_current_densities
end module particle_fields
