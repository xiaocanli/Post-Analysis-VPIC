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
    public init_particle_fields, free_particle_fields, read_particle_fields, &
           set_current_density_zero, adjust_particle_fields, &
           write_particle_fields, calc_current_density, calc_absJ, &
           write_current_densities
    real(fp), allocatable, dimension(:,:,:) :: vx, vy, vz, nrho
    real(fp), allocatable, dimension(:,:,:) :: pxx, pxy, pxz, pyy, pyz, pzz
    real(fp), allocatable, dimension(:,:,:) :: jx, jy, jz, absJ
    real(fp), allocatable, dimension(:,:,:) :: pyx, pzx, pzy, ux, uy, uz
    real(fp), allocatable, dimension(:,:,:,:) :: eb

    contains

    !---------------------------------------------------------------------------
    ! Initialize particle related fields.
    !---------------------------------------------------------------------------
    subroutine init_particle_fields
        use topology, only: ht
        implicit none
        allocate(vx(ht%nx, ht%ny, ht%nz))
        allocate(vy(ht%nx, ht%ny, ht%nz))
        allocate(vz(ht%nx, ht%ny, ht%nz))
        allocate(nrho(ht%nx, ht%ny, ht%nz))
        allocate(pxx(ht%nx, ht%ny, ht%nz))
        allocate(pxy(ht%nx, ht%ny, ht%nz))
        allocate(pxz(ht%nx, ht%ny, ht%nz))
        allocate(pyy(ht%nx, ht%ny, ht%nz))
        allocate(pyz(ht%nx, ht%ny, ht%nz))
        allocate(pzz(ht%nx, ht%ny, ht%nz))
        allocate(jx(ht%nx, ht%ny, ht%nz))
        allocate(jy(ht%nx, ht%ny, ht%nz))
        allocate(jz(ht%nx, ht%ny, ht%nz))
        allocate(absJ(ht%nx, ht%ny, ht%nz))
        if (nbands > 0) then
            allocate(eb(ht%nx, ht%ny, ht%nz, nbands))
            eb = 0.0
        endif

        vx = 0.0; vy = 0.0; vz = 0.0
        pxx = 0.0; pxy = 0.0; pxz = 0.0
        pyy = 0.0; pyz = 0.0; pzz = 0.0
        nrho = 0.0
        call set_current_density_zero

        if (is_rel == 1) then
            ! Relativistic fields
            allocate(ux(ht%nx, ht%ny, ht%nz))
            allocate(uy(ht%nx, ht%ny, ht%nz))
            allocate(uz(ht%nx, ht%ny, ht%nz))
            allocate(pyx(ht%nx, ht%ny, ht%nz))
            allocate(pzx(ht%nx, ht%ny, ht%nz))
            allocate(pzy(ht%nx, ht%ny, ht%nz))
            ux = 0.0; uy = 0.0; uz = 0.0
            pyx = 0.0; pzx = 0.0; pzy = 0.0
        endif
    end subroutine init_particle_fields

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
        deallocate(vx, vy, vz, nrho)
        deallocate(pxx, pxy, pxz, pyy, pyz, pzz)
        deallocate(jx, jy, jz, absJ)
        if (nbands > 0) then
            deallocate(eb)
        endif
        if (is_rel == 1) then
            deallocate(pyx, pzx, pzy)
            deallocate(ux, uy, uz)
        endif
    end subroutine free_particle_fields

    !---------------------------------------------------------------------------
    ! Read electromagnetic fields from file.
    ! Inputs:
    !   tindex0: the time step index.
    !   species: 'e' for electron, 'H' for ion.
    !---------------------------------------------------------------------------
    subroutine read_particle_fields(tindex0, species)
        use rank_index_mapping, only: index_to_rank
        use picinfo, only: domain
        use topology, only: ht
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
    ! Read the particle related fields for a single MPI process of PIC
    ! simulation.
    ! Inputs:
    !   tindex: the time step index.
    !   pic_mpi_id: MPI id for the PIC simulation to identify the file.
    !   species: 'e' for electron, 'H' for ion.
    !---------------------------------------------------------------------------
    subroutine read_particle_fields_single(tindex0, pic_mpi_id, species)
        use constants, only: fp
        use file_header, only: read_boilerplate, read_fields_header, fheader
        use topology, only: idxstart, idxstop
        implicit none
        integer, intent(in) :: tindex0, pic_mpi_id
        character(len=1), intent(in) :: species
        real(fp), allocatable, dimension(:,:,:) :: buffer
        character(len=150) :: fname
        logical :: is_exist
        integer :: fh   ! File handler
        integer :: n, ixl, iyl, izl, ixh, iyh, izh
        integer :: nc1, nc2, nc3
        integer :: tindex, i

        fh = 10

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
            open(unit=10, file=trim(fname), access='stream', status='unknown', &
                 form='unformatted', action='read')
        else
            print *, "Can't find file:", fname
            print *
            print *, " ***  Terminating ***"
            stop
        endif

        call read_boilerplate(fh)
        call read_fields_header(fh)
        allocate(buffer(fheader%nc(1), fheader%nc(2), fheader%nc(3)))     
        
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

        read(fh) buffer
        vx(ixl:ixh, iyl:iyh, izl:izh) = buffer(2:nc1, 2:nc2, 2:nc3)
        read(fh) buffer
        vy(ixl:ixh, iyl:iyh, izl:izh) = buffer(2:nc1, 2:nc2, 2:nc3)
        read(fh) buffer
        vz(ixl:ixh, iyl:iyh, izl:izh) = buffer(2:nc1, 2:nc2, 2:nc3)
        read(fh) buffer
        nrho(ixl:ixh, iyl:iyh, izl:izh) = buffer(2:nc1, 2:nc2, 2:nc3)

        if (is_rel == 1) then
            read(fh) buffer
            ux(ixl:ixh, iyl:iyh, izl:izh) = buffer(2:nc1, 2:nc2, 2:nc3)
            read(fh) buffer
            uy(ixl:ixh, iyl:iyh, izl:izh) = buffer(2:nc1, 2:nc2, 2:nc3)
            read(fh) buffer
            uz(ixl:ixh, iyl:iyh, izl:izh) = buffer(2:nc1, 2:nc2, 2:nc3)
            read(fh) buffer
        endif

        read(fh) buffer
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

        nrho = abs(nrho)    ! Exclude negative values.
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
    !---------------------------------------------------------------------------
    subroutine write_particle_fields(tindex, output_record, species)
        use mpi_io_translate, only: write_data
        implicit none
        integer, intent(in) :: tindex, output_record
        character(len=1), intent(in) :: species
        character(len=150) :: fname
        integer :: ib
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

        if (is_rel == 1) then
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
        endif
                    
        if (nbands > 0) then
            do ib = 1, nbands
                write(fname, '(A,A,A,I2.2)') &
                    trim(adjustl(rootpath))//'data/', species, 'EB', ib
                call write_data(fname, reshape(eb(:, :, :, ib), shape(nrho)), &
                                tindex, output_record)
            enddo
        endif
    end subroutine write_particle_fields

    !---------------------------------------------------------------------------
    ! Save current densities to file.
    !---------------------------------------------------------------------------
    subroutine write_current_densities(tindex, output_record)
        use mpi_io_translate, only: write_data
        implicit none
        integer, intent(in) :: tindex, output_record
        call write_data(trim(adjustl(rootpath))//'data/jx', &
                        jx, tindex, output_record)
        call write_data(trim(adjustl(rootpath))//'data/jy', &
                        jy, tindex, output_record)
        call write_data(trim(adjustl(rootpath))//'data/jz', &
                        jz, tindex, output_record)
        call write_data(trim(adjustl(rootpath))//'data/absJ', &
                        absJ, tindex, output_record)
    end subroutine write_current_densities
end module particle_fields
