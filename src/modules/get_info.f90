!*******************************************************************************
! Module of the parameters related to PIC simulations.
!*******************************************************************************
module picinfo
    use constants, only: dp, fp
    use path_info, only: rootpath
    use read_config, only: get_variable
    implicit none
    save
    private
    public picdomain, broadcast_pic_info, get_total_time_frames, &
           write_pic_info, get_energy_band_number
    public nbands, mime, domain, nt, read_domain
    ! Information of simulation domain. All values are in simulation units.
    ! Length is in de. Time is in 1/wpe unless clarified.
    type picdomain
        real(dp) :: dx, dy, dz            ! Grid sizes
        real(dp) :: idx, idy, idz         ! Inverse of dx, dy, dz
        real(dp) :: idxh, idyh, idzh      ! Half of idx, idy, idz
        real(fp) :: lx_de, ly_de, lz_de   ! Simulation box sizes
        real(fp) :: dtwpe, dtwce          ! Time step in PIC simulation
        real(fp) :: dtwpi, dtwci          ! Time step in PIC simulation
        real(fp) :: dt, idt               ! Fields output interval and its inverse
        integer :: energies_interval      ! Energy output time interval
        integer :: fields_interval        ! Fields output time interval
        integer :: hydro_interval         ! hydro output time interval
        integer :: particle_interval     ! Particles output time interval
        integer :: nx, ny, nz             ! Grid numbers
        integer :: pic_tx, pic_ty, pic_tz ! MPI topology
        integer :: pic_nx, pic_ny, pic_nz ! The domain sizes for each process
        integer :: nppc                   ! Number of particles/cell/species
        integer :: nproc                  ! Number of CPU cores used.
    end type picdomain

    type(picdomain) :: domain

    real(fp) :: mime     ! Mass ratio
    integer :: nt        ! Total number of time frames for field output.
    integer :: nbands    ! Total number of energy bands.

    contains

    !---------------------------------------------------------------------------
    ! Private function to get the main configuration file name for current PIC
    ! simulation from the Makefile of VPIC code.
    !---------------------------------------------------------------------------
    function get_main_fname() result(fname)
        implicit none
        character(len=20) :: fname
        character(len=150) :: buff
        integer :: fh, index1
        fh = 40
        ! Get the main configuration file for current PIC simulation.
        open(unit=fh, file=trim(adjustl(rootpath))//'Makefile', status='old')
        read(fh, '(A)') buff
        do while (index(buff, 'vpic') == 0)
            read(fh, '(A)') buff
        enddo
        index1 = index(buff, 'op')
        fname = trim(buff(index1+3:))
        close(fh)
    end function get_main_fname

    !---------------------------------------------------------------------------
    ! Get the fields output time interval from the initialization file.
    ! The interval is defined as "int interval = ...", for example,
    ! int interval = int(2.5/(wci*dt)). This routine assumes the intervals
    ! are the same for all particle species.
    !
    ! Input:
    !   dtwpe, dtwce, dtwpi, dtwci: time step in different units.
    ! Output:
    !   fields_interval: the fields output interval
    !   hydro_interval: the hydro fields output interval
    !   particles_interval: the particle output interval
    !---------------------------------------------------------------------------
    subroutine get_fields_interval(dtwpe, dtwce, dtwpi, dtwci, fields_interval, &
            hydro_interval, particle_interval)
        use constants, only: fp
        implicit none
        real(fp), intent(in) :: dtwpe, dtwce, dtwpi, dtwci
        integer, intent(out) :: fields_interval, hydro_interval, particle_interval
        integer :: fh, index1, index2, index3
        character(len=150) :: buff, code
        character(len=20) :: fname, another_interval
        logical :: cond1, cond2
        integer :: interval, ratio, interval_base
        fh = 40

        fname = get_main_fname()

        open(unit=fh, file=trim(adjustl(rootpath))//trim(adjustl(fname)), status='old')
        read(fh, '(A)') buff
        ! Make sure the line is not commented out
        cond1 = index(buff, 'int interval = ') == 0
        cond2 = index(buff, '//') /= 0
        do while (cond1 .or. cond2)
            read(fh, '(A)') buff
            cond1 = index(buff, 'int interval = ') == 0
            cond2 = index(buff, '//') /= 0
        enddo
        if (index(buff, '(') == 0) then
            ! This interval is based on another interval
            index1 = index(buff, '=')
            index2 = index(buff, ';')
            if (index(buff, '*') == 0) then
                another_interval = trim(adjustl(buff(index1+1:index2-1)))
                ratio = 1
            else
                index3 = index(buff, '*')
                another_interval = trim(adjustl(buff(index3+1:index2-1)))
                read(buff(index1+1:index3-1), *) ratio
            endif
            code = 'int '//trim(adjustl(another_interval))//' = '
            ! Find another_interval
            rewind(fh)
            cond1 = index(buff, trim(adjustl(code))) == 0
            cond2 = index(buff, '//') /= 0
            do while (cond1 .or. cond2)
                read(fh, '(A)') buff
                cond1 = index(buff, trim(adjustl(code))) == 0
                cond2 = index(buff, '//') /= 0
            enddo
            interval_base = get_time_interval(buff, dtwpe, dtwce, dtwpi, dtwci)
            interval = interval_base * ratio
        else
            interval = get_time_interval(buff, dtwpe, dtwce, dtwpi, dtwci)
        endif

        ! Read fields_interval
        cond1 = index(buff, 'int fields_interval = ') == 0
        cond2 = index(buff, '//') /= 0
        do while (cond1 .or. cond2)
            read(fh, '(A)') buff
            cond1 = index(buff, 'int fields_interval = ') == 0
            cond2 = index(buff, '//') /= 0
        enddo
        if (index(buff, '*') /= 0) then
            index1 = index(buff, '=')
            index2 = index(buff, '*')
            read(buff(index1+1:index2-1), *) ratio
        else
            ratio = 1
        endif
        fields_interval = interval * ratio

        ! Read hydro_interval, assuming ehydro_interval == Hhydro_interval
        cond1 = index(buff, 'int ehydro_interval = ') == 0
        cond2 = index(buff, '//') /= 0
        do while (cond1 .or. cond2)
            read(fh, '(A)') buff
            cond1 = index(buff, 'int ehydro_interval = ') == 0
            cond2 = index(buff, '//') /= 0
        enddo
        if (index(buff, '*') /= 0) then
            index1 = index(buff, '=')
            index2 = index(buff, '*')
            read(buff(index1+1:index2-1), *) ratio
        else
            ratio = 1
        endif
        hydro_interval = interval * ratio

        ! Read particle_interval, assuming eparticle_interval == Hparticle_interval
        cond1 = index(buff, 'int eparticle_interval = ') == 0
        cond2 = index(buff, '//') /= 0
        do while (cond1 .or. cond2)
            read(fh, '(A)') buff
            cond1 = index(buff, 'int eparticle_interval = ') == 0
            cond2 = index(buff, '//') /= 0
        enddo
        if (index(buff, '*') /= 0) then
            index1 = index(buff, '=')
            index2 = index(buff, '*')
            read(buff(index1+1:index2-1), *) ratio
        else
            ratio = 1
        endif
        particle_interval = interval * ratio
        close(fh)
    end subroutine get_fields_interval

    !---------------------------------------------------------------------------
    ! Read domain information from info file directly from the PIC simulation.
    ! Args:
    !   line: one single line
    !   dtwpe: the time step in 1/wpe.
    !   dtwce: the time step in 1/wce.
    !   dtwpi: the time step in 1/wpi.
    !   dtwci: the time step in 1/wci.
    !---------------------------------------------------------------------------
    function get_time_interval(line, dtwpe, dtwce, dtwpi, dtwci) result(interval)
        implicit none
        character(*), intent(in) :: line
        real(fp), intent(in) :: dtwpe, dtwce, dtwpi, dtwci
        integer :: interval, index1, index2, index3
        real(fp) :: dt
        character(len=16) :: buff
        index1 = index(line, '(')
        index2 = index(line, '/')
        index3 = index(line, '*')
        read(line(index1+1:index2-1), *) dt
        buff = line(index2+2:index3-1)
        if (buff == 'wpe') then
            interval = int(dt / dtwpe)
        else if (buff == 'wce') then
            interval = int(dt / dtwce)
        else if (buff == 'wpi') then
            interval = int(dt / dtwpi)
        else if (buff == 'wci') then
            interval = int(dt / dtwci)
        endif
    end function get_time_interval


    !---------------------------------------------------------------------------
    ! Read domain information from info file directly from the PIC simulation.
    ! Information to read:
    !   domain: the domain information of the PIC simulation.
    !   mime: the ion to electron mass ration.
    !---------------------------------------------------------------------------
    subroutine read_domain
        use constants, only: dp, fp
        implicit none
        real(fp) :: temp, dtf_wpe
        integer :: fh
        ! read the time step of the simulation
        fh = 10
        open(unit=fh, file=trim(adjustl(rootpath))//'info', status='old')
        mime = get_variable(fh, 'mi/me', '=')
        domain%lx_de = get_variable(fh, 'Lx/de', '=')
        domain%ly_de = get_variable(fh, 'Ly/de', '=')
        domain%lz_de = get_variable(fh, 'Lz/de', '=')
        temp = get_variable(fh, 'nx', '=')
        domain%nx = int(temp)
        temp = get_variable(fh, 'ny', '=')
        domain%ny = int(temp)
        temp = get_variable(fh, 'nz', '=')
        domain%nz = int(temp)
        temp = get_variable(fh, 'nproc', '=')
        domain%nproc = int(temp)
        temp = get_variable(fh, 'nppc', '=')
        domain%nppc = int(temp)
        domain%dtwpe = get_variable(fh, 'dt*wpe', '=')
        domain%dtwce = get_variable(fh, 'dt*wce', '=')
        domain%dtwci = get_variable(fh, 'dt*wci', '=')
        domain%dtwpi = domain%dtwpe / mime
        temp = get_variable(fh, 'energies_interval', ':')
        domain%energies_interval = int(temp)
        domain%dx = get_variable(fh, 'dx/de', '=')
        domain%dy = get_variable(fh, 'dy/de', '=')
        domain%dz = get_variable(fh, 'dz/de', '=')
        domain%idx = 1.0/domain%dx
        domain%idy = 1.0/domain%dy
        domain%idz = 1.0/domain%dz
        domain%idxh = 0.5*domain%idx
        domain%idyh = 0.5*domain%idy
        domain%idzh = 0.5*domain%idz

        call get_fields_interval(domain%dtwpe, domain%dtwce, domain%dtwpi, &
            domain%dtwci, domain%fields_interval, domain%hydro_interval, &
            domain%particle_interval)

        domain%dt = domain%fields_interval * domain%dtwpe
        domain%idt = 1.0 / domain%dt

        close(fh)

        call read_pic_mpi_topology

        ! Echo this information
        print *, "---------------------------------------------------"
        write(*, "(A)") " PIC simulation information."
        write(*, "(A,F7.2,A,F7.2,A,F7.2)") " lx, ly, lz (de) = ", &
            domain%lx_de, ',', domain%ly_de, ',', domain%lz_de
        write(*, "(A,I0,A,I0,A,I0)") " nx, ny, nz = ", &
            domain%nx, ',', domain%ny, ',', domain%nz
        write(*, "(A,F9.6,A,F9.6,A,F9.6)") " dx, dy, dz (de) = ", &
            domain%dx, ',', domain%dy, ',', domain%dz
        write(*, "(A,E14.6)") " dtwpe = ", domain%dtwpe
        write(*, "(A,E14.6)") " dtwce = ", domain%dtwce
        write(*, "(A,E14.6)") " dtwci = ", domain%dtwci
        write(*, "(A,E14.6)") " Fields output interval (1/wpe) = ", domain%dt
        write(*, "(A,F6.1)") " mi/me = ", mime
        write(*, "(A,I0)") " Energies outptut steps = ", domain%energies_interval
        write(*, "(A,I0)") " Fields outptut steps = ", domain%fields_interval
        write(*, "(A,I0)") " Hydro outptut steps = ", domain%hydro_interval
        write(*, "(A,I0)") " Particle outptut steps = ", domain%Particle_interval
        write(*, "(A,I0)") " Numer of CPU cores used = ", domain%nproc
        write(*, "(A,I0,A,I0,A,I0)") " MPI topology: ", &
            domain%pic_tx, ',', domain%pic_ty, ',', domain%pic_tz
        write(*, "(A,I0,A,I0,A,I0)") " Domain sizes for each MPI process: ", &
            domain%pic_nx, ',', domain%pic_ny, ',', domain%pic_nz
        write(*, "(A,I0)") " nppc = ", domain%nppc
        print *,"---------------------------------------------------"
    end subroutine read_domain

    !---------------------------------------------------------------------------
    ! Read PIC MPI topology from info.bin.
    !---------------------------------------------------------------------------
    subroutine read_pic_mpi_topology
        use constants, only: dp
        implicit none
        real(dp) :: tx, ty, tz
        integer :: file_size, fh, index1, index2
        logical :: ex
        character(len=150) :: fname, buff
        ex = .false.
        fname = trim(adjustl(rootpath))//"info.bin"
        inquire(file=fname, exist=ex, size=file_size)
        fh = 20
        if (ex .and. file_size .ne. 0) then
            open(unit=fh, file=trim(adjustl(rootpath))//"info.bin", &
                 access='stream', status='unknown', form='unformatted', action='read')
            read(fh) tx, ty, tz
            close(fh)
        else
            fname = get_main_fname()
            open(unit=fh, file=trim(adjustl(rootpath))//trim(adjustl(fname)), status='old')
            read(fh, '(A)') buff
            do while (index(buff, 'double topology_x = ') == 0)
                read(fh, '(A)') buff
            enddo
            index1 = index(buff, '=')
            index2 = index(buff, ';')
            print*, buff
            read(buff(index1+1:index2-1), *) tx
            read(fh, '(A)') buff
            index1 = index(buff, '=')
            index2 = index(buff, ';')
            read(buff(index1+1:index2-1), *) ty
            read(fh, '(A)') buff
            index1 = index(buff, '=')
            index2 = index(buff, ';')
            read(buff(index1+1:index2-1), *) tz
            close(fh)
        endif
        ! Convert to integers
        domain%pic_tx = floor(tx + 0.5)
        domain%pic_ty = floor(ty + 0.5)
        domain%pic_tz = floor(tz + 0.5)

        domain%pic_nx = domain%nx / domain%pic_tx
        domain%pic_ny = domain%ny / domain%pic_ty
        domain%pic_nz = domain%nz / domain%pic_tz
    end subroutine read_pic_mpi_topology

    !---------------------------------------------------------------------------
    ! Write the PIC information to a file for IDL viewer.
    !---------------------------------------------------------------------------
    subroutine write_pic_info
        implicit none
        logical :: dir_e
        dir_e = .false.
        inquire(file=trim(adjustl(rootpath))//'data/.', exist=dir_e)
        if (.not. dir_e) then
            call system('mkdir ../data')
        endif
        open(unit=17, file=trim(adjustl(rootpath))//'data/info', &
             access='stream', status='replace', form='unformatted', action='write')
        write(17) domain%nx, domain%ny, domain%nz
        write(17) real(domain%lx_de, kind=4)
        write(17) real(domain%ly_de, kind=4)
        write(17) real(domain%lz_de, kind=4)
        close(17)
    end subroutine write_pic_info

    !---------------------------------------------------------------------------
    ! Broadcast the PIC simulation information to all MPI processes, including
    !   domain: the simulation domain information.
    !   mime: the ion to electron mass ratio.
    !---------------------------------------------------------------------------
    subroutine broadcast_pic_info
        use mpi_module
        implicit none
        integer :: picInfoType, oldtypes(0:2), blockcounts(0:2)
        integer :: offsets(0:2), extent
        ! Setup description of the 9 MPI_DOUBLE fields.
        offsets(0) = 0
        oldtypes(0) = MPI_DOUBLE_PRECISION
        blockcounts(0) = 9
        ! Setup description of the 6 MPI_REAL fields.
        call MPI_TYPE_EXTENT(MPI_DOUBLE_PRECISION, extent, ierr)
        offsets(1) = 9 * extent
        oldtypes(1) = MPI_REAL
        blockcounts(1) = 9
        ! Setup description of the 3 MPI_INTEGER fields.
        call MPI_TYPE_EXTENT(MPI_REAL, extent, ierr)
        offsets(2) = offsets(1) + 9 * extent
        oldtypes(2) = MPI_INTEGER
        blockcounts(2) = 15
        ! Define structured type and commit it. 
        call MPI_TYPE_STRUCT(3, blockcounts, offsets, oldtypes, picInfoType, ierr)
        call MPI_TYPE_COMMIT(picInfoType, ierr)
        call MPI_BCAST(domain, 1, picInfoType, master, MPI_COMM_WORLD, ierr)
        call MPI_BCAST(mime, 1, MPI_INTEGER, master, MPI_COMM_WORLD, ierr)
        call MPI_TYPE_FREE(picinfoType, ierr)
    end subroutine broadcast_pic_info

    !---------------------------------------------------------------------------
    ! Get the total number of time frames from file size.
    ! Method:
    !   Using the size of the post-processed field data and the dimensions of
    !   the PIC simulation (nx, ny, nz). Then, nt = filesize / (nx*ny*nz)
    ! Inputs:
    !   tp2: the maximum time frame defined by the user. It may be different
    !       from the total number of time frames of the PIC simulation.
    ! Updates:
    !   nt: the total time frames.
    !---------------------------------------------------------------------------
    subroutine get_total_time_frames(tp2)
        use mpi_module
        use path_info, only: filepath
        use configuration_translate, only: output_format
        implicit none
        integer, intent(inout) :: tp2
        integer(kind=8) :: filesize
        logical :: is_exist0, is_exist1, is_exist
        character(len=16) :: cfname
        character(len=150) :: fname
        integer :: tindex, nframe
        if (myid == master) then
            if (output_format == 1) then
                ! One field is saved in one file for all time steps.
                inquire(file=trim(adjustl(filepath))//'bx.gda', size=filesize)
                nt = filesize / (domain%nx*domain%ny*domain%nz*4)
            else
                is_exist = .false.
                is_exist0 = .false.
                is_exist1 = .false.
                inquire(file=trim(adjustl(filepath))//'bx_0.gda', exist=is_exist0)
                inquire(file=trim(adjustl(filepath))//'bx_1.gda', exist=is_exist1)
                is_exist = is_exist0 .or. is_exist1
                nframe = 0
                do while (is_exist)
                    nframe = nframe + 1
                    tindex = domain%fields_interval * nframe
                    write(cfname, '(I0)') tindex
                    fname = trim(adjustl(filepath))//'bx_'//trim(cfname)//'.gda'
                    inquire(file=fname, exist=is_exist)
                enddo
                nt = nframe
            endif
            print*, 'Number of output time frames: ', nt
        endif
        call MPI_BCAST(nt, 1, MPI_INTEGER, master, MPI_COMM_WORLD, ierr)
        ! Check if the last frame is larger than nt
        if (tp2 > nt) tp2 = nt
    end subroutine get_total_time_frames

    !---------------------------------------------------------------------------
    ! Get the total number of energy band.
    !---------------------------------------------------------------------------
    subroutine get_energy_band_number
        use mpi_module
        use constants, only: fp
        implicit none
        integer :: fh, index1, index2
        character(len=150) :: buff, single_line
        character(len=20) :: fname

        fh = 40
        fname = get_main_fname()
        open(unit=fh, file=trim(adjustl(rootpath))//trim(adjustl(fname)), status='old')
        read(fh, '(A)') buff
        do while (index(buff, 'global->nex') == 0)
            read(fh, '(A)') buff
        enddo
        index1 = index(buff, '=')
        index2 = index(buff, ';')
        read(buff(index1+1:index2-1), *) nbands

        ! When the energy diagnostics are commented
        do while (index(buff, 'energy.cxx') == 0)
            read(fh, '(A)') buff
        enddo
        single_line = trim(adjustl(buff))
        if (single_line(1:2) == '//') then
            nbands = 0
        endif

        if (myid == master) then
            write(*, "(A,I0)") " Number of energy band: ", nbands
        endif
        close(fh)
    end subroutine get_energy_band_number

end module picinfo
