!*******************************************************************************
! Module of file path information, including the root path of current PIC run,
! the filepath of the data files (bx, by, bz ...) and the outputpath of current
! analysis.
!*******************************************************************************
module path_info
    implicit none
    save
    private
    public rootpath, filepath, outputpath, get_file_paths
    character(len=150) :: rootpath, filepath, outputpath

    contains

    subroutine get_file_paths
        implicit none
        integer :: status1, getcwd, index1
        status1 = getcwd(rootpath)
        index1 = index(rootpath, '/', back=.true.)
        rootpath = rootpath(1:index1)
        filepath = trim(rootpath)//'data/'
        outputpath = trim(rootpath)//'data1/'
    end subroutine get_file_paths
end module path_info


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
        real(fp) :: dtwpe, dtwce, dtwci   ! Time step in PIC simulation
        real(fp) :: dt, idt               ! Fields output interval and its inverse
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
    ! Get the fields output time interval from the initialization file sigma.cxx.
    ! The interval is defined as "int interval = ...", for example,
    ! int interval = int(2.5/(wci*dt)); 
    ! Return:
    !   dtf_wci: the fields output interval in 1/wci.
    !---------------------------------------------------------------------------
    function get_fields_interval() result(dtf_wci)
        use constants, only: fp
        implicit none
        real(fp) :: dtf_wci ! The interval in 1/wci
        integer :: fh, index1, index2
        character(len=150) :: buff
        fh = 40
        open(unit=fh, file=trim(adjustl(rootpath))//'sigma.cxx', status='old')
        read(fh, '(A)') buff
        do while (index(buff, 'int interval = ') == 0)
            read(fh, '(A)') buff
        enddo
        index1 = index(buff, '(')
        index2 = index(buff, '/')
        read(buff(index1+1:index2-1), *) dtf_wci
        close(fh)
    end function get_fields_interval

    !---------------------------------------------------------------------------
    ! Read domain information from info file directly from the PIC simulation.
    ! Information to read:
    !   domain: the domain information of the PIC simulation.
    !   mime: the ion to electron mass ration.
    !---------------------------------------------------------------------------
    subroutine read_domain
        use constants, only: dp, fp
        implicit none
        real(fp) :: temp, dtf_wci
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
        domain%dx = get_variable(fh, 'dx/de', '=')
        domain%dy = get_variable(fh, 'dy/de', '=')
        domain%dz = get_variable(fh, 'dz/de', '=')
        domain%idx = 1.0/domain%dx
        domain%idy = 1.0/domain%dy
        domain%idz = 1.0/domain%dz
        domain%idxh = 0.5*domain%idx
        domain%idyh = 0.5*domain%idy
        domain%idzh = 0.5*domain%idz

        dtf_wci = get_fields_interval()
        domain%dt = dtf_wci * domain%dtwpe / domain%dtwci
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
        open(unit=20, file=trim(adjustl(rootpath))//"info.bin", &
             access='stream', status='unknown', form='unformatted', action='read')
        read(20) tx, ty, tz
        close(20)
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
        blockcounts(1) = 8
        ! Setup description of the 3 MPI_INTEGER fields.
        call MPI_TYPE_EXTENT(MPI_REAL, extent, ierr)
        offsets(2) = offsets(1) + 8 * extent
        oldtypes(2) = MPI_INTEGER
        blockcounts(2) = 11
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
        implicit none
        integer, intent(inout) :: tp2
        integer(kind=8) :: filesize
        if (myid == master) then
            inquire(file=trim(adjustl(filepath))//'bx.gda', size=filesize)
            nt = filesize / (domain%nx*domain%ny*domain%nz*4)
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
        use constants, only: fp
        implicit none
        integer :: fh, index1, index2
        character(len=150) :: buff

        fh = 40
        open(unit=fh, file=trim(adjustl(rootpath))//'sigma.cxx', status='old')
        read(fh, '(A)') buff
        do while (index(buff, 'global->nex') == 0)
            read(fh, '(A)') buff
        enddo
        index1 = index(buff, '=')
        index2 = index(buff, ';')
        read(buff(index1+1:index2-1), *) nbands
        close(fh)
    end subroutine get_energy_band_number

end module picinfo
