!!******************************************************************************
!! Module of doing interpolation in particle fields
!!******************************************************************************
module interpolation_particle_fields
    use mpi_module
    use constants, only: fp
    implicit none
    private
    public init_velocity_fields, free_velocity_fields, calc_vsingle, &
        init_number_density, free_number_density, trilinear_interp_vel
    public vsx, vsy, vsz, vx1, vy1, vz1, vx2, vy2, vz2, nrho1, nrho2, ntot, &
        vsx0, vsy0, vsz0
    real(fp), allocatable, dimension(:,:,:) :: vx1, vy1, vz1, vsx, vsy, vsz
    real(fp), allocatable, dimension(:,:,:) :: vx2, vy2, vz2, nrho1, nrho2, ntot
    real(fp) :: vsx0, vsy0, vsz0
    real(fp), dimension(2,2,2) :: weights  ! The weights for trilinear interpolation
    integer :: nx, ny, nz

    contains

    !!--------------------------------------------------------------------------
    !! Initialize the velocity fields
    !!--------------------------------------------------------------------------
    subroutine init_velocity_fields
        use picinfo, only: domain
        implicit none
        nx = domain%pic_nx + 2  ! Including ghost cells
        ny = domain%pic_ny + 2
        nz = domain%pic_nz + 2

        allocate(vx1(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(vy1(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(vz1(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(vx2(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(vy2(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(vz2(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(vsx(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(vsy(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(vsz(0:nx-1, 0:ny-1, 0:nz-1))

        vx1 = 0.0; vy1 = 0.0; vz1 = 0.0
        vx2 = 0.0; vy2 = 0.0; vz2 = 0.0
        vsx = 0.0; vsy = 0.0; vsz = 0.0
    end subroutine init_velocity_fields

    !!--------------------------------------------------------------------------
    !! Initialize the number density
    !!--------------------------------------------------------------------------
    subroutine init_number_density
        use picinfo, only: domain
        implicit none
        nx = domain%pic_nx + 2  ! Including ghost cells
        ny = domain%pic_ny + 2
        nz = domain%pic_nz + 2

        allocate(nrho1(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(nrho2(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(ntot(0:nx-1, 0:ny-1, 0:nz-1))

        nrho1 = 0.0
        nrho2 = 0.0
        ntot = 0.0
    end subroutine init_number_density

    !!--------------------------------------------------------------------------
    !! Free the velocity fields
    !!--------------------------------------------------------------------------
    subroutine free_velocity_fields
        implicit none
        deallocate(vx1, vy1, vz1)
        deallocate(vx2, vy2, vz2)
        deallocate(vsx, vsy, vsz)
    end subroutine free_velocity_fields

    !!--------------------------------------------------------------------------
    !! Free the number density
    !!--------------------------------------------------------------------------
    subroutine free_number_density
        implicit none
        deallocate(nrho1, nrho2, ntot)
    end subroutine free_number_density

    !!--------------------------------------------------------------------------
    !! Open the hydro fields for both top part and bottom part of the PIC
    !! simulation.
    !!--------------------------------------------------------------------------
    subroutine open_hydro_files_tb(fh, tindex0, pic_mpi_id)
        implicit none
        integer, dimension(*), intent(in) :: fh
        integer, intent(in) :: tindex0, pic_mpi_id
        call open_hydro_file(fh(1), tindex0, pic_mpi_id, 'e', 'Top')
        call open_hydro_file(fh(2), tindex0, pic_mpi_id, 'e', 'Bot')
        call open_hydro_file(fh(3), tindex0, pic_mpi_id, 'H', 'Top')
        call open_hydro_file(fh(4), tindex0, pic_mpi_id, 'H', 'Bot')
    end subroutine open_hydro_files_tb

    !!--------------------------------------------------------------------------
    !! Open one hydro field file for a single MPI process of PIC simulation.
    !! Inputs:
    !!   fh: file handler.
    !!   tindex0: the time step index.
    !!   pic_mpi_id: MPI id for the PIC simulation to identify the file.
    !!   species: particle species. 'e' or 'H'
    !!   pos: position. "Top" or "Bot"
    !!--------------------------------------------------------------------------
    subroutine open_hydro_file(fh, tindex0, pic_mpi_id, species, pos)
        use path_info, only: rootpath
        use file_header, only: read_boilerplate, read_fields_header
        implicit none
        integer, intent(in) :: fh, tindex0, pic_mpi_id
        character(*), intent(in) :: species, pos
        character(len=256) :: fname
        logical :: is_exist
        integer :: tindex, is_exist_int
        tindex = tindex0
        is_exist = .false.
        is_exist_int = 0
        ! Index 0 does not have proper current, so use index 1 if it exists
        if (tindex == 0) then
            write(fname, "(A,I0,A,A,A,A,I0,A,I0)") &
                trim(adjustl(rootpath))//"hydro/T.", &
                1, "/", species, pos, "hydro.", 1, ".", pic_mpi_id
            if (myid == master) then
                inquire(file=trim(fname), exist=is_exist)
                if (is_exist) is_exist_int = 1
            endif
            call MPI_BCAST(is_exist_int, 1, MPI_INTEGER, master, &
                MPI_COMM_WORLD, ierr)
        endif
        if (is_exist_int == 1) tindex = 1
        write(fname, "(A,I0,A,A,A,A,I0,A,I0)") &
            trim(adjustl(rootpath))//"hydro/T.", &
            tindex, "/", species, pos, "hydro.", tindex, ".", pic_mpi_id
        is_exist = .false.
        is_exist_int = 0
        if (myid == master) then
            inquire(file=trim(fname), exist=is_exist)
            if (is_exist) is_exist_int = 1
        endif
        call MPI_BCAST(is_exist_int, 1, MPI_INTEGER, master, &
            MPI_COMM_WORLD, ierr)
      
        if (is_exist_int == 1) then 
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

    end subroutine open_hydro_file

    !!--------------------------------------------------------------------------
    !! Read the hdyro fields for a single MPI process of PIC simulation and
    !! calculate the single flow velocity
    !! Inputs:
    !!   tindex0: the time step index.
    !!   pic_mpi_id: MPI id for the PIC simulation to identify the file.
    !!--------------------------------------------------------------------------
    subroutine calc_vsingle(tindex0, pic_mpi_id)
        use constants, only: fp, dp
        use neighbors_module, only: get_mpi_neighbors
        use picinfo, only: domain, mime
        use file_header, only: v0, pheader
        implicit none
        integer, intent(in) :: tindex0, pic_mpi_id
        integer, dimension(4) :: fh
        integer :: i

        fh = [20, 21, 22, 23]
        call open_hydro_files_tb(fh, tindex0, pic_mpi_id)
        call read_vel_rho(fh(1:2))
        ! Notice: the read velocities are actually q*n*v
        ! Notice: nrho is actually charge density
        vsx = -vx1 - vx2
        vsy = -vy1 - vy2
        vsz = -vz1 - vz2
        ntot = abs(nrho1 + nrho2)
        call read_vel_rho(fh(3:4))
        vsx = vsx + (vx1 + vx2) * mime
        vsy = vsy + (vy1 + vy2) * mime
        vsz = vsz + (vz1 + vz2) * mime
        ntot = ntot + abs((nrho1 + nrho2)) * mime
        where (ntot > 0.0)
            vsx = vsx / ntot
            vsy = vsy / ntot
            vsz = vsz / ntot
        elsewhere
            vsx = 0.0
            vsy = 0.0
            vsz = 0.0
        endwhere

        do i = 1, 4
            close(fh(i))
        enddo
    end subroutine calc_vsingle

    !!--------------------------------------------------------------------------
    !! Read the velocity and number density from one hydro file.
    !! Notice: the velocities are actually q*n*v
    !! Input:
    !!   fh: file handler
    !!--------------------------------------------------------------------------
    subroutine read_vel_rho(fh)
        implicit none
        integer, dimension(*), intent(in) :: fh
        read(fh(1)) vx1
        read(fh(1)) vy1
        read(fh(1)) vz1
        read(fh(1)) nrho1
        read(fh(2)) vx2
        read(fh(2)) vy2
        read(fh(2)) vz2
        read(fh(2)) nrho2
    end subroutine read_vel_rho

    !!--------------------------------------------------------------------------
    !! Calculate the weights for trilinear interpolation.
    !!
    !! Input:
    !!   dx, dy, dz: the distance to the lower-left corner, [0, 1]
    !!--------------------------------------------------------------------------
    subroutine calc_interp_weights(dx, dy, dz)
        implicit none
        real(fp), intent(in) :: dx, dy, dz
        weights(1, 1, 1) = (1 - dx) * (1 - dy) * (1 - dz)
        weights(2, 1, 1) = dx * (1 - dy) * (1 - dz)
        weights(1, 2, 1) = (1 - dx) * dy * (1 - dz)
        weights(2, 2, 1) = dx * dy * (1 - dz)
        weights(1, 1, 2) = (1 - dx) * (1 - dy) * dz
        weights(2, 1, 2) = dx * (1 - dy) * dz
        weights(1, 2, 2) = (1 - dx) * dy * dz
        weights(2, 2, 2) = dx * dy * dz
    end subroutine calc_interp_weights

    !---------------------------------------------------------------------------
    ! Trilinear interpolation for vsx, vsy, vsz.
    ! 
    ! Input:
    !   ix0, iy0, iz0: the indices of the lower-left corner.
    !   dx, dy, dz: the distance to the lower-left corner, [0, 1]
    !---------------------------------------------------------------------------
    subroutine trilinear_interp_vel(ix0, iy0, iz0, dx, dy, dz)
        implicit none
        integer, intent(in) :: ix0, iy0, iz0
        real(fp), intent(in) :: dx, dy, dz
        call calc_interp_weights(dx, dy, dz)
        vsx0 = sum(vsx(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        vsy0 = sum(vsy(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        vsz0 = sum(vsz(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
    end subroutine trilinear_interp_vel

end module interpolation_particle_fields
