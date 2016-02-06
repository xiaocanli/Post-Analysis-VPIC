!*******************************************************************************
! Module of particle maximum energy in each cell.
!*******************************************************************************
module particle_maximum_energy
    use constants, only: fp, dp
    use picinfo, only: domain 
    use path_info, only: rootpath
    implicit none
    private
    public distribute_pic_mpi, init_emax_array, free_emax_array, &
           update_emax_array, write_emax, set_emax_datatype, &
           free_emax_datatype
    public txs, tys, tzs, txe, tye, tze

    integer :: txp, typ, tzp, txs, tys, tzs, txe, tye, tze
    integer :: nx_tot, ny_tot, nz_tot, nxp, nyp, nzp, nxs, nys, nzs
    integer :: offset_nx, offset_ny, offset_nz
    integer, dimension(3) :: sizes, subsizes, starts
    real(fp), allocatable, dimension(:,:,:) :: emax
    integer :: datatype

    contains

    !---------------------------------------------------------------------------
    ! Distribute tasks based on the corners of the MPI processes of the PIC
    ! simulation for the spectrum calculation.
    !---------------------------------------------------------------------------
    subroutine distribute_pic_mpi
        use mpi_module
        use spectrum_config, only: corners_mpi
        use mpi_topology, only: distribute_tasks
        implicit none
        integer :: tx, ty, tz, txc, tyc, tzc, ix, iy, iz
        integer :: pic_nx, pic_ny, pic_nz
        tx = corners_mpi(2,1) - corners_mpi(1,1) + 1
        ty = corners_mpi(2,2) - corners_mpi(1,2) + 1
        tz = corners_mpi(2,3) - corners_mpi(1,3) + 1
        pic_nx = domain%pic_nx
        pic_ny = domain%pic_ny
        pic_nz = domain%pic_nz
        nx_tot = tx * pic_nx
        ny_tot = ty * pic_ny
        nz_tot = tz * pic_nz
        offset_nx = corners_mpi(1,1) * pic_nx
        offset_ny = corners_mpi(1,2) * pic_ny
        offset_nz = corners_mpi(1,3) * pic_nz
        ! Defaults topology for current analysis
        txc = numprocs
        tyc = 1
        tzc = 1
        call set_ana_topology(tx, ty, tz, txc, tyc, tzc)
        iz = myid / (txc*tyc)
        iy = mod(myid, txc*tyc) / txc
        ix = myid - iz*txc*tyc - iy*txc
        call distribute_tasks(tx, txc, ix, txp, txs, txe)
        call distribute_tasks(ty, tyc, iy, typ, tys, tye)
        call distribute_tasks(tz, tzc, iz, tzp, tzs, tze)
        ! Shift to the right position
        txs = txs + corners_mpi(1,1)
        txe = txe + corners_mpi(1,1)
        tys = tys + corners_mpi(1,2)
        tye = tye + corners_mpi(1,2)
        tzs = tzs + corners_mpi(1,3)
        tze = tze + corners_mpi(1,3)
        nxs = txs * pic_nx
        nys = tys * pic_ny
        nzs = tzs * pic_nz
        nxp = txp * pic_nx
        nyp = typ * pic_ny
        nzp = tzp * pic_nz
    end subroutine distribute_pic_mpi

    !---------------------------------------------------------------------------
    ! Initialize emax array
    !---------------------------------------------------------------------------
    subroutine init_emax_array
        implicit none
        allocate(emax(nxp, nyp, nzp))
        emax = 0.0
    end subroutine init_emax_array

    !---------------------------------------------------------------------------
    ! Free emax array
    !---------------------------------------------------------------------------
    subroutine free_emax_array
        implicit none
        deallocate(emax)
    end subroutine free_emax_array

    !---------------------------------------------------------------------------
    ! Update emax data array
    ! Input:
    !   emax_pic_mpi: maximum kinetic energy in each cell for one PIC MPI
    !   otx, oty, otz: the offsets in PIC MPI process
    !---------------------------------------------------------------------------
    subroutine update_emax_array(emax_pic_mpi, otx, oty, otz)
        use picinfo, only: domain
        implicit none
        real(fp), dimension(:, :, :), intent(in) :: emax_pic_mpi
        integer, intent(in) :: otx, oty, otz
        integer :: ixl, ixh, iyl, iyh, izl, izh
        ixl = otx*domain%pic_nx + 1
        iyl = oty*domain%pic_ny + 1
        izl = otz*domain%pic_nz + 1
        ixh = ixl + domain%pic_nx - 1
        iyh = iyl + domain%pic_ny - 1
        izh = izl + domain%pic_nz - 1
        emax(ixl:ixh, iyl:iyh, izl:izh) = emax_pic_mpi
    end subroutine update_emax_array

    !---------------------------------------------------------------------------
    ! Write emax array to file
    !---------------------------------------------------------------------------
    subroutine write_emax(ct, species)
        use mpi_module
        use path_info, only: outputpath
        use mpi_io_module, only: open_data_mpi_io, write_data_mpi_io
        use mpi_info_module, only: fileinfo
        implicit none
        integer, intent(in) :: ct
        character(len=1), intent(in) :: species
        integer(kind=MPI_OFFSET_KIND) :: disp, offset
        character(len=150) :: fname
        integer :: fh
        disp = domain%nx * domain%ny * domain%nz * sizeof(MPI_REAL) * (ct-1)
        offset = 0 
        fname = trim(adjustl(outputpath))//'emax_'//species//'.gda'
        call open_data_mpi_io(fname, MPI_MODE_RDWR+MPI_MODE_CREATE, fileinfo, fh)
        call write_data_mpi_io(fh, datatype, subsizes, disp, offset, emax)
        call MPI_FILE_CLOSE(fh, ierror)
    end subroutine write_emax

    !---------------------------------------------------------------------------
    ! Set emax datatype for MPI IO
    !---------------------------------------------------------------------------
    subroutine set_emax_datatype
        use mpi_datatype_module, only: set_mpi_datatype
        implicit none
        sizes = [nx_tot, ny_tot, nz_tot]
        subsizes = [nxp, nyp, nzp]
        starts = [nxs, nys, nzs]
        datatype = set_mpi_datatype(sizes, subsizes, starts)
    end subroutine set_emax_datatype

    !---------------------------------------------------------------------------
    ! Free emax datatype for MPI IO
    !---------------------------------------------------------------------------
    subroutine free_emax_datatype
        use mpi_module
        implicit none
        call MPI_TYPE_FREE(datatype, ierror)
    end subroutine free_emax_datatype

    !---------------------------------------------------------------------------
    ! Set topology for current analysis
    ! Input:
    !   tx, ty, tz: the box sizes in PIC MPI topology
    !---------------------------------------------------------------------------
    subroutine set_ana_topology(tx, ty, tz, txc, tyc, tzc)
        use mpi_module
        implicit none
        integer, intent(in) :: tx, ty, tz
        integer, intent(out) :: txc, tyc, tzc
        integer :: tot_mpi
        tot_mpi = tx * ty * tz
        if (myid > (tot_mpi - 1)) then
            ! To many processes to use
            txc = 0
            tyc = 0
            tzc = 0
        else
            if (numprocs <= tx) then
                ! Most cases
                txc = numprocs
            else
                ! Possible for 3D
                txc = gcd(tx, numprocs)
                if (numprocs / txc <= ty) then
                    tyc = numprocs / txc
                else
                    tyc = gcd(ty, numprocs/txc)
                    tzc = numprocs/txc/tyc
                    if (tzc > tz) then
                        call MPI_FINALIZE(ierr)
                        print*, "Impossible to balance the topology."
                        stop
                    endif
                endif
            endif
        endif
    end subroutine set_ana_topology

    !---------------------------------------------------------------------------
    ! Get the greatest common divisor of two numbers
    !---------------------------------------------------------------------------
    function gcd(a0, b0) result (a)
        implicit none
        integer, intent(in) :: a0, b0
        integer :: a, b, t
        a = a0
        b = b0
        do while (b /= 0)
            t = b
            b = mod(a, b)
            a = t
        enddo
    end function gcd
end module particle_maximum_energy
