!<******************************************************************************
!< Combine particle energy spectra saved in binary
!<******************************************************************************
program combine_spectrum_binary
    use constants, only: fp, dp
    use mpi_module
    implicit none
    real(fp), allocatable, dimension(:, :, :, :) :: pspect
    real(dp) :: start, finish
    integer :: tstart, tend, tinterval, tframe
    character(len=256) :: rootpath, output_path
    integer :: pic_mpi_size, pic_mpi_sizex, pic_mpi_sizey, pic_mpi_sizez
    integer :: mpi_sizex, mpi_sizey, mpi_sizez
    integer :: nzones, ndata, num_fold
    integer :: nx, ny, nz, nz_local
    integer :: mpi_rankx, mpi_ranky, mpi_rankz

    call MPI_INIT(ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)
    call cpu_time(start)
    call get_cmd_args

    if ((mod(pic_mpi_sizex, mpi_sizex) .NE. 0) .or. &
        (mod(pic_mpi_sizey, mpi_sizey) .NE. 0) .or. &
        (mod(pic_mpi_sizez, mpi_sizez) .NE. 0) .or. &
        ((mpi_sizex * mpi_sizey * mpi_sizez) .NE. numprocs)) then
        print*, "ERROR: PIC MPI size cannot be divided by current mpi_size"
        call MPI_FINALIZE(ierr)
    endif

    nx = pic_mpi_sizex / mpi_sizex
    ny = pic_mpi_sizey / mpi_sizey
    nz = pic_mpi_sizez / mpi_sizez
    nz_local = nz * nzones
    mpi_rankz = myid / (mpi_sizex * mpi_sizey)
    mpi_ranky = mod(myid, mpi_sizex * mpi_sizey) / mpi_sizex
    mpi_rankx = mod(myid, mpi_sizex)

    allocate(pspect(ndata, nx, ny, nz_local))
    pspect = 0.0

    do tframe = tstart, tend, tinterval
        if (myid == master) print*, tframe
        call combine_spectrum_single(tframe, 'e')
        call save_energy_spectrum(tframe, 'e')
        call combine_spectrum_single(tframe, 'H')
        call save_energy_spectrum(tframe, 'H')
    enddo

    deallocate(pspect)

    call cpu_time(finish)
    if (myid == master) then
        print '("Time = ",f9.4," seconds.")',finish-start
    endif

    call MPI_FINALIZE(ierr)

    contains

    !<--------------------------------------------------------------------------
    !< Combine spectrum for a single time frame
    !<--------------------------------------------------------------------------
    subroutine combine_spectrum_single(tframe, species)
        implicit none
        integer, intent(in) :: tframe
        character(*), intent(in) :: species
        character(len=256) :: fname, fdir
        character(len=8) :: tframe_char, findex_char, rank_char
        real(fp), allocatable, dimension(:,:) :: buffer
        integer :: sx, sy, sz, py, pz
        integer :: ix, iy, iz, pic_mpi_rank
        integer :: fh, r_status, access, file_size
        logical :: file_exist

        sx = mpi_rankx * nx
        sy = mpi_ranky * ny
        sz = mpi_rankz * nz

        allocate(buffer(ndata, nzones))

        fh = 30
        write(tframe_char, "(I0)") tframe
        do iz = 0, nz - 1
            pz = (sz + iz) * pic_mpi_sizex * pic_mpi_sizey
            do iy = 0, ny - 1
                py = (sy + iy) * pic_mpi_sizex
                do ix = 0, nx - 1
                    pic_mpi_rank = pz + py + sx + ix
                    write(findex_char, "(I0)") (pic_mpi_rank/num_fold)
                    write(rank_char, "(I0)") pic_mpi_rank
                    fname = trim(adjustl(rootpath))//"/hydro/"//trim(findex_char)
                    fname = trim(fname)//"/T."//trim(tframe_char)
                    fname = trim(fname)//"/spectrum-"//species//"hydro."
                    fname = trim(fname)//trim(tframe_char)//"."//trim(rank_char)
                    inquire(file=trim(fname), exist=file_exist)
                    r_status = access(trim(fname), 'r')
                    if (file_exist .and. r_status == 0) then
                        inquire(file=trim(fname), size=file_size)
                        if (file_size == ndata * nzones * 4) then
                            open(unit=fh, file=trim(fname), access='stream', status='unknown', &
                                 form='unformatted', action='read')
                            read(fh) buffer
                            pspect(:, ix+1, iy+1, iz*nzones+1:(iz+1)*nzones) = buffer
                        else
                            pspect(:, ix+1, iy+1, iz*nzones+1:(iz+1)*nzones) = 0.0
                        endif
                    else
                        pspect(:, ix+1, iy+1, iz*nzones+1:(iz+1)*nzones) = 0.0
                    endif
                    close(fh)
                enddo
            enddo
        enddo
        deallocate(buffer)
    end subroutine combine_spectrum_single

    !<--------------------------------------------------------------------------
    !< Save particle particle energy spectra
    !<--------------------------------------------------------------------------
    subroutine save_energy_spectrum(tindex, species)
        implicit none
        integer, intent(in) :: tindex
        character(*), intent(in) :: species
        integer :: fh1, posf
        character(len=16) :: tindex_str
        character(len=256) :: fname, fpath
        integer, dimension(4) :: sizes, subsizes, starts
        integer :: fileinfo, datatype, fh
        integer(kind=MPI_OFFSET_KIND) :: disp, offset

        if (myid == 0) then
            print*, "Writting spectrum data to file"
        endif

        sizes = (/ ndata, pic_mpi_sizex, pic_mpi_sizey, pic_mpi_sizez*nzones /)
        subsizes = (/ ndata, nx, ny, nz*nzones /)
        starts(1) = 0
        starts(2) = nx * mpi_rankx
        starts(3) = ny * mpi_ranky
        starts(4) = nz * mpi_rankz * nzones

        call MPI_TYPE_CREATE_SUBARRAY(4, sizes, subsizes, starts, &
                MPI_ORDER_FORTRAN, MPI_REAL, datatype, ierror)
        call MPI_TYPE_COMMIT(datatype, ierror)

        fh = 40
        call MPI_INFO_CREATE(fileinfo, ierror)
        call MPI_INFO_SET(fileinfo, "romio_ds_read", "disable", ierror)
        call MPI_INFO_SET(fileinfo, "romio_ds_write", "disable", ierror)
        call MPI_INFO_SET(fileinfo, "romio_cb_read", "enable", ierror)
        call MPI_INFO_SET(fileinfo, "romio_cb_write", "enable", ierror)
        write(tindex_str, "(I0)") tindex
        fname = trim(adjustl(output_path))//"spectrum_"//species
        fname = trim(fname)//"_"//trim(tindex_str)//".gda"
        call MPI_FILE_OPEN(MPI_COMM_WORLD, trim(fname), &
            MPI_MODE_RDWR+MPI_MODE_CREATE, fileinfo, fh, ierror)
        if (ierror /= 0) then
            call MPI_ERROR_STRING(ierror, err_msg, err_length, ierror2)
            print*, "Error in MPI_FILE_OPEN: ", trim(err_msg)
        endif

        offset = 0
        disp = 0
        call MPI_FILE_SET_VIEW(fh, disp, MPI_REAL, datatype, 'native', &
            MPI_INFO_NULL, ierror)
        if (ierror /= 0) then
            call MPI_ERROR_STRING(ierror, err_msg, err_length, ierror2)
            print*, "Error in MPI_FILE_SET_VIEW: ", trim(err_msg)
        endif

        call MPI_FILE_WRITE_AT_ALL(fh, offset, pspect, &
                product(subsizes), MPI_REAL, status, ierror)
        if (ierror /= 0) then
            call MPI_ERROR_STRING(ierror, err_msg, err_length, ierror2)
            print*, "Error in MPI_FILE_WRITE: ", trim(err_msg)
        endif

        call MPI_FILE_CLOSE(fh, ierror)

        call MPI_TYPE_FREE(datatype, ierror)
    end subroutine save_energy_spectrum

    !<--------------------------------------------------------------------------
    !< Get commandline arguments
    !<--------------------------------------------------------------------------
    subroutine get_cmd_args
        use flap                                !< FLAP package
        use penf
        implicit none
        type(command_line_interface) :: cli     !< Command Line Interface (CLI).
        integer(I4P)                 :: error   !< Error trapping flag.
        call cli%init(progname = 'combine_spectrum_binary', &
            authors     = 'Xiaocan Li', &
            help        = 'Usage: ', &
            description = 'Combine particle energy spectrum in binary', &
            examples    = ['combine_spectrum_binary -rp rootpath'])
        call cli%add(switch='--rootpath', switch_ab='-rp', &
            help='simulation root path', required=.true., act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--output_path', switch_ab='-op', &
            help='data output path', required=.true., act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--pic_mpi_size', switch_ab='-pm', &
            help='MPI size for PIC simulation', &
            required=.false., act='store', def='131072', error=error)
        if (error/=0) stop
        call cli%add(switch='--pic_mpi_sizex', switch_ab='-px', &
            help='MPI size for PIC simulation along x', &
            required=.false., act='store', def='256', error=error)
        if (error/=0) stop
        call cli%add(switch='--pic_mpi_sizey', switch_ab='-py', &
            help='MPI size for PIC simulation along y', &
            required=.false., act='store', def='256', error=error)
        if (error/=0) stop
        call cli%add(switch='--pic_mpi_sizez', switch_ab='-pz', &
            help='MPI size for PIC simulation along z', &
            required=.false., act='store', def='2', error=error)
        if (error/=0) stop
        call cli%add(switch='--mpi_sizex', switch_ab='-mx', &
            help='MPI size for current analysis along x', &
            required=.false., act='store', def='4', error=error)
        if (error/=0) stop
        call cli%add(switch='--mpi_sizey', switch_ab='-my', &
            help='MPI size for current analysis along y', &
            required=.false., act='store', def='4', error=error)
        if (error/=0) stop
        call cli%add(switch='--mpi_sizez', switch_ab='-mz', &
            help='MPI size for current analysis along z', &
            required=.false., act='store', def='2', error=error)
        if (error/=0) stop
        call cli%add(switch='--nzones', switch_ab='-nz', &
            help='Number of zones in each PIC MPI rank', &
            required=.false., act='store', def='8', error=error)
        if (error/=0) stop
        call cli%add(switch='--num_fold', switch_ab='-nf', &
            help='Number of files in each sum-directory', &
            required=.false., act='store', def='32', error=error)
        if (error/=0) stop
        call cli%add(switch='--ndata', switch_ab='-nd', &
            help='Number of data points for each energy spectrum', &
            required=.false., act='store', def='600', error=error)
        if (error/=0) stop
        call cli%add(switch='--tstart', switch_ab='-ts', &
            help='Starting time frame', required=.false., act='store', &
            def='0', error=error)
        if (error/=0) stop
        call cli%add(switch='--tend', switch_ab='-te', help='Last time frame', &
            required=.false., act='store', def='5', error=error)
        if (error/=0) stop
        call cli%add(switch='--tinterval', switch_ab='-ti', help='Time interval', &
            required=.false., act='store', def='2732', error=error)
        if (error/=0) stop
        call cli%get(switch='-rp', val=rootpath, error=error)
        if (error/=0) stop
        call cli%get(switch='-op', val=output_path, error=error)
        if (error/=0) stop
        call cli%get(switch='-pm', val=pic_mpi_size, error=error)
        if (error/=0) stop
        call cli%get(switch='-px', val=pic_mpi_sizex, error=error)
        if (error/=0) stop
        call cli%get(switch='-py', val=pic_mpi_sizey, error=error)
        if (error/=0) stop
        call cli%get(switch='-pz', val=pic_mpi_sizez, error=error)
        if (error/=0) stop
        call cli%get(switch='-mx', val=mpi_sizex, error=error)
        if (error/=0) stop
        call cli%get(switch='-my', val=mpi_sizey, error=error)
        if (error/=0) stop
        call cli%get(switch='-mz', val=mpi_sizez, error=error)
        if (error/=0) stop
        call cli%get(switch='-nz', val=nzones, error=error)
        if (error/=0) stop
        call cli%get(switch='-nf', val=num_fold, error=error)
        if (error/=0) stop
        call cli%get(switch='-nd', val=ndata, error=error)
        if (error/=0) stop
        call cli%get(switch='-ts', val=tstart, error=error)
        if (error/=0) stop
        call cli%get(switch='-te', val=tend, error=error)
        if (error/=0) stop
        call cli%get(switch='-ti', val=tinterval, error=error)
        if (error/=0) stop

        if (myid == 0) then
            print '(A,A)', ' The simulation rootpath: ', trim(adjustl(rootpath))
            print '(A,A)', ' Data output path: ', trim(adjustl(output_path))
            print '(A,I0,A,I0,A,I0)', ' Min, max and interval: ', &
                tstart, ' ', tend, ' ', tinterval
            print '(A,I0)', ' MPI size of the PIC simulation: ', pic_mpi_size
            print '(A,I0,A,I0,A,I0)', ' MPI size along each direction for PIC: ', &
                pic_mpi_sizex, ' ', pic_mpi_sizey, ' ', pic_mpi_sizez
            print '(A,I0)', ' Number of zones in each PIC MPI rank: ', nzones
            print '(A,I0)', ' Number of files in each sub-directory: ', num_fold
            print '(A,I0)', ' Number of data points ine each spectrum: ', ndata
            print '(A,I0,A,I0,A,I0)', &
                ' MPI size along each direction for current analysis: ', &
                mpi_sizex, ' ', mpi_sizey, ' ', mpi_sizez
        endif
    end subroutine get_cmd_args

end program combine_spectrum_binary
