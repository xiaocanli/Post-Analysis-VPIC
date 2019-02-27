!<******************************************************************************
!< Reduce particle energy spectra saved in HDF5 to a smaller size
!< We assume the spectrum is well-organized in order
!<  ndata * (pic_topox*nzonex) * (pic_topoy*nzoney) * (pic_topoz*nzonez)
!< This can be achieved when dumping spectra during VPIC simulations. Or, if
!< the spectrum data is originally saved in
!< ndata * nzonex * nzoney * nzonez * pic_topox * pic_topoy * pic_topoz,
!< you can use the program reorganize_spectrum.f90 to reorganize the data.
!< You need to carefully choose your MPI toplogy (topox, topoy, and topoz) and
!< the reduced factors (nreducex, nreducey, nreducez) to avoid communication
!< between different MPI ranks. We will take the x-direction for example in the
!< discussion below. Typically, we want to make sure that
!<  1. pic_topox*nzonex is divisible by topox,
!<  2. pic_topox*nzonex/topox is divisible by nreducex,
!< so we don't have to communicate between different MPI ranks. But, in some
!< cases that might be what you want. For example, we had one simulation with
!<  * pic_topox = pic_topoy = 256, pic_topoz = 2
!<  * nzonex = nzoney = 1, nzonez = 80
!<  * each zone has sizes 12 * 6 * 8.
!< We hope to reduce the spectra to sizes 48*48*48, so the reduced factors are
!< 4, 8, and 6. The problem is that pic_topoz*nzonez is not divisible by nreducez.
!< In such case, we reduce the data to (pic_topoz*nzonez/nreducez) + 2 and assign
!< mod(pic_topoz*nzonez, nreducez) zones to the additional 2 cells at the boundary,
!< and we'd better choose topoz=1 to make life easier.
!<******************************************************************************
program reduce_organized_spectrum_hdf5
    use constants, only: fp, dp
    use mpi_module
    use hdf5
    implicit none
    integer :: pic_topox, pic_topoy, pic_topoz
    integer :: nzonex, nzoney, nzonez, ndata
    integer :: nreducex, nreducey, nreducez
    integer :: topox, topoy, topoz
    integer :: nx_reduce, ny_reduce, nz_reduce, nzones
    integer :: shiftx, shifty, shiftz
    integer :: nx_local, ny_local, nz_local
    integer :: mpi_rankx, mpi_ranky, mpi_rankz
    integer :: tstart, tend, tinterval, tframe
    integer(hsize_t) :: pic_mpi_size
    real(fp), allocatable, dimension(:, :, :, :) :: pspect
    real(fp), allocatable, dimension(:, :, :, :) :: pspect_reduced
    real(fp), allocatable, dimension(:, :, :) :: nzones_reduced
    character(len=256) :: rootpath
    character(len=32) :: input_path, output_path, input_suffix, output_suffix
    integer :: ny_reduce_local
    integer :: t1, t2, clock_rate, clock_max

    call MPI_INIT(ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)
    call get_cmd_args

    if (topox*topoy*topoz /= numprocs) then
        if (myid == master) then
            print*, "Wrong MPI topology"
        endif
        call MPI_FINALIZE(ierr)
    endif

    call system_clock(t1, clock_rate, clock_max)

    call check_reduce_factor(pic_topox, nzonex, nreducex, topox, nx_reduce, shiftx)
    call check_reduce_factor(pic_topoy, nzoney, nreducey, topoy, ny_reduce, shifty)
    call check_reduce_factor(pic_topoz, nzonez, nreducez, topoz, nz_reduce, shiftz)
    if (nx_reduce == 0 .or. ny_reduce == 0 .or. nz_reduce == 0) then
        if (myid == master) then
            print*, "Wrong MPI topology or reduce factors"
        endif
        call MPI_FINALIZE(ierr)
    endif

    pic_mpi_size = pic_topox * pic_topoy * pic_topoz
    nzones = nzonex * nzoney * nzonez
    nx_local = (pic_topox*nzonex) / topox
    ny_local = (pic_topoy*nzoney) / topoy
    nz_local = (pic_topoz*nzonez) / topoz
    mpi_rankx = mod(myid, topox)
    mpi_ranky = mod(myid, topox*topoy) / topox
    mpi_rankz = myid / (topox*topoy)

    call reduce_spectrum

    call system_clock(t2, clock_rate, clock_max)
    if (myid == master) then
        write (*, *) 'Elapsed real time = ', real(t2 - t1) / real(clock_rate)
    endif

    call MPI_FINALIZE(ierr)

    contains

    !<--------------------------------------------------------------------------
    !< Check reduce factor. We hope to avoid some unusual situations.
    !< This may be complete.
    !<--------------------------------------------------------------------------
    subroutine check_reduce_factor(pic_topo, nzone, reduce_factor, topo, nreduce, shift)
        implicit none
        integer, intent(in) :: pic_topo, nzone, reduce_factor, topo
        integer, intent(out) :: nreduce, shift
        nreduce = 0
        shift = 0
        if (mod(pic_topo*nzone, reduce_factor) /= 0) then
            if (topo /= 1) then
                nreduce = 0
            else
                nreduce = (pic_topo*nzone) / reduce_factor + 2
                ! We assume here that pic_topo*nzone is an even number
                shift = (pic_topo*nzone - (nreduce-2)*reduce_factor) / 2
            endif
        else
            if (mod(pic_topo*nzone, topo) /= 0) then
                nreduce = 0
            else
                if (mod((pic_topo*nzone)/topo, reduce_factor) /= 0) then
                    nreduce = 0
                else
                    nreduce = (pic_topo*nzone)/topo/reduce_factor
                    shift = 0
                endif
            endif
        endif
    end subroutine check_reduce_factor

    !<--------------------------------------------------------------------------
    !< Reduce spectrum for all time frames.
    !<--------------------------------------------------------------------------
    subroutine reduce_spectrum
        implicit none
        integer(hsize_t), dimension(4) :: dcount, doffset
        integer(hsize_t), dimension(4) :: dcount_r, doffset_r
        integer :: t1, t2, clock_rate, clock_max

        ! Data count and offset for reading data
        dcount(1) = ndata
        dcount(2) = nx_local
        dcount(3) = ny_local
        dcount(4) = nz_local
        doffset(1) = 0
        doffset(2) = dcount(2) * mpi_rankx
        doffset(3) = dcount(3) * mpi_ranky
        doffset(4) = dcount(4) * mpi_rankz

        ! Data count and offset for reduced data
        dcount_r(1) = ndata
        dcount_r(2) = nx_reduce
        dcount_r(3) = ny_reduce
        dcount_r(4) = nz_reduce
        doffset_r(1) = 0
        doffset_r(2) = dcount_r(2) * mpi_rankx
        doffset_r(3) = dcount_r(3) * mpi_ranky
        doffset_r(4) = dcount_r(4) * mpi_rankz

        allocate(pspect(dcount(1), dcount(2), dcount(3), dcount(4)))
        allocate(pspect_reduced(dcount_r(1), dcount_r(2), dcount_r(3), dcount_r(4)))
        allocate(nzones_reduced(dcount_r(2), dcount_r(3), dcount_r(4)))

        do tframe = tstart, tend, tinterval
            if (myid == master) print*, tframe
            call system_clock(t1, clock_rate, clock_max)
            pspect = 0.0
            pspect_reduced = 0.0
            nzones_reduced = 0.0
            call read_spectrum_single(tframe, 'e', dcount, doffset)
            call reduce_spectrum_local
            call save_energy_spectrum(tframe, 'e', dcount_r, doffset_r)
            pspect = 0.0
            pspect_reduced = 0.0
            nzones_reduced = 0.0
            call read_spectrum_single(tframe, 'i', dcount, doffset)
            call reduce_spectrum_local
            call save_energy_spectrum(tframe, 'i', dcount_r, doffset_r)

            call system_clock(t2, clock_rate, clock_max)
            if (myid == master) then
                write (*, *) 'Time to reduce one frame: ', &
                    real(t2 - t1) / real(clock_rate)
            endif
        enddo

        deallocate(pspect, pspect_reduced, nzones_reduced)
    end subroutine reduce_spectrum

    !<--------------------------------------------------------------------------
    !< Reduce spectrum at each local MPI rank.
    !<--------------------------------------------------------------------------
    subroutine reduce_spectrum_local
        implicit none
        integer :: ix, iy, iz, xr, yr, zr, i
        integer :: offsetx, offsety, offsetz

        if (shiftx > 0) then
            offsetx = 2
        else
            offsetx = 1
        endif

        if (shifty > 0) then
            offsety = 2
        else
            offsety = 1
        endif

        if (shiftz > 0) then
            offsetz = 2
        else
            offsetz = 1
        endif

        do iz = 1, nz_local
            zr = floor((iz-shiftz-1) / (nreducez+0.0)) + offsetz
            do iy = 1, ny_local
                yr = floor((iy-shifty-1) / (nreducey+0.0)) + offsety
                do ix = 1, nx_local
                    xr = floor((ix-shiftx-1) / (nreducex+0.0)) + offsetx
                    pspect_reduced(:, xr, yr, zr) = &
                        pspect_reduced(:, xr, yr, zr) + pspect(:, ix, iy, iz)
                    nzones_reduced(xr, yr, zr) = nzones_reduced(xr, yr, zr) + 1.0
                enddo
            enddo
        enddo

        ! Average the magnetic field
        do i = 1, 3
            pspect_reduced(i, :, :, :) = pspect_reduced(i, :, :, :) / nzones_reduced
        enddo
    end subroutine reduce_spectrum_local

    !<--------------------------------------------------------------------------
    !< Save reduced particle particle energy spectra
    !<--------------------------------------------------------------------------
    subroutine save_energy_spectrum(tindex, species, dcount_r, doffset_r)
        implicit none
        integer, intent(in) :: tindex
        character(*), intent(in) :: species
        integer(hsize_t), dimension(4), intent(in) :: dcount_r, doffset_r
        character(len=16) :: tindex_str
        character(len=256) :: fname, fpath
        character(len=16) :: dataset_name
        character(len=8) :: tframe_char
        integer(hid_t) :: file_id, plist_id
        integer(hid_t) :: filespace, memspace, dataset_id
        integer(hid_t) :: datatype_id
        integer(hsize_t), dimension(4) :: dset_dims, dcount, doffset
        integer :: fileinfo, error
        logical :: dir_e
        fpath=trim(adjustl(rootpath))//'/'//trim(adjustl(output_path))//'/'
        inquire(file=trim(fpath)//'.', exist=dir_e)
        if (.not. dir_e) then
            call system('mkdir -p '//trim(fpath))
        endif
        if (myid == master) then
            print*, "Saving reorganized particle energy spectrum..."
        endif

        call MPI_INFO_CREATE(fileinfo, ierror)
        call MPI_INFO_SET(fileinfo, "romio_cb_read", "automatic", ierror)
        call MPI_INFO_SET(fileinfo, "romio_ds_read", "automatic", ierror)
        ! call MPI_INFO_SET(fileinfo, "romio_cb_read", "enable", ierror)
        ! call MPI_INFO_SET(fileinfo, "romio_ds_read", "disable", ierror)

        write(tindex_str, "(I0)") tindex
        if (species == 'e') then
            fname = trim(fpath)//"spectrum_electron_"
        else if (species == 'H' .or. species == 'h' .or. species == 'i') then
            fname = trim(fpath)//"spectrum_ion_"
        endif
        fname = trim(fname)//trim(tindex_str)//'.h5'

        call h5open_f(error)
        call h5pcreate_f(H5P_FILE_ACCESS_F, plist_id, error)
        call h5pset_fapl_mpio_f(plist_id, MPI_COMM_WORLD, fileinfo, error)
        call h5fcreate_f(fname, H5F_ACC_TRUNC_F, file_id, error, &
            access_prp=plist_id)
        call h5pclose_f(plist_id, error)

        dset_dims = (/ndata, nx_reduce*topox, ny_reduce*topoy, nz_reduce*topoz/)
        call h5screate_simple_f(4, dset_dims, filespace, error)
        call h5dcreate_f(file_id, "spectrum", H5T_NATIVE_REAL, filespace, &
            dataset_id, error)

        CALL h5screate_simple_f(4, dcount_r, memspace, error)
        call h5sselect_hyperslab_f(filespace, H5S_SELECT_SET_F, doffset_r, &
            dcount_r, error)

        call h5pcreate_f(H5P_DATASET_XFER_F, plist_id, error)
        call h5pset_dxpl_mpio_f(plist_id, H5FD_MPIO_COLLECTIVE_F, error)

        call h5dwrite_f(dataset_id, H5T_NATIVE_REAL, pspect_reduced, &
            dset_dims, error, file_space_id=filespace, mem_space_id=memspace, &
            xfer_prp=plist_id)

        call h5pclose_f(plist_id, error)
        call h5sclose_f(memspace, error)
        call h5sclose_f(filespace, error)
        call h5dclose_f(dataset_id, error)
        call h5fclose_f(file_id, error)
        call h5close_f(error)

        call MPI_INFO_FREE(fileinfo, ierror)
    end subroutine save_energy_spectrum

    !<--------------------------------------------------------------------------
    !< Read spectrum for a single time frame
    !<--------------------------------------------------------------------------
    subroutine read_spectrum_single(tframe, species, dcount, doffset)
        implicit none
        integer, intent(in) :: tframe
        character(*), intent(in) :: species
        integer(hsize_t), dimension(4), intent(in) :: dcount, doffset
        character(len=256) :: fname
        character(len=16) :: groupname, dataset_name
        character(len=8) :: tframe_char
        integer(hid_t) :: file_id, group_id, plist_id
        integer(hid_t) :: filespace, memspace, dataset_id
        integer(hid_t) :: datatype_id
        integer(hsize_t), dimension(4) :: dset_dims, dset_dims_max
        integer :: fileinfo, error

        write(tframe_char, "(I0)") tframe
        fname = trim(adjustl(rootpath))//"/"//trim(adjustl(input_path))
        ! fname = trim(adjustl(fname))//"/T."//trim(tframe_char)
        if (species == 'e') then
            fname = trim(fname)//"/spectrum_electron_"
        else if (species == 'H' .or. species == 'h' .or. species == 'i') then
            fname = trim(fname)//"/spectrum_ion_"
        endif
        fname = trim(fname)//trim(tframe_char)//"."//trim(adjustl(input_suffix))
        groupname = "/"
        call MPI_INFO_CREATE(fileinfo, ierror)
        call h5open_f(error)
        call h5pcreate_f(H5P_FILE_ACCESS_F, plist_id, error)

        call MPI_INFO_SET(fileinfo, "romio_cb_read", "automatic", ierror)
        call MPI_INFO_SET(fileinfo, "romio_ds_read", "automatic", ierror)
        ! call MPI_INFO_SET(fileinfo, "romio_cb_read", "enable", ierror)
        ! call MPI_INFO_SET(fileinfo, "romio_ds_read", "disable", ierror)

        call h5pset_fapl_mpio_f(plist_id, MPI_COMM_WORLD, fileinfo, error)
        call MPI_INFO_FREE(fileinfo, ierror)
        call h5fopen_f(fname, H5F_ACC_RDWR_F, file_id, error, access_prp=plist_id)
        call h5pclose_f(plist_id, error)
        call h5gopen_f(file_id, groupname, group_id, error)

        call h5dopen_f(group_id, "spectrum", dataset_id, error)
        call h5dget_type_f(dataset_id, datatype_id, error)

        ! Create property list for collective dataset write
        call h5pcreate_f(H5P_DATASET_XFER_F, plist_id, error)
        call h5pset_dxpl_mpio_f(plist_id, H5FD_MPIO_COLLECTIVE_F, error)
        call h5screate_simple_f(4, dcount, memspace, error)
        call h5dget_space_f(dataset_id, filespace, error)
        call h5Sget_simple_extent_dims_f(filespace, dset_dims, &
            dset_dims_max, error)
        call h5sselect_hyperslab_f(filespace, H5S_SELECT_SET_F, doffset, &
            dcount, error)

        call h5dread_f(dataset_id, H5T_NATIVE_REAL, pspect, dset_dims, error, &
            file_space_id=filespace, mem_space_id=memspace, xfer_prp=plist_id)

        call h5pclose_f(plist_id, error)
        call h5sclose_f(memspace, error)
        call h5sclose_f(filespace, error)
        call h5dclose_f(dataset_id, error)
        call h5gclose_f(group_id, error)
        call h5fclose_f(file_id, error)

        if (myid == master) print*, "Finished reading energy spectrum"
    end subroutine read_spectrum_single

    !<--------------------------------------------------------------------------
    !< Get commandline arguments
    !<--------------------------------------------------------------------------
    subroutine get_cmd_args
        use flap                                !< FLAP package
        use penf
        implicit none
        type(command_line_interface) :: cli     !< Command Line Interface (CLI).
        integer(I4P)                 :: error   !< Error trapping flag.
        call cli%init(progname = 'reduce_organized_spectrum_hdf5', &
            authors     = 'Xiaocan Li', &
            help        = 'Usage: ', &
            description = 'reduce organized particle energy spectrum in HDF5', &
            examples    = ['reduce_organized_spectrum_hdf5 -rp rootpath'])
        call cli%add(switch='--rootpath', switch_ab='-rp', &
            help='simulation root path', required=.true., act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--tstart', switch_ab='-ts', &
            help='Starting time frame', required=.false., act='store', &
            def='0', error=error)
        if (error/=0) stop
        call cli%add(switch='--tend', switch_ab='-te', help='Last time frame', &
            required=.true., act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--tinterval', switch_ab='-ti', help='Time interval', &
            required=.true., act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--pic_topox', switch_ab='-px', &
            help='Topology_x for PIC simulation', required=.false., &
            act='store', def='256', error=error)
        if (error/=0) stop
        call cli%add(switch='--pic_topoy', switch_ab='-py', &
            help='Topology_y for PIC simulation', required=.false., &
            act='store', def='256', error=error)
        if (error/=0) stop
        call cli%add(switch='--pic_topoz', switch_ab='-pz', &
            help='Topology_z for PIC simulation', required=.false., &
            act='store', def='2', error=error)
        if (error/=0) stop
        call cli%add(switch='--nzonex', switch_ab='-nx', &
            help='Number of zones along x-direction in each PIC MPI rank', &
            required=.false., act='store', def='1', error=error)
        if (error/=0) stop
        call cli%add(switch='--nzoney', switch_ab='-ny', &
            help='Number of zones along y-direction in each PIC MPI rank', &
            required=.false., act='store', def='1', error=error)
        if (error/=0) stop
        call cli%add(switch='--nzonez', switch_ab='-nz', &
            help='Number of zones along z-direction in each PIC MPI rank', &
            required=.false., act='store', def='80', error=error)
        if (error/=0) stop
        call cli%add(switch='--nreducex', switch_ab='-rx', &
            help='Reduce factor along x-direction', &
            required=.false., act='store', def='4', error=error)
        if (error/=0) stop
        call cli%add(switch='--nreducey', switch_ab='-ry', &
            help='Reduce factor along y-direction', &
            required=.false., act='store', def='8', error=error)
        if (error/=0) stop
        call cli%add(switch='--nreducez', switch_ab='-rz', &
            help='Reduce factor along z-direction', &
            required=.false., act='store', def='6', error=error)
        if (error/=0) stop
        call cli%add(switch='--ndata', switch_ab='-nd', &
            help='Number of data points for each energy spectrum', &
            required=.false., act='store', def='1003', error=error)
        if (error/=0) stop
        call cli%add(switch='--topox', switch_ab='-tx', &
            help='MPI size along x for current analysis', required=.false., &
            act='store', def='1', error=error)
        if (error/=0) stop
        call cli%add(switch='--topoy', switch_ab='-ty', &
            help='MPI size along y for current analysis', required=.false., &
            act='store', def='1', error=error)
        if (error/=0) stop
        call cli%add(switch='--topoz', switch_ab='-tz', &
            help='MPI size along z for current analysis', required=.false., &
            act='store', def='1', error=error)
        if (error/=0) stop
        call cli%add(switch='--input_path', switch_ab='-ip', &
            help='input path of energy spectrum', required=.false., &
            act='store', def='spectrum', error=error)
        if (error/=0) stop
        call cli%add(switch='--output_path', switch_ab='-op', &
            help='output path of energy spectrum', required=.false., &
            act='store', def='spectrum_reduced', error=error)
        if (error/=0) stop
        call cli%add(switch='--input_suffix', switch_ab='-is', &
            help='input file suffix', required=.false., &
            act='store', def='h5', error=error)
        if (error/=0) stop
        call cli%add(switch='--output_suffix', switch_ab='-os', &
            help='output file suffix', required=.false., &
            act='store', def='h5', error=error)
        if (error/=0) stop
        call cli%get(switch='-rp', val=rootpath, error=error)
        if (error/=0) stop
        call cli%get(switch='-ts', val=tstart, error=error)
        if (error/=0) stop
        call cli%get(switch='-te', val=tend, error=error)
        if (error/=0) stop
        call cli%get(switch='-ti', val=tinterval, error=error)
        if (error/=0) stop
        call cli%get(switch='-px', val=pic_topox, error=error)
        if (error/=0) stop
        call cli%get(switch='-py', val=pic_topoy, error=error)
        if (error/=0) stop
        call cli%get(switch='-pz', val=pic_topoz, error=error)
        if (error/=0) stop
        call cli%get(switch='-nx', val=nzonex, error=error)
        if (error/=0) stop
        call cli%get(switch='-ny', val=nzoney, error=error)
        if (error/=0) stop
        call cli%get(switch='-nz', val=nzonez, error=error)
        if (error/=0) stop
        call cli%get(switch='-rx', val=nreducex, error=error)
        if (error/=0) stop
        call cli%get(switch='-ry', val=nreducey, error=error)
        if (error/=0) stop
        call cli%get(switch='-rz', val=nreducez, error=error)
        if (error/=0) stop
        call cli%get(switch='-nd', val=ndata, error=error)
        if (error/=0) stop
        call cli%get(switch='-tx', val=topox, error=error)
        if (error/=0) stop
        call cli%get(switch='-ty', val=topoy, error=error)
        if (error/=0) stop
        call cli%get(switch='-tz', val=topoz, error=error)
        if (error/=0) stop
        call cli%get(switch='-ip', val=input_path, error=error)
        if (error/=0) stop
        call cli%get(switch='-op', val=output_path, error=error)
        if (error/=0) stop
        call cli%get(switch='-is', val=input_suffix, error=error)
        if (error/=0) stop
        call cli%get(switch='-os', val=output_suffix, error=error)
        if (error/=0) stop

        if (myid == 0) then
            print '(A,A)', ' The simulation rootpath: ', trim(adjustl(rootpath))
            print '(A,I0,A,I0,A,I0)', ' Min, max and interval: ', &
                tstart, ' ', tend, ' ', tinterval
            print '(A,I0,A,I0,A,I0)', ' PIC topology: ', &
                pic_topox, ", ", pic_topoy, ", ", pic_topoz
            print '(A,I0,A,I0,A,I0)', ' Number of zones along each direction: ', &
                nzonex, ", ", nzoney, ", ", nzonez
            print '(A,I0,A,I0,A,I0)', ' Reduce factor along each direction: ', &
                nreducex, ", ", nreducey, ", ", nreducez
            print '(A,I0)', ' Number of data points in each spectrum: ', ndata
            print '(A,I0,A,I0,A,I0)', ' MPI topology for current analysis: ', &
                topox, ", ", topoy, ", ", topoz
            print '(A,A)', ' Input spectrum file path: ', trim(adjustl(input_path))
            print '(A,A)', ' Output spectrum file path: ', trim(adjustl(output_path))
            print '(A,A)', ' Input spectrum file suffix: ', trim(adjustl(input_suffix))
            print '(A,A)', ' Output spectrum file suffix: ', trim(adjustl(output_suffix))
        endif
    end subroutine get_cmd_args

end program reduce_organized_spectrum_hdf5
