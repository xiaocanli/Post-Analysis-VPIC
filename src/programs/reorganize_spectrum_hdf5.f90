!<******************************************************************************
!< Re-organize particle energy spectrum.
!< Spectrum data is originally saved in a 1D array in the order of
!< ndata * nzonex * nzoney * nzonez * pic_topox * pic_topoy * pic_topoz
!< It is difficult to get the spectrum a local position. The program reorganizes
!< the data into the order
!< ndata * (nzonex * pic_topox) * (nzoney * pic_topoy) * (nzonez * pic_topoz)
!<******************************************************************************
program reorganize_spectrum_hdf5
    use constants, only: fp, dp
    use mpi_module
    use hdf5
    implicit none
    integer :: pic_topox, pic_topoy, pic_topoz
    integer :: nzonex, nzoney, nzonez, ndata
    integer :: tstart, tend, tinterval, tframe
    integer :: npx, npy, npz ! Number of PIC mpi ranks along each direction
    integer :: nlx, nly, nlz ! Number of local zones along each direction
    integer :: offsetx, offsety, offsetz, tmp
    integer :: nzones
    integer(hsize_t) :: pic_mpi_size
    real(fp), allocatable, dimension(:, :, :, :, :) :: pspect
    real(fp), allocatable, dimension(:, :, :, :) :: pspect_reorganize
    character(len=256) :: rootpath
    real(dp) :: start, finish
    logical :: single_file

    call MPI_INIT(ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)
    call get_cmd_args

    pic_mpi_size = pic_topox * pic_topoy * pic_topoz
    nzones = nzonex * nzoney * nzonez

    ! Number of PIC mpi ranks along each direction
    if (numprocs <= pic_topoz) then
        npx = pic_topox
        npy = pic_topoy
        npz = pic_topoz / numprocs
        offsetx = 0
        offsety = 0
        offsetz = npz * myid
    else
        if (numprocs <= pic_topoy * pic_topoz) then
            tmp = numprocs / pic_topoz
            npx = pic_topox
            npy = pic_topoy / tmp
            npz = 1
            offsetx = 0
            offsety = npy * mod(myid, tmp)
            offsetz = myid / tmp
        else
            tmp = numprocs / (pic_topoy * pic_topoz)
            npx = pic_topox / tmp
            npy = 1
            npz = 1
            offsetx = npx * mod(myid, tmp)
            offsety = mod(myid / tmp, pic_topoy)
            offsetz = (myid / tmp) / pic_topoy
        endif
    endif

    nlx = npx * nzonex
    nly = npy * nzoney
    nlz = npz * nzonez

    offsetx = offsetx * nzonex
    offsety = offsety * nzoney
    offsetz = offsetz * nzonez

    call reorganize_spectrum

    call MPI_FINALIZE(ierr)

    contains


    !<--------------------------------------------------------------------------
    !< Reorganize spectrum for all time frames.
    !<--------------------------------------------------------------------------
    subroutine reorganize_spectrum
        implicit none
        integer(hsize_t), dimension(1) :: dcount, doffset
        if (mod(pic_mpi_size, numprocs) .NE. 0) then
            print*, "ERROR: PIC MPI size cannot be divided by current mpi_size"
            call MPI_FINALIZE(ierr)
        endif

        call cpu_time(start)

        dcount = pic_mpi_size * nzones * ndata / numprocs
        doffset = dcount * myid

        allocate(pspect(ndata, nzonex, nzoney, nzonez, pic_mpi_size/numprocs))
        allocate(pspect_reorganize(ndata, nlx, nly, nlz))

        do tframe = tstart, tend, tinterval
            if (myid == master) print*, tframe
            pspect = 0.0
            pspect_reorganize = 0.0
            call read_spectrum_single(tframe, 'e', dcount, doffset)
            call reorganize_spectrum_local
            call save_energy_spectrum(tframe, 'e')
            pspect = 0.0
            pspect_reorganize = 0.0
            call read_spectrum_single(tframe, 'i', dcount, doffset)
            call reorganize_spectrum_local
            call save_energy_spectrum(tframe, 'i')
        enddo

        deallocate(pspect, pspect_reorganize)
        call cpu_time(finish)
        if (myid == master) then
            print '("Time = ",f9.4," seconds.")',finish-start
        endif
    end subroutine reorganize_spectrum

    !<--------------------------------------------------------------------------
    !< Reorganize spectrum at each local MPI rank.
    !<--------------------------------------------------------------------------
    subroutine reorganize_spectrum_local
        implicit none
        integer :: ix, iy, iz, xs, ys, zs, xe, ye, ze, ind_global

        do iz = 1, npz
            zs = (iz - 1) * nzonez + 1
            ze = iz * nzonez
            do iy = 1, npy
                ys = (iy - 1) * nzoney + 1
                ye = iy * nzoney
                do ix = 1, npx
                    xs = (ix - 1) * nzonex + 1
                    xe = ix * nzonex
                    ind_global = npx * ((iz - 1) * npy + iy - 1) + ix
                    pspect_reorganize(:, xs:xe, ys:ye, zs:ze) = &
                        pspect(:, :, :, :, ind_global)
                enddo
            enddo
        enddo
    end subroutine reorganize_spectrum_local

    !<--------------------------------------------------------------------------
    !< Save reorganized particle particle energy spectra
    !<--------------------------------------------------------------------------
    subroutine save_energy_spectrum(tindex, species)
        implicit none
        integer, intent(in) :: tindex
        character(*), intent(in) :: species
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
        fpath=trim(adjustl(rootpath))//'spectrum_reorganize/'
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

        dset_dims = (/ndata, pic_topox*nzonex, pic_topoy*nzoney, pic_topoz*nzonez/)
        dcount = (/ndata, nlx, nly, nlz/)
        doffset = (/0, offsetx, offsety, offsetz/)
        call h5screate_simple_f(4, dset_dims, filespace, error)
        call h5dcreate_f(file_id, "spectrum", H5T_NATIVE_REAL, filespace, &
            dataset_id, error)

        CALL h5screate_simple_f(4, dcount, memspace, error)
        call h5sselect_hyperslab_f(filespace, H5S_SELECT_SET_F, doffset, &
            dcount, error)

        call h5pcreate_f(H5P_DATASET_XFER_F, plist_id, error)
        call h5pset_dxpl_mpio_f(plist_id, H5FD_MPIO_COLLECTIVE_F, error)

        call h5dwrite_f(dataset_id, H5T_NATIVE_REAL, pspect_reorganize, &
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
        integer(hsize_t), dimension(1), intent(in) :: dcount, doffset
        character(len=256) :: fname
        character(len=16) :: groupname, dataset_name
        character(len=8) :: tframe_char
        integer(hid_t) :: file_id, group_id, plist_id
        integer(hid_t) :: filespace, memspace, dataset_id
        integer(hid_t) :: datatype_id
        integer(hsize_t), dimension(1) :: dset_dims, dset_dims_max
        integer :: fileinfo, error

        write(tframe_char, "(I0)") tframe
        fname = trim(adjustl(rootpath))//"/spectrum/T."//trim(tframe_char)
        if (single_file) then
            fname = trim(fname)//"/spectrum_"
        else
            if (species == 'e') then
                fname = trim(fname)//"/spectrum_electron_"
            else if (species == 'H' .or. species == 'h' .or. species == 'i') then
                fname = trim(fname)//"/spectrum_ion_"
            endif
        endif
        fname = trim(fname)//trim(tframe_char)//".h5part"
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
        call h5fopen_f(fname, H5F_ACC_RDWR_F, file_id, error, &
            access_prp=plist_id)
        call h5pclose_f(plist_id, error)
        call h5gopen_f(file_id, groupname, group_id, error)

        if (single_file) then
            if (species == 'e') then
                groupname = "spectrum_electron"
                call h5dopen_f(group_id, "spectrum_electron", dataset_id, error)
            else if (species == 'H' .or. species == 'h' .or. species == 'i') then
                call h5dopen_f(group_id, "spectrum_ion", dataset_id, error)
            endif
        else
            call h5dopen_f(group_id, "spectrum", dataset_id, error)
        endif
        call h5dget_type_f(dataset_id, datatype_id, error)

        ! Create property list for collective dataset write
        call h5pcreate_f(H5P_DATASET_XFER_F, plist_id, error)
        call h5pset_dxpl_mpio_f(plist_id, H5FD_MPIO_COLLECTIVE_F, error)
        call h5screate_simple_f(1, dcount, memspace, error)
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
        call cli%init(progname = 'reorganize_spectrum_hdf5', &
            authors     = 'Xiaocan Li', &
            help        = 'Usage: ', &
            description = 'reorganize particle energy spectrum in HDF5', &
            examples    = ['reorganize_spectrum_hdf5 -rp rootpath'])
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
        call cli%add(switch='--ndata', switch_ab='-nd', &
            help='Number of data points for each energy spectrum', &
            required=.false., act='store', def='1003', error=error)
        if (error/=0) stop
        call cli%add(switch='--single_file', switch_ab='-sf', &
            help='Whether spectra are saved in the same file', &
            required=.false., act='store_true', def='.false.', error=error)
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
        call cli%get(switch='-nd', val=ndata, error=error)
        if (error/=0) stop
        call cli%get(switch='-sf', val=single_file, error=error)
        if (error/=0) stop

        if (myid == 0) then
            print '(A,A)', ' The simulation rootpath: ', trim(adjustl(rootpath))
            print '(A,I0,A,I0,A,I0)', ' Min, max and interval: ', &
                tstart, ' ', tend, ' ', tinterval
            print '(A,I0,A,I0,A,I0)', ' PIC topology: ', &
                pic_topox, ", ", pic_topoy, ", ", pic_topoz
            print '(A,I0,A,I0,A,I0)', ' Number of zones along each direction: ', &
                nzonex, ", ", nzoney, ", ", nzonez
            print '(A,I0)', ' Number of data points ine each spectrum: ', ndata
            if (single_file) then
                print '(A)', ' Spectra are saved in the same file'
            endif
        endif
    end subroutine get_cmd_args

end program reorganize_spectrum_hdf5
