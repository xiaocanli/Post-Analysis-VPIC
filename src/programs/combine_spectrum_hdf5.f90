!<******************************************************************************
!< Combine particle energy spectra saved in HDF5
!<******************************************************************************
program combine_spectrum_hdf5
    use constants, only: fp, dp
    use mpi_module
    use hdf5
    implicit none
    integer(hsize_t), dimension(1) :: dcount, doffset
    real(fp), allocatable, dimension(:, :) :: pspect
    real(fp), allocatable, dimension(:) :: pspect_tot
    real(fp), allocatable, dimension(:) :: pspect_tot_global
    real(dp) :: start, finish
    integer :: tstart, tend, tinterval, tframe
    character(len=256) :: rootpath
    integer(hsize_t) :: pic_mpi_size
    integer :: nzones, ndata
    character(len=32) :: species
    logical :: single_h5

    call MPI_INIT(ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)
    call cpu_time(start)
    call get_cmd_args

    if (mod(pic_mpi_size, numprocs) .NE. 0) then
        print*, "ERROR: PIC MPI size cannot be divided by current mpi_size"
        call MPI_FINALIZE(ierr)
    endif

    dcount = pic_mpi_size * nzones * ndata / numprocs
    doffset = dcount * myid

    allocate(pspect(ndata, pic_mpi_size*nzones/numprocs))
    allocate(pspect_tot(ndata))
    allocate(pspect_tot_global(ndata))

    do tframe = tstart, tend, tinterval
        if (myid == master) print*, tframe
        call combine_spectrum_single(tframe, dcount, doffset)
        call save_energy_spectrum(tframe)
    enddo

    deallocate(pspect)
    deallocate(pspect_tot)
    deallocate(pspect_tot_global)

    call cpu_time(finish)
    if (myid == master) then
        print '("Time = ",f9.4," seconds.")',finish-start
    endif

    call MPI_FINALIZE(ierr)

    contains

    !<--------------------------------------------------------------------------
    !< Combine spectrum for a single time frame
    !<--------------------------------------------------------------------------
    subroutine combine_spectrum_single(tframe, dcount, doffset)
        implicit none
        integer, intent(in) :: tframe
        integer(hsize_t), dimension(1), intent(in) :: dcount, doffset
        character(len=256) :: fname
        character(len=32) :: groupname, dataset_name
        character(len=8) :: tframe_char
        integer(hid_t) :: file_id, group_id, plist_id
        integer(hid_t) :: filespace, memspace, dataset_id
        integer(hid_t) :: datatype_id
        integer(hsize_t), dimension(1) :: dset_dims, dset_dims_max
        integer :: fileinfo, error

        write(tframe_char, "(I0)") tframe
        fname = trim(adjustl(rootpath))//"/spectrum/T."//trim(tframe_char)
        if (single_h5) then
            fname = trim(fname)//"/spectrum_"//trim(tframe_char)//".h5part"
            dataset_name = 'spectrum_'//trim(species)
        else
            fname = trim(fname)//"/spectrum_"//trim(species)//"_"
            fname = trim(fname)//trim(tframe_char)//".h5part"
            dataset_name = 'spectrum'
        endif
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

        call h5dopen_f(group_id, trim(dataset_name), dataset_id, error)
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
    end subroutine combine_spectrum_single

    !<--------------------------------------------------------------------------
    !< Save particle particle energy spectra
    !<--------------------------------------------------------------------------
    subroutine save_energy_spectrum(tindex)
        implicit none
        integer, intent(in) :: tindex
        integer :: fh1, posf
        character(len=16) :: tindex_str
        character(len=256) :: fname, fpath
        logical :: dir_e
        pspect_tot = sum(pspect, dim=2)
        call MPI_REDUCE(pspect_tot, pspect_tot_global, ndata, MPI_REAL, &
            MPI_SUM, master, MPI_COMM_WORLD, ierr)
        if (myid == master) then
            fpath=trim(adjustl(rootpath))//'/spectrum_combined/'
            inquire(file=trim(fpath)//'.', exist=dir_e)
            if (.not. dir_e) then
                call system('mkdir -p '//trim(fpath))
            endif
            print*, "Saving particle energy spectrum..."

            fh1 = 66

            write(tindex_str, "(I0)") tindex
            fname = trim(fpath)//"spectrum_"//trim(species)//'_'//trim(tindex_str)//'.dat'
            open(unit=fh1, file=fname, access='stream', status='unknown', &
                form='unformatted', action='write')
            posf = 1
            write(fh1, pos=posf) pspect_tot_global
            close(fh1)
        endif
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
        call cli%init(progname = 'combine_spectrum_hdf5', &
            authors     = 'Xiaocan Li', &
            help        = 'Usage: ', &
            description = 'Combine particle energy spectrum in HDF5', &
            examples    = ['combine_spectrum_hdf5 -rp rootpath'])
        call cli%add(switch='--rootpath', switch_ab='-rp', &
            help='simulation root path', required=.true., act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--species', switch_ab='-sp', &
            help='particle_species', required=.false., act='store', &
            def='electron', error=error)
        if (error/=0) stop
        call cli%add(switch='--pic_mpi_size', switch_ab='-pm', &
            help='MPI size for PIC simulation', &
            required=.false., act='store', def='131072', error=error)
        if (error/=0) stop
        call cli%add(switch='--nzones', switch_ab='-nz', &
            help='Number of zones in each PIC MPI rank', &
            required=.false., act='store', def='80', error=error)
        if (error/=0) stop
        call cli%add(switch='--ndata', switch_ab='-nd', &
            help='Number of data points for each energy spectrum', &
            required=.false., act='store', def='1003', error=error)
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
        call cli%add(switch='--single_h5', switch_ab='-sh', &
            help='whether spectra of all species are saved in the same HDF5 file', &
            required=.false., act='store_true', def='.false.', error=error)
        if (error/=0) stop
        call cli%get(switch='-rp', val=rootpath, error=error)
        if (error/=0) stop
        call cli%get(switch='-sp', val=species, error=error)
        if (error/=0) stop
        call cli%get(switch='-pm', val=pic_mpi_size, error=error)
        if (error/=0) stop
        call cli%get(switch='-nz', val=nzones, error=error)
        if (error/=0) stop
        call cli%get(switch='-nd', val=ndata, error=error)
        if (error/=0) stop
        call cli%get(switch='-ts', val=tstart, error=error)
        if (error/=0) stop
        call cli%get(switch='-te', val=tend, error=error)
        if (error/=0) stop
        call cli%get(switch='-ti', val=tinterval, error=error)
        if (error/=0) stop
        call cli%get(switch='-sh', val=single_h5, error=error)
        if (error/=0) stop

        if (myid == 0) then
            print '(A,A)', ' The simulation rootpath: ', trim(adjustl(rootpath))
            print '(A,A)', ' Particle species: ', trim(adjustl(species))
            print '(A,I0,A,I0,A,I0)', ' Min, max and interval: ', &
                tstart, ' ', tend, ' ', tinterval
            print '(A,I0)', ' MPI size of the PIC simulation: ', pic_mpi_size
            print '(A,I0)', ' Number of zones in each PIC MPI rank: ', nzones
            print '(A,I0)', ' Number of data points ine each spectrum: ', ndata
            if (single_h5) then
                print '(A)', ' Spectra of all species are saved in the same HDF5 file'
            else
                print '(A)', ' Spectrum of different species are saved in different HDF5 file'
            endif
        endif
    end subroutine get_cmd_args

end program combine_spectrum_hdf5
