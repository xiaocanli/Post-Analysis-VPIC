!<******************************************************************************
!< Reduce particle energy spectra saved in HDF5 to a smaller size
!< Spectrum data is originally saved in
!<      ndata * nzonex * nzoney * nzonez * pic_topox * pic_topoy * pic_topoz
!< These are also the order of the data saved in disk.
!< Depending on the dimensions of MPI topologies, this code might need to be
!< modified. Here, we consider a special case with
!<      nzonex == 1, nzoney == 1, nzonez > 1
!<      mod(pic_topox, nreducex) == 0
!<      mod(pic_topoy/MPI_SIZE, nreducey) == 0, where MPI_SIZE is for current analysis
!<      nreducez > pic_topoz
!<******************************************************************************
program reduce_spectrum_hdf5
    use constants, only: fp, dp
    use mpi_module
    use hdf5
    implicit none
    integer :: pic_topox, pic_topoy, pic_topoz
    integer :: nzonex, nzoney, nzonez, ndata
    integer :: nreducex, nreducey, nreducez
    integer :: nx_reduce, ny_reduce, nz_reduce, nzones
    integer(hsize_t) :: pic_mpi_size
    real(dp) :: start, finish
    integer :: tstart, tend, tinterval, tframe
    character(len=256) :: rootpath
    real(fp), allocatable, dimension(:, :) :: pspect
    real(fp), allocatable, dimension(:, :) :: pspect_reduced_xy_local
    real(fp), allocatable, dimension(:, :) :: pspect_reduced_xy
    real(fp), allocatable, dimension(:, :) :: pspect_reduced
    integer :: ny_reduce_local

    call MPI_INIT(ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)
    call get_cmd_args
    nx_reduce = pic_topox / nreducex
    ny_reduce = pic_topoy / nreducey
    nz_reduce = (pic_topoz * nzonez) / nreducez + 2
    pic_mpi_size = pic_topox * pic_topoy * pic_topoz
    nzones = nzonex * nzoney * nzonez

    call reduce_spectrum

    call MPI_FINALIZE(ierr)

    contains


    !<--------------------------------------------------------------------------
    !< Reduce spectrum for all time frames.
    !<--------------------------------------------------------------------------
    subroutine reduce_spectrum
        implicit none
        integer(hsize_t), dimension(1) :: dcount, doffset
        integer :: reduced_size, reduced_size_xy, reduced_size_xy_local
        if (mod(pic_mpi_size, numprocs) .NE. 0) then
            print*, "ERROR: PIC MPI size cannot be divided by current mpi_size"
            call MPI_FINALIZE(ierr)
        endif

        call cpu_time(start)

        dcount = pic_mpi_size * nzones * ndata / numprocs
        doffset = dcount * myid
        reduced_size = nx_reduce * ny_reduce * nz_reduce
        ny_reduce_local = pic_topoy / (numprocs*nreducey)
        reduced_size_xy_local = nx_reduce * ny_reduce_local * pic_topoz
        reduced_size_xy = nx_reduce * ny_reduce * pic_topoz

        allocate(pspect(ndata*nzones, pic_mpi_size/numprocs))
        allocate(pspect_reduced_xy_local(ndata*nzones, reduced_size_xy_local))
        if (myid == master) then
            allocate(pspect_reduced_xy(ndata*nzones, reduced_size_xy))
            allocate(pspect_reduced(ndata, reduced_size))
        endif

        do tframe = tstart, tend, tinterval
            if (myid == master) print*, tframe
            pspect = 0.0
            pspect_reduced_xy_local = 0.0
            if (myid == master) then
                pspect_reduced_xy = 0.0
                pspect_reduced = 0.0
            endif
            call read_spectrum_single(tframe, 'e', dcount, doffset)
            call reduce_spectrum_local(reduced_size_xy, reduced_size_xy_local)
            if (myid == master) then
                call reduce_spectrum_final
                call save_energy_spectrum(tframe, 'e')
            endif
            pspect = 0.0
            pspect_reduced_xy_local = 0.0
            if (myid == master) then
                pspect_reduced_xy = 0.0
                pspect_reduced = 0.0
            endif
            call read_spectrum_single(tframe, 'i', dcount, doffset)
            call reduce_spectrum_local(reduced_size_xy, reduced_size_xy_local)
            if (myid == master) then
                call reduce_spectrum_final
                call save_energy_spectrum(tframe, 'i')
            endif
        enddo

        deallocate(pspect)
        deallocate(pspect_reduced_xy_local)
        if (myid == master) then
            deallocate(pspect_reduced_xy)
            deallocate(pspect_reduced)
        endif

        call cpu_time(finish)
        if (myid == master) then
            print '("Time = ",f9.4," seconds.")',finish-start
        endif
    end subroutine reduce_spectrum

    !<--------------------------------------------------------------------------
    !< First pass to reduce spectrum at each local MPI rank.
    !< Here, we reduce along x and y directions.
    !<--------------------------------------------------------------------------
    subroutine reduce_spectrum_local(reduced_size_xy, reduced_size_xy_local)
        implicit none
        integer, intent(in) :: reduced_size_xy, reduced_size_xy_local
        integer :: ix, iy, iz, xr, yr, ind, ind_global, ny_local

        ny_local = pic_topoy / numprocs

        do iz = 1, pic_topoz
            do iy = 1, ny_local
                yr = (iy - 1) / nreducey + 1
                do ix = 1, pic_topox
                    xr = (ix - 1) / nreducex + 1
                    ind = xr + (yr - 1) * nx_reduce + &
                        (iz - 1) * ny_reduce_local * nx_reduce
                    ind_global = ix + (iy - 1) * pic_topox + &
                        (iz - 1) * ny_local * pic_topox
                    pspect_reduced_xy_local(:, ind) = &
                        pspect_reduced_xy_local(:, ind) + pspect(:, ind_global)
                enddo
            enddo
        enddo
        ! Send the local spectrum to master for the final reduce
        call MPI_GATHER(pspect_reduced_xy_local, ndata*nzones*reduced_size_xy_local, &
            MPI_REAL, pspect_reduced_xy, ndata*nzones*reduced_size_xy_local, MPI_REAL, &
            master, MPI_COMM_WORLD, ierror)
    end subroutine reduce_spectrum_local

    !<--------------------------------------------------------------------------
    !< Reduce spectrum to its final form
    !<--------------------------------------------------------------------------
    subroutine reduce_spectrum_final
        implicit none
        integer :: ix, iy, iz, izone, ind, ind_global, shiftz
        integer :: iz_new, zone_s, zone_e
        shiftz = (pic_topoz * nzonez - (nz_reduce - 2) * nreducez) / 2
        do iz = 1, pic_topoz
            do iy = 1, ny_reduce
                do ix = 1, nx_reduce
                    do izone = 1, nzones
                        zone_s = (izone - 1) * ndata + 1
                        zone_e = zone_s + ndata
                        iz_new = izone + (iz - 1) * nzones + nreducez - shiftz - 1
                        iz_new = iz_new / nreducez
                        ind = ix + (iy - 1) * nx_reduce + iz_new * nx_reduce * ny_reduce
                        ind_global = ix + (iy - 1) * nx_reduce + &
                                     (iz - 1) * nx_reduce * ny_reduce
                        pspect_reduced(:, ind) = &
                            pspect_reduced(:, ind) + pspect_reduced_xy(zone_s:zone_e, ind_global)
                    enddo
                enddo
            enddo
        enddo
    end subroutine reduce_spectrum_final

    !<--------------------------------------------------------------------------
    !< Save reduced particle particle energy spectra
    !<--------------------------------------------------------------------------
    subroutine save_energy_spectrum(tindex, species)
        implicit none
        integer, intent(in) :: tindex
        character(*), intent(in) :: species
        integer :: fh1, posf
        character(len=16) :: tindex_str
        character(len=256) :: fname, fpath
        logical :: dir_e
        fpath=trim(adjustl(rootpath))//'/spectrum_reduced/'
        inquire(file=trim(fpath)//'.', exist=dir_e)
        if (.not. dir_e) then
            call system('mkdir -p '//trim(fpath))
        endif
        print*, "Saving reduced particle energy spectrum..."

        fh1 = 66

        write(tindex_str, "(I0)") tindex
        fname = trim(fpath)//"spectrum_"//species//'_'//trim(tindex_str)//'.dat'
        open(unit=fh1, file=fname, access='stream', status='unknown', &
            form='unformatted', action='write')
        posf = 1
        write(fh1, pos=posf) pspect_reduced
        close(fh1)
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
        if (species == 'e') then
            fname = trim(fname)//"/spectrum_electron_"
        else if (species == 'H' .or. species == 'h' .or. species == 'i') then
            fname = trim(fname)//"/spectrum_ion_"
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

        call h5dopen_f(group_id, "spectrum", dataset_id, error)
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
        call cli%init(progname = 'reduce_spectrum_hdf5', &
            authors     = 'Xiaocan Li', &
            help        = 'Usage: ', &
            description = 'reduce particle energy spectrum in HDF5', &
            examples    = ['reduce_spectrum_hdf5 -rp rootpath'])
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
            print '(A,I0)', ' Number of data points ine each spectrum: ', ndata
        endif
    end subroutine get_cmd_args

end program reduce_spectrum_hdf5
