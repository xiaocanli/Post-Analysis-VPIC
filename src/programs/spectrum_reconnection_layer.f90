!<******************************************************************************
!< Get the spectrum in the reconnection layer. We judge whether a region is in
!< the reconnection layer by checking how many high-energy particles are in the
!< tail. Since we typically set xz as the reconnection lay and z-direction as
!< the reconnection inflow direction, we will find the reconnection layer
!< boundary along the z-direction. That's why we recommend setting topoz to be 1.
!< * We assume that 3 components of magnetic field are saved at the beginning
!<   of the particle spectrum data.
!<******************************************************************************
program spectrum_reconnection_layer
    use constants, only: fp, dp
    use mpi_module
    use hdf5
    implicit none
    integer :: pic_topox, pic_topoy, pic_topoz
    integer :: nzonex, nzoney, nzonez, ndata
    integer :: topox, topoy, topoz
    integer :: nx_local, ny_local, nz_local
    integer :: mpi_rankx, mpi_ranky, mpi_rankz
    integer :: tstart, tend, tinterval, tframe
    integer(hsize_t) :: pic_mpi_size
    real(fp) :: density_threshold, particle_energy, emin, emax, vthe
    real(fp), allocatable, dimension(:, :, :, :) :: pspect_e, pspect_i
    real(fp), allocatable, dimension(:) :: pspect_layer_e, pspect_layer_e_sum
    real(fp), allocatable, dimension(:) :: pspect_layer_i, pspect_layer_i_sum
    integer, allocatable, dimension(:, :, :) :: zbounds
    character(len=256) :: rootpath
    character(len=32) :: input_path, output_path, input_suffix
    integer :: ny_reduce_local
    integer :: t1, t2, clock_rate, clock_max
    integer :: energy_index
    logical :: read_zbounds  ! Read z-boundaries of the reconnection layer

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

    if (mod(pic_topox, topox) /= 0 .or. mod(pic_topoy, topoy) /= 0 .or. &
        mod(pic_topoz, topoz) /= 0) then
        if (myid == master) then
            print*, "Inconsistent MPI topology with PIC MPI topology"
        endif
        call MPI_FINALIZE(ierr)
    endif

    call system_clock(t1, clock_rate, clock_max)

    pic_mpi_size = pic_topox * pic_topoy * pic_topoz
    nx_local = (pic_topox*nzonex) / topox
    ny_local = (pic_topoy*nzoney) / topoy
    nz_local = (pic_topoz*nzonez) / topoz
    mpi_rankx = mod(myid, topox)
    mpi_ranky = mod(myid, topox*topoy) / topox
    mpi_rankz = myid / (topox*topoy)

    call get_energy_index
    call get_spectrum

    call system_clock(t2, clock_rate, clock_max)
    if (myid == master) then
        write (*, *) 'Elapsed real time = ', real(t2 - t1) / real(clock_rate)
    endif

    call MPI_FINALIZE(ierr)

    contains

    !<--------------------------------------------------------------------------
    !< Get particle energy bin index for the threshold energy
    !< We assume here that ndata includes three components of the magnetic field.
    !<--------------------------------------------------------------------------
    subroutine get_energy_index
        implicit none
        real(fp) :: ethe, pene_actual ! particle_energy is the normed energy
        real(fp) :: emin_log, delog, ebin
        ethe = 1.0/sqrt(1.0 - 3 * vthe**2) - 1.0
        emin_log = log10(emin)
        pene_actual = particle_energy * ethe
        delog = (log10(emax) - log10(emin)) / (ndata - 3 - 1)
        ebin = emin
        energy_index = 1
        do while (ebin < pene_actual)
            ebin = 10**(emin_log + delog * energy_index)
            energy_index = energy_index + 1
        enddo
    end subroutine get_energy_index

    !<--------------------------------------------------------------------------
    !< Get spectrum in the reconnection layer
    !<--------------------------------------------------------------------------
    subroutine get_spectrum
        implicit none
        integer(hsize_t), dimension(4) :: dcount, doffset
        integer :: t1, t2, clock_rate, clock_max, tindex

        ! Data count and offset for reading data
        dcount(1) = ndata
        dcount(2) = nx_local
        dcount(3) = ny_local
        dcount(4) = nz_local
        doffset(1) = 0
        doffset(2) = dcount(2) * mpi_rankx
        doffset(3) = dcount(3) * mpi_ranky
        doffset(4) = dcount(4) * mpi_rankz

        allocate(zbounds(nx_local, ny_local, 2))
        allocate(pspect_e(dcount(1), dcount(2), dcount(3), dcount(4)))
        allocate(pspect_i(dcount(1), dcount(2), dcount(3), dcount(4)))
        allocate(pspect_layer_e(ndata))
        allocate(pspect_layer_e_sum(ndata))
        allocate(pspect_layer_i(ndata))
        allocate(pspect_layer_i_sum(ndata))

        do tframe = tstart, tend, tinterval
            if (myid == master) print*, tframe
            tindex = tframe / tinterval
            call system_clock(t1, clock_rate, clock_max)
            zbounds = 0
            pspect_e = 0.0
            pspect_i = 0.0
            pspect_layer_e = 0.0
            pspect_layer_i = 0.0
            pspect_layer_e_sum = 0.0
            pspect_layer_i_sum = 0.0
            call read_spectrum_single(tframe, 'e', dcount, doffset)
            call read_spectrum_single(tframe, 'i', dcount, doffset)
            if (.not. read_zbounds) then
                call get_spectrum_energetic_density
            else
                call get_spectrum_read_zbounds(tindex)
            endif
            call save_energy_spectrum(tframe)
            if (.not. read_zbounds) then
                call save_zbounds(tframe)
            endif
            call system_clock(t2, clock_rate, clock_max)
            if (myid == master) then
                write (*, *) 'Time for one frame: ', &
                    real(t2 - t1) / real(clock_rate)
            endif
        enddo

        deallocate(zbounds)
        deallocate(pspect_e, pspect_i)
        deallocate(pspect_layer_e, pspect_layer_i)
        deallocate(pspect_layer_e_sum, pspect_layer_i_sum)
    end subroutine get_spectrum

    !<--------------------------------------------------------------------------
    !< Get the spectrum at each local MPI rank by checking the high-energy
    !< particle density along the z-direction.
    !<--------------------------------------------------------------------------
    subroutine get_spectrum_energetic_density
        implicit none
        integer :: ix, iy, iz, ztmp
        real(dp) :: frac

        do iy = 1, ny_local
            do ix = 1, nx_local
                ! Bottom half domain
                do iz = 1, nz_local/2
                    frac = sum(pspect_e(3+energy_index:, ix, iy, iz)) / &
                        sum(pspect_e(4:, ix, iy, iz))
                    if (frac > density_threshold) then
                        exit
                    endif
                enddo
                if (frac <= density_threshold) then
                    zbounds(ix, iy, 1) = nz_local/2
                else
                    zbounds(ix, iy, 1) = iz
                endif

                do iz = 1, nz_local/2
                    frac = sum(pspect_i(3+energy_index:, ix, iy, iz)) / &
                        sum(pspect_i(4:, ix, iy, iz))
                    if (frac > density_threshold) then
                        exit
                    endif
                enddo
                if (frac <= density_threshold) then
                    ztmp = nz_local/2
                else
                    ztmp = iz
                endif
                if (ztmp < zbounds(ix, iy, 1)) then
                    zbounds(ix, iy, 1) = ztmp
                endif

                ! Top half domain
                do iz = nz_local, nz_local/2, -1
                    frac = sum(pspect_e(3+energy_index:, ix, iy, iz)) / &
                        sum(pspect_e(4:, ix, iy, iz))
                    if (frac > density_threshold) then
                        exit
                    endif
                enddo
                if (frac <= density_threshold) then
                    zbounds(ix, iy, 2) = nz_local/2
                else
                    zbounds(ix, iy, 2) = iz
                endif

                do iz = nz_local, nz_local/2, -1
                    frac = sum(pspect_i(3+energy_index:, ix, iy, iz)) / &
                        sum(pspect_i(4:, ix, iy, iz))
                    if (frac > density_threshold) then
                        exit
                    endif
                enddo
                if (frac <= density_threshold) then
                    ztmp = nz_local/2
                else
                    ztmp = iz
                endif
                if (ztmp > zbounds(ix, iy, 2)) then
                    zbounds(ix, iy, 2) = ztmp
                endif

                ! Energy spectrum in the reconnection layer
                do iz = zbounds(ix, iy, 1), zbounds(ix, iy, 2)
                    pspect_layer_e = pspect_layer_e + pspect_e(:, ix, iy, iz)
                    pspect_layer_i = pspect_layer_i + pspect_i(:, ix, iy, iz)
                enddo
            enddo
        enddo
    end subroutine get_spectrum_energetic_density

    !<--------------------------------------------------------------------------
    !< Get the spectrum at each local MPI rank using the z-boundaries read from file
    !<--------------------------------------------------------------------------
    subroutine get_spectrum_read_zbounds(tframe)
        implicit none
        integer, intent(in) :: tframe
        integer :: ix, iy, iz

        ! Read zbounds
        call read_zbounds_rec_layer(tframe)

        do iy = 1, ny_local
            do ix = 1, nx_local
                ! Energy spectrum in the reconnection layer
                do iz = zbounds(ix, iy, 1), zbounds(ix, iy, 2)
                    pspect_layer_e = pspect_layer_e + pspect_e(:, ix, iy, iz)
                    pspect_layer_i = pspect_layer_i + pspect_i(:, ix, iy, iz)
                enddo
            enddo
        enddo
    end subroutine get_spectrum_read_zbounds

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
        call MPI_REDUCE(pspect_layer_e, pspect_layer_e_sum, ndata, MPI_REAL, &
            MPI_SUM, master, MPI_COMM_WORLD, ierr)
        call MPI_REDUCE(pspect_layer_i, pspect_layer_i_sum, ndata, MPI_REAL, &
            MPI_SUM, master, MPI_COMM_WORLD, ierr)
        if (myid == master) then
            fpath=trim(adjustl(rootpath))//'/'//trim(adjustl(output_path))//'/'
            inquire(file=trim(fpath)//'.', exist=dir_e)
            if (.not. dir_e) then
                call system('mkdir -p '//trim(fpath))
            endif
            if (myid == master) then
                print*, "Saving particle energy spectrum in the reconnection layer..."
            endif

            fh1 = 66

            write(tindex_str, "(I0)") tindex
            fname = trim(fpath)//"spectrum_layer_electron_"//trim(tindex_str)//".dat"
            open(unit=fh1, file=fname, access='stream', status='unknown', &
                form='unformatted', action='write')
            posf = 1
            write(fh1, pos=posf) pspect_layer_e_sum
            close(fh1)

            fname = trim(fpath)//"spectrum_layer_ion_"//trim(tindex_str)//".dat"
            open(unit=fh1, file=fname, access='stream', status='unknown', &
                form='unformatted', action='write')
            posf = 1
            write(fh1, pos=posf) pspect_layer_i_sum
            close(fh1)
        endif
    end subroutine save_energy_spectrum

    !<--------------------------------------------------------------------------
    !< Read z-boundaries of the reconnection layer for a single time frame
    !<--------------------------------------------------------------------------
    subroutine read_zbounds_rec_layer(tframe)
        implicit none
        integer, intent(in) :: tframe
        character(len=256) :: fname
        character(len=16) :: groupname, dataset_name
        character(len=8) :: tframe_char
        integer(hid_t) :: file_id, group_id, plist_id
        integer(hid_t) :: filespace, memspace, dataset_id
        integer(hsize_t), dimension(2) :: dset_dims, dset_dims_max, dcount, doffset
        integer :: fileinfo, error

        write(tframe_char, "(I0)") tframe
        fname = trim(adjustl(rootpath))//"/reconnection_layer/rec_layer_"
        fname = trim(fname)//trim(tframe_char)//".h5"
        groupname = "/rec_layer_zone"
        call h5open_f(error)
        call h5pcreate_f(H5P_FILE_ACCESS_F, plist_id, error)
        call MPI_INFO_CREATE(fileinfo, ierror)

        call MPI_INFO_SET(fileinfo, "romio_cb_read", "automatic", ierror)
        call MPI_INFO_SET(fileinfo, "romio_ds_read", "automatic", ierror)

        call h5pset_fapl_mpio_f(plist_id, MPI_COMM_WORLD, fileinfo, error)
        call h5fopen_f(fname, H5F_ACC_RDWR_F, file_id, error, access_prp=plist_id)
        call h5pclose_f(plist_id, error)
        call h5gopen_f(file_id, groupname, group_id, error)

        dcount = (/nx_local, ny_local/)
        doffset = (/nx_local*mpi_rankx, ny_local*mpi_ranky/)

        ! Top surface
        call h5dopen_f(group_id, "Top", dataset_id, error)
        call h5pcreate_f(H5P_DATASET_XFER_F, plist_id, error)
        call h5pset_dxpl_mpio_f(plist_id, H5FD_MPIO_COLLECTIVE_F, error)
        call h5screate_simple_f(2, dcount, memspace, error)
        call h5dget_space_f(dataset_id, filespace, error)
        call h5Sget_simple_extent_dims_f(filespace, dset_dims, dset_dims_max, error)
        call h5sselect_hyperslab_f(filespace, H5S_SELECT_SET_F, doffset, dcount, error)
        call h5dread_f(dataset_id, H5T_NATIVE_INTEGER, zbounds(:, :, 2), dset_dims, &
            error, file_space_id=filespace, mem_space_id=memspace, xfer_prp=plist_id)
        call h5pclose_f(plist_id, error)
        call h5sclose_f(memspace, error)
        call h5sclose_f(filespace, error)
        call h5dclose_f(dataset_id, error)
        ! Originally zbounds(:, :, 2) is in [1, pic_topoz*nzonez]
        zbounds(:, :, 2) = zbounds(:, :, 2) + 1

        ! Bottom surface
        call h5dopen_f(group_id, "Bottom", dataset_id, error)
        call h5pcreate_f(H5P_DATASET_XFER_F, plist_id, error)
        call h5pset_dxpl_mpio_f(plist_id, H5FD_MPIO_COLLECTIVE_F, error)
        call h5screate_simple_f(2, dcount, memspace, error)
        call h5dget_space_f(dataset_id, filespace, error)
        call h5Sget_simple_extent_dims_f(filespace, dset_dims, dset_dims_max, error)
        call h5sselect_hyperslab_f(filespace, H5S_SELECT_SET_F, doffset, dcount, error)
        call h5dread_f(dataset_id, H5T_NATIVE_INTEGER, zbounds(:, :, 1), dset_dims, &
            error, file_space_id=filespace, mem_space_id=memspace, xfer_prp=plist_id)
        call h5pclose_f(plist_id, error)
        call h5sclose_f(memspace, error)
        call h5sclose_f(filespace, error)
        call h5dclose_f(dataset_id, error)
        ! Originally zbounds(:, :, 1) is in [0, pic_topoz*nzonez-1]
        zbounds(:, :, 1) = zbounds(:, :, 1) + 1

        call h5gclose_f(group_id, error)
        call h5fclose_f(file_id, error)

        where (zbounds > pic_topoz*nzonez)
            zbounds = pic_topoz*nzonez
        endwhere

        if (myid == master) print*, "Finished reading z-bondaries of the reconnection layer"
    end subroutine read_zbounds_rec_layer

    !<--------------------------------------------------------------------------
    !< Save the boundary of the reconnection along z
    !<--------------------------------------------------------------------------
    subroutine save_zbounds(tindex)
        implicit none
        integer, intent(in) :: tindex
        character(len=16) :: tindex_str
        character(len=256) :: fname, fpath
        character(len=16) :: dataset_name
        character(len=8) :: tframe_char
        integer(hid_t) :: file_id, plist_id
        integer(hid_t) :: filespace, memspace, dataset_id
        integer(hsize_t), dimension(3) :: dset_dims, dcount, doffset
        integer :: fileinfo, error
        logical :: dir_e

        fpath=trim(adjustl(rootpath))//'/'//trim(adjustl(output_path))//'/'
        inquire(file=trim(fpath)//'.', exist=dir_e)
        if (.not. dir_e) then
            call system('mkdir -p '//trim(fpath))
        endif
        if (myid == master) then
            print*, "Saving z-boundary of the reconnection layer..."
        endif

        dcount = (/nx_local, ny_local, 2/)
        doffset = (/nx_local*mpi_rankx, ny_local*mpi_ranky, 0/)

        call MPI_INFO_CREATE(fileinfo, ierror)
        call MPI_INFO_SET(fileinfo, "romio_cb_read", "automatic", ierror)
        call MPI_INFO_SET(fileinfo, "romio_ds_read", "automatic", ierror)
        ! call MPI_INFO_SET(fileinfo, "romio_cb_read", "enable", ierror)
        ! call MPI_INFO_SET(fileinfo, "romio_ds_read", "disable", ierror)

        write(tindex_str, "(I0)") tindex
        fname = trim(fpath)//"zbounds_"//trim(tindex_str)//'.h5'

        call h5open_f(error)
        call h5pcreate_f(H5P_FILE_ACCESS_F, plist_id, error)
        call h5pset_fapl_mpio_f(plist_id, MPI_COMM_WORLD, fileinfo, error)
        call h5fcreate_f(fname, H5F_ACC_TRUNC_F, file_id, error, &
            access_prp=plist_id)
        call h5pclose_f(plist_id, error)

        dset_dims = (/pic_topox*nzonex, pic_topoy*nzoney, 2/)
        call h5screate_simple_f(3, dset_dims, filespace, error)
        call h5dcreate_f(file_id, "zbounds", H5T_NATIVE_REAL, filespace, &
            dataset_id, error)

        CALL h5screate_simple_f(3, dcount, memspace, error)
        call h5sselect_hyperslab_f(filespace, H5S_SELECT_SET_F, doffset, &
            dcount, error)

        call h5pcreate_f(H5P_DATASET_XFER_F, plist_id, error)
        call h5pset_dxpl_mpio_f(plist_id, H5FD_MPIO_COLLECTIVE_F, error)

        call h5dwrite_f(dataset_id, H5T_NATIVE_INTEGER, zbounds, &
            dset_dims, error, file_space_id=filespace, mem_space_id=memspace, &
            xfer_prp=plist_id)

        call h5pclose_f(plist_id, error)
        call h5sclose_f(memspace, error)
        call h5sclose_f(filespace, error)
        call h5dclose_f(dataset_id, error)
        call h5fclose_f(file_id, error)
        call h5close_f(error)

        call MPI_INFO_FREE(fileinfo, ierror)
    end subroutine save_zbounds

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

        if (species == 'e') then
            call h5dread_f(dataset_id, H5T_NATIVE_REAL, pspect_e, dset_dims, error, &
                file_space_id=filespace, mem_space_id=memspace, xfer_prp=plist_id)
        else if (species == 'H' .or. species == 'h' .or. species == 'i') then
            call h5dread_f(dataset_id, H5T_NATIVE_REAL, pspect_i, dset_dims, error, &
                file_space_id=filespace, mem_space_id=memspace, xfer_prp=plist_id)
        endif

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
        call cli%init(progname = 'spectrum_reconnection_layer', &
            authors     = 'Xiaocan Li', &
            help        = 'Usage: ', &
            description = 'get spectrum in the reconnection layer', &
            examples    = ['spectrum_reconnection_layer -rp rootpath'])
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
        call cli%add(switch='--particle_energy', switch_ab='-pe', &
            help='particle energy for checking reconnection boundary', &
            required=.false., act='store', def='10', error=error)
        if (error/=0) stop
        call cli%add(switch='--density_threshold', switch_ab='-dt', &
            help='threshold for the density', required=.false., &
            act='store', def='2E-4', error=error)
        if (error/=0) stop
        call cli%add(switch='--emin', switch_ab='-el', &
            help='Minimum particle energy bin', required=.false., &
            act='store', def='1E-6', error=error)
        if (error/=0) stop
        call cli%add(switch='--emax', switch_ab='-eh', &
            help='Maximum particle energy bin', required=.false., &
            act='store', def='1E4', error=error)
        if (error/=0) stop
        call cli%add(switch='--vthe', switch_ab='-ve', &
            help='Electron thermal speed in light speed', required=.false., &
            act='store', def='0.1', error=error)
        if (error/=0) stop
        call cli%add(switch='--input_path', switch_ab='-ip', &
            help='input path of energy spectrum', required=.false., &
            act='store', def='spectrum', error=error)
        if (error/=0) stop
        call cli%add(switch='--output_path', switch_ab='-op', &
            help='output path of energy spectrum', required=.false., &
            act='store', def='spectrum_reconnection_layer', error=error)
        if (error/=0) stop
        call cli%add(switch='--input_suffix', switch_ab='-is', &
            help='input file suffix', required=.false., &
            act='store', def='h5', error=error)
        if (error/=0) stop
        call cli%add(switch='--read_zbounds', switch_ab='-rz', &
            help='whether z-boundaries of the reconnection layer', required=.false., &
            act='store_true', def='.false.', error=error)
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
        call cli%get(switch='-tx', val=topox, error=error)
        if (error/=0) stop
        call cli%get(switch='-ty', val=topoy, error=error)
        if (error/=0) stop
        call cli%get(switch='-tz', val=topoz, error=error)
        if (error/=0) stop
        call cli%get(switch='-pe', val=particle_energy, error=error)
        if (error/=0) stop
        call cli%get(switch='-dt', val=density_threshold, error=error)
        if (error/=0) stop
        call cli%get(switch='-el', val=emin, error=error)
        if (error/=0) stop
        call cli%get(switch='-eh', val=emax, error=error)
        if (error/=0) stop
        call cli%get(switch='-ve', val=vthe, error=error)
        if (error/=0) stop
        call cli%get(switch='-ip', val=input_path, error=error)
        if (error/=0) stop
        call cli%get(switch='-op', val=output_path, error=error)
        if (error/=0) stop
        call cli%get(switch='-is', val=input_suffix, error=error)
        if (error/=0) stop
        call cli%get(switch='-rz', val=read_zbounds, error=error)
        if (error/=0) stop

        if (myid == 0) then
            print '(A,A)', ' The simulation rootpath: ', trim(adjustl(rootpath))
            print '(A,I0,A,I0,A,I0)', ' Min, max and interval: ', &
                tstart, ' ', tend, ' ', tinterval
            print '(A,I0,A,I0,A,I0)', ' PIC topology: ', &
                pic_topox, ", ", pic_topoy, ", ", pic_topoz
            print '(A,I0,A,I0,A,I0)', ' Number of zones along each direction: ', &
                nzonex, ", ", nzoney, ", ", nzonez
            print '(A,I0)', ' Number of data points in each spectrum: ', ndata
            print '(A,I0,A,I0,A,I0)', ' MPI topology for current analysis: ', &
                topox, ", ", topoy, ", ", topoz
            print '(A,A)', ' Input spectrum file path: ', trim(adjustl(input_path))
            print '(A,A)', ' Output spectrum file path: ', trim(adjustl(output_path))
            print '(A,A)', ' Input spectrum file suffix: ', trim(adjustl(input_suffix))
            print '(A,F)', ' Particle energy for checking reconnection layer boundary: ', &
                particle_energy
            print '(A,F)', ' Particle density threshold: ', density_threshold
            print '(A,F,F)', ' Minimum and maximum particle energy bins: ', emin, emax
            print '(A,F)', ' Electron thermal speed: ', vthe
            if (read_zbounds) then
                print '(A)', ' Read z-boundaries of the reconnection layer instead of calculating them'
            else
                print '(A)', ' Calculate z-boundaries of the reconnection layer'
            endif
        endif
    end subroutine get_cmd_args

end program spectrum_reconnection_layer
