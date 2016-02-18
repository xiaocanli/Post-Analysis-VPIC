!!******************************************************************************
!! Module for reading and writing HDF5 file in parallel
!!******************************************************************************
program parallel_hdf5
    use constants, only: fp, dp
    use mpi_module
    use path_info, only: rootpath
    use particle_info, only: species
    use commandline_arguments, only: dir_tracer_hdf5
    use hdf5
    implicit none
    integer :: ct
    real(dp) :: mp_elapsed
    integer, parameter :: rank = 1
    integer, parameter :: num_dset = 8
    integer(hid_t), dimension(num_dset) :: dset_id
    character(len=16), dimension(num_dset) :: dset_name
    integer, allocatable, dimension(:) :: np_local, offset_local
    integer :: error
    character(len=256) :: filename, filename_metadata
    character(len=16) :: groupname
    character(len=8) :: ct_char
    integer :: current_num_dset

    ct = 1
    species = 'e'

    call init_analysis

    do ct = 0, 5, 5
        if (myid == master) print*, ct
        write(ct_char, "(I0)") ct
        current_num_dset = num_dset
        filename = trim(adjustl(rootpath))//"/"//trim(dir_tracer_hdf5)//"/T."
        filename = trim(filename)//trim(ct_char)//"/electron_tracer.h5p"
        filename_metadata = trim(adjustl(rootpath))//"/"//trim(dir_tracer_hdf5)//"/T."
        filename_metadata = trim(filename_metadata)//trim(ct_char)
        filename_metadata = trim(filename_metadata)//"/grid_metadata_electron_tracer.h5p"
        groupname = "Step#"//trim(ct_char)
        call get_np_local_vpic(filename_metadata, groupname)
        ! call get_particle_emf(filename, groupname, ct, 0)
        call free_np_offset_local
    enddo

    call end_analysis

    contains

    !!--------------------------------------------------------------------------
    !! Get electromagnetic fields at particle position
    !!--------------------------------------------------------------------------
    subroutine get_particle_emf(filename, groupname, tindex0, output_record)
        use interpolation_emf, only: read_emfields_single, &
                calc_emfields_derivatives
        use rank_index_mapping, only: index_to_rank
        use picinfo, only: domain
        use topology_translate, only: ht
        use particle_module, only: ptl, calc_interp_param, &
                iex, jex, kex, iey, jey, key, iez, jez, kez, ibx, jbx, kbx, &
                iby, jby, kby, ibz, jbz, kbz, dx_ex, dy_ex, dz_ex, &
                dx_ey, dy_ey, dz_ey, dx_ez, dy_ez, dz_ez, dx_bx, dx_by, dx_bz, &
                dy_bx, dy_by, dy_bz, dz_bx, dz_by, dz_bz, ino, jno, kno, &
                dnx, dny, dnz
        use interpolation_emf, only: trilinear_interp_only_bx, &
                trilinear_interp_only_by, trilinear_interp_only_bz, &
                trilinear_interp_ex, trilinear_interp_ey, trilinear_interp_ez, &
                bx0, by0, bz0, ex0, ey0, ez0
        use interpolation_particle_fields, only: trilinear_interp_vel, &
                vsx0, vsy0, vsz0, calc_vsingle
        implicit none
        character(*), intent(in) :: filename, groupname
        integer, intent(in) :: tindex0, output_record
        integer(hsize_t), dimension(rank) :: dcount, doffset
        integer(hid_t) :: file_id, group_id
        integer(hid_t) :: bx_id, by_id, bz_id, ex_id, ey_id, ez_id
        integer(hid_t) :: vsx_id, vsy_id, vsz_id
        integer(hsize_t), dimension(1) :: dset_dims, dset_dims_max
        integer(hid_t) :: filespace
        integer :: dom_x, dom_y, dom_z, n
        integer :: ix, iy, iz, i, iptl
        real(fp), allocatable, dimension(:) :: dX, dY, dZ
        real(fp), allocatable, dimension(:) :: Ex, Ey, Ez, Bx, By, Bz
        real(fp), allocatable, dimension(:) :: Vx, Vy, Vz ! Bulk flow velocity
        integer, allocatable, dimension(:) :: icell
        integer :: storage_type, nlinks, max_corder
        dset_id = 0
        dset_name = (/"Ux", "Uy", "Uz", "dX", "dY", "dZ", "i", "q"/)

        call open_hdf5_parallel(filename, groupname, file_id, group_id)
        call h5gget_info_f(group_id, storage_type, nlinks, max_corder, error)

        !! Open dX, dY, dZ and i to determine particle position
        call open_dset_tracer(group_id, dset_id, dset_dims, &
            dset_dims_max, filespace)
        call create_emf_datasets(group_id, dset_dims, ex_id, ey_id, ez_id, &
            bx_id, by_id, bz_id, filespace)
        call create_vel_datasets(group_id, dset_dims, vsx_id, vsy_id, &
            vsz_id, filespace)

        do dom_x = ht%start_x, ht%stop_x
            ix = (dom_x - ht%start_x) * domain%pic_nx
            do dom_y = ht%start_y, ht%stop_y
                iy = (dom_y - ht%start_y) * domain%pic_ny
                do dom_z = ht%start_z, ht%stop_z
                    iz = (dom_z - ht%start_z) * domain%pic_nz
                    call index_to_rank(dom_x, dom_y, dom_z, domain%pic_tx, &
                                       domain%pic_ty, domain%pic_tz, n)
                    dcount(1) = np_local(n)
                    doffset(1) = offset_local(n)
                    allocate(dX(dcount(1)))
                    allocate(dY(dcount(1)))
                    allocate(dZ(dcount(1)))
                    allocate(icell(dcount(1)))
                    allocate(Ex(dcount(1)))
                    allocate(Ey(dcount(1)))
                    allocate(Ez(dcount(1)))
                    allocate(Bx(dcount(1)))
                    allocate(By(dcount(1)))
                    allocate(Bz(dcount(1)))
                    allocate(Vx(dcount(1)))
                    allocate(Vy(dcount(1)))
                    allocate(Vz(dcount(1)))
                    call read_hdf5_parallel_real(dset_id(4), dcount, doffset, &
                        dset_dims, dX)
                    call read_hdf5_parallel_real(dset_id(5), dcount, doffset, &
                        dset_dims, dY)
                    call read_hdf5_parallel_real(dset_id(6), dcount, doffset, &
                        dset_dims, dZ)
                    call read_hdf5_parallel_integer(dset_id(7), dcount, &
                        doffset, dset_dims, icell)
                    call read_emfields_single(tindex0, n-1)
                    call calc_vsingle(tindex0, n-1)
                    do iptl = 1, dcount(1)
                        ptl%dx = dX(iptl)
                        ptl%dy = dY(iptl)
                        ptl%dz = dZ(iptl)
                        ptl%icell = icell(iptl)
                        call calc_interp_param
                        call trilinear_interp_only_bx(ibx, jbx, kbx, dx_bx, dy_bx, dz_bx)
                        call trilinear_interp_only_by(iby, jby, kby, dx_by, dy_by, dz_by)
                        call trilinear_interp_only_bz(ibz, jbz, kbz, dx_bz, dy_bz, dz_bz)
                        call trilinear_interp_ex(iex, jex, kex, dx_ex, dy_ex, dz_ex)
                        call trilinear_interp_ey(iey, jey, key, dx_ey, dy_ey, dz_ey)
                        call trilinear_interp_ez(iez, jez, kez, dx_ez, dy_ez, dz_ez)
                        call trilinear_interp_vel(ino, jno, kno, dnx, dny, dnz)
                        Ex(iptl) = ex0
                        Ey(iptl) = ey0
                        Ez(iptl) = ez0
                        Bx(iptl) = bx0
                        By(iptl) = by0
                        Bz(iptl) = bz0
                        Vx(iptl) = vsx0
                        Vy(iptl) = vsy0
                        Vz(iptl) = vsz0
                    enddo
                    call write_hdf5_parallel_real(ex_id, dcount, doffset, dset_dims, Ex)
                    call write_hdf5_parallel_real(ey_id, dcount, doffset, dset_dims, Ey)
                    call write_hdf5_parallel_real(ez_id, dcount, doffset, dset_dims, Ez)
                    call write_hdf5_parallel_real(bx_id, dcount, doffset, dset_dims, Bx)
                    call write_hdf5_parallel_real(by_id, dcount, doffset, dset_dims, By)
                    call write_hdf5_parallel_real(bz_id, dcount, doffset, dset_dims, Bz)
                    call write_hdf5_parallel_real(vsx_id, dcount, doffset, dset_dims, Vx)
                    call write_hdf5_parallel_real(vsy_id, dcount, doffset, dset_dims, Vy)
                    call write_hdf5_parallel_real(vsz_id, dcount, doffset, dset_dims, Vz)
                    deallocate(dX, dY, dZ, icell)
                    deallocate(Ex, Ey, Ez, Bx, By, Bz)
                    deallocate(Vx, Vy, Vz)
                enddo ! x
            enddo ! y
        enddo ! z

        call h5dclose_f(vsx_id, error)
        call h5dclose_f(vsy_id, error)
        call h5dclose_f(vsz_id, error)
        call h5dclose_f(ex_id, error)
        call h5dclose_f(ey_id, error)
        call h5dclose_f(ez_id, error)
        call h5dclose_f(bx_id, error)
        call h5dclose_f(by_id, error)
        call h5dclose_f(bz_id, error)
        call h5sclose_f(filespace, error)
        call h5dclose_f(dset_id(4), error)
        call h5dclose_f(dset_id(5), error)
        call h5dclose_f(dset_id(6), error)
        call h5dclose_f(dset_id(7), error)
        call h5gclose_f(group_id, error)
        call h5fclose_f(file_id, error)
    end subroutine get_particle_emf

    !!--------------------------------------------------------------------------
    !! Open datasets of the tracer file
    !!--------------------------------------------------------------------------
    subroutine open_dset_tracer(group_id, dset_id, dset_dims, dset_dims_max, &
            filespace)
        implicit none
        integer(hid_t), intent(in) :: group_id
        integer(hid_t), dimension(*), intent(inout) :: dset_id
        integer(hsize_t), dimension(1), intent(out) :: dset_dims, dset_dims_max
        integer(hid_t), intent(out) :: filespace
        call open_hdf5_dataset("dX", group_id, dset_id(4), &
            dset_dims, dset_dims_max, filespace)
        call open_hdf5_dataset("dY", group_id, dset_id(5), &
            dset_dims, dset_dims_max, filespace)
        call open_hdf5_dataset("dZ", group_id, dset_id(6), &
            dset_dims, dset_dims_max, filespace)
        call open_hdf5_dataset("i", group_id, dset_id(7), &
            dset_dims, dset_dims_max, filespace)
    end subroutine open_dset_tracer

    !!--------------------------------------------------------------------------
    !! Create bulk velocity dataset
    !!--------------------------------------------------------------------------
    subroutine create_vel_datasets(group_id, dset_dims, vsx_id, vsy_id, &
            vsz_id, filespace)
        implicit none
        integer(hid_t), intent(in) :: group_id
        integer(hsize_t), dimension(1), intent(in) :: dset_dims
        integer(hid_t), intent(out) :: vsx_id, vsy_id, vsz_id, filespace
        call create_hdf5_dataset("Vx", group_id, vsx_id, &
            dset_dims, H5T_NATIVE_REAL, filespace)
        current_num_dset = current_num_dset + 1
        call create_hdf5_dataset("Vy", group_id, vsy_id, &
            dset_dims, H5T_NATIVE_REAL, filespace)
        current_num_dset = current_num_dset + 1
        call create_hdf5_dataset("Vz", group_id, vsz_id, &
            dset_dims, H5T_NATIVE_REAL, filespace)
        current_num_dset = current_num_dset + 1
    end subroutine create_vel_datasets

    !!--------------------------------------------------------------------------
    !! Create electric field and magnetic field datasets
    !!--------------------------------------------------------------------------
    subroutine create_emf_datasets(group_id, dset_dims, ex_id, ey_id, ez_id, &
            bx_id, by_id, bz_id, filespace)
        implicit none
        integer(hid_t), intent(in) :: group_id
        integer(hsize_t), dimension(1), intent(in) :: dset_dims
        integer(hid_t), intent(out) :: ex_id, ey_id, ez_id, bx_id, by_id, &
            bz_id, filespace
        call create_hdf5_dataset("Ex", group_id, ex_id, &
            dset_dims, H5T_NATIVE_REAL, filespace)
        current_num_dset = current_num_dset + 1
        call create_hdf5_dataset("Ey", group_id, ey_id, &
            dset_dims, H5T_NATIVE_REAL, filespace)
        current_num_dset = current_num_dset + 1
        call create_hdf5_dataset("Ez", group_id, ez_id, &
            dset_dims, H5T_NATIVE_REAL, filespace)
        current_num_dset = current_num_dset + 1
        call create_hdf5_dataset("Bx", group_id, bx_id, &
            dset_dims, H5T_NATIVE_REAL, filespace)
        current_num_dset = current_num_dset + 1
        call create_hdf5_dataset("By", group_id, by_id, &
            dset_dims, H5T_NATIVE_REAL, filespace)
        current_num_dset = current_num_dset + 1
        call create_hdf5_dataset("Bz", group_id, bz_id, &
            dset_dims, H5T_NATIVE_REAL, filespace)
        current_num_dset = current_num_dset + 1
    end subroutine create_emf_datasets

    !!--------------------------------------------------------------------------
    !! Open hdf5 file in parallel
    !!--------------------------------------------------------------------------
    subroutine open_hdf5_parallel(filename, groupname, file_id, group_id)
        use mpi_info_module, only: fileinfo
        implicit none
        character(*), intent(in) :: filename, groupname
        integer(hid_t), intent(out) :: file_id, group_id
        integer(hid_t) :: plist_id
        integer :: storage_type, max_corder
        integer(size_t) :: obj_count_g, obj_count_d
        call h5open_f(error)
        call h5pcreate_f(H5P_FILE_ACCESS_F, plist_id, error)
        call h5pset_fapl_mpio_f(plist_id, MPI_COMM_WORLD, fileinfo, error)
        call h5fopen_f(filename, H5F_ACC_RDWR_F, file_id, error, &
            access_prp=plist_id)
        call h5pclose_f(plist_id, error)
        call h5gopen_f(file_id, groupname, group_id, error)
    end subroutine open_hdf5_parallel

    !---------------------------------------------------------------------------
    ! Read hdf5 dataset in parallel for integer data
    !---------------------------------------------------------------------------
    subroutine read_hdf5_parallel_integer(dset_id, dcount, doffset, &
            dset_dims, fdata)
        implicit none
        integer(hid_t), intent(in) :: dset_id
        integer(hsize_t), dimension(1), intent(in) :: dcount, doffset, dset_dims
        integer, dimension(*), intent(out) :: fdata
        integer(hid_t) :: filespace, memspace, plist_id
        call init_read_hdf5_parallel(dset_id, dcount, doffset, &
            dset_dims, filespace, memspace, plist_id)
        call h5dread_f(dset_id, H5T_NATIVE_INTEGER, fdata, dset_dims, error, &
            file_space_id=filespace, mem_space_id=memspace, xfer_prp=plist_id)
        call final_read_hdf5_parallel(filespace, memspace, plist_id)
    end subroutine read_hdf5_parallel_integer

    !---------------------------------------------------------------------------
    ! Read hdf5 dataset in parallel for real data
    !---------------------------------------------------------------------------
    subroutine read_hdf5_parallel_real(dset_id, dcount, doffset, &
            dset_dims, fdata)
        implicit none
        integer(hid_t), intent(in) :: dset_id
        integer(hsize_t), dimension(1), intent(in) :: dcount, doffset, dset_dims
        real(fp), dimension(*), intent(out) :: fdata
        integer(hid_t) :: filespace, memspace, plist_id
        call init_read_hdf5_parallel(dset_id, dcount, doffset, &
            dset_dims, filespace, memspace, plist_id)
        call h5dread_f(dset_id, H5T_NATIVE_REAL, fdata, dset_dims, error, &
            file_space_id=filespace, mem_space_id=memspace, xfer_prp=plist_id)
        call final_read_hdf5_parallel(filespace, memspace, plist_id)
    end subroutine read_hdf5_parallel_real

    !---------------------------------------------------------------------------
    ! Read hdf5 dataset in parallel for real data
    !---------------------------------------------------------------------------
    subroutine write_hdf5_parallel_real(dset_id, dcount, doffset, &
            dset_dims, fdata)
        implicit none
        integer(hid_t), intent(in) :: dset_id
        integer(hsize_t), dimension(1), intent(in) :: dcount, doffset, dset_dims
        real(fp), dimension(*), intent(in) :: fdata
        integer(hid_t) :: filespace, memspace, plist_id
        call init_read_hdf5_parallel(dset_id, dcount, doffset, &
            dset_dims, filespace, memspace, plist_id)
        call h5dwrite_f(dset_id, H5T_NATIVE_REAL, fdata, dset_dims, error, &
            file_space_id=filespace, mem_space_id=memspace, xfer_prp=plist_id)
        call final_read_hdf5_parallel(filespace, memspace, plist_id)
    end subroutine write_hdf5_parallel_real

    !---------------------------------------------------------------------------
    ! Initial setup for reading hdf5 file in parallel
    !---------------------------------------------------------------------------
    subroutine init_read_hdf5_parallel(dset_id, dcount, doffset, &
            dset_dims, filespace, memspace, plist_id)
        implicit none
        integer(hid_t), intent(in) :: dset_id
        integer(hsize_t), dimension(1), intent(in) :: dcount, doffset, dset_dims
        integer(hid_t), intent(out) :: filespace, memspace, plist_id
        !! Create property list for collective dataset write
        call h5pcreate_f(H5P_DATASET_XFER_F, plist_id, error)
        call h5pset_dxpl_mpio_f(plist_id, H5FD_MPIO_COLLECTIVE_F, error)

        call h5screate_simple_f(rank, dcount, memspace, error)
        call h5dget_space_f(dset_id, filespace, error)
        call h5sselect_hyperslab_f(filespace, H5S_SELECT_SET_F, doffset, &
            dcount, error)
    end subroutine init_read_hdf5_parallel

    !---------------------------------------------------------------------------
    ! Initial setup for reading hdf5 file in parallel
    !---------------------------------------------------------------------------
    subroutine final_read_hdf5_parallel(filespace, memspace, plist_id)
        implicit none
        integer(hid_t), intent(in) :: filespace, memspace, plist_id
        call h5sclose_f(filespace, error)
        call h5sclose_f(memspace, error)
        call h5pclose_f(plist_id, error)
    end subroutine final_read_hdf5_parallel

    !!--------------------------------------------------------------------------
    !! Open hdf5 dataset and get the dataset dimensions
    !!--------------------------------------------------------------------------
    subroutine open_hdf5_dataset(dataset_name, group_id, dataset_id, &
            dset_dims, dset_dims_max, filespace)
        implicit none
        character(*), intent(in) :: dataset_name
        integer(hid_t), intent(in) :: group_id
        integer(hid_t), intent(out) :: dataset_id, filespace
        integer(hsize_t), dimension(1), intent(out) :: dset_dims, &
            dset_dims_max
        integer :: datatype_id
        call h5dopen_f(group_id, dataset_name, dataset_id, error)
        call h5dget_type_f(dataset_id, datatype_id, error)
        call h5dget_space_f(dataset_id, filespace, error)
        call h5Sget_simple_extent_dims_f(filespace, dset_dims, &
            dset_dims_max, error)
    end subroutine open_hdf5_dataset

    !!--------------------------------------------------------------------------
    !! Create hdf5 dataset
    !!--------------------------------------------------------------------------
    subroutine create_hdf5_dataset(dataset_name, group_id, dataset_id, &
            dset_dims, datatype, filespace)
        implicit none
        character(*), intent(in) :: dataset_name
        integer(hid_t), intent(in) :: group_id, datatype
        integer(hsize_t), dimension(1), intent(in) :: dset_dims
        integer(hid_t), intent(out) :: dataset_id, filespace
        integer :: storage_type, nlinks, max_corder
        call h5gget_info_f(group_id, storage_type, nlinks, max_corder, error)
        if (nlinks <= current_num_dset) then
            call h5screate_simple_f(rank, dset_dims, filespace, error)
            call h5dcreate_f(group_id, dataset_name, datatype, filespace, &
                dataset_id, error)
        else
            call h5dopen_f(group_id, dataset_name, dataset_id, error)
            call h5dget_space_f(dataset_id, filespace, error)
        endif
    end subroutine create_hdf5_dataset

    !!--------------------------------------------------------------------------
    !! Open metadata file and dataset of "np_local"
    !!--------------------------------------------------------------------------
    subroutine open_metadata_dset(fname_metadata, groupname, file_id, &
            group_id, dataset_id, dset_dims, dset_dims_max, filespace)
        implicit none
        character(*), intent(in) :: fname_metadata, groupname
        integer(hid_t), intent(out) :: file_id, group_id, dataset_id
        integer(hsize_t), dimension(1), intent(out) :: dset_dims, dset_dims_max
        integer(hid_t), intent(out) :: filespace
        call open_hdf5_serial(fname_metadata, groupname, file_id, group_id)
        call open_hdf5_dataset("np_local", group_id, dataset_id, &
            dset_dims, dset_dims_max, filespace)
    end subroutine open_metadata_dset

    !!--------------------------------------------------------------------------
    !! Close dataset, filespace, group and file of metadata
    !!--------------------------------------------------------------------------
    subroutine close_metadata_dset(file_id, group_id, dataset_id, filespace)
        implicit none
        integer(hid_t), intent(in) :: file_id, group_id, dataset_id, filespace
        call h5sclose_f(filespace, error)
        call h5dclose_f(dataset_id, error)
        call h5gclose_f(group_id, error)
        call h5fclose_f(file_id, error)
    end subroutine close_metadata_dset

    !!--------------------------------------------------------------------------
    !! Initialize the np_local and offset_local array
    !!--------------------------------------------------------------------------
    subroutine init_np_offset_local(dset_dims)
        implicit none
        integer(hsize_t), dimension(1), intent(in) :: dset_dims
        allocate(np_local(dset_dims(1)))
        allocate(offset_local(dset_dims(1)))
        np_local = 0
        offset_local = 0
    end subroutine init_np_offset_local

    !!--------------------------------------------------------------------------
    !! Free the np_local and offset_local array
    !!--------------------------------------------------------------------------
    subroutine free_np_offset_local
        implicit none
        deallocate(np_local)
        deallocate(offset_local)
    end subroutine free_np_offset_local

    !!--------------------------------------------------------------------------
    !! Get the number of particles for each MPI process of PIC simulations
    !!--------------------------------------------------------------------------
    subroutine get_np_local_vpic(fname_metadata, groupname)
        implicit none
        character(*), intent(in) :: fname_metadata, groupname
        integer(hid_t) :: file_id, group_id, dataset_id
        integer(hsize_t), dimension(1) :: dset_dims, dset_dims_max
        integer(hid_t) :: filespace
        integer :: i
        if (myid == master) then
            call open_metadata_dset(filename_metadata, groupname, file_id, &
                group_id, dataset_id, dset_dims, dset_dims_max, filespace)
        endif
        call MPI_BCAST(dset_dims, 1, MPI_INTEGER, master, MPI_COMM_WORLD, &
            ierror)

        call init_np_offset_local(dset_dims)

        if (myid == master) then
            call h5dread_f(dataset_id, H5T_NATIVE_INTEGER, np_local, &
                dset_dims, error)
        endif
        call MPI_BCAST(np_local, dset_dims(1), MPI_INTEGER, master, &
            MPI_COMM_WORLD, ierror)
        offset_local = 0
        do i = 2, dset_dims(1)
            offset_local(i) = offset_local(i-1) + np_local(i-1)
        enddo
        if (myid == master) then
            call h5sclose_f(filespace, error)
            call h5dclose_f(dataset_id, error)
            call h5gclose_f(group_id, error)
            call h5fclose_f(file_id, error)
        endif
    end subroutine get_np_local_vpic

    !!--------------------------------------------------------------------------
    !! Open hdf5 file using one process
    !!--------------------------------------------------------------------------
    subroutine open_hdf5_serial(filename, groupname, file_id, group_id)
        use mpi_info_module, only: fileinfo
        implicit none
        character(*), intent(in) :: filename, groupname
        integer(hid_t), intent(out) :: file_id, group_id
        integer(size_t) :: obj_count_g, obj_count_d
        integer :: error
        call h5open_f(error)
        call h5fopen_f(filename, H5F_ACC_RDWR_F, file_id, error, &
            access_prp=h5p_default_f)
        call h5gopen_f(file_id, groupname, group_id, error)
    end subroutine open_hdf5_serial

    !!--------------------------------------------------------------------------
    !! Assign the number of data points to each MPI process
    !!--------------------------------------------------------------------------
    subroutine assign_data_mpi(dset_dims, dcount, doffset)
        implicit none
        integer(hsize_t), intent(in) :: dset_dims
        integer(hsize_t), intent(out) :: dcount, doffset
        integer(hsize_t) :: rest_size
        dcount = dset_dims / numprocs
        doffset = dcount * myid
        rest_size = mod(dset_dims, numprocs)
        if (myid < rest_size) then
            dcount = dcount + 1
            doffset = doffset + myid
        else
            doffset = doffset + rest_size
        endif
    end subroutine assign_data_mpi

    !!--------------------------------------------------------------------------
    !! Initialize the analysis.
    !!--------------------------------------------------------------------------
    subroutine init_analysis
        use mpi_module
        use particle_info, only: species, get_ptl_mass_charge
        use path_info, only: get_file_paths
        use picinfo, only: read_domain, broadcast_pic_info, &
                get_total_time_frames, get_energy_band_number, &
                read_thermal_params, calc_energy_interval, nbands, &
                write_pic_info, domain
        use configuration_translate, only: read_configuration
        use topology_translate, only: set_topology, set_start_stop_cells
        use time_info, only: get_nout, adjust_tindex_start, set_output_record
        use mpi_io_translate, only: set_mpi_io
        use parameters, only: get_relativistic_flag, get_start_end_time_points, tp2
        use interpolation_emf, only: init_emfields
        use interpolation_particle_fields, only: init_velocity_fields, &
            init_number_density
        use neighbors_module, only: init_neighbors, get_neighbors
        use commandline_arguments, only: get_dir_tracer_hdf5
        implicit none
        integer :: nx, ny, nz

        call MPI_INIT(ierr)
        call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
        call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)

        call get_dir_tracer_hdf5
        call get_ptl_mass_charge(species)
        call get_file_paths
        if (myid == master) then
            call read_domain
            call write_pic_info
        endif
        call broadcast_pic_info
        call get_start_end_time_points
        call get_relativistic_flag
        call get_energy_band_number
        call read_thermal_params
        if (nbands > 0) then
            call calc_energy_interval
        endif
        call read_configuration
        call get_total_time_frames(tp2)
        call set_topology
        call set_start_stop_cells
        call get_nout
        call adjust_tindex_start
        call set_output_record
        call set_mpi_io

        nx = domain%pic_nx + 2  ! Including ghost cells
        ny = domain%pic_ny + 2
        nz = domain%pic_nz + 2
        call init_neighbors(nx, ny, nz)
        call get_neighbors

        call init_emfields
        call init_velocity_fields
        call init_number_density

    end subroutine init_analysis

    !!--------------------------------------------------------------------------
    !! End the analysis by free the memory.
    !!--------------------------------------------------------------------------
    subroutine end_analysis
        use mpi_module
        use topology_translate, only: free_start_stop_cells
        use mpi_io_translate, only: datatype
        use mpi_info_module, only: fileinfo
        use interpolation_emf, only: free_emfields, free_emfields_derivatives
        use interpolation_particle_fields, only: free_velocity_fields, &
            free_number_density
        use particle_drift, only: free_drift_fields, free_para_perp_fields, &
                free_jdote_sum
        use neighbors_module, only: free_neighbors
        use particle_fields, only: free_density_fields
        implicit none
        call free_neighbors
        call free_emfields
        call free_velocity_fields
        call free_number_density
        call free_start_stop_cells
        call MPI_TYPE_FREE(datatype, ierror)
        call MPI_INFO_FREE(fileinfo, ierror)
        call MPI_FINALIZE(ierr)
    end subroutine end_analysis

end program parallel_hdf5
