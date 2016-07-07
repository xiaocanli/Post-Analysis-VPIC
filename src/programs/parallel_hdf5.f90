!!******************************************************************************
!! Module for reading and writing HDF5 file in parallel
!!******************************************************************************
program parallel_hdf5
    use constants, only: fp, dp
    use mpi_module
    use path_info, only: rootpath
    use particle_info, only: species
    use parameters, only: tp1
    use configuration_translate, only: output_format
    use pic_fields, only: open_electric_field_files, open_magnetic_field_files, &
        read_electric_fields, read_magnetic_fields, close_electric_field_files, &
        close_magnetic_field_files
    use usingle, only: open_velocity_density_files, read_velocity_density, &
        calc_usingle, close_velocity_density_files
    use hdf5
    implicit none
    integer :: ct
    real(dp) :: mp_elapsed
    integer, parameter :: rank = 1
    integer, parameter :: num_dset = 8
    integer(hid_t), dimension(num_dset) :: dset_id
    integer, allocatable, dimension(:) :: np_local, offset_local
    integer :: error
    character(len=256) :: filename, filename_metadata
    character(len=64) :: fname_tracer, fname_metadata
    character(len=16) :: groupname
    character(len=8) :: ct_char
    integer :: current_num_dset
    real :: start, finish, step1, step2
    logical :: is_translated_file
    character(len=32) :: dir_tracer_hdf5
    integer :: tstart, tend, tinterval

    call MPI_INIT(ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)

    call cpu_time(start)

    call get_cmd_args

    call init_analysis

    if (output_format == 1) then
        call open_electric_field_files
        call open_magnetic_field_files
        call open_velocity_density_files
    endif

    call cpu_time(step1)
    do ct = tstart, tend, tinterval
        if (myid == master) print*, ct
        write(ct_char, "(I0)") ct
        current_num_dset = num_dset
        filename = trim(adjustl(rootpath))//"/"//trim(dir_tracer_hdf5)//"/T."
        filename = trim(filename)//trim(ct_char)//"/"//trim(fname_tracer)
        filename_metadata = trim(adjustl(rootpath))//"/"//trim(dir_tracer_hdf5)//"/T."
        filename_metadata = trim(filename_metadata)//trim(ct_char)
        filename_metadata = trim(filename_metadata)//"/"//trim(fname_metadata)
        groupname = "Step#"//trim(ct_char)
        call get_np_local_vpic(filename_metadata, groupname)
        if (is_translated_file) then
            if (output_format /= 1) then
                ! Fields at each time step are saved in different files
                call open_electric_field_files(ct)
                call open_magnetic_field_files(ct)
                call open_velocity_density_files(ct)
                call read_electric_fields(tp1)
                call read_magnetic_fields(tp1)
                call read_velocity_density(tp1)
                call close_electric_field_files
                call close_magnetic_field_files
                call close_velocity_density_files
            else
                ! Fields at all time steps are saved in the same file
                call read_electric_fields(ct)
                call read_magnetic_fields(ct)
                call read_velocity_density(ct)
            endif
            call calc_usingle
        endif
        call get_particle_emf(filename, groupname, ct, 0)
        call free_np_offset_local
        call cpu_time(step2)
        if (myid == master) then
            print '("Time for this step = ",f6.3," seconds.")', step2 - step1
        endif
        step1 = step2
    enddo

    if (output_format == 1) then
        call close_electric_field_files
        call close_magnetic_field_files
        call close_velocity_density_files
    endif

    call end_analysis

    call cpu_time(finish)
    if (myid == master) then
        print '("Time = ",f9.4," seconds.")',finish-start
    endif

    call MPI_FINALIZE(ierr)

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
                set_emf, bx0, by0, bz0, ex0, ey0, ez0
        use interpolation_particle_fields, only: trilinear_interp_vel, &
                vsx0, vsy0, vsz0, calc_vsingle, set_usingle
        use file_header, only: set_v0header
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
        integer :: i, iptl, nptl, ptl_offset
        real(fp), allocatable, dimension(:) :: dX, dY, dZ
        real(fp), allocatable, dimension(:) :: Ex, Ey, Ez, Bx, By, Bz
        real(fp), allocatable, dimension(:) :: Vx, Vy, Vz ! Bulk flow velocity
        integer, allocatable, dimension(:) :: icell
        integer :: storage_type, nlinks, max_corder
        integer :: tx, ty, tz, nx, ny, nz
        real(fp) :: x0, y0, z0, dx_grid, dy_grid, dz_grid
        real(fp) :: dx_domain, dy_domain, dz_domain
        dset_id = 0

        call open_hdf5_parallel(filename, groupname, file_id, group_id)
        call h5gget_info_f(group_id, storage_type, nlinks, max_corder, error)

        !! Open dX, dY, dZ and i to determine particle position
        call open_dset_tracer(group_id, dset_id, dset_dims, &
            dset_dims_max, filespace)

        call get_nptl_offset(dcount(1), doffset(1))
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

        tx = domain%pic_tx
        ty = domain%pic_ty
        tz = domain%pic_tz
        nx = domain%pic_nx
        ny = domain%pic_ny
        nz = domain%pic_nz
        dx_grid = domain%dx
        dy_grid = domain%dy
        dz_grid = domain%dz
        dx_domain = domain%lx_de / tx
        dy_domain = domain%ly_de / ty
        dz_domain = domain%lz_de / tz
        ptl_offset = 0
        do dom_z = ht%start_z, ht%stop_z
            do dom_y = ht%start_y, ht%stop_y
                do dom_x = ht%start_x, ht%stop_x
                    call index_to_rank(dom_x, dom_y, dom_z, tx, ty, tz, n)
                    nptl = np_local(n)
                    if (is_translated_file) then
                        x0 = dx_domain * dom_x
                        y0 = dy_domain * dom_y
                        z0 = dz_domain * dom_z
                        call set_v0header(domain%pic_nx, domain%pic_ny, &
                            domain%pic_nz, x0, y0, z0, dx_grid, dy_grid, dz_grid)
                        call set_emf(dom_x, dom_y, dom_z, tx, ty, tz, &
                            ht%start_x, ht%start_y, ht%start_z)
                        call set_usingle(dom_x, dom_y, dom_z, tx, ty, tz, &
                            ht%start_x, ht%start_y, ht%start_z)
                    else
                        call read_emfields_single(tindex0, n-1)
                        call calc_vsingle(tindex0, n-1, 0)
                    endif
                    do iptl = ptl_offset+1, ptl_offset+nptl
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
                    ptl_offset = ptl_offset + nptl
                enddo ! x
            enddo ! y
        enddo ! z

        call create_emf_datasets(group_id, dset_dims, ex_id, ey_id, ez_id, &
            bx_id, by_id, bz_id, filespace)
        call create_vel_datasets(group_id, dset_dims, vsx_id, vsy_id, &
            vsz_id, filespace)
        ! call MPI_BARRIER(MPI_COMM_WORLD, ierror)
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
        call h5close_f(error)
    end subroutine get_particle_emf

    !!--------------------------------------------------------------------------
    !! Get the total number of particles and offset for current MPI process
    !!--------------------------------------------------------------------------
    subroutine get_nptl_offset(nptl, offset)
        use rank_index_mapping, only: index_to_rank
        use picinfo, only: domain
        use topology_translate, only: ht
        implicit none
        integer(hsize_t), intent(out) :: nptl, offset
        integer :: dom_x, dom_y, dom_z, n
        nptl = 0
        offset = 0
        call index_to_rank(ht%start_x, ht%start_y, ht%start_z, &
            domain%pic_tx, domain%pic_ty, domain%pic_tz, n)
        offset = offset_local(n)
        do dom_x = ht%start_x, ht%stop_x
            do dom_y = ht%start_y, ht%stop_y
                do dom_z = ht%start_z, ht%stop_z
                    call index_to_rank(dom_x, dom_y, dom_z, domain%pic_tx, &
                                       domain%pic_ty, domain%pic_tz, n)
                    nptl = nptl + np_local(n)
                enddo
            enddo
        enddo
    end subroutine get_nptl_offset

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
        ! call MPI_INFO_SET(fileinfo, "striping_factor", "2", ierror)
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
        integer(hsize_t) :: numprocs_ht
        dcount = dset_dims / numprocs
        doffset = dcount * myid
        ! Avoiding gcc complains about type mismatch
        numprocs_ht = int(numprocs, hsize_t)
        rest_size = mod(dset_dims, numprocs_ht)
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
        use mpi_topology, only: set_mpi_topology, htg
        use mpi_datatype_fields, only: set_mpi_datatype_fields
        use mpi_info_module, only: fileinfo, set_mpi_info
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
        use pic_fields, only: init_electric_fields, init_magnetic_fields
        use usingle, only: init_usingle
        implicit none
        integer :: nx, ny, nz

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

        call set_mpi_topology(1)   ! MPI topology
        call set_mpi_datatype_fields
        call set_mpi_info

        if (is_translated_file) then
            call init_electric_fields(htg%nx, htg%ny, htg%nz)
            call init_magnetic_fields(htg%nx, htg%ny, htg%nz)
            call init_usingle(species)
        endif
    end subroutine init_analysis

    !!--------------------------------------------------------------------------
    !! End the analysis by free the memory.
    !!--------------------------------------------------------------------------
    subroutine end_analysis
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
        use mpi_datatype_fields, only: filetype_ghost, filetype_nghost
        use mpi_info_module, only: fileinfo
        use pic_fields, only: free_electric_fields, free_magnetic_fields
        use usingle, only: free_usingle
        implicit none
        call free_neighbors
        call free_emfields
        call free_velocity_fields
        call free_number_density
        call free_start_stop_cells
        if (is_translated_file) then
            call free_electric_fields
            call free_magnetic_fields
            call free_usingle(species)
        endif
        call MPI_TYPE_FREE(datatype, ierror)
        call MPI_INFO_FREE(fileinfo, ierror)
        call MPI_TYPE_FREE(filetype_ghost, ierror)
        call MPI_TYPE_FREE(filetype_nghost, ierror)
    end subroutine end_analysis

    !!--------------------------------------------------------------------------
    !! Read PIC simulation fields data
    !!--------------------------------------------------------------------------
    subroutine read_emfields(ct)
        use pic_fields, only: open_electric_field_files, init_electric_fields, &
            read_electric_fields, free_electric_fields, close_electric_field_files
        use rank_index_mapping, only: index_to_rank
        use topology_translate, only: ht_translate => ht
        use mpi_topology, only: htg
        use picinfo, only: domain
        use parameters, only: tp1
        implicit none
        integer, intent(in) :: ct
        integer :: dom_x, dom_y, dom_z, n
        call init_electric_fields(htg%nx, htg%ny, htg%nz)
        call open_electric_field_files(ct)
        call read_electric_fields(tp1)
        do dom_z = ht_translate%start_z, ht_translate%stop_z
            do dom_y = ht_translate%start_y, ht_translate%stop_y
                call index_to_rank(dom_x, dom_y, dom_z, domain%pic_tx, &
                                   domain%pic_ty, domain%pic_tz, n)
            enddo
        enddo
        call close_electric_field_files
    end subroutine read_emfields

    !<--------------------------------------------------------------------------
    !< Get commandline arguments
    !<--------------------------------------------------------------------------
    subroutine get_cmd_args
        use flap                                !< FLAP package
        use penf
        implicit none
        type(command_line_interface) :: cli     !< Command Line Interface (CLI).
        integer(I4P)                 :: error   !< Error trapping flag.
        call cli%init(progname = 'parallel_hdf5', &
            authors     = 'Xiaocan Li', &
            help        = 'Usage: ', &
            description = 'Get eletromagnetic fields and bulk velocity at particle positions', &
            examples    = ['parallel_hdf5 -tf -ts 0 -te 7 -ti 7 -sp e&
                                -ft ion_tracer_reduced_sorted.h5p &
                                -fm grid_metadata_electron_tracer.h5p'])
        call cli%add(switch='--translated_file', switch_ab='-tf', &
            help='whether using translated fields file', required=.false., &
            act='store_true', def='.false.', error=error)
        if (error/=0) stop
        call cli%add(switch='--dir_tracer_hdf5', switch_ab='-dt', &
            help='HDF5 tracer directory', required=.false., &
            act='store', def='tracer', error=error)
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
        call cli%add(switch='--fname_tracer', switch_ab='-ft', &
            help='Particle tracer file name', required=.false., &
            act='store', def='electron_tracer.h5p', error=error)
        if (error/=0) stop
        call cli%add(switch='--fname_metadata', switch_ab='-fm', &
            help='Particle tracer metadata file name', required=.false., &
            act='store', def='grid_metadata_electron_tracer.h5p', error=error)
        call cli%add(switch='--species', switch_ab='-sp', &
            help="Particle species: 'e' or 'h'", required=.false., &
            act='store', def='e', error=error)
        if (error/=0) stop
        call cli%get(switch='-tf', val=is_translated_file, error=error)
        if (error/=0) stop
        call cli%get(switch='-dt', val=dir_tracer_hdf5, error=error)
        if (error/=0) stop
        call cli%get(switch='-ts', val=tstart, error=error)
        if (error/=0) stop
        call cli%get(switch='-te', val=tend, error=error)
        if (error/=0) stop
        call cli%get(switch='-ti', val=tinterval, error=error)
        if (error/=0) stop
        call cli%get(switch='-ft', val=fname_tracer, error=error)
        if (error/=0) stop
        call cli%get(switch='-fm', val=fname_metadata, error=error)
        if (error/=0) stop
        call cli%get(switch='-sp', val=species, error=error)
        if (error/=0) stop

        if (myid == 0) then
            print '(A,L1)', 'Whether using translated fields file: ', is_translated_file
            print '(A,A)', 'Tracer directory: ', dir_tracer_hdf5
            print '(A,I0,A,I0,A,I0)', 'Min, max and interval: ', &
                tstart, ' ', tend, ' ', tinterval
            if (species == 'e') then
                print '(A,A)', 'Particle: electron'
            else if (species == 'h' .or. species == 'i') then
                print '(A,A)', 'Particle: ion'
            endif
            print '(A,A)', 'Tracer filename: ', trim(fname_tracer)
            print '(A,A)', 'Metadata filename: ', trim(fname_metadata)
        endif
    end subroutine get_cmd_args
end program parallel_hdf5
