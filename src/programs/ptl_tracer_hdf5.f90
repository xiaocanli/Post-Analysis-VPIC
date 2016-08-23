!!******************************************************************************
!! Module for analyzing particle tracer data in HDF5 format
!!******************************************************************************
program ptl_tracer_hdf5
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
    ! integer, parameter :: num_dset = 17
    integer, parameter :: num_dset = 8
    integer(hid_t), dimension(num_dset) :: dset_id
    integer, allocatable, dimension(:) :: np_local, offset_local
    ! Tracer data for current time frame
    real(fp), allocatable, dimension(:) :: Ux, Uy, Uz, dX, dY, dZ
    real(fp), allocatable, dimension(:) :: Ex, Ey, Ez, Bx, By, Bz
    real(fp), allocatable, dimension(:) :: Vx, Vy, Vz ! Bulk flow velocity
    integer, allocatable, dimension(:) :: icell, qtag
    ! Tracer data for other two time frames
    real(fp), allocatable, dimension(:) :: Ux1, Uy1, Uz1, dX1, dY1, dZ1
    real(fp), allocatable, dimension(:) :: Ex1, Ey1, Ez1, Bx1, By1, Bz1
    real(fp), allocatable, dimension(:) :: Vx1, Vy1, Vz1 ! Bulk flow velocity
    integer, allocatable, dimension(:) :: icell1, qtag1
    ! Tracer data for other two time frames
    real(fp), allocatable, dimension(:) :: Ux2, Uy2, Uz2, dX2, dY2, dZ2
    real(fp), allocatable, dimension(:) :: Ex2, Ey2, Ez2, Bx2, By2, Bz2
    real(fp), allocatable, dimension(:) :: Vx2, Vy2, Vz2 ! Bulk flow velocity
    integer, allocatable, dimension(:) :: icell2, qtag2
    integer(hsize_t), dimension(rank) :: dcount, doffset
    integer :: error
    character(len=256) :: filename, fpath_t
    character(len=64) :: fname
    character(len=16) :: groupname
    character(len=8) :: ct_char
    integer :: current_num_dset, tmax, tmin, tint, i, j, j1
    integer, parameter :: nbin = 100
    real(fp) :: gam, gam2, ke,dve, emax, delta_ke, rx
    real(fp) :: distribution(nbin), acceleration(nbin), &
        distribution_all(nbin), acceleration_all(nbin), diffusion(nbin), &
        diffusion_all(nbin)
    logical :: dfile

    ct = 1
    species = 'e'
    emax = 200.
    dve = emax/real(nbin)
    distribution(:) = 0.0
    acceleration(:) = 0.0
    distribution_all(:) = 0.0
    acceleration_all(:) = 0.0
    diffusion(:) = 0.0
    diffusion_all(:) = 0.0

    call init_analysis

    tmax = 10640
    tmin = 0
    tint = 10

    do ct = tmin, tmax, tint
        if (myid == master) print*, ct
        write(ct_char, "(I0)") ct
        current_num_dset = num_dset
        fpath_t = trim(adjustl(rootpath))//"/"//trim(dir_tracer_hdf5)//"/T."
        fname = "/ion_tracer_sorted.h5p"

        ! Current frame
        filename = trim(fpath_t)//trim(ct_char)//trim(fname)
        groupname = "Step#"//trim(ct_char)
        inquire(file=trim(filename), exist=dfile)
        if (dfile) then
            call get_tracer_data(filename, groupname, 0, dcount)
            !print*, size(Vx)

            ! Previous time frame. It can be any other time frame.
            !write(ct_char, "(I0)") ct - tint
            !filename = trim(fpath_t)//trim(ct_char)//trim(fname)
            !groupname = "Step#"//trim(ct_char)
            !call get_tracer_data(filename, groupname, 1)
            !print*, qtag1(1)

            ! Next time frame. It can be any other time frame.
            write(ct_char, "(I0)") ct + tint
            filename = trim(fpath_t)//trim(ct_char)//trim(fname)
            groupname = "Step#"//trim(ct_char)
            call get_tracer_data(filename, groupname, 2, dcount)
            !print*, qtag2(1)

            do i=1, dcount(1)
              gam = sqrt(1.0 + Ux(i)**2 + Uy(i)**2 + Uz(i)**2)-1.0
              gam2 = sqrt(1.0 + Ux2(i)**2 + Uy2(i)**2 + Uz2(i)**2)-1.0
              ke = (gam)
              delta_ke = (gam2-gam)
              rx=ke/dve
              j=int(rx)
              if (j .ge. 1. .and. j.le. nbin) then
                distribution(j) = distribution(j) + 1.0
                acceleration(j) = acceleration(j) + delta_ke
                diffusion(j) = diffusion(j) + 0.5*delta_ke**2
              endif
              
            enddo

            call MPI_REDUCE(distribution, distribution_all, nbin, MPI_REAL, &
                MPI_SUM, 0, MPI_COMM_WORLD, ierr)
            call MPI_REDUCE(acceleration, acceleration_all, nbin, MPI_REAL, &
                MPI_SUM, 0, MPI_COMM_WORLD, ierr)
            call MPI_REDUCE(diffusion, diffusion_all, nbin, MPI_REAL, MPI_SUM, &
                0, MPI_COMM_WORLD, ierr)
           
            if (myid==0) then
               do i=1,nbin
                 if (distribution_all(i) > 0.0) then
                 acceleration_all(i) = acceleration_all(i)/distribution_all(i)
                 diffusion_all(i) = diffusion_all(i)/distribution_all(i)
                 endif
               enddo
               !print*, distribution_all
               !print*, acceleration_all
               write(fname,"(A,I0)")"../data/acceleration.",ct/tint
               open(unit=11,file=trim(fname)//'.dat',status='unknown')
               do i=1,nbin
                write(11,66) dve*float(i), &
                    distribution_all(i)/sum(distribution_all), &
                    acceleration_all(i), diffusion_all(i)
               enddo
    66      format (4(f8.3,2x))
            endif
        endif

    enddo

    call free_tracer_data
    call free_tracer_data1
    call free_tracer_data2
    call end_analysis

    contains

    !!--------------------------------------------------------------------------
    !! Get the tracer data for one time frame
    !! Input:
    !!  frame_tag: 0 for current time frame, 1 or 2 for other frames
    !!--------------------------------------------------------------------------
    subroutine get_tracer_data(filename, groupname, frame_tag, dcount)
        use topology_translate, only: ht
        implicit none
        character(*), intent(in) :: filename, groupname
        integer, intent(in) :: frame_tag
        integer(hsize_t), dimension(rank) :: dcount, doffset
        integer(hid_t) :: file_id, group_id
        integer(hsize_t), dimension(1) :: dset_dims, dset_dims_max
        integer(hid_t) :: filespace
        integer :: dom_x, dom_y, dom_z, n
        integer :: ix, iy, iz, i, iptl
        integer :: storage_type, nlinks, max_corder
        dset_id = 0

        call open_hdf5_parallel(filename, groupname, file_id, group_id)
        call h5gget_info_f(group_id, storage_type, nlinks, max_corder, error)

        !! Open dX, dY, dZ and i to determine particle position
        call open_dset_tracer(group_id, dset_id, dset_dims, &
            dset_dims_max, filespace)

        call assign_data_mpi(dset_dims(1), dcount(1), doffset(1))

        select case (frame_tag)
            case (0)
                if (.not. allocated(Ux)) then
                    call init_tracer_data(dcount(1))
                endif
                call read_tracer_data(dset_id, dcount, doffset, dset_dims)
            case (1)
                if (.not. allocated(Ux1)) then
                    call init_tracer_data1(dcount(1))
                endif
                call read_tracer_data1(dset_id, dcount, doffset, dset_dims)
            case (2)
                if (.not. allocated(Ux2)) then
                    call init_tracer_data2(dcount(1))
                endif
                call read_tracer_data2(dset_id, dcount, doffset, dset_dims)
            case default
                print*, 'frame_tag is invalid!!'
        end select

        call close_dset_tracer(dset_id, filespace, num_dset)
        call h5gclose_f(group_id, error)
        call h5fclose_f(file_id, error)
        call h5close_f(error)
    end subroutine get_tracer_data

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
        call open_hdf5_dataset("Ux", group_id, dset_id(1), &
            dset_dims, dset_dims_max, filespace)
        call open_hdf5_dataset("Uy", group_id, dset_id(2), &
            dset_dims, dset_dims_max, filespace)
        call open_hdf5_dataset("Uz", group_id, dset_id(3), &
            dset_dims, dset_dims_max, filespace)
        call open_hdf5_dataset("dX", group_id, dset_id(4), &
            dset_dims, dset_dims_max, filespace)
        call open_hdf5_dataset("dY", group_id, dset_id(5), &
            dset_dims, dset_dims_max, filespace)
        call open_hdf5_dataset("dZ", group_id, dset_id(6), &
            dset_dims, dset_dims_max, filespace)
        call open_hdf5_dataset("i", group_id, dset_id(7), &
            dset_dims, dset_dims_max, filespace)
        call open_hdf5_dataset("q", group_id, dset_id(8), &
            dset_dims, dset_dims_max, filespace)
        ! call open_hdf5_dataset("Bx", group_id, dset_id(9), &
        !     dset_dims, dset_dims_max, filespace)
        ! call open_hdf5_dataset("By", group_id, dset_id(10), &
        !     dset_dims, dset_dims_max, filespace)
        ! call open_hdf5_dataset("Bz", group_id, dset_id(11), &
        !     dset_dims, dset_dims_max, filespace)
        ! call open_hdf5_dataset("Ex", group_id, dset_id(12), &
        !     dset_dims, dset_dims_max, filespace)
        ! call open_hdf5_dataset("Ey", group_id, dset_id(13), &
        !     dset_dims, dset_dims_max, filespace)
        ! call open_hdf5_dataset("Ez", group_id, dset_id(14), &
        !     dset_dims, dset_dims_max, filespace)
        ! call open_hdf5_dataset("Vx", group_id, dset_id(15), &
        !     dset_dims, dset_dims_max, filespace)
        ! call open_hdf5_dataset("Vy", group_id, dset_id(16), &
        !     dset_dims, dset_dims_max, filespace)
        ! call open_hdf5_dataset("Vz", group_id, dset_id(17), &
        !     dset_dims, dset_dims_max, filespace)
    end subroutine open_dset_tracer

    !!--------------------------------------------------------------------------
    !! Close datasets of the tracer file
    !!--------------------------------------------------------------------------
    subroutine close_dset_tracer(dset_id, filespace, num_dset)
        implicit none
        integer(hid_t), dimension(*), intent(in) :: dset_id
        integer(hid_t), intent(in) :: filespace
        integer, intent(in) :: num_dset
        integer :: i
        call h5sclose_f(filespace, error)
        do i = 1, num_dset
            call h5dclose_f(dset_id(i), error)
        enddo
    end subroutine close_dset_tracer

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

    !!--------------------------------------------------------------------------
    !! Initialize the arrays for the tracer data
    !!--------------------------------------------------------------------------
    subroutine init_tracer_data(nd)
        implicit none
        integer(hsize_t), intent(in) :: nd
        allocate(Ux(nd))
        allocate(Uy(nd))
        allocate(Uz(nd))
        allocate(dX(nd))
        allocate(dY(nd))
        allocate(dZ(nd))
        allocate(icell(nd))
        allocate(qtag(nd))
        ! allocate(Ex(nd))
        ! allocate(Ey(nd))
        ! allocate(Ez(nd))
        ! allocate(Bx(nd))
        ! allocate(By(nd))
        ! allocate(Bz(nd))
        ! allocate(Vx(nd))
        ! allocate(Vy(nd))
        ! allocate(Vz(nd))
        Ux = 0.0; Uy = 0.0; Uz = 0.0
        dX = 0.0; dY = 0.0; dZ = 0.0
        ! Bx = 0.0; By = 0.0; Bz = 0.0
        ! Ex = 0.0; Ey = 0.0; Ez = 0.0
        ! Vx = 0.0; Vy = 0.0; Vz = 0.0
        icell = 0; qtag = 0
    end subroutine init_tracer_data

    !!--------------------------------------------------------------------------
    !! Initialize the arrays for the tracer data for another time frame
    !!--------------------------------------------------------------------------
    subroutine init_tracer_data1(nd)
        implicit none
        integer(hsize_t), intent(in) :: nd
        allocate(Ux1(nd))
        allocate(Uy1(nd))
        allocate(Uz1(nd))
        allocate(dX1(nd))
        allocate(dY1(nd))
        allocate(dZ1(nd))
        allocate(icell1(nd))
        allocate(qtag1(nd))
        ! allocate(Ex1(nd))
        ! allocate(Ey1(nd))
        ! allocate(Ez1(nd))
        ! allocate(Bx1(nd))
        ! allocate(By1(nd))
        ! allocate(Bz1(nd))
        ! allocate(Vx1(nd))
        ! allocate(Vy1(nd))
        ! allocate(Vz1(nd))
        Ux1 = 0.0; Uy1 = 0.0; Uz1 = 0.0
        dX1 = 0.0; dY1 = 0.0; dZ1 = 0.0
        ! Bx1 = 0.0; By1 = 0.0; Bz1 = 0.0
        ! Ex1 = 0.0; Ey1 = 0.0; Ez1 = 0.0
        ! Vx1 = 0.0; Vy1 = 0.0; Vz1 = 0.0
        icell1 = 0; qtag1 = 0
    end subroutine init_tracer_data1

    !!--------------------------------------------------------------------------
    !! Initialize the arrays for the tracer data for another time frame
    !!--------------------------------------------------------------------------
    subroutine init_tracer_data2(nd)
        implicit none
        integer(hsize_t), intent(in) :: nd
        allocate(Ux2(nd))
        allocate(Uy2(nd))
        allocate(Uz2(nd))
        allocate(dX2(nd))
        allocate(dY2(nd))
        allocate(dZ2(nd))
        allocate(icell2(nd))
        allocate(qtag2(nd))
        ! allocate(Ex2(nd))
        ! allocate(Ey2(nd))
        ! allocate(Ez2(nd))
        ! allocate(Bx2(nd))
        ! allocate(By2(nd))
        ! allocate(Bz2(nd))
        ! allocate(Vx2(nd))
        ! allocate(Vy2(nd))
        ! allocate(Vz2(nd))
        Ux2 = 0.0; Uy2 = 0.0; Uz2 = 0.0
        dX2 = 0.0; dY2 = 0.0; dZ2 = 0.0
        ! Bx2 = 0.0; By2 = 0.0; Bz2 = 0.0
        ! Ex2 = 0.0; Ey2 = 0.0; Ez2 = 0.0
        ! Vx2 = 0.0; Vy2 = 0.0; Vz2 = 0.0
        icell2 = 0; qtag2 = 0
    end subroutine init_tracer_data2

    !!--------------------------------------------------------------------------
    !! Free tracer data
    !!--------------------------------------------------------------------------
    subroutine free_tracer_data
        implicit none
        if (allocated(Bx)) then
            deallocate(Ux, Uy, Uz, dX, dY, dZ, icell, qtag)
            ! deallocate(Ex, Ey, Ez, Bx, By, Bz)
            ! deallocate(Vx, Vy, Vz)
        endif
    end subroutine free_tracer_data

    !!--------------------------------------------------------------------------
    !! Free tracer data for another time frame
    !!--------------------------------------------------------------------------
    subroutine free_tracer_data1
        implicit none
        if (allocated(Bx1)) then
            deallocate(Ux1, Uy1, Uz1, dX1, dY1, dZ1, icell1, qtag1)
            ! deallocate(Ex1, Ey1, Ez1, Bx1, By1, Bz1)
            ! deallocate(Vx1, Vy1, Vz1)
        endif
    end subroutine free_tracer_data1

    !!--------------------------------------------------------------------------
    !! Free tracer data for another time frame
    !!--------------------------------------------------------------------------
    subroutine free_tracer_data2
        implicit none
        if (allocated(Bx2)) then
            deallocate(Ux2, Uy2, Uz2, dX2, dY2, dZ2, icell2, qtag2)
            ! deallocate(Ex2, Ey2, Ez2, Bx2, By2, Bz2)
            ! deallocate(Vx2, Vy2, Vz2)
        endif
    end subroutine free_tracer_data2

    !!--------------------------------------------------------------------------
    !! Read tracer data
    !!--------------------------------------------------------------------------
    subroutine read_tracer_data(dset_id, dcount, doffset, dset_dims)
        implicit none
        integer(hid_t), dimension(*), intent(in) :: dset_id
        integer(hsize_t), dimension(rank), intent(in) :: dset_dims, dcount, &
            doffset
        call read_hdf5_parallel_real(dset_id(1), dcount, doffset, &
            dset_dims, Ux)
        call read_hdf5_parallel_real(dset_id(2), dcount, doffset, &
            dset_dims, Uy)
        call read_hdf5_parallel_real(dset_id(3), dcount, doffset, &
            dset_dims, Uz)
        call read_hdf5_parallel_real(dset_id(4), dcount, doffset, &
            dset_dims, dX)
        call read_hdf5_parallel_real(dset_id(5), dcount, doffset, &
            dset_dims, dY)
        call read_hdf5_parallel_real(dset_id(6), dcount, doffset, &
            dset_dims, dZ)
        call read_hdf5_parallel_integer(dset_id(7), dcount, doffset, &
            dset_dims, icell)
        call read_hdf5_parallel_integer(dset_id(8), dcount, doffset, &
            dset_dims, qtag)
        ! call read_hdf5_parallel_real(dset_id(9), dcount, doffset, &
        !     dset_dims, Bx)
        ! call read_hdf5_parallel_real(dset_id(10), dcount, doffset, &
        !     dset_dims, By)
        ! call read_hdf5_parallel_real(dset_id(11), dcount, doffset, &
        !     dset_dims, Bz)
        ! call read_hdf5_parallel_real(dset_id(12), dcount, doffset, &
        !     dset_dims, Ex)
        ! call read_hdf5_parallel_real(dset_id(13), dcount, doffset, &
        !     dset_dims, Ey)
        ! call read_hdf5_parallel_real(dset_id(14), dcount, doffset, &
        !     dset_dims, Ez)
        ! call read_hdf5_parallel_real(dset_id(15), dcount, doffset, &
        !     dset_dims, Vx)
        ! call read_hdf5_parallel_real(dset_id(16), dcount, doffset, &
        !     dset_dims, Vy)
        ! call read_hdf5_parallel_real(dset_id(17), dcount, doffset, &
        !     dset_dims, Vz)
    end subroutine read_tracer_data

    !!--------------------------------------------------------------------------
    !! Read tracer data for another time frame
    !!--------------------------------------------------------------------------
    subroutine read_tracer_data1(dset_id, dcount, doffset, dset_dims)
        implicit none
        integer(hid_t), dimension(*), intent(in) :: dset_id
        integer(hsize_t), dimension(rank), intent(in) :: dset_dims, dcount, &
            doffset
        call read_hdf5_parallel_real(dset_id(1), dcount, doffset, &
            dset_dims, Ux1)
        call read_hdf5_parallel_real(dset_id(2), dcount, doffset, &
            dset_dims, Uy1)
        call read_hdf5_parallel_real(dset_id(3), dcount, doffset, &
            dset_dims, Uz1)
        call read_hdf5_parallel_real(dset_id(4), dcount, doffset, &
            dset_dims, dX1)
        call read_hdf5_parallel_real(dset_id(5), dcount, doffset, &
            dset_dims, dY1)
        call read_hdf5_parallel_real(dset_id(6), dcount, doffset, &
            dset_dims, dZ1)
        call read_hdf5_parallel_integer(dset_id(7), dcount, doffset, &
            dset_dims, icell1)
        call read_hdf5_parallel_integer(dset_id(8), dcount, doffset, &
            dset_dims, qtag1)
        ! call read_hdf5_parallel_real(dset_id(9), dcount, doffset, &
        !     dset_dims, Bx1)
        ! call read_hdf5_parallel_real(dset_id(10), dcount, doffset, &
        !     dset_dims, By1)
        ! call read_hdf5_parallel_real(dset_id(11), dcount, doffset, &
        !     dset_dims, Bz1)
        ! call read_hdf5_parallel_real(dset_id(12), dcount, doffset, &
        !     dset_dims, Ex1)
        ! call read_hdf5_parallel_real(dset_id(13), dcount, doffset, &
        !     dset_dims, Ey1)
        ! call read_hdf5_parallel_real(dset_id(14), dcount, doffset, &
        !     dset_dims, Ez1)
        ! call read_hdf5_parallel_real(dset_id(15), dcount, doffset, &
        !     dset_dims, Vx1)
        ! call read_hdf5_parallel_real(dset_id(16), dcount, doffset, &
        !     dset_dims, Vy1)
        ! call read_hdf5_parallel_real(dset_id(17), dcount, doffset, &
        !     dset_dims, Vz1)
    end subroutine read_tracer_data1

    !!--------------------------------------------------------------------------
    !! Read tracer data for even another time frame
    !!--------------------------------------------------------------------------
    subroutine read_tracer_data2(dset_id, dcount, doffset, dset_dims)
        implicit none
        integer(hid_t), dimension(*), intent(in) :: dset_id
        integer(hsize_t), dimension(rank), intent(in) :: dset_dims, dcount, &
            doffset
        call read_hdf5_parallel_real(dset_id(1), dcount, doffset, &
            dset_dims, Ux2)
        call read_hdf5_parallel_real(dset_id(2), dcount, doffset, &
            dset_dims, Uy2)
        call read_hdf5_parallel_real(dset_id(3), dcount, doffset, &
            dset_dims, Uz2)
        call read_hdf5_parallel_real(dset_id(4), dcount, doffset, &
            dset_dims, dX2)
        call read_hdf5_parallel_real(dset_id(5), dcount, doffset, &
            dset_dims, dY2)
        call read_hdf5_parallel_real(dset_id(6), dcount, doffset, &
            dset_dims, dZ2)
        call read_hdf5_parallel_integer(dset_id(7), dcount, doffset, &
            dset_dims, icell2)
        call read_hdf5_parallel_integer(dset_id(8), dcount, doffset, &
            dset_dims, qtag2)
        ! call read_hdf5_parallel_real(dset_id(9), dcount, doffset, &
        !     dset_dims, Bx2)
        ! call read_hdf5_parallel_real(dset_id(10), dcount, doffset, &
        !     dset_dims, By2)
        ! call read_hdf5_parallel_real(dset_id(11), dcount, doffset, &
        !     dset_dims, Bz2)
        ! call read_hdf5_parallel_real(dset_id(12), dcount, doffset, &
        !     dset_dims, Ex2)
        ! call read_hdf5_parallel_real(dset_id(13), dcount, doffset, &
        !     dset_dims, Ey2)
        ! call read_hdf5_parallel_real(dset_id(14), dcount, doffset, &
        !     dset_dims, Ez2)
        ! call read_hdf5_parallel_real(dset_id(15), dcount, doffset, &
        !     dset_dims, Vx2)
        ! call read_hdf5_parallel_real(dset_id(16), dcount, doffset, &
        !     dset_dims, Vy2)
        ! call read_hdf5_parallel_real(dset_id(17), dcount, doffset, &
        !     dset_dims, Vz2)
    end subroutine read_tracer_data2

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
    ! Finalize reading hdf5 file in parallel
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
            call open_metadata_dset(fname_metadata, groupname, file_id, &
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
        ! call get_nout
        ! call adjust_tindex_start
        ! call set_output_record
        call set_mpi_io
    end subroutine init_analysis

    !!--------------------------------------------------------------------------
    !! End the analysis by free the memory.
    !!--------------------------------------------------------------------------
    subroutine end_analysis
        use mpi_module
        use topology_translate, only: free_start_stop_cells
        use mpi_io_translate, only: datatype
        use mpi_info_module, only: fileinfo
        use neighbors_module, only: free_neighbors
        implicit none
        call free_start_stop_cells
        call MPI_TYPE_FREE(datatype, ierror)
        call MPI_INFO_FREE(fileinfo, ierror)
        call MPI_FINALIZE(ierr)
    end subroutine end_analysis

end program ptl_tracer_hdf5
