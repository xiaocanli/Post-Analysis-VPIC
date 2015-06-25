!*******************************************************************************
! This program is to calculate parallel potential due to parallel electric
! field.
! Reference:
!   Egedal, J., et al. "Evidence and theory for trapped electrons in guide field
!   magnetotail reconnection." Journal of Geophysical Research: Space Physics
!   (1978–2012) 113.A12 (2008).
!*******************************************************************************
program parallel_potential
    use mpi_module
    use constants, only: fp
    use fieldline_tracing, only: init_fieldline_tracing, end_fieldline_tracing, &
            Cash_Karp_parameters, Dormand_Prince_parameters
    use analysis_management, only: init_analysis, end_analysis
    use magnetic_field, only: read_magnetic_fields
    use electric_field, only: init_electric_fields, read_electric_fields, &
            free_electric_fields
    implicit none
    real(fp), allocatable, dimension(:,:) :: phi_para
    integer :: nx_local, nx_offset, nx, nz
    integer :: ct

    call init_analysis
    call init_calculation
    call init_fieldline_tracing
    call init_electric_fields
    !call Cash_Karp_parameters
    call Dormand_Prince_parameters

    do ct = 1, 100
        if (myid == master) then
            print*, ct
        endif
        call read_electric_fields(ct)
        call read_magnetic_fields(ct)
        call calc_phi_parallel(ct)
    enddo

    call free_electric_fields
    call end_fieldline_tracing
    call end_calculation
    call end_analysis

    contains

    !---------------------------------------------------------------------------
    ! Initialize the calculation by setting the domain information and
    ! initializing parallel potential.
    !---------------------------------------------------------------------------
    subroutine init_calculation
        use mpi_module
        use picinfo, only: domain
        implicit none

        nx = domain%nx
        nz = domain%nz
        ! The whole domain is equally divided along the x-direction.
        nx_local = nx / numprocs
        nx_offset = nx_local * myid

        allocate(phi_para(nx_local, nz))
        phi_para = 0.0
    end subroutine init_calculation

    !---------------------------------------------------------------------------
    ! Finish the calculation.
    !---------------------------------------------------------------------------
    subroutine end_calculation
        implicit none
        deallocate(phi_para)
    end subroutine end_calculation

    !---------------------------------------------------------------------------
    ! Calculate the parallel potential, which often appears in Jan Egedal's papers.
    ! phi_para is calculated for any point along the magnetic field fields to the
    ! simulation boundary. phi_para = \int_\vec{x}^\infty \vec{E}\cdot d\vec{s}.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine calc_phi_parallel(ct)
        use mpi_module
        use constants, only: fp
        use fieldline_tracing, only: gdx, gdz
        implicit none
        integer, intent(in) :: ct
        integer :: i, k
        real(fp) :: x, z, h, hmax

        hmax = gdx
        phi_para = 0.0
        do k = 1, nz
            !print*, k
            do i = 1, nx_local
                h = hmax
                x = (nx_offset+i-1)*gdx
                z = (k-1)*gdz
                call tracing(x, z, h, phi_para(i, k))
            enddo ! x loop
        enddo ! z loop

        !print*, "__________________", myid

        call save_phi_parallel(ct)

    end subroutine calc_phi_parallel

    !---------------------------------------------------------------------------
    ! Trace the magnetic field starting at one point and accumulate to get
    ! the parallel potential at the same time.
    ! Inputs:
    !   htry: the first-try step size.
    ! Inputs & Outputs:
    !   x, z: the coordinates of current point.
    !   phi_parallel: parallel potential at current point.
    !---------------------------------------------------------------------------
    subroutine tracing(x, z, htry, phi_parallel)
        use fieldline_tracing, only: push, derivs, get_crossing_point, &
                controller, Cash_Karp_parameters, Dormand_Prince_parameters, lx, lz
        implicit none
        real(fp), intent(inout) :: x, z, phi_parallel
        real(fp), intent(in) :: htry
        real(fp), dimension(0:6) :: kx, ky, kz
        real(fp) :: arc_length, xout, zout, xold, zold
        real(fp) :: dxds, dyds, dzds, dxdsnew, dydsnew, dzdsnew
        real(fp) :: xcross, zcross
        real(fp) :: h, hnext, errold, h_old
        logical :: is_accept

        phi_parallel = 0.0
        arc_length = 0.0
        h = htry
        h_old = h
        errold = 1.0e-4
        is_accept = .false.
        call derivs(x, z, dxds, dyds, dzds)
        do while (x > 0 .and. x < lx .and. z > 0 .and. z < lz .and. &
                  arc_length < 2*lx)
            call push(dxds, dyds, dzds, x, z, h, kx, ky, kz, &
                      xout, zout, dxdsnew, dydsnew, dzdsnew)
            call controller(h, hnext, x, z, xout, zout, kx, kz, is_accept, errold)
            if (is_accept) then
                arc_length = arc_length + h
                xold = x
                zold = z
                x = xout
                z = zout
                dxds = dxdsnew
                dyds = dydsnew
                dzds = dzdsnew
                call update_phi_parallel(xold, zold, kx, ky, kz, h, phi_parallel)
                h_old = h
                h = hnext
            endif
        enddo ! while loop

        ! Make sure it integrates to the boundary.
        if (x < 0 .or. x > lx .or. z < 0 .or. z > lz) then
            ! This excludes the closed field inside the domain.
            arc_length = arc_length - h_old 
            xcross = 0.0
            zcross = 0.0
            call get_crossing_point(x, z, xold, zold, lx, lz, xcross, zcross)
            h = sqrt((xcross-xold)**2 + (zcross-zold)**2)
            arc_length = arc_length + h
            call push(dxds, dyds, dzds, x, z, h, kx, ky, kz, &
                      xout, zout, dxdsnew, dydsnew, dzdsnew)
            call update_phi_parallel(xold, zold, kx, ky, kz, h, phi_parallel)
        endif

        ! Exclude the closed field lines
        if (arc_length > (2*lx-10.0)) then
            phi_parallel = 0.0
        endif
    end subroutine tracing

    !---------------------------------------------------------------------------
    ! Update parallel potential.
    ! Input:
    !   xold, zold: the starting point of current step.
    !   kx, ky, kz: the derivatives (actually the direction of the field)
    !   h: the step size. 
    ! Input and Output:
    !   phi_parallel: current parallel potential to be updated.
    !---------------------------------------------------------------------------
    subroutine update_phi_parallel(xold, zold, kx, ky, kz, h, phi_parallel)
        use constants, only: fp
        use fieldline_tracing, only: b, c, gdx, gdz
        use electric_field, only: get_electric_field_at_point, ex0, ey0, ez0
        implicit none
        real(fp), intent(in), dimension(0:6) :: kx, ky, kz
        real(fp), intent(in) :: xold, zold, h
        real(fp), intent(inout) :: phi_parallel
        real(fp), dimension(0:6) :: exs, eys, ezs
        real(fp) :: xtemp, ztemp
        integer :: i

        call get_electric_field_at_point(xold, zold, gdx, gdz)
        exs(0) = ex0
        eys(0) = ey0
        ezs(0) = ez0

        do i = 1, 5
            xtemp = xold + h*dot_product(kx(0:i-1), b(0:i-1,i))
            ztemp = zold + h*dot_product(kz(0:i-1), b(0:i-1,i))
            call get_electric_field_at_point(xtemp, ztemp, gdx, gdz)
            exs(i) = ex0
            eys(i) = ey0
            ezs(i) = ez0
        end do
        xtemp = xold + h*dot_product(kx(2:5), b(2:5,6)) + h*kx(0)*b(0,6)
        ztemp = zold + h*dot_product(kz(2:5), b(2:5,6)) + h*kz(0)*b(0,6)
        call get_electric_field_at_point(xtemp, ztemp, gdx, gdz)
        exs(6) = ex0
        eys(6) = ey0
        ezs(6) = ez0

        phi_parallel = phi_parallel + &
            h * (sum(kx(2:6)*c(2:6)*exs(2:6)) + kx(0)*c(0)*exs(0) + &
                 sum(ky(2:6)*c(2:6)*eys(2:6)) + ky(0)*c(0)*eys(0) + &
                 sum(kz(2:6)*c(2:6)*ezs(2:6)) + kz(0)*c(0)*ezs(0)) 
    end subroutine update_phi_parallel

    !---------------------------------------------------------------------------
    ! Save the calculate parallel potential to a file.
    ! Inputs:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_phi_parallel(ct)
        use mpi_module
        use parameters, only: tp1
        use mpi_io_module, only: open_data_mpi_io
        use mpi_info_module, only: fileinfo
        use path_info, only: rootpath
        implicit none
        integer, intent(in) :: ct
        integer, dimension(2) :: sizes, subsizes, starts
        integer :: filetype, fh
        integer(kind=MPI_OFFSET_KIND) :: disp, offset
        logical :: dir_e

        sizes(1) = nx
        sizes(2) = nz
        subsizes(1) = nx_local
        subsizes(2) = nz
        starts(1) = nx_offset
        starts(2) = 0

        call MPI_TYPE_CREATE_SUBARRAY(2, sizes, subsizes, starts, &
            MPI_ORDER_FORTRAN, MPI_REAL, filetype, ierror)
        call MPI_TYPE_COMMIT(filetype, ierror)

        if (myid == master) then
            dir_e = .false.
            inquire(file='../data1/.', exist=dir_e)
            if (.not. dir_e) then
                call system('mkdir ../data1')
            endif
        endif

        call open_data_mpi_io('../data1/phi_para.gda', &
                              MPI_MODE_RDWR+MPI_MODE_CREATE, fileinfo, fh)

        disp = nx * nz * sizeof(MPI_REAL) * (ct-tp1)
        offset = 0 
        call MPI_FILE_SET_VIEW(fh, disp, MPI_REAL, filetype, 'native', &
            MPI_INFO_NULL, ierror)
        if (ierror /= 0) then
            call MPI_ERROR_STRING(ierror, err_msg, err_length, ierror2)
            print*, "Error in MPI_FILE_SET_VIEW: ", trim(err_msg)
        endif

        call MPI_FILE_WRITE_AT_ALL(fh, offset, phi_para, &
            subsizes(1)*subsizes(2), MPI_REAL, status, ierror)
        if (ierror /= 0) then
            call MPI_ERROR_STRING(ierror, err_msg, err_length, ierror2)
            print*, "Error in MPI_FILE_READ: ", trim(err_msg)
        endif

        call MPI_FILE_CLOSE(fh, ierror)
        call MPI_TYPE_FREE(filetype, ierror)
    end subroutine save_phi_parallel

end program parallel_potential
