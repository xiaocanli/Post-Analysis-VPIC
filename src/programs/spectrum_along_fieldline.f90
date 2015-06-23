!*******************************************************************************
! This module is to calculate particle energy spectrum along a field line.
! We trace one field line first starting at one point. The particle energy
! spectrum at each point along the field line is then calculated.
!*******************************************************************************
program spectrum_along_fieldline
    use mpi_module
    use constants, only: fp
    use path_info, only: get_file_paths
    use picinfo, only: read_domain, broadcast_pic_info
    use fieldline_tracing, only: init_fieldline_tracing, end_fieldline_tracing, &
            Cash_Karp_parameters, Dormand_Prince_parameters
    use magnetic_field, only: read_magnetic_fields
    use particle_frames, only: get_particle_frames, nt, tinterval
    use mpi_topology, only: distribute_tasks
    use spectrum_config, only: nbins 
    use particle_file, only: check_both_particle_fields_exist, &
            get_ratio_interval, ratio_interval
    implicit none
    real(fp), allocatable, dimension(:,:) :: ptl_spectrum
    integer :: ct       ! Current time frame
    real(fp), allocatable, dimension(:) :: xarr, zarr
    integer, parameter :: np_max = 1E3
    integer :: nptot    ! The actual number of points along the field line.
    integer :: np       ! Number of points for current MPI process.
    integer :: startp, endp         ! Starting and ending points.
    ! The spectra at these points.
    real(fp), allocatable, dimension(:, :) :: flog_np, flin_np
    logical :: is_time_valid

    call MPI_INIT(ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)

    call get_file_paths
    if (myid==master) then
        call read_domain
        call get_particle_frames
    endif
    call MPI_BCAST(tinterval, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
    call broadcast_pic_info

    ct = 10
    if (myid == master) then
        call get_ratio_interval
    endif
    call MPI_BCAST(ratio_interval, 1, MPI_INTEGER, master, MPI_COMM_WORLD, ierr)

    is_time_valid = check_both_particle_fields_exist(ct)

    if (is_time_valid) then
        call init_fieldline_tracing
        call Dormand_Prince_parameters
        call read_magnetic_fields(ct)

        allocate(xarr(np_max))
        allocate(zarr(np_max))
        xarr = 0.0
        zarr = 0.0
        if (myid == master) then
            call trace_field_line(1.0, 60.0, nptot)
        endif
        call end_fieldline_tracing

        call MPI_BCAST(nptot, 1, MPI_INTEGER, master, MPI_COMM_WORLD, ierr)
        call MPI_BCAST(xarr(1:nptot), nptot, MPI_REAL, master, MPI_COMM_WORLD, ierr)
        call MPI_BCAST(zarr(1:nptot), nptot, MPI_REAL, master, MPI_COMM_WORLD, ierr)
        call distribute_tasks(nptot, numprocs, myid, np, startp, endp)

        nbins = 100
        allocate(flin_np(nbins, np))
        allocate(flog_np(nbins, np))
        flin_np = 0.0
        flog_np = 0.0
        call calc_particle_energy_spectrum('e')

        deallocate(xarr, zarr)
        deallocate(flin_np, flog_np)
    else
        if (myid == master) then
            write(*, '(A,I0,A)') 'ct = ', ct, ' is invalid.'
            write(*, '(A,I0)') 'Choose a time that is a multiple of ', ratio_interval
        endif
        call MPI_FINALIZE(ierr)
        stop
    endif
    call MPI_FINALIZE(ierr)

    contains

    !---------------------------------------------------------------------------
    ! Trace magnetic field line starting at one point. Record the total number
    ! of points along the field line.
    ! Inputs:
    !   x0, z0: the coordinates of the starting point.
    ! Output:
    !   npoints: total number of points along the field line.
    !---------------------------------------------------------------------------
    subroutine trace_field_line(x0, z0, npoints)
        use constants, only: fp
        use picinfo, only: domain, mime
        use fieldline_tracing, only: tracing, gdx
        implicit none
        real(fp), intent(in) :: x0, z0
        integer, intent(out) :: npoints
        real(fp) :: htry, x, z
        integer :: i
        htry = gdx
        x = x0
        z = z0
        npoints = 0
        call tracing(x, z, htry, npoints, xarr, zarr)
        xarr = xarr * sqrt(mime)    ! di -> de
        zarr = zarr * sqrt(mime)
        open(unit=20, file='xz_field.dat', status='unknown')
        do i = 1, npoints
            write(20, '(2e12.4)') xarr(i), zarr(i)
        enddo
        close(20)
    end subroutine trace_field_line

    !---------------------------------------------------------------------------
    ! Calculate the particle energy spectrum along a line.
    ! Input:
    !   species: 'e' for electron; 'i' for ion.
    !---------------------------------------------------------------------------
    subroutine calc_particle_energy_spectrum(species)
        use spectrum_config, only: emax, emin, center, sizes
        use spectrum_config, only: read_config, set_spatial_range_de, &
                calc_energy_interval, calc_pic_mpi_ids
        use particle_energy_spectrum, only: init_energy_spectra_single, &
                free_energy_spectra_single, calc_energy_spectrum_single, &
                calc_energy_bins, set_energy_spectra_zero_single, f, flog
        implicit none
        character(len=1), intent(in) :: species
        integer :: i

        emax = 1.0E2
        emin = 1.0E-4
        call calc_energy_interval
        call init_energy_spectra_single
        call calc_energy_bins

        sizes = [5.0, 1.0, 5.0]
        do i = startp, endp
            center = [xarr(i), 0.0, zarr(i)]
            call set_spatial_range_de
            call calc_pic_mpi_ids
            call calc_energy_spectrum_single(ct*tinterval/ratio_interval, species)
            flin_np(:, i-startp+1) = f
            flog_np(:, i-startp+1) = flog
            call set_energy_spectra_zero_single
        end do
        call write_particle_spectrum(species)
        call free_energy_spectra_single
    end subroutine calc_particle_energy_spectrum

    !---------------------------------------------------------------------------
    ! Write the spectra data to file.
    !---------------------------------------------------------------------------
    subroutine write_particle_spectrum(species)
        use mpi_module
        use mpi_io_module, only: open_data_mpi_io
        use mpi_info_object, only: fileinfo
        use particle_energy_spectrum, only: ebins_lin, ebins_log
        implicit none
        character(len=1), intent(in) :: species
        integer, dimension(2) :: sizes, subsizes, starts
        integer(kind=MPI_OFFSET_KIND) :: disp, offset
        character(len=150) :: fname
        integer :: filetype, fh
        integer :: pos1
        logical :: dir_e

        sizes(1) = nbins
        sizes(2) = nptot
        subsizes(1) = nbins
        subsizes(2) = np
        starts(1) = 0
        starts(2) = startp

        call MPI_TYPE_CREATE_SUBARRAY(2, sizes, subsizes, starts, &
            MPI_ORDER_FORTRAN, MPI_REAL, filetype, ierror)
        call MPI_TYPE_COMMIT(filetype, ierror)

        if (myid == master) then
            dir_e = .false.
            inquire(file='./data_double_layer/.', exist=dir_e)
            if (.not. dir_e) then
                call system('mkdir ./data_double_layer')
            endif
        endif

        fname = './data_double_layer/spect_fieldline_'//species//'.gda'

        ! Save nbins, npoints, ebins_lin, ebins_log
        if (myid == master) then
            open(unit=41, file=fname, access='stream',&
                status='unknown', form='unformatted', action='write')     
            pos1 = 1
            write(41, pos=pos1) nbins, nptot
            pos1 = 2*sizeof(fp) + pos1
            write(41, pos=pos1) ebins_lin, ebins_log
            close(41)
        endif
        call MPI_BCAST(pos1, 1, MPI_INTEGER, master, MPI_COMM_WORLD, ierr)

        call open_data_mpi_io(fname, MPI_MODE_WRONLY, fileinfo, fh)

        ! Save spectrum with linear bins
        disp = pos1 + 2*sizeof(fp)*nbins - 1
        offset = 0 
        call MPI_FILE_SET_VIEW(fh, disp, MPI_REAL, filetype, 'native', &
            MPI_INFO_NULL, ierror)
        if (ierror /= 0) then
            call MPI_ERROR_STRING(ierror, err_msg, err_length, ierror2)
            print*, "Error in MPI_FILE_SET_VIEW: ", trim(err_msg)
        endif

        call MPI_FILE_WRITE_AT_ALL(fh, offset, flin_np, &
            subsizes(1)*subsizes(2), MPI_REAL, status, ierror)
        if (ierror /= 0) then
            call MPI_ERROR_STRING(ierror, err_msg, err_length, ierror2)
            print*, "Error in MPI_FILE_READ: ", trim(err_msg)
        endif

        ! Save spectrum with logarithmic bins.
        disp = disp + sizeof(fp)*nbins*nptot
        offset = 0 
        call MPI_FILE_SET_VIEW(fh, disp, MPI_REAL, filetype, 'native', &
            MPI_INFO_NULL, ierror)
        if (ierror /= 0) then
            call MPI_ERROR_STRING(ierror, err_msg, err_length, ierror2)
            print*, "Error in MPI_FILE_SET_VIEW: ", trim(err_msg)
        endif

        call MPI_FILE_WRITE_AT_ALL(fh, offset, flog_np, &
            subsizes(1)*subsizes(2), MPI_REAL, status, ierror)
        if (ierror /= 0) then
            call MPI_ERROR_STRING(ierror, err_msg, err_length, ierror2)
            print*, "Error in MPI_FILE_READ: ", trim(err_msg)
        endif

        call MPI_FILE_CLOSE(fh, ierror)
        call MPI_TYPE_FREE(filetype, ierror)
    end subroutine write_particle_spectrum

end program spectrum_along_fieldline
