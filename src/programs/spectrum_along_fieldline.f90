!*******************************************************************************
! This module is to calculate particle energy spectrum along a field line.
! We trace one field line first starting at one point. The particle energy
! spectrum at each point along the field line is then calculated.
!*******************************************************************************
program spectrum_along_fieldline
    use mpi_module
    use constants, only: fp
    use particle_frames, only: nt
    use spectrum_config, only: nbins 
    use particle_fieldline, only: init_analysis, end_analysis, &
            np, get_fieldline_points
    implicit none
    integer :: ct       ! Current time frame
    ! The spectra at these points.
    real(fp), allocatable, dimension(:, :) :: flog_np, flin_np
    real(fp) :: x0, z0

    ct = 10
    call init_analysis(ct)
    x0 = 1.0
    z0 = 60.0
    call get_fieldline_points(x0, z0)

    nbins = 100
    allocate(flin_np(nbins, np))
    allocate(flog_np(nbins, np))
    flin_np = 0.0
    flog_np = 0.0
    call calc_particle_energy_spectrum('e')

    deallocate(flin_np, flog_np)
    call end_analysis

    contains

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
        use fieldline_tracing, only: xarr, zarr
        use particle_frames, only: tinterval
        use particle_file, only: ratio_interval
        use particle_fieldline, only: startp, endp
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
        use mpi_info_module, only: fileinfo
        use particle_energy_spectrum, only: ebins_lin, ebins_log
        use particle_fieldline, only: nptot, np, startp
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
