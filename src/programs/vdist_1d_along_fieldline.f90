!*******************************************************************************
! This module is to calculate 1D velocity distribution along a field line. We
! trace one field line first starting at one point. The particle 1D velocity
! distributions at each point along the field line is then calculated.
!*******************************************************************************
program vdist_1d_along_fieldline
    use mpi_module
    use constants, only: fp
    use particle_frames, only: nt
    use spectrum_config, only: nbins_vdist
    use particle_fieldline, only: init_analysis, end_analysis, &
            np, get_fieldline_points
    implicit none
    integer :: ct       ! Current time frame
    ! The spectra at these points.
    real(fp), allocatable, dimension(:, :) :: vdist_para, vdist_perp
    real(fp) :: x0, z0

    ct = 10
    call init_analysis(ct)
    x0 = 1.0
    z0 = 60.0
    call get_fieldline_points(x0, z0)

    nbins_vdist = 100
    allocate(vdist_para(2*nbins_vdist, np))
    allocate(vdist_perp(nbins_vdist, np))
    vdist_para = 0.0
    vdist_perp = 0.0

    call calc_vdist_1d_fieldline('e')

    deallocate(vdist_para, vdist_perp)
    call end_analysis

    contains

    !---------------------------------------------------------------------------
    ! Calculate the velocity distributions along a line.
    ! Input:
    !   species: 'e' for electron; 'i' for ion.
    !---------------------------------------------------------------------------
    subroutine calc_vdist_1d_fieldline(species)
        use spectrum_config, only: vmax, vmin, center, sizes
        use spectrum_config, only: set_spatial_range_de, &
                calc_velocity_interval, calc_pic_mpi_ids
        use velocity_distribution, only: fvel_para, fvel_perp, &
                init_vdist_1d_single, free_vdist_1d_single, &
                calc_vdist_1d_single, init_velocity_bins, free_velocity_bins, &
                set_vdist_1d_zero_single
        use fieldline_tracing, only: xarr, zarr
        use particle_frames, only: tinterval
        use particle_file, only: ratio_interval
        use particle_fieldline, only: startp, endp
        implicit none
        character(len=1), intent(in) :: species
        integer :: i

        vmax = 2.0
        vmin = 0.0
        call calc_velocity_interval
        call init_vdist_1d_single
        call init_velocity_bins

        sizes = [5.0, 1.0, 5.0]
        do i = startp, endp
            center = [xarr(i), 0.0, zarr(i)]
            call set_spatial_range_de
            call calc_pic_mpi_ids
            call calc_vdist_1d_single(ct*tinterval/ratio_interval, species)
            vdist_para(:, i-startp+1) = fvel_para
            vdist_perp(:, i-startp+1) = fvel_perp
            call set_vdist_1d_zero_single
        end do

        if (myid == master) then
            call check_folder_exist
        endif
        call write_vdist_1d(species)
        call free_velocity_bins
        call free_vdist_1d_single
    end subroutine calc_vdist_1d_fieldline

    !---------------------------------------------------------------------------
    ! Check if the folder for the data exist. If not, make one.
    !---------------------------------------------------------------------------
    subroutine check_folder_exist
        implicit none
        logical :: dir_e
        dir_e = .false.
        inquire(file='./data_vdist_1d/.', exist=dir_e)
        if (.not. dir_e) then
            call system('mkdir ./data_vdist_1d')
        endif
    end subroutine check_folder_exist

    !---------------------------------------------------------------------------
    ! Write the spectra data to file.
    !---------------------------------------------------------------------------
    subroutine write_vdist_1d(species)
        use mpi_module
        use mpi_io_module, only: open_data_mpi_io, write_data_mpi_io
        use mpi_datatype_module, only: set_mpi_datatype
        use mpi_info_module, only: fileinfo
        use velocity_distribution, only: vbins_short, vbins_long
        use particle_fieldline, only: nptot, np, startp
        use spectrum_config, only: nbins_vdist
        implicit none
        character(len=1), intent(in) :: species
        integer, dimension(2) :: sizes_short, sizes_long
        integer, dimension(2) :: subsizes_short, subsizes_long
        integer, dimension(2) :: starts
        integer(kind=MPI_OFFSET_KIND) :: disp, offset
        character(len=150) :: fname
        integer :: datatype_short, datatype_long, fh
        integer :: pos1, nbins

        nbins = nbins_vdist
        sizes_short(1) = nbins
        sizes_short(2) = nptot
        sizes_long(1) = 2 * nbins
        sizes_long(2) = nptot
        subsizes_short(1) = nbins
        subsizes_short(2) = np
        subsizes_long(1) = 2 * nbins
        subsizes_long(2) = np
        starts(1) = 0
        starts(2) = startp

        datatype_short = set_mpi_datatype(sizes_short, subsizes_short, starts)
        datatype_long = set_mpi_datatype(sizes_long, subsizes_long, starts)

        fname = './data_vdist_1d/vdist_1d_fieldline_'//species//'.gda'

        ! Save nbins
        if (myid == master) then
            open(unit=41, file=fname, access='stream',&
                status='unknown', form='unformatted', action='write')     
            pos1 = 1
            write(41, pos=pos1) nbins, nptot
            pos1 = 2*sizeof(fp) + pos1
            write(41, pos=pos1) vbins_short, vbins_long
            close(41)
        endif
        call MPI_BCAST(pos1, 1, MPI_INTEGER, master, MPI_COMM_WORLD, ierr)

        call open_data_mpi_io(fname, MPI_MODE_WRONLY, fileinfo, fh)

        ! Save 1D velocity distribution parallel to local field.
        disp = pos1 + 3*sizeof(fp)*nbins - 1
        offset = 0 
        call write_data_mpi_io(fh, datatype_long, &
                subsizes_long, disp, offset, vdist_para)

        ! Save 1D velocity distribution perpendicular to local field.
        disp = disp + sizeof(fp)*nbins*nptot*2
        call write_data_mpi_io(fh, datatype_short, &
                subsizes_short, disp, offset, vdist_perp)

        call MPI_FILE_CLOSE(fh, ierror)
        call MPI_TYPE_FREE(datatype_short, ierror)
        call MPI_TYPE_FREE(datatype_long, ierror)
    end subroutine write_vdist_1d

end program vdist_1d_along_fieldline
