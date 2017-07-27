!*******************************************************************************
! This module is to calculate 2D velocity distribution along a field line. We
! trace one field line first starting at one point. The particle 2D velocity
! distributions at each point along the field line is then calculated.
!*******************************************************************************
program vdist_2d_along_fieldline
    use mpi_module
    use constants, only: fp
    use particle_frames, only: nt
    use spectrum_config, only: nbins_vdist
    use particle_fieldline, only: init_analysis, end_analysis, &
            np, get_fieldline_points
    implicit none
    integer :: ct       ! Current time frame
    ! The spectra at these points.
    real(fp), allocatable, dimension(:, :, :) :: vdist_2d  ! Para and perp to B.
    real(fp), allocatable, dimension(:, :, :) :: vdist_xy, vdist_xz, vdist_yz
    real(fp) :: x0, z0
    character(len=256) :: rootpath

    ct = 10
    call get_cmd_args
    call init_analysis(ct, rootpath)
    x0 = 1.0
    z0 = 60.0
    call get_fieldline_points(x0, z0)

    nbins_vdist = 100
    allocate(vdist_2d(2*nbins_vdist, nbins_vdist, np))
    allocate(vdist_xy(2*nbins_vdist, 2*nbins_vdist, np))
    allocate(vdist_xz(2*nbins_vdist, 2*nbins_vdist, np))
    allocate(vdist_yz(2*nbins_vdist, 2*nbins_vdist, np))
    vdist_2d = 0.0
    vdist_xy = 0.0
    vdist_xz = 0.0
    vdist_yz = 0.0

    call calc_vdist_2d_fieldline('e')

    deallocate(vdist_2d, vdist_xy, vdist_xz, vdist_yz)
    call end_analysis

    contains

    !---------------------------------------------------------------------------
    ! Calculate the 2D velocity distributions along a line.
    ! Input:
    !   species: 'e' for electron; 'i' for ion.
    !---------------------------------------------------------------------------
    subroutine calc_vdist_2d_fieldline(species)
        use spectrum_config, only: vmax, vmin, center, sizes
        use spectrum_config, only: set_spatial_range_de, &
                calc_velocity_interval, calc_pic_mpi_ids
        use velocity_distribution, only: fvel_2d, fvel_xy, fvel_xz, fvel_yz, & 
                init_vdist_2d_single, free_vdist_2d_single, &
                calc_vdist_2d_single, init_velocity_bins, free_velocity_bins, &
                set_vdist_2d_zero_single
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
        call init_vdist_2d_single
        call init_velocity_bins

        sizes = [5.0, 1.0, 5.0]
        do i = startp, endp
            center = [xarr(i), 0.0, zarr(i)]
            call set_spatial_range_de
            call calc_pic_mpi_ids
            call calc_vdist_2d_single(ct*tinterval/ratio_interval, species)
            vdist_2d(:, :, i-startp+1) = fvel_2d
            vdist_xy(:, :, i-startp+1) = fvel_xy
            vdist_xz(:, :, i-startp+1) = fvel_xz
            vdist_yz(:, :, i-startp+1) = fvel_yz
            call set_vdist_2d_zero_single
        end do

        if (myid == master) then
            call check_folder_exist
        endif
        call write_vdist_2d(species)
        call free_velocity_bins
        call free_vdist_2d_single
    end subroutine calc_vdist_2d_fieldline

    !---------------------------------------------------------------------------
    ! Check if the folder for the data exist. If not, make one.
    !---------------------------------------------------------------------------
    subroutine check_folder_exist
        implicit none
        logical :: dir_e
        dir_e = .false.
        inquire(file='./data_vdist_2d/.', exist=dir_e)
        if (.not. dir_e) then
            call system('mkdir ./data_vdist_2d')
        endif
    end subroutine check_folder_exist

    !---------------------------------------------------------------------------
    ! Write the spectra data to file.
    !---------------------------------------------------------------------------
    subroutine write_vdist_2d(species)
        use mpi_module
        use mpi_io_module, only: open_data_mpi_io, write_data_mpi_io
        use mpi_datatype_module, only: set_mpi_datatype
        use mpi_info_module, only: fileinfo
        use velocity_distribution, only: vbins_short, vbins_long
        use particle_fieldline, only: nptot, np, startp
        use spectrum_config, only: nbins_vdist
        implicit none
        character(len=1), intent(in) :: species
        integer, dimension(3) :: sizes_short, sizes_long
        integer, dimension(3) :: subsizes_short, subsizes_long
        integer, dimension(3) :: starts
        integer(kind=MPI_OFFSET_KIND) :: disp, offset
        character(len=150) :: fname
        integer :: datatype_short, datatype_long, fh
        integer :: pos1, nbins

        nbins = nbins_vdist
        sizes_short(1) = 2 * nbins
        sizes_short(2) = nbins
        sizes_short(3) = nptot
        sizes_long(1) = 2 * nbins
        sizes_long(2) = 2 * nbins
        sizes_long(3) = nptot
        subsizes_short(1) = 2 * nbins
        subsizes_short(2) = nbins
        subsizes_short(3) = np
        subsizes_long(1) = 2 * nbins
        subsizes_long(2) = 2 * nbins
        subsizes_long(3) = np
        starts(1) = 0
        starts(2) = 0
        starts(3) = startp

        datatype_short = set_mpi_datatype(sizes_short, subsizes_short, starts)
        datatype_long = set_mpi_datatype(sizes_long, subsizes_long, starts)

        fname = './data_vdist_2d/vdist_2d_fieldline_'//species//'.gda'

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

        ! Save 2D velocity distribution para and perp to local field.
        disp = pos1 + 3*sizeof(fp)*nbins - 1
        offset = 0 
        call write_data_mpi_io(fh, datatype_short, &
                subsizes_short, disp, offset, vdist_2d)

        ! Save 2D velocity distributions in xy, xz, yz plane.
        disp = disp + sizeof(fp)*nbins*nbins*nptot*2
        call write_data_mpi_io(fh, datatype_long, &
                subsizes_long, disp, offset, vdist_xy)

        disp = disp + sizeof(fp)*nbins*nbins*nptot*4
        call write_data_mpi_io(fh, datatype_long, &
                subsizes_long, disp, offset, vdist_xz)

        disp = disp + sizeof(fp)*nbins*nbins*nptot*4
        call write_data_mpi_io(fh, datatype_long, &
                subsizes_long, disp, offset, vdist_yz)

        call MPI_FILE_CLOSE(fh, ierror)
        call MPI_TYPE_FREE(datatype_short, ierror)
        call MPI_TYPE_FREE(datatype_long, ierror)
    end subroutine write_vdist_2d

    !<--------------------------------------------------------------------------
    !< Get commandline arguments
    !<--------------------------------------------------------------------------
    subroutine get_cmd_args
        use flap                                !< FLAP package
        use penf
        implicit none
        type(command_line_interface) :: cli     !< Command Line Interface (CLI).
        integer(I4P)                 :: error   !< Error trapping flag.
        call cli%init(progname = 'translate', &
            authors     = 'Xiaocan Li', &
            help        = 'Usage: ', &
            description = 'Merge VPIC simulation output from all MPI processes', &
            examples    = ['translate -rp simulation_root_path'])
        call cli%add(switch='--rootpath', switch_ab='-rp', &
            help='simulation root path', required=.true., act='store', error=error)
        if (error/=0) stop
        call cli%get(switch='-rp', val=rootpath, error=error)
        if (error/=0) stop

        if (myid == 0) then
            print '(A,A)', 'The simulation rootpath: ', trim(adjustl(rootpath))
        endif
    end subroutine get_cmd_args
end program vdist_2d_along_fieldline
