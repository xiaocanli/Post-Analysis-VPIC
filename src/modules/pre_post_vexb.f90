!<******************************************************************************
!< Module for smoothed ExB drift velocity of previous and post time frames.
!<******************************************************************************
module pre_post_vexb
    use constants, only: fp
    use mpi_module
    use path_info, only: outputpath
    use mpi_info_module, only: fileinfo
    use mpi_io_module, only: open_data_mpi_io, read_data_mpi_io
    implicit none
    private
    public vexbx1, vexby1, vexbz1, vexbx2, vexby2, vexbz2

    public init_pre_post_vexb, free_pre_post_vexb, open_vexb_pre_post, &
           close_vexb_pre_post, read_pre_post_vexb, shift_vexb_pre_post, &
           calc_vexb_pre_post

    real(fp), allocatable, dimension(:, :, :) :: vexbx1, vexby1, vexbz1
    real(fp), allocatable, dimension(:, :, :) :: vexbx2, vexby2, vexbz2
    integer, dimension(3) :: vexb_pre_fh, vexb_post_fh

    interface open_vexb_pre_post
        module procedure &
            open_vexb_pre_post_single, open_vexb_pre_post_multi
    end interface open_vexb_pre_post

    contains

    !<--------------------------------------------------------------------------
    !< Initialize ExB drift of the previous time frame and post time frame.
    !< They are going be be used to calculate polarization drift current.
    !<--------------------------------------------------------------------------
    subroutine init_pre_post_vexb(nx, ny, nz)
        use mpi_topology, only: htg
        implicit none
        integer, intent(in) :: nx, ny, nz

        allocate(vexbx1(nx,ny,nz))
        allocate(vexby1(nx,ny,nz))
        allocate(vexbz1(nx,ny,nz))
        allocate(vexbx2(nx,ny,nz))
        allocate(vexby2(nx,ny,nz))
        allocate(vexbz2(nx,ny,nz))

        vexbx1 = 0.0; vexby1 = 0.0; vexbz1 = 0.0
        vexbx2 = 0.0; vexby2 = 0.0; vexbz2 = 0.0
    end subroutine init_pre_post_vexb

    !<--------------------------------------------------------------------------
    !< Free the ExB drift of the previous time frame and post time frame.
    !<--------------------------------------------------------------------------
    subroutine free_pre_post_vexb
        implicit none
        deallocate(vexbx1, vexby1, vexbz1, vexbx2, vexby2, vexbz2)
    end subroutine free_pre_post_vexb

    !<--------------------------------------------------------------------------
    !< Calculate ExB drift velocity at previous and next time step
    !<--------------------------------------------------------------------------
    subroutine calc_vexb_pre_post
        use pre_post_emf, only: ex1, ey1, ez1, bx1, by1, bz1, absB1, &
            ex2, ey2, ez2, bx2, by2, bz2, absB2
        implicit none
        vexbx1 = (ey1 * bz1 - ez1 * by1) / absB1**2
        vexby1 = (ez1 * bx1 - ex1 * bz1) / absB1**2
        vexbz1 = (ex1 * by1 - ey1 * bx1) / absB1**2
        vexbx2 = (ey2 * bz2 - ez2 * by2) / absB2**2
        vexby2 = (ez2 * bx2 - ex2 * bz2) / absB2**2
        vexbz2 = (ex2 * by2 - ey2 * bx2) / absB2**2
    end subroutine calc_vexb_pre_post

    !<--------------------------------------------------------------------------
    !< Open v field files if all time frames are saved in the same file
    !< Input:
    !<   separated_pre_post: 1 for separated pre and post files, 0 for not
    !<--------------------------------------------------------------------------
    subroutine open_vexb_pre_post_single(separated_pre_post)
        use exb_drift, only: vexb_fh
        implicit none
        integer, intent(in) :: separated_pre_post
        character(len=256) :: fname
        character(len=16) :: cfname

        if (separated_pre_post == 0) then
            ! Finish latter when needed
        else
            vexb_pre_fh = 0
            vexb_post_fh = 0
            fname = trim(adjustl(outputpath))//'vexb_x_pre.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, &
                vexb_pre_fh(1))
            fname = trim(adjustl(outputpath))//'vexb_y_pre.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, &
                vexb_pre_fh(2))
            fname = trim(adjustl(outputpath))//'vexb_z_pre.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, &
                vexb_pre_fh(3))
            fname = trim(adjustl(outputpath))//'vexb_x_post.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, &
                vexb_post_fh(1))
            fname = trim(adjustl(outputpath))//'vexb_y_post.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, &
                vexb_post_fh(2))
            fname = trim(adjustl(outputpath))//'vexb_z_post.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, &
                vexb_post_fh(3))
        endif
    end subroutine open_vexb_pre_post_single

    !<--------------------------------------------------------------------------
    !< Open v field files if different time frames are saved in different files
    !< Input:
    !<   tindex: time index for current time step
    !<   tindex_pre: time index for previous time step
    !<   tindex_post: time index for next time step
    !<   separated_pre_post: 1 for separated pre and post files, 0 for not
    !<--------------------------------------------------------------------------
    subroutine open_vexb_pre_post_multi(separated_pre_post, tindex, &
            tindex_pre, tindex_post)
        use exb_drift, only: vexb_fh
        implicit none
        integer, intent(in) :: tindex, tindex_pre, tindex_post, separated_pre_post
        character(len=256) :: fname
        character(len=16) :: cfname

        if (separated_pre_post == 0) then
            ! Finish latter when needed
        else
            if (tindex_pre == tindex) then
                vexb_pre_fh = vexb_fh
            else
                vexb_pre_fh = 0
                write(cfname, "(I0)") tindex_pre
                fname = trim(adjustl(outputpath))//'vexb_x_pre_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, vexb_pre_fh(1))
                fname = trim(adjustl(outputpath))//'vexb_y_pre_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, vexb_pre_fh(2))
                fname = trim(adjustl(outputpath))//'vexb_z_pre_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, vexb_pre_fh(3))
            endif
            if (tindex_post == tindex) then
                vexb_post_fh = vexb_fh
            else
                vexb_post_fh = 0
                write(cfname, "(I0)") tindex_post
                fname = trim(adjustl(outputpath))//'vexb_x_post_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, vexb_post_fh(1))
                fname = trim(adjustl(outputpath))//'vexb_y_post_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, vexb_post_fh(2))
                fname = trim(adjustl(outputpath))//'vexb_z_post_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, vexb_post_fh(3))
            endif
        endif
    end subroutine open_vexb_pre_post_multi

    !<--------------------------------------------------------------------------
    !< Close v field files if different time frames are saved in different files
    !<--------------------------------------------------------------------------
    subroutine close_vexb_pre_post
        implicit none

        call MPI_FILE_CLOSE(vexb_pre_fh(1), ierror)
        call MPI_FILE_CLOSE(vexb_pre_fh(2), ierror)
        call MPI_FILE_CLOSE(vexb_pre_fh(3), ierror)

        call MPI_FILE_CLOSE(vexb_post_fh(1), ierror)
        call MPI_FILE_CLOSE(vexb_post_fh(2), ierror)
        call MPI_FILE_CLOSE(vexb_post_fh(3), ierror)
    end subroutine close_vexb_pre_post

    !<--------------------------------------------------------------------------
    !< Read previous and post v field.
    !< Input:
    !<   tframe: time frame. It can be adjusted to make "disp"  0.
    !<   output_format: 2=file per slice, 1=all slices in one file
    !<   separated_pre_post: 1 for separated pre and post files, 0 for not
    !<--------------------------------------------------------------------------
    subroutine read_pre_post_vexb(tframe, output_format, separated_pre_post)
        use constants, only: fp
        use picinfo, only: domain
        use parameters, only: tp1, tp2
        use mpi_datatype_fields, only: filetype_ghost, subsizes_ghost
        use mpi_io_module, only: read_data_mpi_io
        use exb_drift, only: vexbx, vexby, vexbz
        implicit none
        integer, intent(in) :: tframe, output_format, separated_pre_post
        integer(kind=MPI_OFFSET_KIND) :: disp, offset
        offset = 0

        if ((tframe >= tp1) .and. (tframe < tp2)) then
            if (output_format == 1) then
                if (separated_pre_post == 1) then
                    disp = domain%nx * domain%ny * domain%nz * sizeof(fp) * (tframe-tp1)
                else
                    disp = domain%nx * domain%ny * domain%nz * sizeof(fp) * (tframe-tp1+1)
                endif
            else
                disp = 0
            endif
            call read_data_mpi_io(vexb_post_fh(1), filetype_ghost, &
                subsizes_ghost, disp, offset, vexbx2)
            call read_data_mpi_io(vexb_post_fh(2), filetype_ghost, &
                subsizes_ghost, disp, offset, vexby2)
            call read_data_mpi_io(vexb_post_fh(3), filetype_ghost, &
                subsizes_ghost, disp, offset, vexbz2)
        else
            ! tframe = tp2, last time frame.
            vexbx2 = vexbx
            vexby2 = vexby
            vexbz2 = vexbz
        endif

        if ((tframe <= tp2) .and. (tframe > tp1)) then
            if (output_format == 1) then
                if (separated_pre_post == 1) then
                    disp = domain%nx * domain%ny * domain%nz * sizeof(fp) * (tframe-tp1)
                else
                    disp = domain%nx * domain%ny * domain%nz * sizeof(fp) * (tframe-tp1-1)
                endif
            else
                disp = 0
            endif
            call read_data_mpi_io(vexb_pre_fh(1), filetype_ghost, &
                subsizes_ghost, disp, offset, vexbx1)
            call read_data_mpi_io(vexb_pre_fh(2), filetype_ghost, &
                subsizes_ghost, disp, offset, vexby1)
            call read_data_mpi_io(vexb_pre_fh(3), filetype_ghost, &
                subsizes_ghost, disp, offset, vexbz1)
        else
            ! tframe = tp1, The first time frame.
            vexbx1 = vexbx
            vexby1 = vexby
            vexbz1 = vexbz
        endif
    end subroutine read_pre_post_vexb

    !<--------------------------------------------------------------------------
    !< Shift v field to remove ghost cells at lower end along x-, y-,
    !< and z-directions.
    !<--------------------------------------------------------------------------
    subroutine shift_vexb_pre_post
        use mpi_topology, only: ht
        implicit none
        ! x-direction
        if (ht%ix > 0) then
            vexbx1(1:ht%nx, :, :) = vexbx1(2:ht%nx+1, :, :)
            vexby1(1:ht%nx, :, :) = vexby1(2:ht%nx+1, :, :)
            vexbz1(1:ht%nx, :, :) = vexbz1(2:ht%nx+1, :, :)
            vexbx2(1:ht%nx, :, :) = vexbx2(2:ht%nx+1, :, :)
            vexby2(1:ht%nx, :, :) = vexby2(2:ht%nx+1, :, :)
            vexbz2(1:ht%nx, :, :) = vexbz2(2:ht%nx+1, :, :)
        endif

        ! y-direction
        if (ht%iy > 0) then
            vexbx1(:, 1:ht%ny, :) = vexbx1(:, 2:ht%ny+1, :)
            vexby1(:, 1:ht%ny, :) = vexby1(:, 2:ht%ny+1, :)
            vexbz1(:, 1:ht%ny, :) = vexbz1(:, 2:ht%ny+1, :)
            vexbx2(:, 1:ht%ny, :) = vexbx2(:, 2:ht%ny+1, :)
            vexby2(:, 1:ht%ny, :) = vexby2(:, 2:ht%ny+1, :)
            vexbz2(:, 1:ht%ny, :) = vexbz2(:, 2:ht%ny+1, :)
        endif

        ! z-direction
        if (ht%iz > 0) then
            vexbx1(:, :, 1:ht%nz) = vexbx1(:, :, 2:ht%nz+1)
            vexby1(:, :, 1:ht%nz) = vexby1(:, :, 2:ht%nz+1)
            vexbz1(:, :, 1:ht%nz) = vexbz1(:, :, 2:ht%nz+1)
            vexbx2(:, :, 1:ht%nz) = vexbx2(:, :, 2:ht%nz+1)
            vexby2(:, :, 1:ht%nz) = vexby2(:, :, 2:ht%nz+1)
            vexbz2(:, :, 1:ht%nz) = vexbz2(:, :, 2:ht%nz+1)
        endif
    end subroutine shift_vexb_pre_post

end module pre_post_vexb
