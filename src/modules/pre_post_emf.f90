!<******************************************************************************
!< Module for electric and magnetic fields of previous and post time frames.
!<******************************************************************************
module pre_post_emf
    use constants, only: fp
    use mpi_module
    use path_info, only: rootpath, filepath
    use mpi_info_module, only: fileinfo
    use mpi_io_module, only: open_data_mpi_io, read_data_mpi_io
    use hdf5_io, only: open_file_h5, close_file_h5, read_data_h5
    use mpi_datatype_fields, only: filetype_ghost, sizes_ghost, &
        subsizes_ghost, starts_ghost
    use hdf5
    implicit none
    private
    save

    public init_pre_post_efield, free_pre_post_efield, read_pre_post_efield, &
        open_efield_pre_post, close_efield_pre_post
    public init_pre_post_bfield, free_pre_post_bfield, read_pre_post_bfield, &
        open_bfield_pre_post, close_bfield_pre_post
    public open_field_files_pre_post_h5, close_field_files_pre_post_h5
    public interp_bfield_node_ghost, interp_efield_node_ghost
    public ex1, ey1, ez1, absE1, ex2, ey2, ez2, absE2
    public bx1, by1, bz1, absB1, bx2, by2, bz2, absB2

    real(fp), allocatable, dimension(:, :, :) :: ex1, ey1, ez1, absE1
    real(fp), allocatable, dimension(:, :, :) :: ex2, ey2, ez2, absE2
    real(fp), allocatable, dimension(:, :, :) :: bx1, by1, bz1, absB1
    real(fp), allocatable, dimension(:, :, :) :: bx2, by2, bz2, absB2
    integer, dimension(3) :: efields_pre_fh, efields_post_fh
    integer, dimension(3) :: bfields_pre_fh, bfields_post_fh
    integer(hid_t) :: field_file_pre_id, field_group_pre_id
    integer(hid_t) :: field_file_post_id, field_group_post_id

    interface open_efield_pre_post
        module procedure &
            open_efield_pre_post_single, open_efield_pre_post_multi
    end interface open_efield_pre_post

    interface open_bfield_pre_post
        module procedure &
            open_bfield_pre_post_single, open_bfield_pre_post_multi
    end interface open_bfield_pre_post

    contains

    !<--------------------------------------------------------------------------
    !< Initialize electric fields of the previous time frame and post time frames.
    !<--------------------------------------------------------------------------
    subroutine init_pre_post_efield(nx, ny, nz)
        implicit none
        integer, intent(in) :: nx, ny, nz

        allocate(ex1(nx,ny,nz))
        allocate(ey1(nx,ny,nz))
        allocate(ez1(nx,ny,nz))
        allocate(ex2(nx,ny,nz))
        allocate(ey2(nx,ny,nz))
        allocate(ez2(nx,ny,nz))
        allocate(absE1(nx,ny,nz))
        allocate(absE2(nx,ny,nz))

        ex1 = 0.0; ey1 = 0.0; ez1 = 0.0
        ex2 = 0.0; ey2 = 0.0; ez2 = 0.0
        absE1 = 0.0
        absE2 = 0.02
    end subroutine init_pre_post_efield

    !<--------------------------------------------------------------------------
    !< Initialize magnetic fields of the previous time frame and post time frames.
    !<--------------------------------------------------------------------------
    subroutine init_pre_post_bfield(nx, ny, nz)
        implicit none
        integer, intent(in) :: nx, ny, nz

        allocate(bx1(nx,ny,nz))
        allocate(by1(nx,ny,nz))
        allocate(bz1(nx,ny,nz))
        allocate(bx2(nx,ny,nz))
        allocate(by2(nx,ny,nz))
        allocate(bz2(nx,ny,nz))
        allocate(absB1(nx,ny,nz))
        allocate(absB2(nx,ny,nz))

        bx1 = 0.0; by1 = 0.0; bz1 = 0.0
        bx2 = 0.0; by2 = 0.0; bz2 = 0.0
        absB1 = 0.0
        absB2 = 0.02
    end subroutine init_pre_post_bfield

    !<--------------------------------------------------------------------------
    !< Free electric fields of the previous time frame and post time frames.
    !<--------------------------------------------------------------------------
    subroutine free_pre_post_efield
        implicit none
        deallocate(ex1, ey1, ez1, absE1)
        deallocate(ex2, ey2, ez2, absE2)
    end subroutine free_pre_post_efield

    !<--------------------------------------------------------------------------
    !< Free magnetic fields of the previous time frame and post time frames.
    !<--------------------------------------------------------------------------
    subroutine free_pre_post_bfield
        implicit none
        deallocate(bx1, by1, bz1, absB1)
        deallocate(bx2, by2, bz2, absB2)
    end subroutine free_pre_post_bfield

    !<--------------------------------------------------------------------------
    !< Open files containing electric and magnetic fields
    !< Inputs:
    !<   tindex: the time index.
    !<   tindex_pre: time index for previous time step
    !<   tindex_post: time index for next time step
    !<--------------------------------------------------------------------------
    subroutine open_field_files_pre_post_h5(tindex, tindex_pre, tindex_post)
        use pic_fields, only: field_file_id, field_group_id
        implicit none
        integer, intent(in) :: tindex, tindex_pre, tindex_post
        character(len=256) :: fname
        character(len=32) :: gname, cfname
        integer :: ierror

        if (tindex_pre == tindex) then
            field_file_pre_id = field_file_id
            field_group_pre_id = field_group_id
        else
            field_file_pre_id = 0
            field_group_pre_id = 0
            write(cfname, "(I0)") tindex_pre
            fname = trim(adjustl(rootpath))//'field_hdf5/T.'//trim(cfname)//'/'
            fname = trim(adjustl(fname))//'fields_'//trim(cfname)//'.h5'
            call open_file_h5(trim(fname), H5F_ACC_RDONLY_F, field_file_pre_id, .true.)
            gname = "Timestep_"//trim(cfname)
            call h5gopen_f(field_file_pre_id, trim(gname), field_group_pre_id, ierror)
        endif

        if (tindex_post == tindex) then
            field_file_post_id = field_file_id
            field_group_post_id = field_group_id
        else
            field_file_post_id = 0
            field_group_post_id = 0
            write(cfname, "(I0)") tindex_post
            fname = trim(adjustl(rootpath))//'field_hdf5/T.'//trim(cfname)//'/'
            fname = trim(adjustl(fname))//'fields_'//trim(cfname)//'.h5'
            call open_file_h5(trim(fname), H5F_ACC_RDONLY_F, field_file_post_id, .true.)
            gname = "Timestep_"//trim(cfname)
            call h5gopen_f(field_file_post_id, trim(gname), field_group_post_id, ierror)
        endif
    end subroutine open_field_files_pre_post_h5

    !<--------------------------------------------------------------------------
    !< Close files containing electric and magnetic fields
    !<--------------------------------------------------------------------------
    subroutine close_field_files_pre_post_h5(tindex, tindex_pre, tindex_post)
        implicit none
        integer :: error
        integer, intent(in) :: tindex, tindex_pre, tindex_post
        if (tindex /= tindex_pre) then
            call h5gclose_f(field_group_pre_id, error)
            call h5fclose_f(field_file_pre_id, error)
        endif
        if (tindex /= tindex_post) then
            call h5gclose_f(field_group_post_id, error)
            call h5fclose_f(field_file_post_id, error)
        endif
    end subroutine close_field_files_pre_post_h5

    !<--------------------------------------------------------------------------
    !< Open electric field files if all time frames are saved in the same file
    !< Input:
    !<   separated_pre_post: 1 for separated pre and post files, 0 for not
    !<--------------------------------------------------------------------------
    subroutine open_efield_pre_post_single(separated_pre_post)
        use pic_fields, only: efields_fh
        implicit none
        integer, intent(in) :: separated_pre_post
        character(len=256) :: fname
        character(len=16) :: cfname

        if (separated_pre_post == 0) then
            efields_pre_fh = efields_fh
            efields_post_fh = efields_fh
        else
            efields_pre_fh = 0
            efields_post_fh = 0
            fname = trim(adjustl(filepath))//'ex_pre.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, &
                efields_pre_fh(1))
            fname = trim(adjustl(filepath))//'ey_pre.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, &
                efields_pre_fh(2))
            fname = trim(adjustl(filepath))//'ez_pre.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, &
                efields_pre_fh(3))
            fname = trim(adjustl(filepath))//'ex_post.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, &
                efields_post_fh(1))
            fname = trim(adjustl(filepath))//'ey_post.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, &
                efields_post_fh(2))
            fname = trim(adjustl(filepath))//'ez_post.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, &
                efields_post_fh(3))
        endif
    end subroutine open_efield_pre_post_single

    !<--------------------------------------------------------------------------
    !< Open electric field files if different time frames are saved in different files
    !< Input:
    !<   tindex: time index for current time step
    !<   tindex_pre: time index for previous time step
    !<   tindex_post: time index for next time step
    !<   separated_pre_post: 1 for separated pre and post files, 0 for not
    !<--------------------------------------------------------------------------
    subroutine open_efield_pre_post_multi(separated_pre_post, tindex, &
            tindex_pre, tindex_post)
        use pic_fields, only: efields_fh
        implicit none
        integer, intent(in) :: tindex, tindex_pre, tindex_post, separated_pre_post
        character(len=256) :: fname
        character(len=16) :: cfname

        if (separated_pre_post == 0) then
            if (tindex_pre == tindex) then
                efields_pre_fh = efields_fh
            else
                efields_pre_fh = 0
                write(cfname, "(I0)") tindex_pre
                fname = trim(adjustl(filepath))//'ex_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, efields_pre_fh(1))
                fname = trim(adjustl(filepath))//'ey_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, efields_pre_fh(2))
                fname = trim(adjustl(filepath))//'ez_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, efields_pre_fh(3))
            endif
            if (tindex_post == tindex) then
                efields_post_fh = efields_fh
            else
                efields_post_fh = 0
                write(cfname, "(I0)") tindex_post
                fname = trim(adjustl(filepath))//'ex_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, efields_post_fh(1))
                fname = trim(adjustl(filepath))//'ey_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, efields_post_fh(2))
                fname = trim(adjustl(filepath))//'ez_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, efields_post_fh(3))
            endif
        else
            efields_pre_fh = 0
            write(cfname, "(I0)") tindex_pre
            fname = trim(adjustl(filepath))//'ex_pre_'//trim(cfname)//'.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, efields_pre_fh(1))
            fname = trim(adjustl(filepath))//'ey_pre_'//trim(cfname)//'.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, efields_pre_fh(2))
            fname = trim(adjustl(filepath))//'ez_pre_'//trim(cfname)//'.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, efields_pre_fh(3))

            efields_post_fh = 0
            write(cfname, "(I0)") tindex_post
            fname = trim(adjustl(filepath))//'ex_post_'//trim(cfname)//'.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, efields_post_fh(1))
            fname = trim(adjustl(filepath))//'ey_post_'//trim(cfname)//'.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, efields_post_fh(2))
            fname = trim(adjustl(filepath))//'ez_post_'//trim(cfname)//'.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, efields_post_fh(3))
        endif
    end subroutine open_efield_pre_post_multi

    !<--------------------------------------------------------------------------
    !< Close electric field files if different time frames are saved in different
    !< files
    !<--------------------------------------------------------------------------
    subroutine close_efield_pre_post
        implicit none

        call MPI_FILE_CLOSE(efields_pre_fh(1), ierror)
        call MPI_FILE_CLOSE(efields_pre_fh(2), ierror)
        call MPI_FILE_CLOSE(efields_pre_fh(3), ierror)

        call MPI_FILE_CLOSE(efields_post_fh(1), ierror)
        call MPI_FILE_CLOSE(efields_post_fh(2), ierror)
        call MPI_FILE_CLOSE(efields_post_fh(3), ierror)
    end subroutine close_efield_pre_post

    !<--------------------------------------------------------------------------
    !< Read previous and post electric field.
    !< Input:
    !<   tframe: time frame. It can be adjusted to make "disp"  0.
    !<   output_format: 2=file per slice, 1=all slices in one file
    !<   separated_pre_post: 1 for separated pre and post files, 0 for not
    !<   hdf5_flag(optional): whether to read to a HDF5 file
    !<   collective_io(optional): whether to read to data collectively
    !<--------------------------------------------------------------------------
    subroutine read_pre_post_efield(tframe, output_format, separated_pre_post, &
            hdf5_flag, collective_io)
        use constants, only: fp
        use picinfo, only: domain
        use parameters, only: tp1, tp2
        use mpi_io_module, only: read_data_mpi_io
        use pic_fields, only: ex, ey, ez
        implicit none
        integer, intent(in) :: tframe, output_format, separated_pre_post
        logical, intent(in), optional :: hdf5_flag, collective_io
        integer(kind=MPI_OFFSET_KIND) :: disp, offset, disp0
        logical :: use_hdf5, use_collective_io
        integer(hid_t) :: ex_id, ey_id, ez_id
        integer(hsize_t), dimension(3) :: dcount, doffset, dset_dims
        integer :: error
        if (.not. present(hdf5_flag)) then
            use_hdf5 = .false.
        else
            use_hdf5 = hdf5_flag
        endif
        if (.not. present(collective_io)) then
            use_collective_io = .false.
        else
            use_collective_io = collective_io
        endif

        if (use_hdf5) then
            dcount = subsizes_ghost(3:1:-1)
            doffset = starts_ghost(3:1:-1)
            dset_dims = sizes_ghost(3:1:-1)
        endif

        offset = 0
        disp0 = domain%nx * domain%ny * domain%nz * sizeof(fp)
        if ((tframe >= tp1) .and. (tframe < tp2)) then
            if (use_hdf5) then
                call h5dopen_f(field_group_post_id, "ex", ex_id, error)
                call h5dopen_f(field_group_post_id, "ey", ey_id, error)
                call h5dopen_f(field_group_post_id, "ez", ez_id, error)
                call read_data_h5(ex_id, dcount, doffset, dset_dims, &
                    ex2, .true., use_collective_io)
                call read_data_h5(ey_id, dcount, doffset, dset_dims, &
                    ey2, .true., use_collective_io)
                call read_data_h5(ez_id, dcount, doffset, dset_dims, &
                    ez2, .true., use_collective_io)
                ex2 = reshape(ex2, shape(ex2), order=[3, 2, 1])
                ey2 = reshape(ey2, shape(ey2), order=[3, 2, 1])
                ez2 = reshape(ez2, shape(ez2), order=[3, 2, 1])
                call h5dclose_f(ex_id, error)
                call h5dclose_f(ey_id, error)
                call h5dclose_f(ez_id, error)
            else
                if (output_format == 1) then
                    if (separated_pre_post == 1) then
                        disp =  disp0 * (tframe-tp1)
                    else
                        disp = disp0 * (tframe-tp1+1)
                    endif
                else
                    disp = 0
                endif
                call read_data_mpi_io(efields_post_fh(1), filetype_ghost, &
                    subsizes_ghost, disp, offset, ex2)
                call read_data_mpi_io(efields_post_fh(2), filetype_ghost, &
                    subsizes_ghost, disp, offset, ey2)
                call read_data_mpi_io(efields_post_fh(3), filetype_ghost, &
                    subsizes_ghost, disp, offset, ez2)
            endif
        else
            ! tframe = tp2, last time frame.
            ex2 = ex
            ey2 = ey
            ez2 = ez
        endif

        if ((tframe <= tp2) .and. (tframe > tp1)) then
            if (use_hdf5) then
                call h5dopen_f(field_group_pre_id, "ex", ex_id, error)
                call h5dopen_f(field_group_pre_id, "ey", ey_id, error)
                call h5dopen_f(field_group_pre_id, "ez", ez_id, error)
                call read_data_h5(ex_id, dcount, doffset, dset_dims, &
                    ex1, .true., use_collective_io)
                call read_data_h5(ey_id, dcount, doffset, dset_dims, &
                    ey1, .true., use_collective_io)
                call read_data_h5(ez_id, dcount, doffset, dset_dims, &
                    ez1, .true., use_collective_io)
                ex1 = reshape(ex1, shape(ex1), order=[3, 2, 1])
                ey1 = reshape(ey1, shape(ey1), order=[3, 2, 1])
                ez1 = reshape(ez1, shape(ez1), order=[3, 2, 1])
                call h5dclose_f(ex_id, error)
                call h5dclose_f(ey_id, error)
                call h5dclose_f(ez_id, error)
            else
                if (output_format == 1) then
                    if (separated_pre_post == 1) then
                        disp = disp0 * (tframe-tp1)
                    else
                        disp = disp0 * (tframe-tp1-1)
                    endif
                else
                    disp = 0
                endif
                call read_data_mpi_io(efields_pre_fh(1), filetype_ghost, &
                    subsizes_ghost, disp, offset, ex1)
                call read_data_mpi_io(efields_pre_fh(2), filetype_ghost, &
                    subsizes_ghost, disp, offset, ey1)
                call read_data_mpi_io(efields_pre_fh(3), filetype_ghost, &
                    subsizes_ghost, disp, offset, ez1)
            endif
        else
            ! tframe = tp1, The first time frame.
            ex1 = ex
            ey1 = ey
            ez1 = ez
        endif

        absE1 = sqrt(ex1**2 + ey1**2 + ez1**2)
        absE2 = sqrt(ex2**2 + ey2**2 + ez2**2)
    end subroutine read_pre_post_efield

    !<--------------------------------------------------------------------------
    !< Open magnetic field files if all time frames are saved in the same file
    !< Input:
    !<   separated_pre_post: 1 for separated pre and post files, 0 for not
    !<--------------------------------------------------------------------------
    subroutine open_bfield_pre_post_single(separated_pre_post)
        use pic_fields, only: bfields_fh
        implicit none
        integer, intent(in) :: separated_pre_post
        character(len=256) :: fname
        character(len=16) :: cfname

        if (separated_pre_post == 0) then
            bfields_pre_fh = bfields_fh(1:3)
            bfields_post_fh = bfields_fh(1:3)
        else
            bfields_pre_fh = 0
            bfields_post_fh = 0
            fname = trim(adjustl(filepath))//'bx_pre.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, &
                bfields_pre_fh(1))
            fname = trim(adjustl(filepath))//'by_pre.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, &
                bfields_pre_fh(2))
            fname = trim(adjustl(filepath))//'bz_pre.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, &
                bfields_pre_fh(3))
            fname = trim(adjustl(filepath))//'bx_post.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, &
                bfields_post_fh(1))
            fname = trim(adjustl(filepath))//'by_post.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, &
                bfields_post_fh(2))
            fname = trim(adjustl(filepath))//'bz_post.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, &
                bfields_post_fh(3))
        endif
    end subroutine open_bfield_pre_post_single

    !<--------------------------------------------------------------------------
    !< Open magnetic field files if different time frames are saved in different files
    !< Input:
    !<   tindex: time index for current time step
    !<   tindex_pre: time index for previous time step
    !<   tindex_post: time index for next time step
    !<   separated_pre_post: 1 for separated pre and post files, 0 for not
    !<--------------------------------------------------------------------------
    subroutine open_bfield_pre_post_multi(separated_pre_post, tindex, &
            tindex_pre, tindex_post)
        use pic_fields, only: bfields_fh
        implicit none
        integer, intent(in) :: tindex, tindex_pre, tindex_post, separated_pre_post
        character(len=256) :: fname
        character(len=16) :: cfname

        if (separated_pre_post == 0) then
            if (tindex_pre == tindex) then
                bfields_pre_fh = bfields_fh(1:3)
            else
                bfields_pre_fh = 0
                write(cfname, "(I0)") tindex_pre
                fname = trim(adjustl(filepath))//'bx_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, bfields_pre_fh(1))
                fname = trim(adjustl(filepath))//'by_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, bfields_pre_fh(2))
                fname = trim(adjustl(filepath))//'bz_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, bfields_pre_fh(3))
            endif

            if (tindex_post == tindex) then
                bfields_post_fh = bfields_fh(1:3)
            else
                bfields_post_fh = 0
                write(cfname, "(I0)") tindex_post
                fname = trim(adjustl(filepath))//'bx_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, bfields_post_fh(1))
                fname = trim(adjustl(filepath))//'by_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, bfields_post_fh(2))
                fname = trim(adjustl(filepath))//'bz_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, bfields_post_fh(3))
            endif
        else
            bfields_pre_fh = 0
            write(cfname, "(I0)") tindex_pre
            fname = trim(adjustl(filepath))//'bx_pre_'//trim(cfname)//'.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, bfields_pre_fh(1))
            fname = trim(adjustl(filepath))//'by_pre_'//trim(cfname)//'.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, bfields_pre_fh(2))
            fname = trim(adjustl(filepath))//'bz_pre_'//trim(cfname)//'.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, bfields_pre_fh(3))

            bfields_post_fh = 0
            write(cfname, "(I0)") tindex_post
            fname = trim(adjustl(filepath))//'bx_post_'//trim(cfname)//'.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, bfields_post_fh(1))
            fname = trim(adjustl(filepath))//'by_post_'//trim(cfname)//'.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, bfields_post_fh(2))
            fname = trim(adjustl(filepath))//'bz_post_'//trim(cfname)//'.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, bfields_post_fh(3))
        endif
    end subroutine open_bfield_pre_post_multi

    !<--------------------------------------------------------------------------
    !< Close magnetic field files if different time frames are saved in different
    !< files
    !<--------------------------------------------------------------------------
    subroutine close_bfield_pre_post
        implicit none

        call MPI_FILE_CLOSE(bfields_pre_fh(1), ierror)
        call MPI_FILE_CLOSE(bfields_pre_fh(2), ierror)
        call MPI_FILE_CLOSE(bfields_pre_fh(3), ierror)

        call MPI_FILE_CLOSE(bfields_post_fh(1), ierror)
        call MPI_FILE_CLOSE(bfields_post_fh(2), ierror)
        call MPI_FILE_CLOSE(bfields_post_fh(3), ierror)
    end subroutine close_bfield_pre_post

    !<--------------------------------------------------------------------------
    !< Read previous and post magnetic field.
    !< Input:
    !<   tframe: time frame. It can be adjusted to make "disp"  0.
    !<   output_format: 2=file per slice, 1=all slices in one file
    !<   separated_pre_post: 1 for separated pre and post files, 0 for not
    !<   hdf5_flag(optional): whether to read to a HDF5 file
    !<   collective_io(optional): whether to read to data collectively
    !<--------------------------------------------------------------------------
    subroutine read_pre_post_bfield(tframe, output_format, separated_pre_post, &
            hdf5_flag, collective_io)
        use constants, only: fp
        use picinfo, only: domain
        use parameters, only: tp1, tp2
        use mpi_io_module, only: read_data_mpi_io
        use pic_fields, only: bx, by, bz
        implicit none
        integer, intent(in) :: tframe, output_format, separated_pre_post
        logical, intent(in), optional :: hdf5_flag, collective_io
        integer(kind=MPI_OFFSET_KIND) :: disp, offset, disp0
        logical :: use_hdf5, use_collective_io
        integer(hid_t) :: bx_id, by_id, bz_id
        integer(hsize_t), dimension(3) :: dcount, doffset, dset_dims
        integer :: error
        if (.not. present(hdf5_flag)) then
            use_hdf5 = .false.
        else
            use_hdf5 = hdf5_flag
        endif
        if (.not. present(collective_io)) then
            use_collective_io = .false.
        else
            use_collective_io = collective_io
        endif

        if (use_hdf5) then
            dcount = subsizes_ghost(3:1:-1)
            doffset = starts_ghost(3:1:-1)
            dset_dims = sizes_ghost(3:1:-1)
        endif

        offset = 0
        disp0 = domain%nx * domain%ny * domain%nz * sizeof(fp)
        if ((tframe >= tp1) .and. (tframe < tp2)) then
            if (use_hdf5) then
                call h5dopen_f(field_group_post_id, "cbx", bx_id, error)
                call h5dopen_f(field_group_post_id, "cby", by_id, error)
                call h5dopen_f(field_group_post_id, "cbz", bz_id, error)
                call read_data_h5(bx_id, dcount, doffset, dset_dims, &
                    bx2, .true., use_collective_io)
                call read_data_h5(by_id, dcount, doffset, dset_dims, &
                    by2, .true., use_collective_io)
                call read_data_h5(bz_id, dcount, doffset, dset_dims, &
                    bz2, .true., use_collective_io)
                bx2 = reshape(bx2, shape(bx2), order=[3, 2, 1])
                by2 = reshape(by2, shape(by2), order=[3, 2, 1])
                bz2 = reshape(bz2, shape(bz2), order=[3, 2, 1])
                call h5dclose_f(bx_id, error)
                call h5dclose_f(by_id, error)
                call h5dclose_f(bz_id, error)
            else
                if (output_format == 1) then
                    if (separated_pre_post == 1) then
                        disp = disp0 * (tframe-tp1)
                    else
                        disp = disp0 * (tframe-tp1+1)
                    endif
                else
                    disp = 0
                endif
                call read_data_mpi_io(bfields_post_fh(1), filetype_ghost, &
                    subsizes_ghost, disp, offset, bx2)
                call read_data_mpi_io(bfields_post_fh(2), filetype_ghost, &
                    subsizes_ghost, disp, offset, by2)
                call read_data_mpi_io(bfields_post_fh(3), filetype_ghost, &
                    subsizes_ghost, disp, offset, bz2)
            endif
        else
            ! tframe = tp2, last time frame.
            bx2 = bx
            by2 = by
            bz2 = bz
        endif

        if ((tframe <= tp2) .and. (tframe > tp1)) then
            if (use_hdf5) then
                call h5dopen_f(field_group_pre_id, "cbx", bx_id, error)
                call h5dopen_f(field_group_pre_id, "cby", by_id, error)
                call h5dopen_f(field_group_pre_id, "cbz", bz_id, error)
                call read_data_h5(bx_id, dcount, doffset, dset_dims, &
                    bx1, .true., use_collective_io)
                call read_data_h5(by_id, dcount, doffset, dset_dims, &
                    by1, .true., use_collective_io)
                call read_data_h5(bz_id, dcount, doffset, dset_dims, &
                    bz1, .true., use_collective_io)
                bx1 = reshape(bx1, shape(bx1), order=[3, 2, 1])
                by1 = reshape(by1, shape(by1), order=[3, 2, 1])
                bz1 = reshape(bz1, shape(bz1), order=[3, 2, 1])
                call h5dclose_f(bx_id, error)
                call h5dclose_f(by_id, error)
                call h5dclose_f(bz_id, error)
            else
                if (output_format == 1) then
                    if (separated_pre_post == 1) then
                        disp = disp0 * (tframe-tp1)
                    else
                        disp = disp0 * (tframe-tp1-1)
                    endif
                else
                    disp = 0
                endif
                call read_data_mpi_io(bfields_pre_fh(1), filetype_ghost, &
                    subsizes_ghost, disp, offset, bx1)
                call read_data_mpi_io(bfields_pre_fh(2), filetype_ghost, &
                    subsizes_ghost, disp, offset, by1)
                call read_data_mpi_io(bfields_pre_fh(3), filetype_ghost, &
                    subsizes_ghost, disp, offset, bz1)
            endif
        else
            ! tframe = tp1, The first time frame.
            bx1 = bx
            by1 = by
            bz1 = bz
        endif

        absB1 = sqrt(bx1**2 + by1**2 + bz1**2)
        absB2 = sqrt(bx2**2 + by2**2 + bz2**2)
    end subroutine read_pre_post_bfield

    !<--------------------------------------------------------------------------
    !< Linearly interpolate electric field to the node positions.
    !< We need to calculate fields at ghost cells too.
    !<--------------------------------------------------------------------------
    subroutine interp_efield_node_ghost
        use mpi_topology, only: ht, htg
        implicit none
        integer :: nx, ny, nz, nxg, nyg, nzg
        nx = ht%nx
        ny = ht%ny
        nz = ht%nz
        nxg = htg%nx
        nyg = htg%ny
        nzg = htg%nz
        ! Ex
        if (nxg > 1) then
            ex1(2:nxg, :, :) = (ex1(1:nxg-1, :, :) + ex1(2:nxg, :, :)) * 0.5
            ex1(1, :, :) = 2 * ex1(1, :, :) - ex1(2, :, :)
            ex2(2:nxg, :, :) = (ex2(1:nxg-1, :, :) + ex2(2:nxg, :, :)) * 0.5
            ex2(1, :, :) = 2 * ex2(1, :, :) - ex2(2, :, :)
        endif
        ! Ey
        if (nyg > 1) then
            ey1(:, 2:nyg, :) = (ey1(:, 1:nyg-1, :) + ey1(:, 2:nyg, :)) * 0.5
            ey1(:, 1, :) = 2 * ey1(:, 1, :) - ey1(:, 2, :)
            ey2(:, 2:nyg, :) = (ey2(:, 1:nyg-1, :) + ey2(:, 2:nyg, :)) * 0.5
            ey2(:, 1, :) = 2 * ey2(:, 1, :) - ey2(:, 2, :)
        endif
        ! Ez
        if (nzg > 1) then
            ez1(:, :, 2:nzg) = (ez1(:, :, 1:nzg-1) + ez1(:, :, 2:nzg)) * 0.5
            ez1(:, :, 1) = 2 * ez1(:, :, 1) - ez1(:, :, 2)
            ez2(:, :, 2:nzg) = (ez2(:, :, 1:nzg-1) + ez2(:, :, 2:nzg)) * 0.5
            ez2(:, :, 1) = 2 * ez2(:, :, 1) - ez2(:, :, 2)
        endif

        absE1 = sqrt(ex1**2 + ey1**2 + ez1**2)
        absE2 = sqrt(ex2**2 + ey2**2 + ez2**2)
    end subroutine interp_efield_node_ghost

    !<--------------------------------------------------------------------------
    !< Linearly interpolate magnetic field to the node positions.
    !< We need to calculate fields at ghost cells too.
    !<--------------------------------------------------------------------------
    subroutine interp_bfield_node_ghost
        use mpi_topology, only: ht, htg
        implicit none
        integer :: nx, ny, nz, nxg, nyg, nzg
        nx = ht%nx
        ny = ht%ny
        nz = ht%nz
        nxg = htg%nx
        nyg = htg%ny
        nzg = htg%nz
        ! Bx
        if (nyg > 1 .and. nzg > 1) then
            bx1(:, 2:nyg, 2:nzg) = (bx1(:, 1:nyg-1, 1:nzg-1) + &
                                    bx1(:, 1:nyg-1, 2:nzg) + &
                                    bx1(:, 2:nyg, 1:nzg-1) + &
                                    bx1(:, 2:nyg, 2:nzg)) * 0.25
            bx1(:, 2:nyg, 1) = bx1(:, 1:nyg-1, 1) + bx1(:, 2:nyg, 1) - bx1(:, 2:nyg, 2)
            bx1(:, 1, 2:nzg) = bx1(:, 1, 1:nzg-1) + bx1(:, 1, 2:nzg) - bx1(:, 2, 2:nzg)
            bx1(:, 1, 1) = bx1(:, 1, 2) + bx1(:, 2, 1) - bx1(:, 2, 2)
            bx2(:, 2:nyg, 2:nzg) = (bx2(:, 1:nyg-1, 1:nzg-1) + &
                                    bx2(:, 1:nyg-1, 2:nzg) + &
                                    bx2(:, 2:nyg, 1:nzg-1) + &
                                    bx2(:, 2:nyg, 2:nzg)) * 0.25
            bx2(:, 2:nyg, 1) = bx2(:, 1:nyg-1, 1) + bx2(:, 2:nyg, 1) - bx2(:, 2:nyg, 2)
            bx2(:, 1, 2:nzg) = bx2(:, 1, 1:nzg-1) + bx2(:, 1, 2:nzg) - bx2(:, 2, 2:nzg)
            bx2(:, 1, 1) = bx2(:, 1, 2) + bx2(:, 2, 1) - bx2(:, 2, 2)
        else if (nyg == 1 .and. nzg > 1) then
            bx1(:, 1, 2:nzg) = (bx1(:, 1, 1:nzg-1) + bx1(:, 1, 2:nzg)) * 0.5
            bx1(:, 1, 1) = 2 * bx1(:, 1, 1) - bx1(:, 1, 2)
            bx2(:, 1, 2:nzg) = (bx2(:, 1, 1:nzg-1) + bx2(:, 1, 2:nzg)) * 0.5
            bx2(:, 1, 1) = 2 * bx2(:, 1, 1) - bx2(:, 1, 2)
        else if (nyg > 1 .and. nzg == 1) then
            bx1(:, 2:nyg, 1) = (bx1(:, 1:nyg-1, 1) + bx1(:, 2:nyg, 1)) * 0.5
            bx1(:, 1, 1) = 2 * bx1(:, 1, 1) - bx1(:, 2, 1)
            bx2(:, 2:nyg, 1) = (bx2(:, 1:nyg-1, 1) + bx2(:, 2:nyg, 1)) * 0.5
            bx2(:, 1, 1) = 2 * bx2(:, 1, 1) - bx2(:, 2, 1)
        endif

        ! By
        if (nxg > 1 .and. nzg > 1) then
            by1(2:nxg, :, 2:nzg) = (by1(1:nxg-1, :, 1:nzg-1) + &
                                    by1(1:nxg-1, :, 2:nzg) + &
                                    by1(2:nxg, :, 1:nzg-1) + &
                                    by1(2:nxg, :, 2:nzg)) * 0.25
            by1(2:nxg, :, 1) = by1(1:nxg-1, :, 1) + by1(2:nxg, :, 1) - by1(2:nxg, :, 2)
            by1(1, :, 2:nzg) = by1(1, :, 1:nzg-1) + by1(1, :, 2:nzg) - by1(2, :, 2:nzg)
            by1(1, :, 1) = by1(1, :, 2) + by1(2, :, 1) - by1(2, :, 2)
            by2(2:nxg, :, 2:nzg) = (by2(1:nxg-1, :, 1:nzg-1) + &
                                    by2(1:nxg-1, :, 2:nzg) + &
                                    by2(2:nxg, :, 1:nzg-1) + &
                                    by2(2:nxg, :, 2:nzg)) * 0.25
            by2(2:nxg, :, 1) = by2(1:nxg-1, :, 1) + by2(2:nxg, :, 1) - by2(2:nxg, :, 2)
            by2(1, :, 2:nzg) = by2(1, :, 1:nzg-1) + by2(1, :, 2:nzg) - by2(2, :, 2:nzg)
            by2(1, :, 1) = by2(1, :, 2) + by2(2, :, 1) - by2(2, :, 2)
        else if (nxg == 1 .and. nzg > 1) then
            by1(1, :, 2:nzg) = (by1(1, :, 1:nzg-1) + by1(1, :, 2:nzg)) * 0.5
            by1(1, :, 1) = 2 * by1(1, :, 1) - by1(1, :, 2)
            by2(1, :, 2:nzg) = (by2(1, :, 1:nzg-1) + by2(1, :, 2:nzg)) * 0.5
            by2(1, :, 1) = 2 * by2(1, :, 1) - by2(1, :, 2)
        else if (nxg > 1 .and. nzg == 1) then
            by1(2:nxg, :, 1) = (by1(1:nxg-1, :, 1) + by1(2:nxg, :, 1)) * 0.5
            by1(1, :, 1) = 2 * by1(1, :, 1) - by1(2, :, 1)
            by2(2:nxg, :, 1) = (by2(1:nxg-1, :, 1) + by2(2:nxg, :, 1)) * 0.5
            by2(1, :, 1) = 2 * by2(1, :, 1) - by2(2, :, 1)
        endif

        ! Bz
        if (nxg > 1 .and. nyg > 1) then
            bz1(2:nxg, 2:nyg, :) = (bz1(1:nxg-1, 1:nyg-1, :) + &
                                    bz1(1:nxg-1, 2:nyg, :) + &
                                    bz1(2:nxg, 1:nyg-1, :) + &
                                    bz1(2:nxg, 2:nyg, :)) * 0.25
            bz1(2:nxg, 1, :) = bz1(1:nxg-1, 1, :) + bz1(2:nxg, 1, :) - bz1(2:nxg, 2, :)
            bz1(1, 2:nyg, :) = bz1(1, 1:nyg-1, :) + bz1(1, 2:nyg, :) - bz1(2, 2:nyg, :)
            bz1(1, 1, :) = bz1(1, 2, :) + bz1(2, 1, :) - bz1(2, 2, :)
            bz2(2:nxg, 2:nyg, :) = (bz2(1:nxg-1, 1:nyg-1, :) + &
                                    bz2(1:nxg-1, 2:nyg, :) + &
                                    bz2(2:nxg, 1:nyg-1, :) + &
                                    bz2(2:nxg, 2:nyg, :)) * 0.25
            bz2(2:nxg, 1, :) = bz2(1:nxg-1, 1, :) + bz2(2:nxg, 1, :) - bz2(2:nxg, 2, :)
            bz2(1, 2:nyg, :) = bz2(1, 1:nyg-1, :) + bz2(1, 2:nyg, :) - bz2(2, 2:nyg, :)
            bz2(1, 1, :) = bz2(1, 2, :) + bz2(2, 1, :) - bz2(2, 2, :)
        else if (nxg == 1 .and. nyg > 1) then
            bz1(1, 2:nyg, :) = (bz1(1, 1:nyg-1, :) + bz1(1, 2:nyg, :)) * 0.5
            bz1(1, 1, :) = 2 * bz1(1, 1, :) - bz1(1, 2, :)
            bz2(1, 2:nyg, :) = (bz2(1, 1:nyg-1, :) + bz2(1, 2:nyg, :)) * 0.5
            bz2(1, 1, :) = 2 * bz2(1, 1, :) - bz2(1, 2, :)
        else if (nxg > 1 .and. nyg == 1) then
            bz1(2:nxg, 1, :) = (bz1(1:nxg-1, 1, :) + bz1(2:nxg, 1, :)) * 0.5
            bz1(1, 1, :) = 2 * bz1(1, 1, :) - bz1(2, 1, :)
            bz2(2:nxg, 1, :) = (bz2(1:nxg-1, 1, :) + bz2(2:nxg, 1, :)) * 0.5
            bz2(1, 1, :) = 2 * bz2(1, 1, :) - bz2(2, 1, :)
        endif

        absB1 = sqrt(bx1**2 + by1**2 + bz1**2)
        absB2 = sqrt(bx2**2 + by2**2 + bz2**2)
    end subroutine interp_bfield_node_ghost

end module pre_post_emf
