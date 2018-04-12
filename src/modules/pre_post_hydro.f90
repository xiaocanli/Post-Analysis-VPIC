!<******************************************************************************
!< Module for hydro fields of previous and post time frames.
!<******************************************************************************
module pre_post_hydro
    use constants, only: fp
    use mpi_module
    use path_info, only: filepath
    use mpi_info_module, only: fileinfo
    use mpi_io_module, only: open_data_mpi_io, read_data_mpi_io
    implicit none
    private
    public vdx1, vdy1, vdz1, vdx2, vdy2, vdz2
    public udx1, udy1, udz1, udx2, udy2, udz2
    public nrho1, nrho2

    public init_pre_post_velocities, free_pre_post_velocities, &
           read_pre_post_velocities
    public init_pre_post_u, free_pre_post_u, open_ufield_pre_post, &
           close_ufield_pre_post, read_pre_post_u, shift_ufield_pre_post
    public init_pre_post_density, free_pre_post_density, read_pre_post_density

    real(fp), allocatable, dimension(:, :, :) :: vdx1, vdy1, vdz1
    real(fp), allocatable, dimension(:, :, :) :: vdx2, vdy2, vdz2
    real(fp), allocatable, dimension(:, :, :) :: udx1, udy1, udz1  ! gamma * v
    real(fp), allocatable, dimension(:, :, :) :: udx2, udy2, udz2
    real(fp), allocatable, dimension(:, :, :) :: nrho1, nrho2
    integer, dimension(3) :: ufields_pre_fh, ufields_post_fh

    interface open_ufield_pre_post
        module procedure &
            open_ufield_pre_post_single, open_ufield_pre_post_multi
    end interface open_ufield_pre_post

    contains

    !<--------------------------------------------------------------------------
    !< Initialize velocities of the previous time frame and post time frame.
    !< They are going be be used to calculate polarization drift current.
    !<--------------------------------------------------------------------------
    subroutine init_pre_post_velocities
        use mpi_topology, only: htg
        implicit none
        integer :: nx, ny, nz

        nx = htg%nx
        ny = htg%ny
        nz = htg%nz

        allocate(vdx1(nx,ny,nz))
        allocate(vdy1(nx,ny,nz))
        allocate(vdz1(nx,ny,nz))
        allocate(vdx2(nx,ny,nz))
        allocate(vdy2(nx,ny,nz))
        allocate(vdz2(nx,ny,nz))

        vdx1 = 0.0; vdy1 = 0.0; vdz1 = 0.0
        vdx2 = 0.0; vdy2 = 0.0; vdz2 = 0.0
    end subroutine init_pre_post_velocities

    !<--------------------------------------------------------------------------
    !< Initialize gamma * V of the previous time frame and post time frame.
    !<--------------------------------------------------------------------------
    subroutine init_pre_post_u(nx, ny, nz)
        implicit none
        integer, intent(in) :: nx, ny, nz

        allocate(udx1(nx,ny,nz))
        allocate(udy1(nx,ny,nz))
        allocate(udz1(nx,ny,nz))
        allocate(udx2(nx,ny,nz))
        allocate(udy2(nx,ny,nz))
        allocate(udz2(nx,ny,nz))

        udx1 = 0.0; udy1 = 0.0; udz1 = 0.0
        udx2 = 0.0; udy2 = 0.0; udz2 = 0.0
    end subroutine init_pre_post_u

    !<--------------------------------------------------------------------------
    !< Initialize density of the previous time frame and post time frame.
    !< They are going be be used to calculate polarization drift current.
    !<--------------------------------------------------------------------------
    subroutine init_pre_post_density
        use mpi_topology, only: htg
        implicit none
        integer :: nx, ny, nz

        nx = htg%nx
        ny = htg%ny
        nz = htg%nz

        allocate(nrho1(nx,ny,nz))
        allocate(nrho2(nx,ny,nz))

        nrho1 = 0.0
        nrho2 = 0.0
    end subroutine init_pre_post_density

    !<--------------------------------------------------------------------------
    !< Free the velocities of the previous time frame and post time frame.
    !<--------------------------------------------------------------------------
    subroutine free_pre_post_velocities
        implicit none
        deallocate(vdx1, vdy1, vdz1, vdx2, vdy2, vdz2)
    end subroutine free_pre_post_velocities

    !<--------------------------------------------------------------------------
    !< Free the velocities of the previous time frame and post time frame.
    !<--------------------------------------------------------------------------
    subroutine free_pre_post_u
        implicit none
        deallocate(udx1, udy1, udz1, udx2, udy2, udz2)
    end subroutine free_pre_post_u

    !<--------------------------------------------------------------------------
    !< Free the density of the previous time frame and post time frame.
    !<--------------------------------------------------------------------------
    subroutine free_pre_post_density
        implicit none
        deallocate(nrho1, nrho2)
    end subroutine free_pre_post_density

    !<--------------------------------------------------------------------------
    !< Read previous and post velocities. Only one of them is read in the
    !< first and last time frame.
    !< Input:
    !<   ct: current time frame.
    !<   fh: the file handlers for the velocities.
    !<--------------------------------------------------------------------------
    subroutine read_pre_post_velocities(ct, fh)
        use constants, only: fp
        use parameters, only: tp1
        use picinfo, only: domain, nt   ! Total number of output time frames.
        use mpi_datatype_fields, only: filetype_ghost, subsizes_ghost
        use mpi_io_module, only: read_data_mpi_io
        use pic_fields, only: vx, vy, vz
        implicit none
        integer, intent(in) :: ct
        integer, dimension(3), intent(in) :: fh
        integer(kind=MPI_OFFSET_KIND) :: disp, offset
        offset = 0
        if ((ct >= tp1) .and. (ct < nt)) then
            disp = domain%nx * domain%ny * domain%nz * sizeof(fp) * (ct-tp1+1)
            call read_data_mpi_io(fh(1), filetype_ghost, subsizes_ghost, &
                disp, offset, vdx2)
            call read_data_mpi_io(fh(2), filetype_ghost, subsizes_ghost, &
                disp, offset, vdy2)
            call read_data_mpi_io(fh(3), filetype_ghost, subsizes_ghost, &
                disp, offset, vdz2)
        else
            ! ct = nt, last time frame.
            vdx2 = vx
            vdy2 = vy
            vdz2 = vz
        endif

        if ((ct <= nt) .and. (ct > tp1)) then
            disp = domain%nx * domain%ny * domain%nz * sizeof(fp) * (ct-tp1-1)
            call read_data_mpi_io(fh(1), filetype_ghost, subsizes_ghost, &
                disp, offset, vdx1)
            call read_data_mpi_io(fh(2), filetype_ghost, subsizes_ghost, &
                disp, offset, vdy1)
            call read_data_mpi_io(fh(3), filetype_ghost, subsizes_ghost, &
                disp, offset, vdz1)
        else
            ! ct = tp1, The first time frame.
            vdx1 = vx
            vdy1 = vy
            vdz1 = vz
        endif
    end subroutine read_pre_post_velocities

    !<--------------------------------------------------------------------------
    !< Read previous and post densities. Only one of them is read in the
    !< first and last time frame.
    !< Input:
    !<   ct: current time frame.
    !<   fh: the file handlers for the velocities.
    !<--------------------------------------------------------------------------
    subroutine read_pre_post_density(ct, fh)
        use constants, only: fp
        use parameters, only: tp1
        use picinfo, only: domain, nt   ! Total number of output time frames.
        use mpi_datatype_fields, only: filetype_ghost, subsizes_ghost
        use mpi_io_module, only: read_data_mpi_io
        use pic_fields, only: num_rho
        implicit none
        integer, intent(in) :: ct
        integer, intent(in) :: fh
        integer(kind=MPI_OFFSET_KIND) :: disp, offset
        offset = 0
        if ((ct >= tp1) .and. (ct < nt)) then
            disp = domain%nx * domain%ny * domain%nz * sizeof(fp) * (ct-tp1+1)
            call read_data_mpi_io(fh, filetype_ghost, subsizes_ghost, &
                disp, offset, nrho2)
        else
            ! ct = nt, last time frame.
            nrho2 = num_rho
        endif

        if ((ct <= nt) .and. (ct > tp1)) then
            disp = domain%nx * domain%ny * domain%nz * sizeof(fp) * (ct-tp1-1)
            call read_data_mpi_io(fh, filetype_ghost, subsizes_ghost, &
                disp, offset, nrho1)
        else
            ! ct = tp1, The first time frame.
            nrho1 = num_rho
        endif
    end subroutine read_pre_post_density

    !<--------------------------------------------------------------------------
    !< Open u field files if all time frames are saved in the same file
    !< Input:
    !<   species: particle species
    !<   separated_pre_post: 1 for separated pre and post files, 0 for not
    !<--------------------------------------------------------------------------
    subroutine open_ufield_pre_post_single(species, separated_pre_post)
        use pic_fields, only: ufields_fh
        implicit none
        character(*), intent(in) :: species
        integer, intent(in) :: separated_pre_post
        character(len=256) :: fname
        character(len=16) :: cfname

        if (separated_pre_post == 0) then
            ufields_pre_fh = ufields_fh
            ufields_post_fh = ufields_fh
        else
            ufields_pre_fh = 0
            ufields_post_fh = 0
            fname = trim(adjustl(filepath))//'u'//species//'x_pre.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, &
                ufields_pre_fh(1))
            fname = trim(adjustl(filepath))//'u'//species//'y_pre.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, &
                ufields_pre_fh(2))
            fname = trim(adjustl(filepath))//'u'//species//'z_pre.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, &
                ufields_pre_fh(3))
            fname = trim(adjustl(filepath))//'u'//species//'x_post.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, &
                ufields_post_fh(1))
            fname = trim(adjustl(filepath))//'u'//species//'y_post.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, &
                ufields_post_fh(2))
            fname = trim(adjustl(filepath))//'u'//species//'z_post.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, &
                ufields_post_fh(3))
        endif
    end subroutine open_ufield_pre_post_single

    !<--------------------------------------------------------------------------
    !< Open u field files if different time frames are saved in different files
    !< Input:
    !<   species: particle species
    !<   tindex: time index for current time step
    !<   tindex_pre: time index for previous time step
    !<   tindex_post: time index for next time step
    !<   separated_pre_post: 1 for separated pre and post files, 0 for not
    !<--------------------------------------------------------------------------
    subroutine open_ufield_pre_post_multi(species, separated_pre_post, tindex, &
            tindex_pre, tindex_post)
        use pic_fields, only: ufields_fh
        implicit none
        character(*), intent(in) :: species
        integer, intent(in) :: tindex, tindex_pre, tindex_post, separated_pre_post
        character(len=256) :: fname
        character(len=16) :: cfname

        if (separated_pre_post == 0) then
            if (tindex_pre == tindex) then
                ufields_pre_fh = ufields_fh
            else
                ufields_pre_fh = 0
                write(cfname, "(I0)") tindex_pre
                fname = trim(adjustl(filepath))//'u'//species//'x_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, ufields_pre_fh(1))
                fname = trim(adjustl(filepath))//'u'//species//'y_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, ufields_pre_fh(2))
                fname = trim(adjustl(filepath))//'u'//species//'z_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, ufields_pre_fh(3))
            endif
            
            if (tindex_post == tindex) then
                ufields_post_fh = ufields_fh
            else
                ufields_post_fh = 0
                write(cfname, "(I0)") tindex_post
                fname = trim(adjustl(filepath))//'u'//species//'x_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, ufields_post_fh(1))
                fname = trim(adjustl(filepath))//'u'//species//'y_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, ufields_post_fh(2))
                fname = trim(adjustl(filepath))//'u'//species//'z_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, ufields_post_fh(3))
            endif
        else
            if (tindex_pre == tindex) then
                ufields_pre_fh = ufields_fh
            else
                ufields_pre_fh = 0
                write(cfname, "(I0)") tindex_pre
                fname = trim(adjustl(filepath))//'u'//species//'x_'//trim(cfname)//'_pre.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, ufields_pre_fh(1))
                fname = trim(adjustl(filepath))//'u'//species//'y_'//trim(cfname)//'_pre.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, ufields_pre_fh(2))
                fname = trim(adjustl(filepath))//'u'//species//'z_'//trim(cfname)//'_pre.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, ufields_pre_fh(3))
            endif
            if (tindex_post == tindex) then
                ufields_post_fh = ufields_fh
            else
                ufields_post_fh = 0
                write(cfname, "(I0)") tindex_post
                fname = trim(adjustl(filepath))//'u'//species//'x_'//trim(cfname)//'_post.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, ufields_post_fh(1))
                fname = trim(adjustl(filepath))//'u'//species//'y_'//trim(cfname)//'_post.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, ufields_post_fh(2))
                fname = trim(adjustl(filepath))//'u'//species//'z_'//trim(cfname)//'_post.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, ufields_post_fh(3))
            endif
        endif
    end subroutine open_ufield_pre_post_multi

    !<--------------------------------------------------------------------------
    !< Close u field files if different time frames are saved in different files
    !<--------------------------------------------------------------------------
    subroutine close_ufield_pre_post
        implicit none
        logical :: is_opened

        inquire(ufields_pre_fh(1), opened=is_opened)
        if (is_opened) then
            call MPI_FILE_CLOSE(ufields_pre_fh(1), ierror)
            call MPI_FILE_CLOSE(ufields_pre_fh(2), ierror)
            call MPI_FILE_CLOSE(ufields_pre_fh(3), ierror)
        endif

        inquire(ufields_post_fh(1), opened=is_opened)
        if (is_opened) then
            call MPI_FILE_CLOSE(ufields_post_fh(1), ierror)
            call MPI_FILE_CLOSE(ufields_post_fh(2), ierror)
            call MPI_FILE_CLOSE(ufields_post_fh(3), ierror)
        endif
    end subroutine close_ufield_pre_post

    !<--------------------------------------------------------------------------
    !< Read previous and post u field.
    !< Input:
    !<   tframe: time frame. It can be adjusted to make "disp"  0.
    !<   output_format: 2=file per slice, 1=all slices in one file
    !<   separated_pre_post: 1 for separated pre and post files, 0 for not
    !<--------------------------------------------------------------------------
    subroutine read_pre_post_u(tframe, output_format, separated_pre_post)
        use constants, only: fp
        use picinfo, only: domain
        use parameters, only: tp1, tp2
        use mpi_datatype_fields, only: filetype_ghost, subsizes_ghost
        use mpi_io_module, only: read_data_mpi_io
        use pic_fields, only: ux, uy, uz
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
            call read_data_mpi_io(ufields_post_fh(1), filetype_ghost, &
                subsizes_ghost, disp, offset, udx2)
            call read_data_mpi_io(ufields_post_fh(2), filetype_ghost, &
                subsizes_ghost, disp, offset, udy2)
            call read_data_mpi_io(ufields_post_fh(3), filetype_ghost, &
                subsizes_ghost, disp, offset, udz2)
        else
            ! tframe = tp2, last time frame.
            udx2 = ux
            udy2 = uy
            udz2 = uz
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
            call read_data_mpi_io(ufields_pre_fh(1), filetype_ghost, &
                subsizes_ghost, disp, offset, udx1)
            call read_data_mpi_io(ufields_pre_fh(2), filetype_ghost, &
                subsizes_ghost, disp, offset, udy1)
            call read_data_mpi_io(ufields_pre_fh(3), filetype_ghost, &
                subsizes_ghost, disp, offset, udz1)
        else
            ! tframe = tp1, The first time frame.
            udx1 = ux
            udy1 = uy
            udz1 = uz
        endif
    end subroutine read_pre_post_u

    !<--------------------------------------------------------------------------
    !< Shift u field to remove ghost cells at lower end along x-, y-,
    !< and z-directions.
    !<--------------------------------------------------------------------------
    subroutine shift_ufield_pre_post
        use mpi_topology, only: ht
        implicit none
        ! x-direction
        if (ht%ix > 0) then
            udx1(1:ht%nx, :, :) = udx1(2:ht%nx+1, :, :)
            udy1(1:ht%nx, :, :) = udy1(2:ht%nx+1, :, :)
            udz1(1:ht%nx, :, :) = udz1(2:ht%nx+1, :, :)
            udx2(1:ht%nx, :, :) = udx2(2:ht%nx+1, :, :)
            udy2(1:ht%nx, :, :) = udy2(2:ht%nx+1, :, :)
            udz2(1:ht%nx, :, :) = udz2(2:ht%nx+1, :, :)
        endif

        ! y-direction
        if (ht%iy > 0) then
            udx1(:, 1:ht%ny, :) = udx1(:, 2:ht%ny+1, :)
            udy1(:, 1:ht%ny, :) = udy1(:, 2:ht%ny+1, :)
            udz1(:, 1:ht%ny, :) = udz1(:, 2:ht%ny+1, :)
            udx2(:, 1:ht%ny, :) = udx2(:, 2:ht%ny+1, :)
            udy2(:, 1:ht%ny, :) = udy2(:, 2:ht%ny+1, :)
            udz2(:, 1:ht%ny, :) = udz2(:, 2:ht%ny+1, :)
        endif

        ! z-direction
        if (ht%iz > 0) then
            udx1(:, :, 1:ht%nz) = udx1(:, :, 2:ht%nz+1)
            udy1(:, :, 1:ht%nz) = udy1(:, :, 2:ht%nz+1)
            udz1(:, :, 1:ht%nz) = udz1(:, :, 2:ht%nz+1)
            udx2(:, :, 1:ht%nz) = udx2(:, :, 2:ht%nz+1)
            udy2(:, :, 1:ht%nz) = udy2(:, :, 2:ht%nz+1)
            udz2(:, :, 1:ht%nz) = udz2(:, :, 2:ht%nz+1)
        endif
    end subroutine shift_ufield_pre_post

end module pre_post_hydro
