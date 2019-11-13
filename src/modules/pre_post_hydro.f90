!<******************************************************************************
!< Module for hydro fields of previous and post time frames.
!<******************************************************************************
module pre_post_hydro
    use constants, only: fp
    use mpi_module
    use path_info, only: rootpath, filepath
    use mpi_info_module, only: fileinfo
    use mpi_io_module, only: open_data_mpi_io, read_data_mpi_io
    use hdf5_io, only: open_file_h5, close_file_h5, read_data_h5
    use mpi_datatype_fields, only: filetype_ghost, sizes_ghost, &
        subsizes_ghost, starts_ghost
    use particle_info, only: species, ptl_mass, ptl_charge
    use hdf5
    implicit none
    private
    public vdx1, vdy1, vdz1, vdx2, vdy2, vdz2
    public udx1, udy1, udz1, udx2, udy2, udz2
    public nrho1, nrho2

    public init_pre_post_v, free_pre_post_v, open_vfield_pre_post, &
        close_vfield_pre_post, read_pre_post_v, shift_vfield_pre_post
    public init_pre_post_u, free_pre_post_u, open_ufield_pre_post, &
        close_ufield_pre_post, read_pre_post_u, shift_ufield_pre_post
    public init_pre_post_density, free_pre_post_density, &
        open_num_density_pre_post, close_num_density_pre_post, &
        read_pre_post_density, shift_density_pre_post
    public open_hydro_files_pre_post_h5, close_hydro_files_pre_post_h5

    real(fp), allocatable, dimension(:, :, :) :: vdx1, vdy1, vdz1
    real(fp), allocatable, dimension(:, :, :) :: vdx2, vdy2, vdz2
    real(fp), allocatable, dimension(:, :, :) :: udx1, udy1, udz1  ! gamma * v
    real(fp), allocatable, dimension(:, :, :) :: udx2, udy2, udz2
    real(fp), allocatable, dimension(:, :, :) :: nrho1, nrho2
    integer, dimension(3) :: ufields_pre_fh, ufields_post_fh
    integer, dimension(3) :: vfields_pre_fh, vfields_post_fh
    integer :: nrho_pre_fh, nrho_post_fh
    integer(hid_t) :: hydro_file_pre_id, hydro_group_pre_id
    integer(hid_t) :: hydro_file_post_id, hydro_group_post_id

    interface open_ufield_pre_post
        module procedure &
            open_ufield_pre_post_single, open_ufield_pre_post_multi
    end interface open_ufield_pre_post

    interface open_vfield_pre_post
        module procedure &
            open_vfield_pre_post_single, open_vfield_pre_post_multi
    end interface open_vfield_pre_post

    interface open_num_density_pre_post
        module procedure &
            open_num_density_pre_post_single, open_num_density_pre_post_multi
    end interface open_num_density_pre_post

    interface init_pre_post_density
        module procedure &
            init_pre_post_density_noarg, init_pre_post_density_arg
    end interface init_pre_post_density

    interface read_pre_post_density
        module procedure &
            read_pre_post_density_same_file, read_pre_post_number_density
    end interface read_pre_post_density

    contains

    !<--------------------------------------------------------------------------
    !< Initialize velocities of the previous time frame and post time frame.
    !< They are going be be used to calculate polarization drift current.
    !<--------------------------------------------------------------------------
    subroutine init_pre_post_v(nx, ny, nz)
        use mpi_topology, only: htg
        implicit none
        integer, intent(in) :: nx, ny, nz

        allocate(vdx1(nx,ny,nz))
        allocate(vdy1(nx,ny,nz))
        allocate(vdz1(nx,ny,nz))
        allocate(vdx2(nx,ny,nz))
        allocate(vdy2(nx,ny,nz))
        allocate(vdz2(nx,ny,nz))

        vdx1 = 0.0; vdy1 = 0.0; vdz1 = 0.0
        vdx2 = 0.0; vdy2 = 0.0; vdz2 = 0.0
    end subroutine init_pre_post_v

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
    !< Old version without arguments
    !<--------------------------------------------------------------------------
    subroutine init_pre_post_density_noarg
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
    end subroutine init_pre_post_density_noarg

    !<--------------------------------------------------------------------------
    !< Initialize density of the previous time frame and post time frame.
    !< New version: with arguments
    !<--------------------------------------------------------------------------
    subroutine init_pre_post_density_arg(nx, ny, nz)
        implicit none
        integer, intent(in) :: nx, ny, nz

        allocate(nrho1(nx,ny,nz))
        allocate(nrho2(nx,ny,nz))

        nrho1 = 0.0
        nrho2 = 0.0
    end subroutine init_pre_post_density_arg

    !<--------------------------------------------------------------------------
    !< Free the velocities of the previous time frame and post time frame.
    !<--------------------------------------------------------------------------
    subroutine free_pre_post_v
        implicit none
        deallocate(vdx1, vdy1, vdz1, vdx2, vdy2, vdz2)
    end subroutine free_pre_post_v

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
    !< Open v field files if all time frames are saved in the same file
    !< Input:
    !<   species: particle species
    !<   separated_pre_post: 1 for separated pre and post files, 0 for not
    !<--------------------------------------------------------------------------
    subroutine open_vfield_pre_post_single(species, separated_pre_post)
        use pic_fields, only: vfields_fh
        implicit none
        character(*), intent(in) :: species
        integer, intent(in) :: separated_pre_post
        character(len=256) :: fname
        character(len=16) :: cfname

        if (separated_pre_post == 0) then
            vfields_pre_fh = vfields_fh
            vfields_post_fh = vfields_fh
        else
            vfields_pre_fh = 0
            vfields_post_fh = 0
            fname = trim(adjustl(filepath))//'v'//species//'x_pre.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, &
                vfields_pre_fh(1))
            fname = trim(adjustl(filepath))//'v'//species//'y_pre.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, &
                vfields_pre_fh(2))
            fname = trim(adjustl(filepath))//'v'//species//'z_pre.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, &
                vfields_pre_fh(3))
            fname = trim(adjustl(filepath))//'v'//species//'x_post.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, &
                vfields_post_fh(1))
            fname = trim(adjustl(filepath))//'v'//species//'y_post.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, &
                vfields_post_fh(2))
            fname = trim(adjustl(filepath))//'v'//species//'z_post.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, &
                vfields_post_fh(3))
        endif
    end subroutine open_vfield_pre_post_single

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
    !< Open number density files if all time frames are saved in the same file
    !< Input:
    !<   species: particle species
    !<   separated_pre_post: 1 for separated pre and post files, 0 for not
    !<--------------------------------------------------------------------------
    subroutine open_num_density_pre_post_single(species, separated_pre_post)
        use pic_fields, only: nrho_fh
        implicit none
        character(*), intent(in) :: species
        integer, intent(in) :: separated_pre_post
        character(len=256) :: fname
        character(len=16) :: cfname

        if (separated_pre_post == 0) then
            nrho_pre_fh = nrho_fh
            nrho_post_fh = nrho_fh
        else
            nrho_pre_fh = 0
            nrho_post_fh = 0
            fname = trim(adjustl(filepath))//'n'//species//'_pre.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, nrho_pre_fh)
            fname = trim(adjustl(filepath))//'n'//species//'_post.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, nrho_post_fh)
        endif
    end subroutine open_num_density_pre_post_single

    !<--------------------------------------------------------------------------
    !< Open v field files if different time frames are saved in different files
    !< Input:
    !<   species: particle species
    !<   tindex: time index for current time step
    !<   tindex_pre: time index for previous time step
    !<   tindex_post: time index for next time step
    !<   separated_pre_post: 1 for separated pre and post files, 0 for not
    !<--------------------------------------------------------------------------
    subroutine open_vfield_pre_post_multi(species, separated_pre_post, tindex, &
            tindex_pre, tindex_post)
        use pic_fields, only: vfields_fh
        implicit none
        character(*), intent(in) :: species
        integer, intent(in) :: tindex, tindex_pre, tindex_post, separated_pre_post
        character(len=256) :: fname
        character(len=16) :: cfname

        if (separated_pre_post == 0) then
            if (tindex_pre == tindex) then
                vfields_pre_fh = vfields_fh
            else
                vfields_pre_fh = 0
                write(cfname, "(I0)") tindex_pre
                fname = trim(adjustl(filepath))//'v'//species//'x_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, vfields_pre_fh(1))
                fname = trim(adjustl(filepath))//'v'//species//'y_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, vfields_pre_fh(2))
                fname = trim(adjustl(filepath))//'v'//species//'z_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, vfields_pre_fh(3))
            endif

            if (tindex_post == tindex) then
                vfields_post_fh = vfields_fh
            else
                vfields_post_fh = 0
                write(cfname, "(I0)") tindex_post
                fname = trim(adjustl(filepath))//'v'//species//'x_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, vfields_post_fh(1))
                fname = trim(adjustl(filepath))//'v'//species//'y_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, vfields_post_fh(2))
                fname = trim(adjustl(filepath))//'v'//species//'z_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, vfields_post_fh(3))
            endif
        else
            if (tindex_pre == tindex) then
                vfields_pre_fh = vfields_fh
            else
                vfields_pre_fh = 0
                write(cfname, "(I0)") tindex_pre
                fname = trim(adjustl(filepath))//'v'//species//'x_pre_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, vfields_pre_fh(1))
                fname = trim(adjustl(filepath))//'v'//species//'y_pre_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, vfields_pre_fh(2))
                fname = trim(adjustl(filepath))//'v'//species//'z_pre_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, vfields_pre_fh(3))
            endif
            if (tindex_post == tindex) then
                vfields_post_fh = vfields_fh
            else
                vfields_post_fh = 0
                write(cfname, "(I0)") tindex_post
                fname = trim(adjustl(filepath))//'v'//species//'x_post_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, vfields_post_fh(1))
                fname = trim(adjustl(filepath))//'v'//species//'y_post_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, vfields_post_fh(2))
                fname = trim(adjustl(filepath))//'v'//species//'z_post_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, vfields_post_fh(3))
            endif
        endif
    end subroutine open_vfield_pre_post_multi

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
            ufields_pre_fh = 0
            write(cfname, "(I0)") tindex_pre
            fname = trim(adjustl(filepath))//'u'//species//'x_pre_'//trim(cfname)//'.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, ufields_pre_fh(1))
            fname = trim(adjustl(filepath))//'u'//species//'y_pre_'//trim(cfname)//'.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, ufields_pre_fh(2))
            fname = trim(adjustl(filepath))//'u'//species//'z_pre_'//trim(cfname)//'.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, ufields_pre_fh(3))

            ufields_post_fh = 0
            write(cfname, "(I0)") tindex_post
            fname = trim(adjustl(filepath))//'u'//species//'x_post_'//trim(cfname)//'.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, ufields_post_fh(1))
            fname = trim(adjustl(filepath))//'u'//species//'y_post_'//trim(cfname)//'.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, ufields_post_fh(2))
            fname = trim(adjustl(filepath))//'u'//species//'z_post_'//trim(cfname)//'.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, ufields_post_fh(3))
        endif
    end subroutine open_ufield_pre_post_multi

    !<--------------------------------------------------------------------------
    !< Open number density files if different time frames are saved in different files
    !< Input:
    !<   species: particle species
    !<   tindex: time index for current time step
    !<   tindex_pre: time index for previous time step
    !<   tindex_post: time index for next time step
    !<   separated_pre_post: 1 for separated pre and post files, 0 for not
    !<--------------------------------------------------------------------------
    subroutine open_num_density_pre_post_multi(species, separated_pre_post, &
            tindex, tindex_pre, tindex_post)
        use pic_fields, only: nrho_fh
        implicit none
        character(*), intent(in) :: species
        integer, intent(in) :: tindex, tindex_pre, tindex_post, separated_pre_post
        character(len=256) :: fname
        character(len=16) :: cfname

        if (separated_pre_post == 0) then
            if (tindex_pre == tindex) then
                nrho_pre_fh = nrho_fh
            else
                nrho_pre_fh = 0
                write(cfname, "(I0)") tindex_pre
                fname = trim(adjustl(filepath))//'n'//species//'_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, nrho_pre_fh)
            endif

            if (tindex_post == tindex) then
                nrho_post_fh = nrho_fh
            else
                nrho_post_fh = 0
                write(cfname, "(I0)") tindex_post
                fname = trim(adjustl(filepath))//'n'//species//'_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, nrho_post_fh)
            endif
        else
            if (tindex_pre == tindex) then
                nrho_pre_fh = nrho_fh
            else
                nrho_pre_fh = 0
                write(cfname, "(I0)") tindex_pre
                fname = trim(adjustl(filepath))//'n'//species//'_pre_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, nrho_pre_fh)
            endif
            if (tindex_post == tindex) then
                nrho_post_fh = nrho_fh
            else
                nrho_post_fh = 0
                write(cfname, "(I0)") tindex_post
                fname = trim(adjustl(filepath))//'n'//species//'_post_'//trim(cfname)//'.gda'
                call open_data_mpi_io(fname, MPI_MODE_RDONLY, fileinfo, nrho_post_fh)
            endif
        endif
    end subroutine open_num_density_pre_post_multi

    !<--------------------------------------------------------------------------
    !< Open files containing hydro fields
    !< Inputs:
    !<   species: particle species ('e' or 'i')
    !<   tindex: the time index.
    !<   tindex_pre: time index for previous time step
    !<   tindex_post: time index for next time step
    !<--------------------------------------------------------------------------
    subroutine open_hydro_files_pre_post_h5(species, tindex, tindex_pre, tindex_post)
        use pic_fields, only: ehydro_file_id, ehydro_group_id, &
            Hhydro_file_id, Hhydro_group_id
        implicit none
        character(*), intent(in) :: species
        integer, intent(in) :: tindex, tindex_pre, tindex_post
        character(len=256) :: fname
        character(len=32) :: gname, cfname, sname
        integer :: ierror

        if (species .eq. 'e') then
            sname = "electron"
        else
            sname = "ion"
        endif

        if (tindex_pre == tindex) then
            if (species .eq. 'e') then
                hydro_file_pre_id = ehydro_file_id
                hydro_group_pre_id = ehydro_group_id
            else
                hydro_file_pre_id = Hhydro_file_id
                hydro_group_pre_id = Hhydro_group_id
            endif
        else
            hydro_file_pre_id = 0
            hydro_group_pre_id = 0
            write(cfname, "(I0)") tindex_pre
            fname = trim(adjustl(rootpath))//'hydro_hdf5/T.'//trim(cfname)//'/'
            fname = trim(adjustl(fname))//'hydro_'//trim(sname)//"_"//trim(cfname)//'.h5'
            call open_file_h5(trim(fname), H5F_ACC_RDONLY_F, hydro_file_pre_id, .true.)
            gname = "Timestep_"//trim(cfname)
            call h5gopen_f(hydro_file_pre_id, trim(gname), hydro_group_pre_id, ierror)
        endif

        if (tindex_post == tindex) then
            if (species .eq. 'e') then
                hydro_file_post_id = ehydro_file_id
                hydro_group_post_id = ehydro_group_id
            else
                hydro_file_post_id = Hhydro_file_id
                hydro_group_post_id = Hhydro_group_id
            endif
        else
            hydro_file_post_id = 0
            hydro_group_post_id = 0
            write(cfname, "(I0)") tindex_post
            fname = trim(adjustl(rootpath))//'hydro_hdf5/T.'//trim(cfname)//'/'
            fname = trim(adjustl(fname))//'hydro_'//trim(sname)//"_"//trim(cfname)//'.h5'
            call open_file_h5(trim(fname), H5F_ACC_RDONLY_F, hydro_file_post_id, .true.)
            gname = "Timestep_"//trim(cfname)
            call h5gopen_f(hydro_file_post_id, trim(gname), hydro_group_post_id, ierror)
        endif
    end subroutine open_hydro_files_pre_post_h5

    !<--------------------------------------------------------------------------
    !< Close u field files if different time frames are saved in different files
    !<--------------------------------------------------------------------------
    subroutine close_ufield_pre_post
        implicit none

        call MPI_FILE_CLOSE(ufields_pre_fh(1), ierror)
        call MPI_FILE_CLOSE(ufields_pre_fh(2), ierror)
        call MPI_FILE_CLOSE(ufields_pre_fh(3), ierror)

        call MPI_FILE_CLOSE(ufields_post_fh(1), ierror)
        call MPI_FILE_CLOSE(ufields_post_fh(2), ierror)
        call MPI_FILE_CLOSE(ufields_post_fh(3), ierror)
    end subroutine close_ufield_pre_post

    !<--------------------------------------------------------------------------
    !< Close v field files if different time frames are saved in different files
    !<--------------------------------------------------------------------------
    subroutine close_vfield_pre_post
        implicit none

        call MPI_FILE_CLOSE(vfields_pre_fh(1), ierror)
        call MPI_FILE_CLOSE(vfields_pre_fh(2), ierror)
        call MPI_FILE_CLOSE(vfields_pre_fh(3), ierror)

        call MPI_FILE_CLOSE(vfields_post_fh(1), ierror)
        call MPI_FILE_CLOSE(vfields_post_fh(2), ierror)
        call MPI_FILE_CLOSE(vfields_post_fh(3), ierror)
    end subroutine close_vfield_pre_post

    !<--------------------------------------------------------------------------
    !< Close number density files if different time frames are saved in different files
    !<--------------------------------------------------------------------------
    subroutine close_num_density_pre_post
        implicit none

        call MPI_FILE_CLOSE(nrho_pre_fh, ierror)
        call MPI_FILE_CLOSE(nrho_post_fh, ierror)
    end subroutine close_num_density_pre_post

    !<--------------------------------------------------------------------------
    !< Close files containing hydro fields
    !<--------------------------------------------------------------------------
    subroutine close_hydro_files_pre_post_h5(tindex, tindex_pre, tindex_post)
        implicit none
        integer :: error
        integer, intent(in) :: tindex, tindex_pre, tindex_post
        if (tindex /= tindex_pre) then
            call h5gclose_f(hydro_group_pre_id, error)
            call h5fclose_f(hydro_file_pre_id, error)
        endif
        if (tindex /= tindex_post) then
            call h5gclose_f(hydro_group_post_id, error)
            call h5fclose_f(hydro_file_post_id, error)
        endif
    end subroutine close_hydro_files_pre_post_h5

    !<--------------------------------------------------------------------------
    !< Read previous and post u field.
    !< Note that we assume that when reading from HDF5 files, we have read
    !< particle number density from the files
    !< Input:
    !<   tframe: time frame. It can be adjusted to make "disp"  0.
    !<   output_format: 2=file per slice, 1=all slices in one file
    !<   separated_pre_post: 1 for separated pre and post files, 0 for not
    !<   hdf5_flag(optional): whether to read to a HDF5 file
    !<   collective_io(optional): whether to read to data collectively
    !<--------------------------------------------------------------------------
    subroutine read_pre_post_u(tframe, output_format, separated_pre_post, &
            hdf5_flag, collective_io)
        use constants, only: fp
        use picinfo, only: domain
        use parameters, only: tp1, tp2
        use mpi_io_module, only: read_data_mpi_io
        use pic_fields, only: ux, uy, uz
        implicit none
        integer, intent(in) :: tframe, output_format, separated_pre_post
        logical, intent(in), optional :: hdf5_flag, collective_io
        integer(kind=MPI_OFFSET_KIND) :: disp, offset, disp0
        logical :: use_hdf5, use_collective_io
        integer(hid_t) :: px_id, py_id, pz_id
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
        if (use_hdf5) then
            call h5dopen_f(hydro_group_post_id, "px", px_id, error)
            call h5dopen_f(hydro_group_post_id, "py", py_id, error)
            call h5dopen_f(hydro_group_post_id, "pz", pz_id, error)
            call read_data_h5(px_id, dcount, doffset, dset_dims, &
                udx2, .true., use_collective_io)
            call read_data_h5(py_id, dcount, doffset, dset_dims, &
                udy2, .true., use_collective_io)
            call read_data_h5(pz_id, dcount, doffset, dset_dims, &
                udz2, .true., use_collective_io)
            udx2 = reshape(udx2, shape(udx2), order=[3, 2, 1])
            udy2 = reshape(udy2, shape(udy2), order=[3, 2, 1])
            udz2 = reshape(udz2, shape(udz2), order=[3, 2, 1])
            call h5dclose_f(px_id, error)
            call h5dclose_f(py_id, error)
            call h5dclose_f(pz_id, error)
            ! We assume that num_rho has been loaded from files
            where (nrho2 > 0.0)
                udx2 = (udx2/nrho2) / ptl_mass
                udy2 = (udy2/nrho2) / ptl_mass
                udz2 = (udz2/nrho2) / ptl_mass
            elsewhere
                udx2 = 0.0
                udy2 = 0.0
                udz2 = 0.0
            endwhere
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
            call read_data_mpi_io(ufields_post_fh(1), filetype_ghost, &
                subsizes_ghost, disp, offset, udx2)
            call read_data_mpi_io(ufields_post_fh(2), filetype_ghost, &
                subsizes_ghost, disp, offset, udy2)
            call read_data_mpi_io(ufields_post_fh(3), filetype_ghost, &
                subsizes_ghost, disp, offset, udz2)
        endif

        if (use_hdf5) then
            call h5dopen_f(hydro_group_pre_id, "px", px_id, error)
            call h5dopen_f(hydro_group_pre_id, "py", py_id, error)
            call h5dopen_f(hydro_group_pre_id, "pz", pz_id, error)
            call read_data_h5(px_id, dcount, doffset, dset_dims, &
                udx1, .true., use_collective_io)
            call read_data_h5(py_id, dcount, doffset, dset_dims, &
                udy1, .true., use_collective_io)
            call read_data_h5(pz_id, dcount, doffset, dset_dims, &
                udz1, .true., use_collective_io)
            udx1 = reshape(udx1, shape(udx1), order=[3, 2, 1])
            udy1 = reshape(udy1, shape(udy1), order=[3, 2, 1])
            udz1 = reshape(udz1, shape(udz1), order=[3, 2, 1])
            call h5dclose_f(px_id, error)
            call h5dclose_f(py_id, error)
            call h5dclose_f(pz_id, error)
            ! We assume that num_rho has been loaded from files
            where (nrho1 > 0.0)
                udx1 = (udx1/nrho1) / ptl_mass
                udy1 = (udy1/nrho1) / ptl_mass
                udz1 = (udz1/nrho1) / ptl_mass
            elsewhere
                udx1 = 0.0
                udy1 = 0.0
                udz1 = 0.0
            endwhere
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
            call read_data_mpi_io(ufields_pre_fh(1), filetype_ghost, &
                subsizes_ghost, disp, offset, udx1)
            call read_data_mpi_io(ufields_pre_fh(2), filetype_ghost, &
                subsizes_ghost, disp, offset, udy1)
            call read_data_mpi_io(ufields_pre_fh(3), filetype_ghost, &
                subsizes_ghost, disp, offset, udz1)
        endif
    end subroutine read_pre_post_u

    !<--------------------------------------------------------------------------
    !< Read previous and post v field.
    !< Note that we assume that when reading from HDF5 files, we have read
    !< particle number density from the files
    !< Input:
    !<   tframe: time frame. It can be adjusted to make "disp"  0.
    !<   output_format: 2=file per slice, 1=all slices in one file
    !<   separated_pre_post: 1 for separated pre and post files, 0 for not
    !<   hdf5_flag(optional): whether to read to a HDF5 file
    !<   collective_io(optional): whether to read to data collectively
    !<--------------------------------------------------------------------------
    subroutine read_pre_post_v(tframe, output_format, separated_pre_post, &
            hdf5_flag, collective_io)
        use constants, only: fp
        use picinfo, only: domain
        use parameters, only: tp1, tp2
        use mpi_io_module, only: read_data_mpi_io
        use pic_fields, only: vx, vy, vz
        implicit none
        integer, intent(in) :: tframe, output_format, separated_pre_post
        logical, intent(in), optional :: hdf5_flag, collective_io
        integer(kind=MPI_OFFSET_KIND) :: disp, offset, disp0
        logical :: use_hdf5, use_collective_io
        integer(hid_t) :: jx_id, jy_id, jz_id
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
        if (use_hdf5) then
            call h5dopen_f(hydro_group_post_id, "jx", jx_id, error)
            call h5dopen_f(hydro_group_post_id, "jy", jy_id, error)
            call h5dopen_f(hydro_group_post_id, "jz", jz_id, error)
            call read_data_h5(jx_id, dcount, doffset, dset_dims, &
                vdx2, .true., use_collective_io)
            call read_data_h5(jy_id, dcount, doffset, dset_dims, &
                vdy2, .true., use_collective_io)
            call read_data_h5(jz_id, dcount, doffset, dset_dims, &
                vdz2, .true., use_collective_io)
            vdx2 = reshape(vdx2, shape(vdx2), order=[3, 2, 1])
            vdy2 = reshape(vdy2, shape(vdy2), order=[3, 2, 1])
            vdz2 = reshape(vdz2, shape(vdz2), order=[3, 2, 1])
            call h5dclose_f(jx_id, error)
            call h5dclose_f(jy_id, error)
            call h5dclose_f(jz_id, error)
            ! We assume that num_rho has been loaded from files
            where (nrho2 > 0.0)
                vdx2 = (vdx2/nrho2) * ptl_charge
                vdy2 = (vdy2/nrho2) * ptl_charge
                vdz2 = (vdz2/nrho2) * ptl_charge
            elsewhere
                vdx2 = 0.0
                vdy2 = 0.0
                vdz2 = 0.0
            endwhere
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
            call read_data_mpi_io(vfields_post_fh(1), filetype_ghost, &
                subsizes_ghost, disp, offset, vdx2)
            call read_data_mpi_io(vfields_post_fh(2), filetype_ghost, &
                subsizes_ghost, disp, offset, vdy2)
            call read_data_mpi_io(vfields_post_fh(3), filetype_ghost, &
                subsizes_ghost, disp, offset, vdz2)
        endif

        if (use_hdf5) then
            call h5dopen_f(hydro_group_pre_id, "jx", jx_id, error)
            call h5dopen_f(hydro_group_pre_id, "jy", jy_id, error)
            call h5dopen_f(hydro_group_pre_id, "jz", jz_id, error)
            call read_data_h5(jx_id, dcount, doffset, dset_dims, &
                vdx1, .true., use_collective_io)
            call read_data_h5(jy_id, dcount, doffset, dset_dims, &
                vdy1, .true., use_collective_io)
            call read_data_h5(jz_id, dcount, doffset, dset_dims, &
                vdz1, .true., use_collective_io)
            vdx1 = reshape(vdx1, shape(vdx1), order=[3, 2, 1])
            vdy1 = reshape(vdy1, shape(vdy1), order=[3, 2, 1])
            vdz1 = reshape(vdz1, shape(vdz1), order=[3, 2, 1])
            call h5dclose_f(jx_id, error)
            call h5dclose_f(jy_id, error)
            call h5dclose_f(jz_id, error)
            ! We assume that num_rho has been loaded from files
            where (nrho1 > 0.0)
                vdx1 = (vdx1/nrho1) * ptl_charge
                vdy1 = (vdy1/nrho1) * ptl_charge
                vdz1 = (vdz1/nrho1) * ptl_charge
            elsewhere
                vdx1 = 0.0
                vdy1 = 0.0
                vdz1 = 0.0
            endwhere
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
            call read_data_mpi_io(vfields_pre_fh(1), filetype_ghost, &
                subsizes_ghost, disp, offset, vdx1)
            call read_data_mpi_io(vfields_pre_fh(2), filetype_ghost, &
                subsizes_ghost, disp, offset, vdy1)
            call read_data_mpi_io(vfields_pre_fh(3), filetype_ghost, &
                subsizes_ghost, disp, offset, vdz1)
        endif
    end subroutine read_pre_post_v

    !<--------------------------------------------------------------------------
    !< Read previous and post density field
    !< Input:
    !<   tframe: time frame. It can be adjusted to make "disp"  0.
    !<   output_format: 2=file per slice, 1=all slices in one file
    !<   separated_pre_post: 1 for separated pre and post files, 0 for not
    !<   hdf5_flag(optional): whether to read to a HDF5 file
    !<   collective_io(optional): whether to read to data collectively
    !<--------------------------------------------------------------------------
    subroutine read_pre_post_number_density(tframe, output_format, &
            separated_pre_post, hdf5_flag, collective_io)
        use constants, only: fp
        use picinfo, only: domain
        use parameters, only: tp1, tp2
        use mpi_io_module, only: read_data_mpi_io
        use pic_fields, only: num_rho
        implicit none
        integer, intent(in) :: tframe, output_format, separated_pre_post
        logical, intent(in), optional :: hdf5_flag, collective_io
        integer(kind=MPI_OFFSET_KIND) :: disp, offset, disp0
        logical :: use_hdf5, use_collective_io
        integer(hid_t) :: nrho_id
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
        if (use_hdf5) then
            call h5dopen_f(hydro_group_post_id, "rho", nrho_id, error)
            call read_data_h5(nrho_id, dcount, doffset, dset_dims, &
                nrho2, .true., use_collective_io)
            nrho2 = reshape(nrho2, shape(nrho2), order=[3, 2, 1])
            call h5dclose_f(nrho_id, error)
            nrho2 = abs(nrho2 / ptl_charge)
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
            call read_data_mpi_io(nrho_post_fh, filetype_ghost, &
                subsizes_ghost, disp, offset, nrho2)
        endif

        if (use_hdf5) then
            call h5dopen_f(hydro_group_pre_id, "rho", nrho_id, error)
            call read_data_h5(nrho_id, dcount, doffset, dset_dims, &
                nrho1, .true., use_collective_io)
            nrho1 = reshape(nrho1, shape(nrho1), order=[3, 2, 1])
            call h5dclose_f(nrho_id, error)
            nrho1 = abs(nrho1 / ptl_charge)
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
            call read_data_mpi_io(nrho_pre_fh, filetype_ghost, &
                subsizes_ghost, disp, offset, nrho1)
        endif
    end subroutine read_pre_post_number_density

    !<--------------------------------------------------------------------------
    !< Read previous and post densities. Only one of them is read in the
    !< first and last time frame.
    !< Input:
    !<   ct: current time frame.
    !<   fh: the file handlers for the velocities.
    !<--------------------------------------------------------------------------
    subroutine read_pre_post_density_same_file(ct, fh)
        use constants, only: fp
        use parameters, only: tp1
        use picinfo, only: domain, nt   ! Total number of output time frames.
        use mpi_io_module, only: read_data_mpi_io
        use pic_fields, only: num_rho
        implicit none
        integer, intent(in) :: ct
        integer, intent(in) :: fh
        integer(kind=MPI_OFFSET_KIND) :: disp, offset, disp0
        offset = 0
        disp0 = domain%nx * domain%ny * domain%nz * sizeof(fp)
        if ((ct >= tp1) .and. (ct < nt)) then
            disp = disp0 * (ct-tp1+1)
            call read_data_mpi_io(fh, filetype_ghost, subsizes_ghost, &
                disp, offset, nrho2)
        else
            ! ct = nt, last time frame.
            nrho2 = num_rho
        endif

        if ((ct <= nt) .and. (ct > tp1)) then
            disp = disp0 * (ct-tp1-1)
            call read_data_mpi_io(fh, filetype_ghost, subsizes_ghost, &
                disp, offset, nrho1)
        else
            ! ct = tp1, The first time frame.
            nrho1 = num_rho
        endif
    end subroutine read_pre_post_density_same_file

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

    !<--------------------------------------------------------------------------
    !< Shift v field to remove ghost cells at lower end along x-, y-,
    !< and z-directions.
    !<--------------------------------------------------------------------------
    subroutine shift_vfield_pre_post
        use mpi_topology, only: ht
        implicit none
        ! x-direction
        if (ht%ix > 0) then
            vdx1(1:ht%nx, :, :) = vdx1(2:ht%nx+1, :, :)
            vdy1(1:ht%nx, :, :) = vdy1(2:ht%nx+1, :, :)
            vdz1(1:ht%nx, :, :) = vdz1(2:ht%nx+1, :, :)
            vdx2(1:ht%nx, :, :) = vdx2(2:ht%nx+1, :, :)
            vdy2(1:ht%nx, :, :) = vdy2(2:ht%nx+1, :, :)
            vdz2(1:ht%nx, :, :) = vdz2(2:ht%nx+1, :, :)
        endif

        ! y-direction
        if (ht%iy > 0) then
            vdx1(:, 1:ht%ny, :) = vdx1(:, 2:ht%ny+1, :)
            vdy1(:, 1:ht%ny, :) = vdy1(:, 2:ht%ny+1, :)
            vdz1(:, 1:ht%ny, :) = vdz1(:, 2:ht%ny+1, :)
            vdx2(:, 1:ht%ny, :) = vdx2(:, 2:ht%ny+1, :)
            vdy2(:, 1:ht%ny, :) = vdy2(:, 2:ht%ny+1, :)
            vdz2(:, 1:ht%ny, :) = vdz2(:, 2:ht%ny+1, :)
        endif

        ! z-direction
        if (ht%iz > 0) then
            vdx1(:, :, 1:ht%nz) = vdx1(:, :, 2:ht%nz+1)
            vdy1(:, :, 1:ht%nz) = vdy1(:, :, 2:ht%nz+1)
            vdz1(:, :, 1:ht%nz) = vdz1(:, :, 2:ht%nz+1)
            vdx2(:, :, 1:ht%nz) = vdx2(:, :, 2:ht%nz+1)
            vdy2(:, :, 1:ht%nz) = vdy2(:, :, 2:ht%nz+1)
            vdz2(:, :, 1:ht%nz) = vdz2(:, :, 2:ht%nz+1)
        endif
    end subroutine shift_vfield_pre_post

    !<--------------------------------------------------------------------------
    !< Shift number density to remove ghost cells at lower end along x-, y-,
    !< and z-directions.
    !<--------------------------------------------------------------------------
    subroutine shift_density_pre_post
        use mpi_topology, only: ht
        implicit none
        ! x-direction
        if (ht%ix > 0) then
            nrho1(1:ht%nx, :, :) = nrho1(2:ht%nx+1, :, :)
            nrho2(1:ht%nx, :, :) = nrho2(2:ht%nx+1, :, :)
        endif

        ! y-direction
        if (ht%iy > 0) then
            nrho1(:, 1:ht%ny, :) = nrho1(:, 2:ht%ny+1, :)
            nrho2(:, 1:ht%ny, :) = nrho2(:, 2:ht%ny+1, :)
        endif

        ! z-direction
        if (ht%iz > 0) then
            nrho1(:, :, 1:ht%nz) = nrho1(:, :, 2:ht%nz+1)
            nrho2(:, :, 1:ht%nz) = nrho2(:, :, 2:ht%nz+1)
        endif
    end subroutine shift_density_pre_post
end module pre_post_hydro
