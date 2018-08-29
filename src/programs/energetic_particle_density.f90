!<******************************************************************************
!< Program for accumulating high-energy particle density.
!<******************************************************************************
program energetic_particle_density
    use constants, only: fp, dp
    use mpi_module
    use path_info, only: set_filepath
    use particle_info, only: species, ptl_mass, ptl_charge
    use parameters, only: tp1, tp2
    use configuration_translate, only: output_format
    use particle_module, only: particle
    use hdf5
    implicit none
    character(len=256) :: rootpath
    integer :: tstart, tend, tinterval, tframe, fields_interval
    integer :: reduce_factor, nbands
    integer :: nx, ny, nz, filetype
    integer, dimension(3) :: sizes, subsizes, starts
    integer :: xreduce, yreduce, zreduce
    ! The energy bands will be: [0, starting_ene],
    ! [starting_ene, power_base*starting_ene] ...
    real(fp) :: starting_ene, power_base
    type(particle), allocatable, dimension(:) :: ptls

    ! Particles in HDF5 format
    integer, allocatable, dimension(:) :: np_local
    integer(hsize_t), allocatable, dimension(:) :: offset_local
    logical :: particle_hdf5, parallel_read, collective_io
    integer, parameter :: num_dset = 8
    integer(hid_t), dimension(num_dset) :: dset_ids
    integer(hid_t) :: file_id, group_id
    integer(hid_t) :: filespace
    integer(hsize_t), dimension(1) :: dset_dims, dset_dims_max
    integer :: t1, t2, clock_rate, clock_max
    real(fp), allocatable, dimension(:, :, :, :) :: nrho

    call MPI_INIT(ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)

    call system_clock(t1, clock_rate, clock_max)

    call get_cmd_args

    call init_analysis

    call accumulate_density

    call end_analysis

    call system_clock(t2, clock_rate, clock_max)
    if (myid == master) then
        write (*, *) 'Elapsed real time = ', real(t2 - t1) / real(clock_rate)
    endif

    call MPI_FINALIZE(ierr)

    contains

    !<--------------------------------------------------------------------------
    !< Accumulate the high-energy particle densities at different band
    !<--------------------------------------------------------------------------
    subroutine accumulate_density
        use picinfo, only: domain, vthe, vthi
        use topology_translate, only: ht_translate => ht
        use mpi_topology, only: ht
        use file_header, only: pheader
        use rank_index_mapping, only: index_to_rank
        implicit none
        integer :: dom_x, dom_y, dom_z
        integer :: tp_emf, tindex
        integer :: t1, t2, t3, t4, clock_rate, clock_max
        integer :: n, iptl, nptl
        real(fp) :: dx_domain, dy_domain, dz_domain
        character(len=16) :: cid
        type(particle) :: ptl
        integer :: icell, nxg, nyg, nzg, ino, jno, kno
        integer :: ibin, xshift, yshift, zshift
        real(fp) :: vth, eth, gama, gama_norm, ux, uy, uz
        real(fp) :: emin_log, delog, weight
        real(fp) :: idv

        ! Thermal energy
        if (species == 'e') then
            vth = vthe
        else
            vth = vthi
        endif
        gama = 1.0 / sqrt(1.0 - 3 * vth**2)
        eth = gama - 1.0

        ! Logarithmic parameter
        emin_log = log10(starting_ene)
        delog = log10(power_base)

        if (domain%nx == 1) then
            nx = 1
            xreduce = 1
        else
            nx = ht%nx / reduce_factor
            xreduce = reduce_factor
        endif
        if (domain%ny == 1) then
            ny = 1
            yreduce = 1
        else
            ny = ht%ny / reduce_factor
            yreduce = reduce_factor
        endif
        if (domain%nz == 1) then
            nz = 1
            zreduce = 1
        else
            nz = ht%nz / reduce_factor
            zreduce = reduce_factor
        endif
        if ((nx * xreduce .ne. ht%nx) .or. &
            (ny * yreduce .ne. ht%ny) .or. &
            (nz * zreduce .ne. ht%nz)) then
            if (myid == master) print *, "invalid converter topology"
            call end_analysis
            call MPI_FINALIZE(ierr)
            stop
        endif
        call mpi_datatype
        call init_densities(nx, ny, nz)
        nxg = domain%pic_nx + 2  ! Including ghost cells
        nyg = domain%pic_ny + 2
        nzg = domain%pic_nz + 2

        call system_clock(t1, clock_rate, clock_max)
        do tframe = tstart, tend, tinterval
            if (myid == master) print*, tframe
            call set_densities_zero
            tp_emf = tframe / fields_interval + 1

            ! Time frame and interval
            tindex = domain%fields_interval * (tp_emf - tp1)

            dx_domain = domain%lx_de / domain%pic_tx
            dy_domain = domain%ly_de / domain%pic_ty
            dz_domain = domain%lz_de / domain%pic_tz
            idv = 1.0 / (domain%dx * domain%dy * domain%dz)
            idv = idv / (xreduce * yreduce * zreduce)

            ! Particles are saved in HDF5
            if (particle_hdf5) then
                call system_clock(t3, clock_rate, clock_max)
                call get_np_local_vpic(tframe, species)
                call open_particle_file_h5(tframe, species)
                call system_clock(t4, clock_rate, clock_max)
                if (myid == master) then
                    write (*, *) 'Time for openning HDF5 = ', &
                        real(t4 - t3) / real(clock_rate)
                endif
            endif

            call system_clock(t3, clock_rate, clock_max)
            do dom_z = ht_translate%start_z, ht_translate%stop_z
                zshift = (dom_z - ht_translate%start_z) * domain%pic_nz
                do dom_y = ht_translate%start_y, ht_translate%stop_y
                    yshift = (dom_y - ht_translate%start_y) * domain%pic_ny
                    do dom_x = ht_translate%start_x, ht_translate%stop_x
                        xshift = (dom_x - ht_translate%start_x) * domain%pic_nx
                        call index_to_rank(dom_x, dom_y, dom_z, domain%pic_tx, &
                            domain%pic_ty, domain%pic_tz, n)
                        write(cid, "(I0)") n - 1

                        if (particle_hdf5) then
                            if (parallel_read) then
                                call read_particle_h5_parallel(n - 1)
                            else
                                call read_particle_h5(n - 1)
                            endif
                            nptl = np_local(n)
                        else
                            call read_particle_binary(tindex, species, cid)
                            nptl = pheader%dim
                        endif

                        do iptl = 1, nptl, 1
                            ptl = ptls(iptl)
                            icell = ptl%icell
                            kno = icell / (nxg*nyg)          ! [1,nzg-2]
                            jno = mod(icell, nxg*nyg) / nxg  ! [1,nyg-2]
                            ino = mod(icell, nxg)            ! [1,nxg-2]
                            ux = ptl%vx  ! v in ptl is actually gamma*v
                            uy = ptl%vy
                            uz = ptl%vz
                            weight = abs(ptl%q)
                            gama = sqrt(1.0 + ux**2 + uy**2 + uz**2)
                            gama_norm = log10((gama - 1) / eth)
                            if (gama_norm > emin_log) then
                                ibin = ceiling((gama_norm - emin_log) / delog) + 1
                            else
                                ibin = 1
                            endif
                            if (ibin > nbands) ibin = nbands
                            ino = (ino + xshift + xreduce - 1) / xreduce
                            jno = (jno + yshift + yreduce - 1) / yreduce
                            kno = (kno + zshift + zreduce - 1) / zreduce
                            nrho(ino, jno, kno, ibin) = &
                                nrho(ino, jno, kno, ibin) + weight
                        enddo ! Loop over particles
                        deallocate(ptls)
                    enddo ! x
                enddo ! y
            enddo ! z
            nrho = nrho * idv
            call system_clock(t4, clock_rate, clock_max)
            if (myid == master) then
                write (*, *) 'Time for computing = ', &
                    real(t4 - t3) / real(clock_rate)
            endif

            ! Particles are saved in HDF5
            if (particle_hdf5) then
                call system_clock(t3, clock_rate, clock_max)
                call free_np_offset_local
                call close_particle_file_h5
                call system_clock(t4, clock_rate, clock_max)
                if (myid == master) then
                    write (*, *) 'Time for closing HDF5 = ', &
                        real(t4 - t3) / real(clock_rate)
                endif
            endif

            call system_clock(t3, clock_rate, clock_max)
            if (output_format /= 0) then
                call save_densities(tp1, tframe)
            else
                call save_densities(tp_emf, tframe)
            endif
            call system_clock(t4, clock_rate, clock_max)
            if (myid == master) then
                write (*, *) 'Time for saving data = ', &
                    real(t4 - t3) / real(clock_rate)
            endif

            call system_clock(t2, clock_rate, clock_max)
            if (myid == master) then
                write (*, *) 'Time for this step = ', &
                    real(t2 - t1) / real(clock_rate)
            endif
            t1 = t2
        enddo  ! Time loop
        call free_densities
        call MPI_TYPE_FREE(filetype, ierror)
    end subroutine accumulate_density

    !<--------------------------------------------------------------------------
    !< Initialize arrays for energetic particle densities
    !<--------------------------------------------------------------------------
    subroutine init_densities(nx, ny, nz)
        implicit none
        integer, intent(in) :: nx, ny, nz
        allocate(nrho(nx, ny, nz, nbands))
        call set_densities_zero
    end subroutine init_densities

    !<--------------------------------------------------------------------------
    !< Set densities to zeros
    !<--------------------------------------------------------------------------
    subroutine set_densities_zero
        implicit none
        nrho = 0.0
    end subroutine set_densities_zero

    !<--------------------------------------------------------------------------
    !< Free arrays for energetic particle densities
    !<--------------------------------------------------------------------------
    subroutine free_densities
        implicit none
        deallocate(nrho)
    end subroutine free_densities

    !<--------------------------------------------------------------------------
    !< Set MPI datatype
    !<--------------------------------------------------------------------------
    subroutine mpi_datatype
        use picinfo, only: domain
        use mpi_topology, only: ht
        use mpi_datatype_module, only: set_mpi_datatype
        implicit none

        sizes(1) = domain%nx / xreduce
        sizes(2) = domain%ny / yreduce
        sizes(3) = domain%nz / zreduce
        subsizes(1) = nx
        subsizes(2) = ny
        subsizes(3) = nz
        starts(1) = nx * ht%ix
        starts(2) = ny * ht%iy
        starts(3) = nz * ht%iz

        filetype = set_mpi_datatype(sizes, subsizes, starts)

    end subroutine mpi_datatype

    !<--------------------------------------------------------------------------
    !< Save densities
    !<--------------------------------------------------------------------------
    subroutine save_densities(tframe, tindex)
        use picinfo, only: domain
        use mpi_info_module, only: fileinfo
        use mpi_io_module, only: open_data_mpi_io, write_data_mpi_io
        implicit none
        integer, intent(in) :: tframe, tindex
        integer :: fh, posf
        character(len=16) :: tindex_str, iband_str
        character(len=256) :: fdir, fname, cmd
        integer(kind=MPI_OFFSET_KIND) :: disp, offset
        integer :: nxr, nyr, nzr
        integer :: iband
        logical :: dir_e
        fdir = trim(adjustl(rootpath))//'data-smooth2/'
        inquire(file=trim(fdir), exist=dir_e)
        if (.not. dir_e) then
            cmd = 'mkdir -p '//trim(fdir)
            call system(cmd)
        endif
        if (myid == master) then
            print*, "Saving energetic-particle density..."
        endif

        write(tindex_str, "(I0)") tindex
        nxr = domain%nx / xreduce
        nyr = domain%ny / yreduce
        nzr = domain%nz / zreduce
        disp = nxr * nyr * nzr * sizeof(MPI_REAL) * (tframe-tp1)
        offset = 0
        do iband = 1, nbands
            write(iband_str, "(I0)") iband - 1
            fname = trim(fdir)//'n'//species//'_'//iband_str
            fname = trim(fname)//'_'//trim(tindex_str)//'.gda'
            call open_data_mpi_io(fname, MPI_MODE_RDWR+MPI_MODE_CREATE, &
                fileinfo, fh)
            call write_data_mpi_io(fh, filetype, subsizes, &
                disp, offset, nrho(:, :, :, iband))
            call MPI_FILE_CLOSE(fh, ierror)
        enddo
    end subroutine save_densities

    !<--------------------------------------------------------------------------
    !< Initialize the analysis.
    !<--------------------------------------------------------------------------
    subroutine init_analysis
        use mpi_topology, only: set_mpi_topology, htg
        use mpi_datatype_fields, only: set_mpi_datatype_fields
        use mpi_info_module, only: set_mpi_info
        use particle_info, only: get_ptl_mass_charge
        use path_info, only: get_file_paths
        use picinfo, only: read_domain, broadcast_pic_info, &
                get_total_time_frames, get_energy_band_number, &
                read_thermal_params, calc_energy_interval, nbands, &
                write_pic_info, domain
        use configuration_translate, only: read_configuration
        use topology_translate, only: set_topology, set_start_stop_cells
        use mpi_io_translate, only: set_mpi_io
        use parameters, only: get_relativistic_flag, get_start_end_time_points, tp2
        use neighbors_module, only: init_neighbors, get_neighbors
        implicit none
        integer :: nx, ny, nz

        call get_file_paths(rootpath)
        if (myid == master) then
            call read_domain
        endif
        call broadcast_pic_info
        call get_ptl_mass_charge(species)
        call get_start_end_time_points
        call get_relativistic_flag
        ! call get_energy_band_number
        call read_thermal_params
        if (nbands > 0) then
            call calc_energy_interval
        endif
        call read_configuration
        call get_total_time_frames(tp2)
        call set_topology
        call set_start_stop_cells
        call set_mpi_io

        call set_mpi_topology(1)   ! MPI topology
        call set_mpi_datatype_fields
        call set_mpi_info

        call init_neighbors(htg%nx, htg%ny, htg%nz)
        call get_neighbors

    end subroutine init_analysis

    !!--------------------------------------------------------------------------
    !! End the analysis by free the memory.
    !!--------------------------------------------------------------------------
    subroutine end_analysis
        use topology_translate, only: free_start_stop_cells
        use mpi_io_translate, only: datatype
        use mpi_info_module, only: fileinfo
        use neighbors_module, only: free_neighbors
        use mpi_datatype_fields, only: filetype_ghost, filetype_nghost
        implicit none
        call free_neighbors
        call free_start_stop_cells
        call MPI_TYPE_FREE(datatype, ierror)
        call MPI_INFO_FREE(fileinfo, ierror)
        call MPI_TYPE_FREE(filetype_ghost, ierror)
        call MPI_TYPE_FREE(filetype_nghost, ierror)
    end subroutine end_analysis

    !<--------------------------------------------------------------------------
    !< Read particle data in binary format
    !<--------------------------------------------------------------------------
    subroutine read_particle_binary(tindex, species, cid)
        use particle_file, only: open_particle_file, close_particle_file, fh
        use file_header, only: pheader
        implicit none
        integer, intent(in) :: tindex
        character(*), intent(in) :: species
        character(*), intent(in) :: cid
        integer :: IOstatus
        ! Read particle data
        if (species == 'e') then
            call open_particle_file(tindex, species, cid)
        else
            call open_particle_file(tindex, 'h', cid)
        endif
        allocate(ptls(pheader%dim))
        read(fh, IOSTAT=IOstatus) ptls
        call close_particle_file
    end subroutine read_particle_binary

    !<--------------------------------------------------------------------------
    !< Get commandline arguments
    !<--------------------------------------------------------------------------
    subroutine get_cmd_args
        use flap                                !< FLAP package
        use penf
        implicit none
        type(command_line_interface) :: cli     !< Command Line Interface (CLI).
        integer(I4P)                 :: error   !< Error trapping flag.
        call cli%init(progname = 'energetic_particle_density', &
            authors     = 'Xiaocan Li', &
            help        = 'Usage: ', &
            description = 'Accumulate energetic particle density', &
            examples    = ['energetic_particle_density -rp rootpath'])
        call cli%add(switch='--rootpath', switch_ab='-rp', &
            help='simulation root path', required=.true., act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--tstart', switch_ab='-ts', &
            help='Starting time frame', required=.false., act='store', &
            def='0', error=error)
        if (error/=0) stop
        call cli%add(switch='--tend', switch_ab='-te', help='Last time frame', &
            required=.true., act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--tinterval', switch_ab='-ti', help='Time interval', &
            required=.true., act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--fields_interval', switch_ab='-fi', &
            help='Time interval for PIC fields', &
            required=.true., act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--species', switch_ab='-sp', &
            help="Particle species: 'e' or 'h'", required=.false., &
            act='store', def='e', error=error)
        if (error/=0) stop
        call cli%add(switch='--particle_hdf5', switch_ab='-ph', &
            help='Whether particles are saved in HDF5', &
            required=.false., act='store_true', def='.false.', error=error)
        if (error/=0) stop
        call cli%add(switch='--parallel_read', switch_ab='-pr', &
            help='Whether to read HDF5 partile file in parallel', &
            required=.false., act='store_true', def='.false.', error=error)
        if (error/=0) stop
        call cli%add(switch='--collective_io', switch_ab='-ci', &
            help='Whether to use collective IO to read HDF5 partile file', &
            required=.false., act='store_true', def='.false.', error=error)
        if (error/=0) stop
        call cli%add(switch='--reduce_factor', switch_ab='-rf', &
            help='Reduce factor of the data along each direction', &
            required=.false., def='4', act='store', error=error)
        call cli%add(switch='--starting_ene', switch_ab='-se', &
            help='Staring energy of energy bands', &
            required=.false., def='10.0', act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--power_base', switch_ab='-pb', &
            help='The base of power increment', &
            required=.false., def='2.0', act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--nbands', switch_ab='-nb', &
            help='Number of energy bands', &
            required=.false., def='7', act='store', error=error)
        if (error/=0) stop
        call cli%get(switch='-rp', val=rootpath, error=error)
        if (error/=0) stop
        call cli%get(switch='-ts', val=tstart, error=error)
        if (error/=0) stop
        call cli%get(switch='-te', val=tend, error=error)
        if (error/=0) stop
        call cli%get(switch='-ti', val=tinterval, error=error)
        if (error/=0) stop
        call cli%get(switch='-fi', val=fields_interval, error=error)
        if (error/=0) stop
        call cli%get(switch='-sp', val=species, error=error)
        if (error/=0) stop
        call cli%get(switch='-ph', val=particle_hdf5, error=error)
        if (error/=0) stop
        call cli%get(switch='-pr', val=parallel_read, error=error)
        if (error/=0) stop
        call cli%get(switch='-ci', val=collective_io, error=error)
        if (error/=0) stop
        call cli%get(switch='-rf', val=reduce_factor, error=error)
        if (error/=0) stop
        call cli%get(switch='-se', val=starting_ene, error=error)
        if (error/=0) stop
        call cli%get(switch='-pb', val=power_base, error=error)
        if (error/=0) stop
        call cli%get(switch='-nb', val=nbands, error=error)
        if (error/=0) stop

        if (myid == 0) then
            print '(A,A)', ' The simulation rootpath: ', trim(adjustl(rootpath))
            print '(A,I0,A,I0,A,I0)', ' Min, max and interval: ', &
                tstart, ' ', tend, ' ', tinterval
            print '(A,I0)', ' Time interval for electric and magnetic fields: ', &
                fields_interval
            if (species == 'e') then
                print '(A,A)', ' Particle: electron'
            else if (species == 'h' .or. species == 'i') then
                print '(A,A)', ' Particle: ion'
            endif
            if (particle_hdf5) then
                print '(A)', ' Particles are saved in HDF5 format'
                if (parallel_read) then
                    print '(A)', ' Read HDF5 particle file in parallel'
                    if (collective_io) then
                        print '(A)', ' Using colletive IO to read HDF5 particle file'
                    endif
                endif
            endif
            print '(A,F)', ' Starting energy/Initial thermal energy: ', starting_ene
            print '(A,F)', ' Power base: ', power_base
            print '(A,I0)', ' Number of energy bands: ', nbands
        endif
    end subroutine get_cmd_args

    !<--------------------------------------------------------------------------
    !< Initialize the np_local and offset_local array
    !<--------------------------------------------------------------------------
    subroutine init_np_offset_local(dset_dims)
        implicit none
        integer(hsize_t), dimension(1), intent(in) :: dset_dims
        allocate(np_local(dset_dims(1)))
        allocate(offset_local(dset_dims(1)))
        np_local = 0
        offset_local = 0
    end subroutine init_np_offset_local

    !<--------------------------------------------------------------------------
    !< Free the np_local and offset_local array
    !<--------------------------------------------------------------------------
    subroutine free_np_offset_local
        implicit none
        deallocate(np_local)
        deallocate(offset_local)
    end subroutine free_np_offset_local

    !<--------------------------------------------------------------------------
    !< Open metadata file and dataset of "np_local"
    !<--------------------------------------------------------------------------
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

    !<--------------------------------------------------------------------------
    !< Close dataset, filespace, group and file of metadata
    !<--------------------------------------------------------------------------
    subroutine close_metadata_dset(file_id, group_id, dataset_id, filespace)
        implicit none
        integer(hid_t), intent(in) :: file_id, group_id, dataset_id, filespace
        integer :: error
        call h5sclose_f(filespace, error)
        call h5dclose_f(dataset_id, error)
        call h5gclose_f(group_id, error)
        call h5fclose_f(file_id, error)
    end subroutine close_metadata_dset

    !<--------------------------------------------------------------------------
    !< Get the number of particles for each MPI process of PIC simulations
    !<--------------------------------------------------------------------------
    subroutine get_np_local_vpic(tframe, species)
        implicit none
        integer, intent(in) :: tframe
        character(*), intent(in) :: species
        character(len=256) :: fname_meta
        character(len=16) :: groupname
        integer(hid_t) :: file_id, group_id, dataset_id
        integer(hsize_t), dimension(1) :: dset_dims, dset_dims_max
        integer(hid_t) :: filespace
        integer :: i, error
        character(len=8) :: tframe_char
        write(tframe_char, "(I0)") tframe
        fname_meta = trim(adjustl(rootpath))//"/particle/T."//trim(tframe_char)
        if (species == 'e') then
            fname_meta = trim(fname_meta)//"/grid_metadata_electron_"
        else if (species == 'H' .or. species == 'h' .or. species == 'i') then
            fname_meta = trim(fname_meta)//"/grid_metadata_ion_"
        endif
        fname_meta = trim(fname_meta)//trim(tframe_char)//".h5part"
        groupname = "Step#"//trim(tframe_char)
        if (myid == master) then
            call open_metadata_dset(fname_meta, groupname, file_id, &
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

    !<--------------------------------------------------------------------------
    !< Open hdf5 file using one process
    !<--------------------------------------------------------------------------
    subroutine open_hdf5_serial(filename, groupname, file_id, group_id)
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

    !<--------------------------------------------------------------------------
    !< Open hdf5 file in parallel
    !<--------------------------------------------------------------------------
    subroutine open_hdf5_parallel(filename, groupname, file_id, group_id)
        implicit none
        character(*), intent(in) :: filename, groupname
        integer(hid_t), intent(out) :: file_id, group_id
        integer(hid_t) :: plist_id
        integer :: storage_type, max_corder
        integer(size_t) :: obj_count_g, obj_count_d
        integer :: fileinfo, error
        call MPI_INFO_CREATE(fileinfo, ierror)
        call h5open_f(error)
        call h5pcreate_f(H5P_FILE_ACCESS_F, plist_id, error)
        if (collective_io) then
            ! Disable ROMIO's data-sieving
            call MPI_INFO_SET(fileinfo, "romio_ds_read", "disable", ierror)
            call MPI_INFO_SET(fileinfo, "romio_ds_write", "disable", ierror)
            ! Enable ROMIO's collective buffering
            call MPI_INFO_SET(fileinfo, "romio_cb_read", "enable", ierror)
            call MPI_INFO_SET(fileinfo, "romio_cb_write", "enable", ierror)
            ! call MPI_INFO_SET(fileinfo, "cb_buffer_size", "1048576", ierror)
            ! call MPI_INFO_SET(fileinfo, "striping_factor", "32", ierror)
            ! call MPI_INFO_SET(fileinfo, "striping_unit", "4194304", ierror)
            ! call MPI_INFO_SET(fileinfo, "romio_no_indep_rw", "true", ierror)
            ! call MPI_INFO_SET(fileinfo, "cb_nodes", "4", ierror)
        else
            call MPI_INFO_SET(fileinfo, "romio_ds_read", "automatic", ierror)
        endif
        call h5pset_fapl_mpio_f(plist_id, MPI_COMM_WORLD, fileinfo, error)
        call MPI_INFO_FREE(fileinfo, ierror)
        call h5fopen_f(filename, H5F_ACC_RDWR_F, file_id, error, &
            access_prp=plist_id)
        call h5pclose_f(plist_id, error)
        call h5gopen_f(file_id, groupname, group_id, error)
    end subroutine open_hdf5_parallel

    !<--------------------------------------------------------------------------
    !< Open hdf5 dataset and get the dataset dimensions
    !<--------------------------------------------------------------------------
    subroutine open_hdf5_dataset(dataset_name, group_id, dataset_id, &
            dset_dims, dset_dims_max, filespace)
        implicit none
        character(*), intent(in) :: dataset_name
        integer(hid_t), intent(in) :: group_id
        integer(hid_t), intent(out) :: dataset_id, filespace
        integer(hsize_t), dimension(1), intent(out) :: dset_dims, &
            dset_dims_max
        integer(hid_t) :: datatype_id
        integer :: error
        call h5dopen_f(group_id, dataset_name, dataset_id, error)
        call h5dget_type_f(dataset_id, datatype_id, error)
        call h5dget_space_f(dataset_id, filespace, error)
        call h5Sget_simple_extent_dims_f(filespace, dset_dims, &
            dset_dims_max, error)
    end subroutine open_hdf5_dataset

    !<--------------------------------------------------------------------------
    !< Open particle file, group, and datasets in HDF5 format
    !<--------------------------------------------------------------------------
    subroutine open_particle_file_h5(tframe, species)
        implicit none
        integer, intent(in) :: tframe
        character(*), intent(in) :: species
        character(len=256) :: fname
        character(len=16) :: groupname
        character(len=8) :: tframe_char
        write(tframe_char, "(I0)") tframe
        fname = trim(adjustl(rootpath))//"/particle/T."//trim(tframe_char)
        if (species == 'e') then
            fname = trim(fname)//"/electron_"
        else if (species == 'H' .or. species == 'h' .or. species == 'i') then
            fname = trim(fname)//"/ion_"
        endif
        fname = trim(fname)//trim(tframe_char)//".h5part"
        groupname = "Step#"//trim(tframe_char)

        if (parallel_read) then
            call open_hdf5_parallel(fname, groupname, file_id, group_id)
        else
            call open_hdf5_serial(fname, groupname, file_id, group_id)
        endif
        call open_hdf5_dataset("Ux", group_id, dset_ids(1), &
            dset_dims, dset_dims_max, filespace)
        call open_hdf5_dataset("Uy", group_id, dset_ids(2), &
            dset_dims, dset_dims_max, filespace)
        call open_hdf5_dataset("Uz", group_id, dset_ids(3), &
            dset_dims, dset_dims_max, filespace)
        call open_hdf5_dataset("dX", group_id, dset_ids(4), &
            dset_dims, dset_dims_max, filespace)
        call open_hdf5_dataset("dY", group_id, dset_ids(5), &
            dset_dims, dset_dims_max, filespace)
        call open_hdf5_dataset("dZ", group_id, dset_ids(6), &
            dset_dims, dset_dims_max, filespace)
        call open_hdf5_dataset("i", group_id, dset_ids(7), &
            dset_dims, dset_dims_max, filespace)
        call open_hdf5_dataset("q", group_id, dset_ids(8), &
            dset_dims, dset_dims_max, filespace)
    end subroutine open_particle_file_h5

    !<--------------------------------------------------------------------------
    !< Close particle file, group, and datasets in HDF5 format
    !<--------------------------------------------------------------------------
    subroutine close_particle_file_h5
        implicit none
        integer :: i, error
        call h5sclose_f(filespace, error)
        do i = 1, num_dset
            call h5dclose_f(dset_ids(i), error)
        enddo
        call h5gclose_f(group_id, error)
        call h5fclose_f(file_id, error)
    end subroutine close_particle_file_h5

    !<--------------------------------------------------------------------------
    !< Initial setup for reading hdf5 file
    !<--------------------------------------------------------------------------
    subroutine init_read_hdf5(dset_id, dcount, doffset, dset_dims, &
            filespace, memspace)
        implicit none
        integer(hid_t), intent(in) :: dset_id
        integer(hsize_t), dimension(1), intent(in) :: dcount, doffset, dset_dims
        integer(hid_t), intent(out) :: filespace, memspace
        integer :: error
        call h5screate_simple_f(1, dcount, memspace, error)
        call h5dget_space_f(dset_id, filespace, error)
        call h5sselect_hyperslab_f(filespace, H5S_SELECT_SET_F, doffset, &
            dcount, error)
    end subroutine init_read_hdf5

    !<--------------------------------------------------------------------------
    !< Finalize reading hdf5 file
    !<--------------------------------------------------------------------------
    subroutine final_read_hdf5(filespace, memspace)
        implicit none
        integer(hid_t), intent(in) :: filespace, memspace
        integer :: error
        call h5sclose_f(filespace, error)
        call h5sclose_f(memspace, error)
    end subroutine final_read_hdf5

    !---------------------------------------------------------------------------
    ! Read hdf5 dataset for integer data
    !---------------------------------------------------------------------------
    subroutine read_hdf5_integer(dset_id, dcount, doffset, dset_dims, fdata)
        implicit none
        integer(hid_t), intent(in) :: dset_id
        integer(hsize_t), dimension(1), intent(in) :: dcount, doffset, dset_dims
        integer, dimension(*), intent(out) :: fdata
        integer(hid_t) :: filespace, memspace
        integer :: error
        call init_read_hdf5(dset_id, dcount, doffset, dset_dims, filespace, memspace)
        call h5dread_f(dset_id, H5T_NATIVE_INTEGER, fdata, dset_dims, error, &
            file_space_id=filespace, mem_space_id=memspace)
        call final_read_hdf5(filespace, memspace)
    end subroutine read_hdf5_integer

    !---------------------------------------------------------------------------
    ! Read hdf5 dataset for real data
    !---------------------------------------------------------------------------
    subroutine read_hdf5_real(dset_id, dcount, doffset, dset_dims, fdata)
        implicit none
        integer(hid_t), intent(in) :: dset_id
        integer(hsize_t), dimension(1), intent(in) :: dcount, doffset, dset_dims
        real(fp), dimension(*), intent(out) :: fdata
        integer(hid_t) :: filespace, memspace
        integer :: error
        call init_read_hdf5(dset_id, dcount, doffset, dset_dims, filespace, memspace)
        call h5dread_f(dset_id, H5T_NATIVE_REAL, fdata, dset_dims, error, &
            file_space_id=filespace, mem_space_id=memspace)
        call final_read_hdf5(filespace, memspace)
    end subroutine read_hdf5_real

    !<--------------------------------------------------------------------------
    !< Read particle data in HDF5 format
    !<--------------------------------------------------------------------------
    subroutine read_particle_h5(pic_mpi_rank)
        implicit none
        integer, intent(in) :: pic_mpi_rank
        integer(hsize_t), dimension(1) :: dcount, doffset
        allocate(ptls(np_local(pic_mpi_rank + 1)))
        dcount(1) = np_local(pic_mpi_rank + 1)
        doffset(1) = offset_local(pic_mpi_rank + 1)
        call read_hdf5_real(dset_ids(1), dcount, doffset, dset_dims, ptls%vx)
        call read_hdf5_real(dset_ids(2), dcount, doffset, dset_dims, ptls%vy)
        call read_hdf5_real(dset_ids(3), dcount, doffset, dset_dims, ptls%vz)
        call read_hdf5_real(dset_ids(4), dcount, doffset, dset_dims, ptls%dx)
        call read_hdf5_real(dset_ids(5), dcount, doffset, dset_dims, ptls%dy)
        call read_hdf5_real(dset_ids(6), dcount, doffset, dset_dims, ptls%dz)
        call read_hdf5_integer(dset_ids(7), dcount, doffset, dset_dims, ptls%icell)
        call read_hdf5_real(dset_ids(8), dcount, doffset, dset_dims, ptls%q)
    end subroutine read_particle_h5

    !<--------------------------------------------------------------------------
    !< Initial setup for reading hdf5 file in parallel
    !<--------------------------------------------------------------------------
    subroutine init_read_hdf5_parallel(dset_id, dcount, doffset, dset_dims, &
            filespace, memspace, plist_id)
        implicit none
        integer(hid_t), intent(in) :: dset_id
        integer(hsize_t), dimension(1), intent(in) :: dcount, doffset, dset_dims
        integer(hid_t), intent(out) :: filespace, memspace, plist_id
        integer :: error
        ! Create property list for collective dataset write
        call h5pcreate_f(H5P_DATASET_XFER_F, plist_id, error)
        if (collective_io) then
            call h5pset_dxpl_mpio_f(plist_id, H5FD_MPIO_COLLECTIVE_F, error)
        else
            call h5pset_dxpl_mpio_f(plist_id, H5FD_MPIO_INDEPENDENT_F, error)
        endif

        call h5screate_simple_f(1, dcount, memspace, error)
        call h5dget_space_f(dset_id, filespace, error)
        call h5sselect_hyperslab_f(filespace, H5S_SELECT_SET_F, doffset, &
            dcount, error)
    end subroutine init_read_hdf5_parallel

    !<--------------------------------------------------------------------------
    !< Finalize reading hdf5 file in parallel
    !<--------------------------------------------------------------------------
    subroutine final_read_hdf5_parallel(filespace, memspace, plist_id)
        implicit none
        integer(hid_t), intent(in) :: filespace, memspace, plist_id
        integer :: error
        call h5sclose_f(filespace, error)
        call h5sclose_f(memspace, error)
        call h5pclose_f(plist_id, error)
    end subroutine final_read_hdf5_parallel

    !---------------------------------------------------------------------------
    ! Read hdf5 dataset for integer data in parallel
    !---------------------------------------------------------------------------
    subroutine read_hdf5_integer_parallel(dset_id, dcount, doffset, dset_dims, fdata)
        implicit none
        integer(hid_t), intent(in) :: dset_id
        integer(hsize_t), dimension(1), intent(in) :: dcount, doffset, dset_dims
        integer, dimension(*), intent(out) :: fdata
        integer(hid_t) :: filespace, memspace, plist_id
        integer :: error, actual_io_mode
        call init_read_hdf5_parallel(dset_id, dcount, doffset, dset_dims, &
            filespace, memspace, plist_id)
        call h5dread_f(dset_id, H5T_NATIVE_INTEGER, fdata, dset_dims, error, &
            file_space_id=filespace, mem_space_id=memspace, xfer_prp=plist_id)
        call final_read_hdf5_parallel(filespace, memspace, plist_id)
    end subroutine read_hdf5_integer_parallel

    !---------------------------------------------------------------------------
    ! Read hdf5 dataset for real data in parallel
    !---------------------------------------------------------------------------
    subroutine read_hdf5_real_parallel(dset_id, dcount, doffset, dset_dims, fdata)
        implicit none
        integer(hid_t), intent(in) :: dset_id
        integer(hsize_t), dimension(1), intent(in) :: dcount, doffset, dset_dims
        real(fp), dimension(*), intent(out) :: fdata
        integer(hid_t) :: filespace, memspace, plist_id
        integer :: error
        call init_read_hdf5_parallel(dset_id, dcount, doffset, dset_dims, &
            filespace, memspace, plist_id)
        call h5dread_f(dset_id, H5T_NATIVE_REAL, fdata, dset_dims, error, &
            file_space_id=filespace, mem_space_id=memspace, xfer_prp=plist_id)
        call final_read_hdf5_parallel(filespace, memspace, plist_id)
    end subroutine read_hdf5_real_parallel

    !<--------------------------------------------------------------------------
    !< Read particle data in HDF5 format in parallel
    !<--------------------------------------------------------------------------
    subroutine read_particle_h5_parallel(pic_mpi_rank)
        implicit none
        integer, intent(in) :: pic_mpi_rank
        integer(hsize_t), dimension(1) :: dcount, doffset
        allocate(ptls(np_local(pic_mpi_rank + 1)))
        dcount(1) = np_local(pic_mpi_rank + 1)
        doffset(1) = offset_local(pic_mpi_rank + 1)
        call read_hdf5_real_parallel(dset_ids(1), dcount, doffset, dset_dims, ptls%vx)
        call read_hdf5_real_parallel(dset_ids(2), dcount, doffset, dset_dims, ptls%vy)
        call read_hdf5_real_parallel(dset_ids(3), dcount, doffset, dset_dims, ptls%vz)
        call read_hdf5_real_parallel(dset_ids(4), dcount, doffset, dset_dims, ptls%dx)
        call read_hdf5_real_parallel(dset_ids(5), dcount, doffset, dset_dims, ptls%dy)
        call read_hdf5_real_parallel(dset_ids(6), dcount, doffset, dset_dims, ptls%dz)
        call read_hdf5_integer_parallel(dset_ids(7), dcount, doffset, dset_dims, ptls%icell)
        call read_hdf5_real_parallel(dset_ids(8), dcount, doffset, dset_dims, ptls%q)
    end subroutine read_particle_h5_parallel

end program energetic_particle_density
