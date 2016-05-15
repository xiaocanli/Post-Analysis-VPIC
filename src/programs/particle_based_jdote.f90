!*******************************************************************************
! Main program for particle based drift analysis
!*******************************************************************************
program particle_based_jdote
    use mpi_module
    use constants, only: dp
    use particle_info, only: species
    implicit none
    integer :: ct
    real(dp) :: mp_elapsed

    ct = 1

    ! species = 'e'
    ! call init_analysis
    ! call commit_analysis
    ! call end_analysis

    ! call MPI_BARRIER(MPI_COMM_WORLD, ierror)

    species = 'i'
    call init_analysis
    call commit_analysis
    call end_analysis

    contains

    !---------------------------------------------------------------------------
    ! Initialize the analysis.
    !---------------------------------------------------------------------------
    subroutine init_analysis
        use mpi_module
        use particle_info, only: species, get_ptl_mass_charge
        use path_info, only: get_file_paths
        use picinfo, only: read_domain, broadcast_pic_info, &
                get_total_time_frames, get_energy_band_number, &
                read_thermal_params, calc_energy_interval, nbands, &
                write_pic_info, domain
        use configuration_translate, only: read_configuration
        use topology_translate, only: set_topology, set_start_stop_cells
        use time_info, only: get_nout, adjust_tindex_start, set_output_record
        use mpi_io_translate, only: set_mpi_io
        use parameters, only: get_relativistic_flag, get_start_end_time_points, tp2
        use interpolation_emf, only: init_emfields, init_emfields_derivatives 
        use particle_drift, only: init_drift_fields, init_para_perp_fields, &
                init_jdote_sum
        use neighbors_module, only: init_neighbors, get_neighbors
        use particle_fields, only: init_density_fields
        implicit none
        integer :: nx, ny, nz

        call MPI_INIT(ierr)
        call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
        call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)

        call get_ptl_mass_charge(species)
        call get_file_paths
        if (myid == master) then
            call read_domain
            call write_pic_info
        endif
        call broadcast_pic_info
        call get_start_end_time_points
        call get_relativistic_flag
        call get_energy_band_number
        call read_thermal_params
        if (nbands > 0) then
            call calc_energy_interval
        endif
        call read_configuration
        call get_total_time_frames(tp2)
        call set_topology
        call set_start_stop_cells
        call get_nout
        call adjust_tindex_start
        call set_output_record
        call set_mpi_io

        nx = domain%pic_nx + 2  ! Including ghost cells
        ny = domain%pic_ny + 2
        nz = domain%pic_nz + 2
        call init_neighbors(nx, ny, nz)
        call get_neighbors

        call init_density_fields
        call init_emfields
        call init_emfields_derivatives
        call init_drift_fields
        call init_para_perp_fields
        call init_jdote_sum

    end subroutine init_analysis

    !---------------------------------------------------------------------------
    ! This subroutine does the analysis.
    !---------------------------------------------------------------------------
    subroutine commit_analysis
        use mpi_module
        use constants, only: dp
        use configuration_translate, only: tindex_start, tindex_stop
        use time_info, only: nout, output_record
        use path_info, only: rootpath
        use particle_drift, only: save_jdote_sum
        use picinfo, only: domain
        implicit none
        integer :: tindex, tindex_new, ct
        character(len=150) :: fname
        real(dp) :: mp_elapsed
        logical :: dfile

        mp_elapsed = MPI_WTIME()

        ! Loop over time slices
        dfile= .true.
        tindex = domain%particle_interval
        ct = 1
        do while(dfile)
            if (myid==master) print *, " Time slice: ", tindex

            call analysis_single_frame(tindex, ct, output_record)

            ! Might as well just wait here
            call MPI_BARRIER(MPI_COMM_WORLD, ierror)

            ! Check if there is another time slice to read
            dfile = .false.
            tindex_new = tindex + domain%particle_interval
            if (tindex_new <= tindex_stop) then
            ! if (tindex_new <= 5000) then
                write(fname, "(A,I0,A,I0,A)") &
                    trim(adjustl(rootpath))//"fields/T.", tindex_new, &
                    "/fields.", tindex_new, ".0"
                inquire(file=trim(fname), exist=dfile)
            else
                dfile = .false.
            endif
            tindex = tindex_new     
            if (dfile) then 
                output_record = output_record + 1
                ct = ct + 1
            endif
        enddo

        call save_jdote_sum

        mp_elapsed = MPI_WTIME() - mp_elapsed

        if (myid==master) then
            write(*,'(A, F6.1)') " Total time used (s): ", mp_elapsed
        endif
    end subroutine commit_analysis

    !---------------------------------------------------------------------------
    ! For one time frame.
    ! Input:
    !   tindex0: the time step index.
    !   ct: current time frame.
    !   output_record: the record number for data output.
    !---------------------------------------------------------------------------
    subroutine analysis_single_frame(tindex0, ct, output_record)
        use particle_info, only: species
        use interpolation_emf, only: read_emfields_single, &
                calc_emfields_derivatives
        use particle_drift, only: calc_particle_energy_change_rate, &
                set_drift_fields_zero, set_para_perp_fields_zero, &
                sum_data_arrays, save_data_arrays
        use rank_index_mapping, only: index_to_rank
        use picinfo, only: domain
        use topology_translate, only: ht
        use particle_fields, only: read_density_fields_single
        implicit none
        integer, intent(in) :: tindex0, ct, output_record
        integer :: dom_x, dom_y, dom_z, n
        integer :: ix, iy, iz
        do dom_x = ht%start_x, ht%stop_x
            ix = (dom_x - ht%start_x) * domain%pic_nx
            do dom_y = ht%start_y, ht%stop_y
                iy = (dom_y - ht%start_y) * domain%pic_ny
                do dom_z = ht%start_z, ht%stop_z
                    iz = (dom_z - ht%start_z) * domain%pic_nz
                    call index_to_rank(dom_x, dom_y, dom_z, domain%pic_tx, &
                                       domain%pic_ty, domain%pic_tz, n)
                    if (species == 'i') then
                        species = 'H'
                    endif
                    call read_emfields_single(tindex0, n-1)
                    call calc_emfields_derivatives
                    call read_density_fields_single(tindex0, n-1, species)
                    if (species == 'H') then
                        species = 'i'
                    endif
                    if (species == 'i') then
                        species = 'h'
                    endif
                    call calc_particle_energy_change_rate(tindex0, species, &
                            n-1, ix, iy, iz)
                    if (species == 'h') then
                        species = 'i'
                    endif
                enddo ! x
            enddo ! y
        enddo ! z
        call sum_data_arrays(ct)
        call save_data_arrays(tindex0, output_record)
        call set_drift_fields_zero
        call set_para_perp_fields_zero
    end subroutine analysis_single_frame

    !---------------------------------------------------------------------------
    ! End the analysis by free the memory.
    !---------------------------------------------------------------------------
    subroutine end_analysis
        use mpi_module
        use topology_translate, only: free_start_stop_cells
        use mpi_io_translate, only: datatype
        use mpi_info_module, only: fileinfo
        use interpolation_emf, only: free_emfields, free_emfields_derivatives
        use particle_drift, only: free_drift_fields, free_para_perp_fields, &
                free_jdote_sum
        use neighbors_module, only: free_neighbors
        use particle_fields, only: free_density_fields
        implicit none
        call free_neighbors
        call free_jdote_sum
        call free_para_perp_fields
        call free_drift_fields
        call free_emfields_derivatives
        call free_emfields
        call free_density_fields
        call free_start_stop_cells
        call MPI_TYPE_FREE(datatype, ierror)
        call MPI_INFO_FREE(fileinfo, ierror)
        call MPI_FINALIZE(ierr)
    end subroutine end_analysis

end program particle_based_jdote
