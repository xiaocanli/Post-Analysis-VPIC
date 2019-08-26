!<*******************************************************************************
!< Parallel conversion; Using Bill's data type at LANL
!<
!< This code convert VPIC output into gda files, which are "bricks" of data
!<
!<*******************************************************************************
program translate
    use mpi_module
    use constants, only: dp
    use configuration_translate, only: tindex_start, tindex_stop
    use time_info, only: nout, nout_fd, output_record_initial => output_record
    implicit none
    integer :: tindex, tindex_new
    character(len=256) :: rootpath, fname
    real(dp) :: mp_elapsed
    logical :: dfile, frequent_dump, sub_directory
    logical :: current_step, pre_step, post_step
    logical :: turbulence_mixing, write_top_bottom
    integer :: out_record, numfold
    character(len=8) :: suffix

    call init_analysis

    mp_elapsed = MPI_WTIME()

    out_record = output_record_initial

    ! Loop over time slices
    if (current_step) then
        dfile= .true.
        tindex = tindex_start
        do while(dfile)
            if (myid==master) print *, " Time slice: ", tindex
            call translate_fields(.false.)
            ! Check if there is another time slice to read
            tindex_new = tindex + nout
            call check_file_existence
        enddo
    endif

    if (frequent_dump) then
        if (pre_step) then
            out_record = output_record_initial
            suffix = '_pre'
            tindex = tindex_start
            if (myid==master) print *, " Translate prevous time step"
            dfile= .true.
            if (tindex > 0) then
                tindex = tindex_start - nout_fd
            endif
            do while(dfile)
                if (myid==master) print *, " Time slice: ", tindex
                call translate_fields(frequent_dump)
                ! Check if there is another time slice to read
                if (tindex == 0) then
                    tindex_new = nout - nout_fd ! Previous time step
                else
                    tindex_new = tindex + nout
                endif
                call check_file_existence
            enddo
        endif

        if (post_step) then
            out_record = output_record_initial
            suffix = '_post'
            if (tindex_start == 0) then
                tindex = 1
            else
                tindex = tindex_start + nout_fd
            endif
            if (myid==master) print *, " Translate latter time step"
            dfile= .true.
            do while(dfile)
                if (myid==master) print *, " Time slice: ", tindex
                call translate_fields(frequent_dump)
                ! Check if there is another time slice to read
                if (tindex == 1) then
                    tindex_new = nout + nout_fd
                else
                    tindex_new = tindex + nout
                endif
                call check_file_existence
            enddo
        endif
    endif

    mp_elapsed = MPI_WTIME() - mp_elapsed

    if (myid==master) then
        write(*,'(A, F6.1)') " Total time used (s): ", mp_elapsed
    endif

    call end_analysis

    contains

    !<---------------------------------------------------------------------------
    !< Check whether next time step exists
    !<---------------------------------------------------------------------------
    subroutine check_file_existence
        implicit none
        character(len=256) :: fname1, fname2
        logical :: dfile1, dfile2
        dfile = .false.
        if (tindex_new <= tindex_stop) then
            write(fname1, "(A,I0,A,I0,A)") &
                trim(adjustl(rootpath))//"fields/T.", tindex_new, &
                "/fields.", tindex_new, ".0"
            write(fname2, "(A,I0,A,I0,A)") &
                trim(adjustl(rootpath))//"fields/0/T.", tindex_new, &
                "/fields.", tindex_new, ".0"
            inquire(file=trim(fname1), exist=dfile1)
            inquire(file=trim(fname2), exist=dfile2)
            dfile = dfile1 .or. dfile2
        endif
        tindex = tindex_new
        if (dfile) out_record = out_record + 1
    end subroutine check_file_existence

    !<---------------------------------------------------------------------------
    !< translate both EM fields and particle fields
    !<---------------------------------------------------------------------------
    subroutine translate_fields(with_suffix)
        use emfields, only: read_emfields, write_emfields
        use particle_fields, only: read_particle_fields, set_current_density_zero, &
                calc_current_density, calc_absJ, write_current_densities, &
                adjust_particle_fields, write_particle_fields, read_particle_fields_mixing, &
                adjust_particle_fields_mixing, write_particle_fields_mixing, &
                write_mixing_rate
        implicit none
        logical, intent(in) :: with_suffix
        ! EMF
        if (sub_directory) then
            call read_emfields(tindex, numfold)
        else
            call read_emfields(tindex)
        endif
        call write_emfields(tindex, out_record, with_suffix, trim(adjustl(suffix)))

        ! Particle fields
        if (turbulence_mixing) then
            if (sub_directory) then
                call read_particle_fields_mixing(tindex, 'e', numfold)
            else
                call read_particle_fields_mixing(tindex, 'e')
            endif
        else
            if (sub_directory) then
                call read_particle_fields(tindex, 'e', numfold)
            else
                call read_particle_fields(tindex, 'e')
            endif
        endif
        call calc_current_density
        call adjust_particle_fields('e')
        call write_particle_fields(tindex, out_record, 'e', &
            with_suffix, trim(adjustl(suffix)))
        if (turbulence_mixing) then
            call adjust_particle_fields_mixing('e')
            call write_mixing_rate(tindex, out_record, 'e', &
                with_suffix, trim(adjustl(suffix)))
            if (write_top_bottom) then
                call write_particle_fields_mixing(tindex, out_record, 'e', &
                    with_suffix, trim(adjustl(suffix)))
            endif
        endif

        if (turbulence_mixing) then
            if (sub_directory) then
                call read_particle_fields_mixing(tindex, 'H', numfold)
            else
                call read_particle_fields_mixing(tindex, 'H')
            endif
        else
            if (sub_directory) then
                call read_particle_fields(tindex, 'H', numfold)
            else
                call read_particle_fields(tindex, 'H')
            endif
        endif
        call calc_current_density
        call adjust_particle_fields('H')
        call write_particle_fields(tindex, out_record, 'i', &
            with_suffix, trim(adjustl(suffix)))
        if (turbulence_mixing) then
            call adjust_particle_fields_mixing('H')
            call write_mixing_rate(tindex, out_record, 'i', &
                with_suffix, trim(adjustl(suffix)))
            if (write_top_bottom) then
                call write_particle_fields_mixing(tindex, out_record, 'i', &
                    with_suffix, trim(adjustl(suffix)))
            endif
        endif
        call calc_absJ
        call write_current_densities(tindex, out_record, with_suffix, suffix)
        ! Avoid accumulation in calc_current_density
        call set_current_density_zero

        ! Might as well just wait here
        call MPI_BARRIER(MPI_COMM_WORLD, ierror)
    end subroutine translate_fields

    !<---------------------------------------------------------------------------
    !< Initialize the analysis.
    !<---------------------------------------------------------------------------
    subroutine init_analysis
        use mpi_module
        use path_info, only: get_file_paths
        use picinfo, only: read_domain, broadcast_pic_info, write_pic_info, &
            get_energy_band_number
        use configuration_translate, only: read_configuration
        use topology_translate, only: set_topology, set_start_stop_cells
        use time_info, only: get_nout, adjust_tindex_start, set_output_record
        use mpi_io_translate, only: set_mpi_io
        use emfields, only: init_emfields
        use particle_fields, only: init_particle_fields, init_mixing_rate, &
            init_top_bottom_hydro
        use parameters, only: get_relativistic_flag, get_emf_flag
        implicit none

        call MPI_INIT(ierr)
        call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
        call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)

        call get_cmd_args

        call get_file_paths(rootpath)
        if (myid == master) then
            call read_domain
            call write_pic_info
        endif
        call broadcast_pic_info
        call get_relativistic_flag
        call get_emf_flag
        ! call get_energy_band_number
        call read_configuration
        call set_topology
        call set_start_stop_cells
        call get_nout(frequent_dump)
        call adjust_tindex_start
        call set_output_record
        call set_mpi_io
        call init_emfields
        call init_particle_fields
        if (turbulence_mixing) then
            call init_mixing_rate
            call init_top_bottom_hydro
        endif

    end subroutine init_analysis

    !<---------------------------------------------------------------------------
    !< End the analysis by free the memory.
    !<---------------------------------------------------------------------------
    subroutine end_analysis
        use mpi_module
        use topology_translate, only: free_start_stop_cells
        use mpi_io_translate, only: datatype
        use mpi_info_module, only: fileinfo
        use emfields, only: free_emfields
        use particle_fields, only: free_particle_fields, free_mixing_rate, &
            free_top_bottom_hydro
        implicit none
        call free_particle_fields
        if (turbulence_mixing) then
            call free_mixing_rate
            call free_top_bottom_hydro
        endif
        call free_emfields
        call free_start_stop_cells
        call MPI_TYPE_FREE(datatype, ierror)
        call MPI_INFO_FREE(fileinfo, ierror)
        call MPI_FINALIZE(ierr)
    end subroutine end_analysis

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
            examples    = ['translate -rp simulation_root_path -fd frequent_dump'])
        call cli%add(switch='--rootpath', switch_ab='-rp', &
            help='simulation root path', required=.true., act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--frequent_dump', switch_ab='-fd', &
            help='whether VPIC dumps fields frequently', required=.false., &
            act='store_true', def='.false.', error=error)
        if (error/=0) stop
        call cli%add(switch='--turbulence_mixing', switch_ab='-tm', &
            help='whether simulation has diagnostics on turbulence mixing', &
            required=.false., act='store_true', def='.false.', error=error)
        if (error/=0) stop
        call cli%add(switch='--write_top_bottom', switch_ab='-wt', &
            help='whether writting top and bottom hydro fields', &
            required=.false., act='store_true', def='.false.', error=error)
        if (error/=0) stop
        call cli%add(switch='--sub_dir', switch_ab='-sd', &
            help='whether VPIC fields are saved in sub-directory', &
            required=.false., act='store_true', def='.false.', error=error)
        if (error/=0) stop
        call cli%add(switch='--numfold', switch_ab='-nf', &
            help='Every numfold domains are saved in one sub-directory', &
            required=.false., def='32', act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--current_step', switch_ab='-cs', &
            help='whether to translate current step', required=.false., &
            act='store_true', def='.false.', error=error)
        if (error/=0) stop
        call cli%add(switch='--pre_step', switch_ab='-ps', &
            help='whether to translate previous step', required=.false., &
            act='store_true', def='.false.', error=error)
        if (error/=0) stop
        call cli%add(switch='--post_step', switch_ab='-ns', &
            help='whether to translate next step', required=.false., &
            act='store_true', def='.false.', error=error)
        if (error/=0) stop
        call cli%get(switch='-rp', val=rootpath, error=error)
        if (error/=0) stop
        call cli%get(switch='-fd', val=frequent_dump, error=error)
        if (error/=0) stop
        call cli%get(switch='-tm', val=turbulence_mixing, error=error)
        if (error/=0) stop
        call cli%get(switch='-wt', val=write_top_bottom, error=error)
        if (error/=0) stop
        call cli%get(switch='-sd', val=sub_directory, error=error)
        if (error/=0) stop
        call cli%get(switch='-nf', val=numfold, error=error)
        if (error/=0) stop
        call cli%get(switch='-cs', val=current_step, error=error)
        if (error/=0) stop
        call cli%get(switch='-ps', val=pre_step, error=error)
        if (error/=0) stop
        call cli%get(switch='-ns', val=post_step, error=error)
        if (error/=0) stop

        if (myid == 0) then
            print '(A,A)', 'The simulation rootpath: ', trim(adjustl(rootpath))
            if (frequent_dump) then
                print '(A)', 'VPIC saves fieds from both previous and next time step'
            endif
            if (turbulence_mixing) then
                print '(A)', 'VPIC simulation has diagnostics on turbulence mixing'
                if (write_top_bottom) then
                    print '(A)', 'Write top and bottom hydro fields for turbulence mixing runs'
                endif
            endif
            if (sub_directory) then
                print '(A,I0,A)', 'Every ', numfold, " domains are saved in one sub-directory"
            endif
            if (current_step) then
                print '(A)', 'translate current time step'
            endif
            if (pre_step) then
                print '(A)', 'translate previous time step'
            endif
            if (post_step) then
                print '(A)', 'translate next time step'
            endif
        endif
    end subroutine get_cmd_args

end program translate
