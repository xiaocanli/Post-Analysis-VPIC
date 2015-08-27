!*******************************************************************************
! Module for data analysis setup, including the initialization of MPI processes,
! reading PIC simulation domain, setting MPI topology, setting MPI datatype,
! setting MPI_INFO for I/O, opening and closing PIC field fields,
! initialization and free of the PIC fields, freeing the MPI datatype, MPI_INFO,
! and finalizing the MPI process.
!*******************************************************************************
module analysis_management
    implicit none
    private
    public init_analysis, end_analysis

    contains

    !---------------------------------------------------------------------------
    ! Initialize the analysis by reading the PIC simulation domain information,
    ! get file paths for the field data and the outputs.
    !---------------------------------------------------------------------------
    subroutine init_analysis
        use mpi_module
        use path_info, only: get_file_paths
        use mpi_topology, only: set_mpi_topology
        use mpi_datatype_fields, only: set_mpi_datatype_fields
        use mpi_info_module, only: fileinfo, set_mpi_info
        use picinfo, only: read_domain, broadcast_pic_info, &
                get_total_time_frames, get_energy_band_number
        use parameters, only: get_start_end_time_points, get_inductive_flag, &
                tp2, get_relativistic_flag
        use configuration_translate, only: read_configuration
        use time_info, only: get_nout

        implicit none

        call MPI_INIT(ierr)
        call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
        call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)

        call get_file_paths
        if (myid == master) then
            call read_domain
        endif
        call broadcast_pic_info
        call get_start_end_time_points
        call get_inductive_flag
        call get_relativistic_flag
        call read_configuration
        call get_nout
        call get_total_time_frames(tp2)
        call get_energy_band_number
        call set_mpi_topology   ! MPI topology
        call set_mpi_datatype_fields
        call set_mpi_info
    end subroutine init_analysis

    !---------------------------------------------------------------------------
    ! Finalizing the analysis by release the memory, MPI data types, MPI info.
    !---------------------------------------------------------------------------
    subroutine end_analysis
        use mpi_module
        use mpi_datatype_fields, only: filetype_ghost, filetype_nghost
        use mpi_info_module, only: fileinfo
        implicit none

        call MPI_TYPE_FREE(filetype_ghost, ierror)
        call MPI_TYPE_FREE(filetype_nghost, ierror)
        call MPI_INFO_FREE(fileinfo, ierror)
        call MPI_FINALIZE(ierr)
    end subroutine end_analysis

end module analysis_management
