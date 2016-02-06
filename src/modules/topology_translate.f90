!*******************************************************************************
! MPI topology for current analysis. The PIC simulation has one MPI topology.
! When doing post-analysis, we use less CPU cores. So we must make sure the
! tasks are equally divided among all the MPI processes of current analysis.
!*******************************************************************************
module topology_translate
    use mpi_module
    use constants, only: fp, dp
    use rank_index_mapping, only: rank_to_index
    implicit none
    private
    public idxstart, idxstop, ht, set_topology, set_start_stop_cells, &
           free_start_stop_cells

    ! Starting and stopping cells for current MPI process.
    integer, allocatable, dimension(:,:) :: idxstart, idxstop 

    ! MPI 3D topology for current analysis. The MPI topology of the PIC
    ! simulation is also 3D, but with larger sizes. The start_x and stop_x
    ! indices are based one the MPI topology of the PIC simulation.
    type ht_type
        integer :: tx, ty, tz       ! Number of processes in x and y
        integer :: ix, iy, iz       ! MPI process ID in each dimension. 
        integer :: nx, ny, nz       ! Number of cells in each direction
        integer :: start_x, stop_x  ! Where to start/stop in x
        integer :: start_z, stop_z, start_y, stop_y
    end type ht_type

    type(ht_type) :: ht

    contains

    !---------------------------------------------------------------------------
    ! Set topology for current analysis.
    !---------------------------------------------------------------------------
    subroutine set_topology
        use configuration_translate, only: httx, htty, httz
        use picinfo, only: domain
        implicit none

        ht%tx = httx
        ht%ty = htty
        ht%tz = httz

        ! Map myid to a 3D topology.
        call rank_to_index(myid, ht%tx, ht%ty, ht%tz, ht%ix, ht%iy, ht%iz)

        ! Domain start/stop for this process
        ht%start_x = (domain%pic_tx/ht%tx) * ht%ix  
        ht%stop_x = (domain%pic_tx/ht%tx) * (ht%ix + 1) - 1 

        ht%start_y = (domain%pic_ty/ht%ty) * ht%iy  
        ht%stop_y = (domain%pic_ty/ht%ty) * (ht%iy + 1) - 1 

        ht%start_z = (domain%pic_tz/ht%tz) * ht%iz 
        ht%stop_z = (domain%pic_tz/ht%tz) * (ht%iz + 1) - 1 

        ! Number of cells for each process
        ht%nx = domain%nx / ht%tx
        ht%ny = domain%ny / ht%ty
        ht%nz = domain%nz / ht%tz

        if (myid==master) then
            ! echo this information
            print *, "---------------------------------------------------"
            print *, "The topology information for current analysis"
            write(*, "(A,I0,A,I0,A,I0)") " Topology: tx, ty, tz = ", &
                ht%tx, ', ', ht%ty, ', ', ht%tz
            write(*, "(A,I0,A,I0,A,I0)") " Cells: nx, ny, nz = ", &
                ht%nx, ', ', ht%ny, ', ', ht%nz
            print *, "---------------------------------------------------"
        endif
    end subroutine set_topology

    !---------------------------------------------------------------------------
    ! Set the starting and stopping cell indices for current MPI process.
    !---------------------------------------------------------------------------
    subroutine set_start_stop_cells
        use picinfo, only: domain
        implicit none
        integer :: ndomains, ix, iy, iz, n

        ndomains = domain%pic_tx * domain%pic_ty * domain%pic_tz
        allocate(idxstart(ndomains,3))
        allocate(idxstop(ndomains,3))

        ! Determine total size of global problem
        do n = 1, ndomains
            call rank_to_index(n-1, domain%pic_tx, domain%pic_ty, &
                               domain%pic_tz, ix, iy, iz)

            idxstart(n,1) = ((domain%nx/domain%pic_tx))*ix+1 - ht%nx*ht%ix
            idxstart(n,2) = ((domain%ny/domain%pic_ty))*iy+1 - ht%ny*ht%iy
            idxstart(n,3) = ((domain%nz/domain%pic_tz))*iz+1 - ht%nz*ht%iz

            idxstop(n,1)  = idxstart(n,1) +  (domain%nx/domain%pic_tx) - 1
            idxstop(n,2)  = idxstart(n,2) +  (domain%ny/domain%pic_ty) - 1 
            idxstop(n,3)  = idxstart(n,3) +  (domain%nz/domain%pic_tz) - 1
        enddo

        ! Check the topology for consistency
        if ((ht%tx*ht%ty*ht%tz /= numprocs) .or. &
            ((domain%pic_tx/ht%tx)*ht%tx /= domain%pic_tx) .or. &
            ((domain%pic_tz/ht%tz)*ht%tz /= domain%pic_tz) .or. &
            ((domain%pic_ty/ht%ty)*ht%ty /= domain%pic_ty)) then

            if (myid == master) print *, "invalid converter topology"
            call MPI_FINALIZE(ierr)
            stop
        endif
    end subroutine set_start_stop_cells

    !---------------------------------------------------------------------------
    ! Free starting and stopping cell indices.
    !---------------------------------------------------------------------------
    subroutine free_start_stop_cells
        implicit none
        deallocate(idxstart, idxstop)
    end subroutine free_start_stop_cells

end module topology_translate
