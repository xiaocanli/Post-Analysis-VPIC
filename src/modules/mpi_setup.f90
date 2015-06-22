!*******************************************************************************
! Module of the topology of the MPI processes.
!*******************************************************************************
module mpi_topology
    implicit none
    private
    public ht_type, ht, htg, set_mpi_topology, range_out, distribute_tasks
    type ht_type
        integer :: tx, ty, tz   ! number of processes in each dimension.
        integer :: nx, ny, nz   ! number of cells in each dimension.
        integer :: start_x, start_y, start_z    ! where to start in x/y/z.
        integer :: stop_x, stop_y, stop_z       ! where to stop in x/y/z.
        integer :: ix, iy, iz   ! IDs of current MPI process.
    end type ht_type

    type(ht_type) :: ht  ! The real topology for the MPI processes.
    type(ht_type) :: htg ! The topology including the ghost cells.

    ! Data range for saving. The ghost cells are not needed when saving data.
    ! The MPI process including the boundary has different ghost cells.
    ! That's why we need some calculation of the staring and ending indices
    ! for data saving. (l and h indicating lower and higher bound).
    type data_range
        integer :: ixl, ixh
        integer :: iyl, iyh
        integer :: izl, izh
    end type data_range

    type(data_range) :: range_out

    contains

    !---------------------------------------------------------------------------
    ! Set the topology of the MPI processes with ghost cells.
    !---------------------------------------------------------------------------
    subroutine set_topology_with_ghost
        implicit none
        htg = ht
        call adjust_topology(htg%ix, htg%tx, htg%nx, htg%start_x, htg%stop_x)
        call adjust_topology(htg%iy, htg%ty, htg%ny, htg%start_y, htg%stop_y)
        call adjust_topology(htg%iz, htg%tz, htg%nz, htg%start_z, htg%stop_z)
    end subroutine set_topology_with_ghost

    !---------------------------------------------------------------------------
    ! Adjust the topology according to whether the current MPI process is 
    ! dealing with cells at the center or at the boundary of the domain.
    ! Input:
    !   mpiId: MPI ID in current dimension.
    !   nptot: total number of MPI processes in current dimension.
    ! Input & output:
    !   nc: number of cells in current dimension for current MPI process..
    !   startc: ID for the starting cells in current dimension.
    !   stopc: ID for the stopping cels in current dimension.
    !---------------------------------------------------------------------------
    subroutine adjust_topology(mpiId, nptot, nc, startc, stopc)
        implicit none
        integer, intent(in) :: mpiId, nptot
        integer, intent(inout) :: nc, startc, stopc
        if ((mpiId > 0) .and. (mpiId < (nptot-1))) then
            ! Center
            nc = nc + 2
            startc = startc - 1
            stopc = stopc + 1
        else if ((mpiId == 0) .and. (mpiId < (nptot-1))) then
            ! Left/bottom boundary.
            nc = nc + 1
            stopc = stopc + 1
        else if ((mpiId == (nptot-1)) .and. (mpiId > 0)) then
            ! Right/top boundary.
            nc = nc + 1
            startc = startc - 1
        endif
    end subroutine adjust_topology

    !---------------------------------------------------------------------------
    ! Get the middle two divisors of an integer. e.g. 24=4*6
    !---------------------------------------------------------------------------
    subroutine get_middle_divisors(num, d1, d2)
        implicit none
        integer, intent(in) :: num
        integer, intent(out) :: d1, d2
        integer :: nsq, i
        nsq = int(sqrt(num+0.0))
        do i = nsq, 1, -1
            if (mod(num, i) == 0) then
                d1 = i
                exit
            endif
        enddo
        d2 = num / d1
        return
    end subroutine get_middle_divisors

    !---------------------------------------------------------------------------
    ! Get the number of cells, the starting and stopping cells IDs in one
    ! dimension for current MPI process.
    ! Inputs:
    !   ntasks: total number of tasks for all.
    !   nworkers: number of the worker we have.
    !   worker_id: the ID of current worker. [0, nworkers-1]
    ! Outputs:
    !   ntasks_current: number of the tasks for current worker.
    !   task_start, task_end: starting and ending task ID for current worker.
    !---------------------------------------------------------------------------
    subroutine distribute_tasks(ntasks, nworkers, worker_id, ntask_current, &
                                task_start, task_end)
        implicit none
        integer, intent(in) :: ntasks, nworkers, worker_id
        integer, intent(out) :: ntask_current, task_start, task_end
        integer :: mod_tasks ! The tasks that cannot be averagely distributed.
        ntask_current = ntasks / nworkers
        task_start = worker_id * ntask_current
        mod_tasks = mod(ntasks, nworkers)
        if (worker_id < mod_tasks) then
            ntask_current = ntask_current + 1
            task_start = task_start + worker_id
        else
            task_start = task_start + mod_tasks
        endif
        task_end = task_start + ntask_current - 1
    end subroutine distribute_tasks

    !---------------------------------------------------------------------------
    ! Distribute tasks and set the topology of the MPI processes for this analysis.
    ! Updates:
    !   ht: the MPI topology without ghost cells.
    !   htg: the MPI topology with ghost cells.
    !   range_out the range for data output.
    !---------------------------------------------------------------------------
    subroutine set_mpi_topology
        use mpi_module
        use picinfo, only: domain
        implicit none
        integer :: d1, d2
        call get_middle_divisors(numprocs, d1, d2)
        if (domain%ny == 1) then
            ! 2D case
            ht%tx = d1
            ht%ty = 1
            ht%tz = d2
            ht%iz = myid / ht%tx
            ht%ix = mod(myid, ht%tx)
            ht%iy = 0
        else
            ! 3D case
            ht%tx = 1
            ht%ty = d1
            ht%tz = d2
            ht%iz = myid / ht%ty 
            ht%iy = mod(myid, ht%ty)
            ht%ix = 0
        endif
        call distribute_tasks(domain%nx, ht%tx, ht%ix, ht%nx, &
                              ht%start_x, ht%stop_x)
        call distribute_tasks(domain%ny, ht%ty, ht%iy, ht%ny, &
                              ht%start_y, ht%stop_y)
        call distribute_tasks(domain%nz, ht%tz, ht%iz, ht%nz, &
                              ht%start_z, ht%stop_z)
        !print*, myid, ht%start_x, ht%start_y, ht%start_z
        !print*, myid, ht%stop_x, ht%stop_y, ht%stop_z
        !print*, myid, ht%nx, ht%ny, ht%nz
        call set_topology_with_ghost

        ! Set data range for saving.
        range_out%ixl = ht%start_x - htg%start_x + 1
        range_out%iyl = ht%start_y - htg%start_y + 1
        range_out%izl = ht%start_z - htg%start_z + 1
        range_out%ixh = range_out%ixl + ht%nx - 1
        range_out%iyh = range_out%iyl + ht%ny - 1
        range_out%izh = range_out%izl + ht%nz - 1
    end subroutine set_mpi_topology

end module mpi_topology
