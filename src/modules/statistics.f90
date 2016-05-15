!*******************************************************************************
! Statistics for the field data. It includes subroutines to calculate the mean,
! the total, the distribution with logarithmic bins.
!*******************************************************************************
module statistics
    implicit none
    private
    public get_average_and_total, get_log_distribution

    contains

    !---------------------------------------------------------------------------
    ! Get the average and total of a field data set across MPI process
    ! Input:
    !   fdata: the data array.
    !   ntot: total number of data points.
    ! Output:
    !   avg, tot: the average and total of the field data.
    !---------------------------------------------------------------------------
    subroutine get_average_and_total(fdata, avg, tot)
        use mpi_module
        use constants, only: fp
        use picinfo, only: domain
        use mpi_topology, only: range_out
        implicit none
        real(fp), dimension(:, :, :), intent(in) :: fdata
        real(fp), intent(out) :: avg, tot
        real(fp), allocatable, dimension(:) :: tot_array
        real(fp) :: tot_local
        integer :: ixl, ixh, iyl, iyh, izl, izh

        ixl = range_out%ixl
        iyl = range_out%iyl
        izl = range_out%izl
        ixh = range_out%ixh
        iyh = range_out%iyh
        izh = range_out%izh

        allocate(tot_array(numprocs))
        tot_array = 0.0
        tot_local = sum(fdata(ixl:ixh, iyl:iyh, izl:izh))
        call MPI_BARRIER(MPI_COMM_WORLD, ierror)
        call MPI_GATHER(tot_local, 1, MPI_REAL, tot_array, 1, MPI_REAL, &
                master, MPI_COMM_WORLD, ierror)
        if (myid == master) then
            tot = sum(tot_array) * domain%dx * domain%dy * domain%dz
            avg = tot / (domain%nx*domain%ny*domain%nz)
        else
            avg = 0.1; tot = 1.0 ! Some random number except rank master.
        endif
        deallocate(tot_array)
    end subroutine get_average_and_total

    !---------------------------------------------------------------------------
    ! Get the distribution of a data set using a logarithm scale bins.
    ! Input:
    !   dataSet: the data set.
    !   minValue: the minimum value in the energy bins.
    !   maxValue: the maximum value in the energy bins.
    !   nbins: number of energy bins.
    ! Return:
    !   distData: the distribution of the data set.
    !---------------------------------------------------------------------------
    subroutine get_log_distribution(dataSet, minValue, maxValue, nbins, distData)
        use constants, only: fp
        use mpi_topology, only: ht, htg
        implicit none
        real(fp), intent(in) :: minValue, maxValue
        integer, intent(in) :: nbins
        real(fp), dimension(:, :, :), intent(in) :: dataSet
        real(fp), intent(out), dimension(nbins+2) :: distData
        integer :: ix, iy, iz, ibin
        integer :: ix_start, iy_start, iz_start, ix_stop, iy_stop, iz_stop
        real(fp) :: deltaLog, minLog, maxLog
        minLog = log10(minValue)
        maxLog = log10(maxValue)
        deltaLog = (maxLog - minLog) / nbins
        distData = 0.0

        ix_start = ht%start_x - htg%start_x + 1
        iy_start = ht%start_y - htg%start_y + 1
        iz_start = ht%start_z - htg%start_z + 1
        ix_stop = ix_start + ht%nx - 1
        iy_stop = iy_start + ht%ny - 1
        iz_stop = iz_start + ht%nz - 1

        do iz = iz_start, iz_stop
            do iy = iy_start, iy_stop
                do ix = ix_start, ix_stop
                    ibin = (log10(dataSet(ix,iy,iz)) - minLog) / deltaLog + 1
                    if (ibin < 1) ibin = 1
                    if (ibin > nbins+1) ibin = nbins + 2
                    distData(ibin) = distData(ibin) + 1
                enddo
            enddo
        enddo
    end subroutine get_log_distribution

end module statistics
