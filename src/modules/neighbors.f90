!*******************************************************************************
! Module including the indices of the two adjoint points for finite difference
! method.
!*******************************************************************************
module neighbors_module
    use constants, only: fp, dp
    use mpi_topology, only: htg
    use picinfo, only: domain
    implicit none
    private
    public ixl, ixh, iyl, iyh, izl, izh, idx, idy, idz
    public adjoint_points

    ! The indices of the neighbors.
    integer, allocatable, dimension(:) :: ixl, ixh, iyl, iyh, izl, izh
    ! The inverse of the distance between the neighbors.
    real(dp), allocatable, dimension(:) :: idx, idy, idz

    contains

    !---------------------------------------------------------------------------
    ! Decide the two adjoint points.
    ! Input:
    !   ntot: total number of points in this dimension.
    !   cindex: index of current point.
    ! Output:
    !   index1: index of left/bottom point. 
    !   index2: index of the right/top point.
    !---------------------------------------------------------------------------
    subroutine adjoint_points(ntot, cindex, index1, index2)
        implicit none
        integer, intent(in) :: ntot, cindex
        integer, intent(out) :: index1, index2
        if (ntot == 1) then
            index1 = 1
            index2 = 1
        else if (cindex == 1) then
            index1 = 1
            index2 = 2
        else if (cindex == ntot) then
            index1 = ntot-1
            index2 = ntot
        else
            index1 = cindex - 1
            index2 = cindex + 1
        endif
    end subroutine adjoint_points

    !---------------------------------------------------------------------------
    ! Initialize the indices of the neighbors and the inverse of the distance
    ! between them.
    !---------------------------------------------------------------------------
    subroutine init_neighbors
        implicit none
        allocate(ixl(htg%nx))
        allocate(ixh(htg%nx))
        allocate(idx(htg%nx))
        allocate(iyl(htg%ny))
        allocate(iyh(htg%ny))
        allocate(idy(htg%ny))
        allocate(izl(htg%nz))
        allocate(izh(htg%nz))
        allocate(idz(htg%nz))
        ixl = 0; iyl = 0; izl = 0
        ixh = 0; iyh = 0; izh = 0
        idx = 0.0; idy = 0.0; idz = 0.0
    end subroutine init_neighbors

    !---------------------------------------------------------------------------
    ! Free the indices of the neighbors and the inverse of the distance between
    ! them.
    !---------------------------------------------------------------------------
    subroutine free_neighbors
        implicit none
        deallocate(ixl, iyl, izl)
        deallocate(ixh, iyh, izh)
        deallocate(idx, idy, idz)
    end subroutine free_neighbors

    !---------------------------------------------------------------------------
    ! Decide the indices of two neighbors.
    ! Input:
    !   ntot: total number of points in this dimension.
    !   indexc: index of current point.
    ! Output:
    !   indexl: index of the neighbor with lower index.
    !   indexh: index of the neighbor with higher index.
    !---------------------------------------------------------------------------
    subroutine neighbors(ntot, indexc, indexl, indexh)
        implicit none
        integer, intent(in) :: ntot, indexc
        integer, intent(out) :: indexl, indexh
        if (ntot == 1) then
            indexl = 1
            indexh = 1
        else if (indexc == 1) then
            indexl = 1
            indexh = 2
        else if (indexc == ntot) then
            indexl = ntot - 1
            indexh = ntot
        else
            indexl = indexc - 1
            indexh = indexc + 1
        endif
    end subroutine neighbors

    !---------------------------------------------------------------------------
    ! Get the indices of the neighbors and the inverse of the distance between
    ! them.
    !---------------------------------------------------------------------------
    subroutine get_neighbors
        implicit none
        integer :: nx, ny, nz, ix, iy, iz
        nx = htg%nx
        ny = htg%ny
        nz = htg%nz

        do ix = 1, nx
            call neighbors(nx, ix, ixl(ix), ixh(ix))
            if ((ixh(ix) - ixl(ix)) /= 0) then
                idx(ix) = domain%idx / (ixh(ix) - ixl(ix))
            else
                idx(ix) = 0.0
            endif
        enddo

        do iy = 1, ny
            call neighbors(ny, iy, iyl(iy), iyh(iy))
            if ((iyh(iy) - iyl(iy)) /= 0) then
                idy(iy) = domain%idy / (iyh(iy) - iyl(iy))
            else
                idy(iy) = 0.0
            endif
        enddo

        do iz = 1, nz
            call neighbors(nz, iz, izl(iz), izh(iz))
            if ((izh(iz) - izl(iz)) /= 0) then
                idz(iz) = domain%idz / (izh(iz) - izl(iz))
            else
                idz(iz) = 0.0
            endif
        enddo
    end subroutine get_neighbors

end module neighbors_module
