!*******************************************************************************
! Decide the two adjoint points for finite different method at current point.
! Input:
!   ntot: total number of points in this dimension.
!   cindex: index of current point.
! Output:
!   index1: index of left/bottom point. 
!   index2: index of the right/top point.
!*******************************************************************************
module adjoint_points_module
    implicit none

    contains

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

end module adjoint_points_module
