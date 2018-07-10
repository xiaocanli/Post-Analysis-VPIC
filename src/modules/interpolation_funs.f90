!<******************************************************************************
!< Module of functions used in interpolation
!<******************************************************************************
module interpolation_funs
    use constants, only: fp
    implicit none
    private
    save
    public bounding_indcies

    contains

    !<--------------------------------------------------------------------------
    !< Decide the starting and ending indices
    !< Input:
    !<  ix: MPI rank for the PIC simulation
    !<  pic_nx: number of cells along one direction in a MPI rank in PIC
    !<  tx: MPI size along one direction in the PIC simulation
    !<  sx: starting MPI rank along one direction in the PIC simulation
    !       for the MPI rank for current analysis
    !< Output:
    !<  ixs_local, ixe_local: starting and ending cell indices in a local
    !<      MPI rank of the PIC simulation
    !<  ixs_global, ixe_global: starting and ending cell indices in the MPI
    !       rank of current analysis
    !<--------------------------------------------------------------------------
    subroutine bounding_indcies(ix, pic_nx, tx, sx, ixs_local, ixe_local, &
            ixs_global, ixe_global)
        implicit none
        integer, intent(in) :: ix, pic_nx, tx, sx
        integer, intent(out) :: ixs_local, ixe_local, ixs_global, ixe_global
        if (tx == 1) then
            ixs_local = 1
            ixe_local = pic_nx
            ixs_global = 1
            ixe_global = pic_nx
        else if (ix == 0 .and. ix < tx - 1) then
            ixs_local = 1
            ixe_local = pic_nx + 1
            ixs_global = 1
            ixe_global = pic_nx + 1
        else if (ix == tx - 1) then
            ixs_local = 0
            ixe_local = pic_nx
            if (sx > 0) then
                ixs_global = pic_nx * (ix - sx) + 1
                ixe_global = pic_nx * (ix - sx + 1) + 1
            else
                ixs_global = pic_nx * (ix - sx)
                ixe_global = pic_nx * (ix - sx + 1)
            endif
        else
            ixs_local = 0
            ixe_local = pic_nx + 1
            if (sx > 0) then
                ixs_global = pic_nx * (ix - sx) + 1
                ixe_global = pic_nx * (ix - sx + 1) + 2
            else
                ixs_global = pic_nx * (ix - sx)
                ixe_global = pic_nx * (ix - sx + 1) + 1
            endif
        endif
    end subroutine bounding_indcies

end module interpolation_funs
