!*******************************************************************************
! The model for data with in different energy band, including the band number
! the fraction of particles, the bulk flow energy and the pressure tensor.
!*******************************************************************************
module energy_band_data
    use parameters, only: is_ebands, nbands
    implicit none
    private

    ! allocate(ntot(nx,ny,nz))
    ! ntot = 0.0
    ! if (is_ebands .EQ. 1) then
    !     allocate(pEB(nx,ny,nz))
    !     pEB = 0.0
    ! endif

    ! ntot = num_rho
    ! if (is_ebands .EQ. 1) then
    !     deallocate(pEB)
    ! endif
    ! deallocate(ntot)
 module energy_band_data
