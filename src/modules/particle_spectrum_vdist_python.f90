!*******************************************************************************
! Main program to calculate particle spectrum and velocity distributions in a
! user defined box.
!*******************************************************************************
subroutine particle_spectrum_vdist_box
    use particle_spectrum_vdist_module, only: particle_spectrum_vdist_main
    implicit none
    call particle_spectrum_vdist_main
end subroutine particle_spectrum_vdist_box
