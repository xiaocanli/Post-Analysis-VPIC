!*******************************************************************************
! Mass and charge for particles used in current analysis.
!*******************************************************************************
module particle_info
    use constants, only: fp
    implicit none
    private
    public ptl_mass, ptl_charge, get_ptl_mass_charge, species, ibtag, &
           sqrt_ptl_mass

    real(fp) :: ptl_mass, ptl_charge, sqrt_ptl_mass
    character(len=1) :: species
    character(len=2) :: ibtag ! Current energy band (transferred to string).

    contains

    subroutine get_ptl_mass_charge(species)
        use constants, only: qi, qe
        use picinfo, only: mime
        implicit none
        character(*), intent(in) :: species

        if (species == 'e') then
            ptl_mass = 1.0
            sqrt_ptl_mass = 1.0
            ptl_charge = qe
        else if (species == 'i') then
            ptl_mass = mime
            sqrt_ptl_mass = sqrt(mime)
            ptl_charge = qi
        else
            print*, "Error: particles don't exist."
            stop
        endif
    end subroutine get_ptl_mass_charge

end module particle_info
