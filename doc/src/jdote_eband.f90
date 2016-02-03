!*******************************************************************************
! The main procedure to calculate jdote for different energy band.
!*******************************************************************************
program compression
    use jdote_energy_band, only: read_config_jdote_eband
    implicit none
    call read_config_jdote_eband
end program compression
