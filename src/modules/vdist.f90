!*******************************************************************************
! Module for velocity distribution.
!*******************************************************************************
module velocity_distribution
    use constants, only: fp
    implicit none
    private

    real(fp), allocatable, dimension(:, :) :: fvel_2d, fvel_xy, fvel_xz, fvel_yz
    real(fp), allocatable, dimension(:) :: fvel_para, fvel_perp

    contains

    !---------------------------------------------------------------------------
    ! Initialize 2D velocity distributions.
    !---------------------------------------------------------------------------
    subroutine init_vdist_2d
        use spectrum_config, only: nbins_vdist
        implicit none
        allocate(fvel_2d(nbins_vdist*2, nbins_vdist))
        allocate(fvel_xy(nbins_vdist*2, nbins_vdist*2))
        allocate(fvel_xz(nbins_vdist*2, nbins_vdist*2))
        allocate(fvel_yz(nbins_vdist*2, nbins_vdist*2))
        call set_vdist_2d_zero
    end subroutine init_vdist_2d

    !---------------------------------------------------------------------------
    ! Set 2D velocity distributions to zero.
    !---------------------------------------------------------------------------
    subroutine set_vdist_2d_zero
        implicit none
        fvel_2d = 0.0
        fvel_xy = 0.0
        fvel_xz = 0.0
        fvel_yz = 0.0
    end subroutine set_vdist_2d_zero

    !---------------------------------------------------------------------------
    ! Free 2D velocity distribution.
    !---------------------------------------------------------------------------
    subroutine free_vdist_2d
        implicit none
        deallocate(fvel_2d)
        deallocate(fvel_xy, fvel_xz, fvel_yz)
    end subroutine free_vdist_2d

    !---------------------------------------------------------------------------
    ! Initialize 1D velocity distributions (parallel and perpendicular to the
    ! local magnetic field.
    !---------------------------------------------------------------------------
    subroutine init_vdist_1d
        use spectrum_config, only: nbins_vdist
        implicit none
        allocate(fvel_para(nbins_vdist*2))
        allocate(fvel_perp(nbins_vdist))
        call set_vdist_1d_zero
    end subroutine init_vdist_1d

    !---------------------------------------------------------------------------
    ! Set 1D velocity distributions to zero.
    !---------------------------------------------------------------------------
    subroutine set_vdist_1d_zero
        implicit none
        fvel_para = 0.0
        fvel_perp = 0.0
    end subroutine set_vdist_1d_zero
    !---------------------------------------------------------------------------
    ! Free 1D velocity distributions.
    !---------------------------------------------------------------------------
    subroutine free_vdist_1d
        implicit none
        deallocate(fvel_para, fvel_perp)
    end subroutine free_vdist_1d

    !---------------------------------------------------------------------------
    ! Read particle data and calculate the energy spectrum.
    ! Input:
    !   tindex: the time index, indicating the time step numbers in PIC simulation.
    !   species: 'e' for electron. 'h' for others.
    !---------------------------------------------------------------------------
    subroutine CalVdistribution(species)
        use constants, only: fp
        use picinfo, only: pTopo
        use structures, only: h0
        implicit none
        character(len=1), intent(in) :: species
        integer :: tindex, tinterval, nt
        character(len=50) :: ctindex, dataset, fname, cid
        integer :: ix, iy, iz, np, iptl
        real(fp) :: Bx_avg, By_avg, Bz_avg, Bxc, Byc, Bzc

        call ReadMagneticField(Bx_avg, By_avg, Bz_avg, Bxc, Byc, Bzc)
        call get_tinfo(tinterval, nt)
        tindex = tp * tinterval
        write(ctindex, "(I0)") tindex
        dataset = "particle/T."//trim(ctindex)//"/"//species//"particle."
        ! Read particle data in parallel to generate distributions
        do iz = iz1_pic, iz2_pic
            do iy = iy1_pic, iy2_pic
                do ix = ix1_pic, ix2_pic
                    np = ix + iy*pTopo%tx + iz*pTopo%tx*pTopo%ty
                    write(cid, "(I0)") np
                    fname = trim(dataset)//trim(ctindex)//"."//trim(cid)
                    open(unit=10, file=trim(fname), status='unknown', &
                        form='unformatted', access='stream', action='read')
                    print *,"Reading  --> ", trim(fname), "    np = ", np
                    call ReadHeaders(10)
                    ! Loop over particles
                    do iptl = 1, h0%dim, 1
                        !call GetSingleParticleEnergy(10)
                    enddo
                    close(10)
                enddo ! x-loop
            enddo ! y-loop
        enddo ! z-loop
    end subroutine CalVdistribution

end module velocity_distribution

! !*******************************************************************************
! ! To get the ratio of particle output interval and the fields_interval.
! ! These two are different for these two. And the ratio is given in sigma.cxx
! !*******************************************************************************
! subroutine GetRatio_OutputInterval(ratio_interval)
!     implicit none
!     integer, intent(out) :: ratio_interval
!     character(len=150) :: buff
!     character(len=50) :: buff1, format1
!     integer :: len1, len2, len3
!     ratio_interval = 0
!     open(unit=40,file='sigma.cxx', status='old')
!     read(40,'(A)') buff
!     do while (index(buff, 'int Hhydro_interval = ') == 0)
!         read(40,'(A)') buff
!     enddo
!     read(40, '(A)') buff
!     len1 = len(trim(buff))
!     ! "int eparticle_interval = " has 25 characters
!     len2 = index(buff, 'int') + 24
!     ! The last 10 characters are "*interval;"
!     len3 = len1 - len2 - 10
!     write(format1, "(A,I2.2,A,I1.1,A,I1.1,A)") "(A", len2, ",I", len3, ".", len3, ")'"
!     read(buff, trim(adjustl(format1))) buff1, ratio_interval
!     close(40)
!     print*, "The ratio of particle output interval and field output interval:", &
!         ratio_interval
! end subroutine GetRatio_OutputInterval

! !*******************************************************************************
! ! Read the B fields at current time.
! ! 1. Bx, By, Bz at ixc, iyc, izc
! ! 2. Bx_avg, By_avg, Bz_avg got form the average of the B fields in the whole box.
! !*******************************************************************************
! subroutine ReadMagneticField(Bx_avg, By_avg, Bz_avg, Bxc, Byc, Bzc)
!     use constants, only: MyLongIntType, fp
!     use parameters, only: ix1, iy1, iz1, ix2, iy2, iz2, ixc, iyc, izc, tp
!     use picinfo, only: domain
!     implicit none
!     real(fp), intent(out) :: Bx_avg, By_avg, Bz_avg, Bxc, Byc, Bzc
!     integer :: ratio_interval, ix, iy, iz, nx, ny, nz, npoints
!     integer(kind=MyLongIntType) :: pos0, pos1, pos2
!     real(fp) :: Bx, By, Bz 
!     call GetRatio_OutputInterval(ratio_interval)
!     nx = domain%nx
!     ny = domain%ny
!     nz = domain%nz
!     open(unit=51,file='data/bx.gda', access='stream', &
!         form='unformatted',action='read') 
!     open(unit=52,file='data/by.gda', access='stream', &
!         form='unformatted',action='read') 
!     open(unit=53,file='data/bz.gda', access='stream', &
!         form='unformatted',action='read') 
!     ! The number is too huge. Two steps to avoid truncation.
!     pos0 = nx * ny * nz * 4
!     pos0 = pos0 * (ratio_interval*tp - 1)
!     npoints = (ix2-ix1+1) * (iy2-iy1+1) * (iz2-iz1+1)
!     Bx_avg = 0.0
!     By_avg = 0.0
!     Bz_avg = 0.0
!     do iz = iz1, iz2
!         do iy = iy1, iy2
!             do ix = ix1, ix2
!                 pos2 = (ix-1) + (iy-1)*nx + (iz-1)*nx*ny
!                 pos1 = pos0 + pos2*4 + 1
!                 read(51, pos=pos1) Bx
!                 read(52, pos=pos1) By
!                 read(53, pos=pos1) Bz
!                 Bx_avg = Bx_avg + Bx
!                 By_avg = By_avg + By
!                 Bz_avg = Bz_avg + Bz
!             enddo
!         enddo
!     enddo
!     Bx_avg = Bx_avg / npoints
!     By_avg = By_avg / npoints
!     Bz_avg = Bz_avg / npoints

!     ! Magnetic field at the center of the box.
!     pos2 = (ixc-1) + (iyc-1)*nx + (izc-1)*nx*ny
!     pos1 = pos0 + pos2*4 + 1
!     read(51, pos=pos1) Bxc
!     read(52, pos=pos1) Byc
!     read(53, pos=pos1) Bzc

!     close(51)
!     close(52)
!     close(53)

!     print *, "---------------------------------------------------"
!     print *, "The B field at the center:       ", Bxc, Byc, Bzc
!     print *, "The averaged B field in the box: ", Bx_avg, By_avg, Bz_avg
!     print *, "---------------------------------------------------" 
! end subroutine ReadMagneticField
