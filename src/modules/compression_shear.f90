!*******************************************************************************
! Module dealing with compressional heating.
!*******************************************************************************
module compression
    use constants, only: fp
    use mpi_topology, only: htg
    implicit none
    private

    real(fp), allocatable, dimension(:, :, :) :: pdiv_u, pshear
    real(fp), allocatable, dimension(:, :, :) :: udot_div_ptensor
    real(fp), allocatable, dimension(:, :, :) :: divu, bbsigma

    contains

    !---------------------------------------------------------------------------
    ! Initialize the data arrays.
    !---------------------------------------------------------------------------
    subroutine init_compression
        implicit none
        allocate(pdiv_u(htg%nx, htg%ny, htg%nz))
        allocate(pshear(htg%nx, htg%ny, htg%nz))
        allocate(udot_div_ptensor(htg%nx, htg%ny, htg%nz))
        pdiv_u = 0.0
        pshear = 0.0
        udot_div_ptensor = 0.0
    end subroutine init_compression

    !---------------------------------------------------------------------------
    ! Initialize divu, pdiv_u. 
    !---------------------------------------------------------------------------
    subroutine init_

    !---------------------------------------------------------------------------
    ! Free the data arrays.
    !---------------------------------------------------------------------------
    subroutine free_compression
        implicit none
        deallocate(pdiv_u, pshear, udot_div_ptensor)
    end subroutine free_compression

    !---------------------------------------------------------------------------
    !---------------------------------------------------------------------------

    !---------------------------------------------------------------------------
    ! Calculate p\nabla\cdot\vec{u}. Here, p is the scalar pressure.
    !---------------------------------------------------------------------------
    subroutine calc_pdiv_u
        use pressure_tensor, only: pscalar
        use pic_fields, only: ux, uy, uz
        use neighbors_module, only: ixl, iyl, izl, ixh, iyh, izh, idx, idy, idz
        implicit none
        integer :: nx, ny, nz, ix, iy, iz
        real(fp) :: divu

        nx = htg%nx
        ny = htg%ny
        nz = htg%nz

        do iz = 1, nz
            do iy = 1, ny
                do ix = 1, nx
                    divu = (ux(ixh(ix),iy,iz)-ux(ixl(ix),iy,iz))*idx(ix) + &
                           (uy(ix,iyh(iy),iz)-uy(ix,iyl(iy),iz))*idy(iy) + &
                           (uz(ix,iy,izh(iz))-uz(ix,iy,izl(iz)))*idz(iz)
                    pdiv_u(ix, iy, iz) = pscalar(ix, iy, iz) * divu
                enddo
            enddo
        enddo
    end subroutine calc_pdiv_u

    !---------------------------------------------------------------------------
    ! Calculate (p_\parallel - p_\perp)b_ib_j\sigma_{ij}, where \sigma_{ij}
    ! is the shear tensor.
    !---------------------------------------------------------------------------
    subroutine calc_pshear
        use para_perp_pressure, only: ppara, pperp
        use pic_fields, only: bx, by, bz, ux, uy, uz
        implicit none
    end subroutine calc_pshear

end module compression
