!*******************************************************************************
! Module of pressure tensor. It contains subroutines to calculate the scalar
! pressure from the pressure tensor, to calculate the divergence of the pressure
! tensor, the gradient of the scalar pressure.
!*******************************************************************************
module pressure_tensor
    use constants, only: fp
    use mpi_topology, only: htg
    use pic_fields, only: pxx, pyy, pzz, pxy, pxz, pyz
    implicit none
    private
    public pscalar, init_scalar_pressure, init_div_ptensor, init_grad_pscalar, &
           free_scalar_pressure, free_div_ptensor, free_grad_pscalar, &
           calc_scalar_pressure, calc_grad_pscalar, calc_div_ptensor

    real(fp), allocatable, dimension(:, :, :) :: pscalar
    ! Divergence of the pressure tensor.
    real(fp), allocatable, dimension(:, :, :) :: divp_x, divp_y, divp_z
    ! Gradient of the scalar pressure.
    real(fp), allocatable, dimension(:, :, :) :: gradp_x, gradp_y, gradp_z
    
    contains

    !---------------------------------------------------------------------------
    ! Initialize the scalar pressure.
    !---------------------------------------------------------------------------
    subroutine init_scalar_pressure
        implicit none
        allocate(pscalar(htg%nx, htg%ny, htg%nz))
    end subroutine init_scalar_pressure

    !---------------------------------------------------------------------------
    ! Initialize the divergence of the pressure tensor.
    !---------------------------------------------------------------------------
    subroutine init_div_ptensor
        implicit none
        allocate(divp_x(htg%nx, htg%ny, htg%nz))
        allocate(divp_y(htg%nx, htg%ny, htg%nz))
        allocate(divp_z(htg%nx, htg%ny, htg%nz))
    end subroutine init_div_ptensor

    !---------------------------------------------------------------------------
    ! Initialize the gradient of the scalar pressure.
    !---------------------------------------------------------------------------
    subroutine init_grad_pscalar
        implicit none
        allocate(gradp_x(htg%nx, htg%ny, htg%nz))
        allocate(gradp_y(htg%nx, htg%ny, htg%nz))
        allocate(gradp_z(htg%nx, htg%ny, htg%nz))
    end subroutine init_grad_pscalar

    !---------------------------------------------------------------------------
    ! Free the scalar pressure.
    !---------------------------------------------------------------------------
    subroutine free_scalar_pressure
        implicit none
        deallocate(pscalar)
    end subroutine free_scalar_pressure

    !---------------------------------------------------------------------------
    ! Free the divergence of the pressure tensor.
    !---------------------------------------------------------------------------
    subroutine free_div_ptensor
        implicit none
        deallocate(divp_x, divp_y, divp_z)
    end subroutine free_div_ptensor

    !---------------------------------------------------------------------------
    ! Free the gradient of the scalar pressure.
    !---------------------------------------------------------------------------
    subroutine free_grad_pscalar
        implicit none
        deallocate(gradp_x, gradp_y, gradp_z)
    end subroutine free_grad_pscalar

    !---------------------------------------------------------------------------
    ! Calculate the scalar pressure from the pressure tensor.
    !---------------------------------------------------------------------------
    subroutine calc_scalar_pressure
        implicit none
        pscalar = (pxx + pyy + pzz) / 3.0
    end subroutine calc_scalar_pressure

    !---------------------------------------------------------------------------
    ! Calculate the divergence of the pressure tensor.
    !---------------------------------------------------------------------------
    subroutine calc_div_ptensor
        use constants, only: dp
        use neighbors_module, only: ixl, iyl, izl, ixh, iyh, izh, idx, idy, idz
        implicit none
        integer :: nx, ny, nz, ix, iy, iz

        nx = htg%nx
        ny = htg%ny
        nz = htg%nz

        do iz = 1, nz
            do iy = 1, ny
                do ix = 1, nx
                    divp_x(ix,iy,iz) = (pxx(ixh(ix),iy,iz)-pxx(ixl(ix),iy,iz))*idx(ix) + &
                                       (pxy(ix,iyh(iy),iz)-pxy(ix,iyl(iy),iz))*idy(iy) + &
                                       (pxz(ix,iy,izh(iz))-pxz(ix,iy,izl(iz)))*idz(iz)
                    divp_y(ix,iy,iz) = (pxy(ixh(ix),iy,iz)-pxy(ixl(ix),iy,iz))*idx(ix) + &
                                       (pyy(ix,iyh(iy),iz)-pyy(ix,iyl(iy),iz))*idy(iy) + &
                                       (pyz(ix,iy,izh(iz))-pyz(ix,iy,izl(iz)))*idz(iz)
                    divp_z(ix,iy,iz) = (pxz(ixh(ix),iy,iz)-pxz(ixl(ix),iy,iz))*idx(ix) + &
                                       (pyz(ix,iyh(iy),iz)-pyz(ix,iyl(iy),iz))*idy(iy) + &
                                       (pzz(ix,iy,izh(iz))-pzz(ix,iy,izl(iz)))*idz(iz)
                enddo  ! X
            enddo  ! Y
        enddo  ! Z
    end subroutine calc_div_ptensor

    !---------------------------------------------------------------------------
    ! Calculate the gradient of the scalar pressure.
    !---------------------------------------------------------------------------
    subroutine calc_grad_pscalar
        use constants, only: dp
        use neighbors_module, only: ixl, iyl, izl, ixh, iyh, izh, idx, idy, idz
        implicit none
        integer :: nx, ny, nz, ix, iy, iz

        nx = htg%nx
        ny = htg%ny
        nz = htg%nz

        do iz = 1, nz
            do iy = 1, ny
                do ix = 1, nx
                    gradp_x(ix,iy,iz) = (pscalar(ixh(ix),iy,iz)-pscalar(ixl(ix),iy,iz)) * idx(ix)
                    gradp_y(ix,iy,iz) = (pscalar(ix,iyh(iy),iz)-pscalar(ix,iyl(iy),iz)) * idy(iy)
                    gradp_z(ix,iy,iz) = (pscalar(ix,iy,izh(iz))-pscalar(ix,iy,izl(iz))) * idz(iz)
                enddo  ! X
            enddo  ! Y
        enddo  ! Z
    end subroutine calc_grad_pscalar

end module pressure_tensor
