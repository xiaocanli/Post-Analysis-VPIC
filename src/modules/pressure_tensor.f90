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
    public pscalar

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
        use picinfo, only: domain
        use adjoint_points_module, only: adjoint_points 
        implicit none
        real(dp) :: idxh, idyh, idzh
        integer :: nx, ny, nz, ix, iy, iz
        integer :: ix1, ix2, iy1, iy2, iz1, iz2

        nx = htg%nx
        ny = htg%ny
        nz = htg%nz
        idxh = domain%idxh  ! 1/(2.0*dx)
        idyh = domain%idyh
        idzh = domain%idzh

        do iz = 1, nz
            call adjoint_points(nz, iz, iz1, iz2)
            do iy = 1, ny
                call adjoint_points(ny, iy, iy1, iy2)
                do ix = 1, nx
                    call adjoint_points(nx, ix, ix1, ix2)
                    divp_x(ix,iy,iz) = (pxx(ix2,iy,iz)-pxx(ix1,iy,iz))*idxh + &
                                       (pxy(ix,iy2,iz)-pxy(ix,iy1,iz))*idyh + &
                                       (pxz(ix,iy,iz2)-pxz(ix,iy,iz1))*idzh
                    divp_y(ix,iy,iz) = (pxy(ix2,iy,iz)-pxy(ix1,iy,iz))*idxh + &
                                       (pyy(ix,iy2,iz)-pyy(ix,iy1,iz))*idyh + &
                                       (pyz(ix,iy,iz2)-pyz(ix,iy,iz1))*idzh
                    divp_z(ix,iy,iz) = (pxz(ix2,iy,iz)-pxz(ix1,iy,iz))*idxh + &
                                       (pyz(ix,iy2,iz)-pyz(ix,iy1,iz))*idyh + &
                                       (pzz(ix,iy,iz2)-pzz(ix,iy,iz1))*idzh
                enddo  ! X
            enddo  ! Y
        enddo  ! Z
    end subroutine calc_div_ptensor

    !---------------------------------------------------------------------------
    ! Calculate the gradient of the scalar pressure.
    !---------------------------------------------------------------------------
    subroutine calc_grad_pscalar
        use constants, only: dp
        use picinfo, only: domain
        implicit none
        integer :: nx, ny, nz
        real(dp) :: idxh, idyh, idzh, idx, idy, idz

        nx = htg%nx
        ny = htg%ny
        nz = htg%nz
        idx = domain%idx
        idy = domain%idy
        idz = domain%idz
        idxh = domain%idxh  ! 1/(2.0*dx)
        idyh = domain%idyh
        idzh = domain%idzh

        gradp_x(2:nx-1,:,:) = (pscalar(3:nx,:,:) - pscalar(1:nx-2,:,:)) * idxh
        gradp_x(1,:,:) = (pscalar(2,:,:) - pscalar(1,:,:)) * idx
        gradp_x(nx,:,:) = (pscalar(nx,:,:) - pscalar(nx-1,:,:)) * idx

        gradp_y(:,2:ny-1,:) = (pscalar(:,3:ny,:) - pscalar(:,1:ny-2,:)) * idyh
        gradp_y(:,1,:) = (pscalar(:,2,:) - pscalar(:,1,:)) * idy
        gradp_y(:,ny,:) = (pscalar(:,ny,:) - pscalar(:,ny-1,:)) * idy

        gradp_z(:,:,2:nz-1) = (pscalar(:,:,3:nz) - pscalar(:,:,1:nz-2)) * idzh
        gradp_z(:,:,1) = (pscalar(:,:,2) - pscalar(:,:,1)) * idz
        gradp_z(:,:,nz) = (pscalar(:,:,nz) - pscalar(:,:,nz-1)) * idz
    end subroutine calc_grad_pscalar

end module pressure_tensor
