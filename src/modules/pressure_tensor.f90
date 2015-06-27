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
        implicit none
    end subroutine calc_div_ptensor

end module pressure_tensor
