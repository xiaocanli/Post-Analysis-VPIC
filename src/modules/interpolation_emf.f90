!*******************************************************************************
! Module of doing interpolation in electromagnetic fields and their derivatives
!*******************************************************************************
module interpolation_emf
    use constants, only: fp
    implicit none
    private
    public init_emfields, free_emfields, init_emfields_derivatives, &
           free_emfields_derivatives, read_emfields_single, &
           calc_interp_weights, calc_emfields_derivatives, trilinear_interp, &
           calc_b_norm, calc_gradient_B, calc_curvature
    public bx0, by0, bz0, ex0, ey0, ez0, absB0, dbxdx0, dbxdy0, dbxdz0, &
           dbydx0, dbydy0, dbydz0, dbzdx0, dbzdy0, dbzdz0, bxn, byn, bzn, &
           dBdx, dBdy, dBdz, kappax, kappay, kappaz
    real(fp), allocatable, dimension(:,:,:) :: ex, ey, ez, bx, by, bz
    real(fp), allocatable, dimension(:,:,:) :: dbxdx, dbxdy, dbxdz
    real(fp), allocatable, dimension(:,:,:) :: dbydx, dbydy, dbydz
    real(fp), allocatable, dimension(:,:,:) :: dbzdx, dbzdy, dbzdz
    real(fp) :: bx0, by0, bz0, ex0, ey0, ez0, absB0
    real(fp) :: dbxdx0, dbxdy0, dbxdz0
    real(fp) :: dbydx0, dbydy0, dbydz0
    real(fp) :: dbzdx0, dbzdy0, dbzdz0
    real(fp) :: bxn, byn, bzn     ! Norm of the magnetic field
    real(fp) :: dBdx, dBdy, dBdz  ! The gradient of B
    real(fp) :: kappax, kappay, kappaz  ! The curvature of the magnetic field
    real(fp), dimension(2,2,2) :: weights  ! The weights for trilinear interpolation

    contains

    !---------------------------------------------------------------------------
    ! Initialize the electromagnetic fields
    !---------------------------------------------------------------------------
    subroutine init_emfields
        use picinfo, only: domain
        implicit none
        integer :: nx, ny, nz
        nx = domain%pic_nx + 2  ! Including ghost cells
        ny = domain%pic_ny + 2
        nz = domain%pic_nz + 2

        allocate(ex(nx, ny, nz))
        allocate(ey(nx, ny, nz))
        allocate(ez(nx, ny, nz))
        allocate(bx(nx, ny, nz))
        allocate(by(nx, ny, nz))
        allocate(bz(nx, ny, nz))

        ex = 0.0; ey = 0.0; ez = 0.0
        bx = 0.0; by = 0.0; bz = 0.0
    end subroutine init_emfields

    !---------------------------------------------------------------------------
    ! Initialize the derivatives of the electromagnetic fields
    !---------------------------------------------------------------------------
    subroutine init_emfields_derivatives
        use picinfo, only: domain
        implicit none
        integer :: nx, ny, nz
        nx = domain%pic_nx + 2  ! Including ghost cells
        ny = domain%pic_ny + 2
        nz = domain%pic_nz + 2

        allocate(dbxdx(nx, ny, nz))
        allocate(dbxdy(nx, ny, nz))
        allocate(dbxdz(nx, ny, nz))
        allocate(dbydx(nx, ny, nz))
        allocate(dbydy(nx, ny, nz))
        allocate(dbydz(nx, ny, nz))
        allocate(dbzdx(nx, ny, nz))
        allocate(dbzdy(nx, ny, nz))
        allocate(dbzdz(nx, ny, nz))

        dbxdx = 0.0; dbxdy = 0.0; dbxdz = 0.0
        dbydx = 0.0; dbydy = 0.0; dbydz = 0.0
        dbzdx = 0.0; dbzdy = 0.0; dbzdz = 0.0
    end subroutine init_emfields_derivatives

    !---------------------------------------------------------------------------
    ! Free the electromagnetic fields
    !---------------------------------------------------------------------------
    subroutine free_emfields
        implicit none
        deallocate(ex, ey, ez)
        deallocate(bx, by, bz)
    end subroutine free_emfields

    !---------------------------------------------------------------------------
    ! Free the derivatives of the electromagnetic fields
    !---------------------------------------------------------------------------
    subroutine free_emfields_derivatives
        implicit none
        deallocate(dbxdx, dbxdy, dbxdz)
        deallocate(dbydx, dbydy, dbydz)
        deallocate(dbzdx, dbzdy, dbzdz)
    end subroutine free_emfields_derivatives

    !---------------------------------------------------------------------------
    ! Read the fields for a single MPI process of PIC simulation.
    ! Inputs:
    !   tindex0: the time step index.
    !   pic_mpi_id: MPI id for the PIC simulation to identify the file.
    !---------------------------------------------------------------------------
    subroutine read_emfields_single(tindex0, pic_mpi_id)
        use path_info, only: rootpath
        use constants, only: fp
        use file_header, only: read_boilerplate, read_fields_header, fheader
        implicit none
        integer, intent(in) :: tindex0, pic_mpi_id
        character(len=150) :: fname
        logical :: is_exist
        integer :: fh   ! File handler
        integer :: n
        integer :: tindex

        fh = 10

        tindex = tindex0
        ! Index 0 does not have proper current, so use index 1 if it exists
        if (tindex == 0) then
            write(fname, "(A,I0,A8,I0,A1,I0)") trim(adjustl(rootpath))//"fields/T.", &
                  1, "/fields.", 1, ".", pic_mpi_id
            is_exist = .false.
            inquire(file=trim(fname), exist=is_exist)
            if (is_exist) tindex = 1
        endif
        write(fname, "(A,I0,A8,I0,A1,I0)") trim(adjustl(rootpath))//"fields/T.", &
              tindex, "/fields.", tindex, ".", pic_mpi_id
        is_exist = .false.
        inquire(file=trim(fname), exist=is_exist)
      
        if (is_exist) then 
            open(unit=10, file=trim(fname), access='stream', status='unknown', &
                 form='unformatted', action='read')
        else
            print *, "Can't find file:", fname
            print *
            print *, " ***  Terminating ***"
            stop
        endif

        call read_boilerplate(fh)
        call read_fields_header(fh)
        
        n = pic_mpi_id + 1  ! MPI ID starts at 0. The 1D rank starts at 1.

        read(fh) ex
        read(fh) ey
        read(fh) ez
        read(fh) bx
        read(fh) by
        read(fh) bz
        close(fh)
    end subroutine read_emfields_single

    !---------------------------------------------------------------------------
    ! Calculate the derivatives of the electromagnetic fields.
    !---------------------------------------------------------------------------
    subroutine calc_emfields_derivatives
        use picinfo, only: domain
        use neighbors_module, only: ixl, iyl, izl, ixh, iyh, izh, idx, idy, idz
        implicit none
        integer :: nx, ny, nz
        integer :: ix, iy, iz
        nx = domain%pic_nx + 2  ! Including ghost cells
        ny = domain%pic_ny + 2
        nz = domain%pic_nz + 2

        do ix = 1, nx
            dbxdx(ix, :, :) = (bx(ixh(ix), :, :) - bx(ixl(ix), :, :)) * idx(ix)
            dbydx(ix, :, :) = (by(ixh(ix), :, :) - by(ixl(ix), :, :)) * idx(ix)
            dbzdx(ix, :, :) = (bz(ixh(ix), :, :) - bz(ixl(ix), :, :)) * idx(ix)
        enddo

        do iy = 1, ny
            dbxdy(:, iy, :) = (bx(:, iyh(iy), :) - bx(:, iyl(iy), :)) * idy(iy)
            dbydy(:, iy, :) = (by(:, iyh(iy), :) - by(:, iyl(iy), :)) * idy(iy)
            dbzdy(:, iy, :) = (bz(:, iyh(iy), :) - bz(:, iyl(iy), :)) * idy(iy)
        enddo

        do iz = 1, nz
            dbxdz(:, :, iz) = (bx(:, :, izh(iz)) - bx(:, :, izl(iz))) * idz(iz)
            dbydz(:, :, iz) = (by(:, :, izh(iz)) - by(:, :, izl(iz))) * idz(iz)
            dbzdz(:, :, iz) = (bz(:, :, izh(iz)) - bz(:, :, izl(iz))) * idz(iz)
        enddo
    end subroutine calc_emfields_derivatives

    !---------------------------------------------------------------------------
    ! Calculate the weights for trilinear interpolation.
    !
    ! Input:
    !   dx, dy, dz: the distance to the lower-left corner, [0, 1]
    !---------------------------------------------------------------------------
    subroutine calc_interp_weights(dx, dy, dz)
        implicit none
        real(fp), intent(in) :: dx, dy, dz
        weights(1, 1, 1) = (1 - dx) * (1 - dy) * (1 - dz)
        weights(2, 1, 1) = dx * (1 - dy) * (1 - dz)
        weights(1, 2, 1) = (1 - dx) * dy * (1 - dz)
        weights(2, 2, 1) = dx * dy * (1 - dz)
        weights(1, 1, 2) = (1 - dx) * (1 - dy) * dz
        weights(2, 1, 2) = dx * (1 - dy) * dz
        weights(1, 2, 2) = (1 - dx) * dy * dz
        weights(2, 2, 2) = dx * dy * dz
    end subroutine calc_interp_weights

    !---------------------------------------------------------------------------
    ! Trilinear interpolation for the fields.
    ! 
    ! Input:
    !   ix0, iy0, iz0: the indices of the lower-left corner.
    !---------------------------------------------------------------------------
    subroutine trilinear_interp(ix0, iy0, iz0)
        implicit none
        integer, intent(in) :: ix0, iy0, iz0
        bx0 = sum(bx(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights) * 0.125
        by0 = sum(by(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights) * 0.125
        bz0 = sum(bz(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights) * 0.125
        ex0 = sum(ex(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights) * 0.125
        ey0 = sum(ey(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights) * 0.125
        ez0 = sum(ez(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights) * 0.125
        dbxdx0 = sum(dbxdx(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights) * 0.125
        dbxdy0 = sum(dbxdy(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights) * 0.125
        dbxdz0 = sum(dbxdz(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights) * 0.125
        dbydx0 = sum(dbydx(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights) * 0.125
        dbydy0 = sum(dbydy(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights) * 0.125
        dbydz0 = sum(dbydz(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights) * 0.125
        dbzdx0 = sum(dbzdx(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights) * 0.125
        dbzdy0 = sum(dbzdy(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights) * 0.125
        dbzdz0 = sum(dbzdz(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights) * 0.125
    end subroutine trilinear_interp

    !---------------------------------------------------------------------------
    ! Calculate the norm of the magnetic field.
    !---------------------------------------------------------------------------
    subroutine calc_b_norm
        implicit none
        real(fp) :: ib
        absB0 = sqrt(bx0**2 + by0**2 + bz0**2)
        ib = 1 / absB0
        bxn = bx0 * ib
        byn = by0 * ib
        bzn = bz0 * ib
    end subroutine calc_b_norm

    !---------------------------------------------------------------------------
    ! Calculate the gradient of B. calc_b_norm should be called previously.
    !---------------------------------------------------------------------------
    subroutine calc_gradient_B
        implicit none
        dBdx = bxn * dbxdx0 + byn * dbydx0 + bzn * dbzdx0
        dBdy = bxn * dbxdy0 + byn * dbydy0 + bzn * dbzdy0
        dBdz = bxn * dbxdz0 + byn * dbydz0 + bzn * dbzdz0
    end subroutine calc_gradient_B

    !---------------------------------------------------------------------------
    ! Calculate the curvature of the magnetic field. calc_gradient_B should be
    ! called previously.
    !---------------------------------------------------------------------------
    subroutine calc_curvature
        implicit none
        real(fp) :: b_dot_gradB, ib2, ib
        b_dot_gradB = bxn * dBdx + byn * dBdy + bzn * dBdz
        ib = 1 / absB0
        ib2 = ib * ib
        kappax = (bxn*dbxdx0 + byn*dbxdy0 + bzn*dbxdz0)*ib - bx0*b_dot_gradB*ib2
        kappay = (bxn*dbydx0 + byn*dbydy0 + bzn*dbydz0)*ib - by0*b_dot_gradB*ib2
        kappaz = (bxn*dbzdx0 + byn*dbzdy0 + bzn*dbzdz0)*ib - bz0*b_dot_gradB*ib2
    end subroutine calc_curvature
end module interpolation_emf
