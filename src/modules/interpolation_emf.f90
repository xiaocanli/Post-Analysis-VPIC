!*******************************************************************************
! Module of doing interpolation in electromagnetic fields and their derivatives
!*******************************************************************************
module interpolation_emf
    use constants, only: fp
    use interpolation_funs, only: bounding_indcies
    implicit none
    private
    save

    public init_emfields, free_emfields, init_emfields_derivatives, &
           free_emfields_derivatives, read_emfields_single, &
           calc_interp_weights, calc_emfields_derivatives, calc_b_norm, &
           calc_gradient_B, calc_curvature, trilinear_interp_bx,&
           trilinear_interp_by, trilinear_interp_bz, trilinear_interp_ex, &
           trilinear_interp_ey, trilinear_interp_ez, trilinear_interp_only_bx, &
           trilinear_interp_only_by, trilinear_interp_only_bz, set_emf, &
           set_emf_derivatives
    public init_absb_derivatives_single, free_absb_derivatives_single, &
        set_absb_derivatives, trilinear_interp_absb_derivatives
    public bx0, by0, bz0, ex0, ey0, ez0, absB0, dbxdx0, dbxdy0, dbxdz0, &
           dbydx0, dbydy0, dbydz0, dbzdx0, dbzdy0, dbzdz0, bxn, byn, bzn, &
           dBdx, dBdy, dBdz, kappax, kappay, kappaz
    public dbdx0, dbdy0, dbdz0
    real(fp), allocatable, dimension(:,:,:) :: ex, ey, ez, bx, by, bz
    real(fp), allocatable, dimension(:,:,:) :: ex1, ey1, ez1, bx1, by1, bz1
    real(fp), allocatable, dimension(:,:,:) :: dbxdx, dbxdy, dbxdz
    real(fp), allocatable, dimension(:,:,:) :: dbydx, dbydy, dbydz
    real(fp), allocatable, dimension(:,:,:) :: dbzdx, dbzdy, dbzdz
    real(fp), allocatable, dimension(:,:,:) :: dbdx_s, dbdy_s, dbdz_s
    real(fp) :: bx0, by0, bz0, ex0, ey0, ez0, absB0
    real(fp) :: dbxdx0, dbxdy0, dbxdz0
    real(fp) :: dbydx0, dbydy0, dbydz0
    real(fp) :: dbzdx0, dbzdy0, dbzdz0
    real(fp) :: dbdx0, dbdy0, dbdz0
    real(fp) :: bxn, byn, bzn     ! Norm of the magnetic field
    real(fp) :: dBdx, dBdy, dBdz  ! The gradient of B
    real(fp) :: kappax, kappay, kappaz  ! The curvature of the magnetic field
    real(fp), dimension(2,2,2) :: weights  ! The weights for trilinear interpolation
    integer :: nx, ny, nz

    contains

    !---------------------------------------------------------------------------
    ! Initialize the electromagnetic fields
    !---------------------------------------------------------------------------
    subroutine init_emfields
        use picinfo, only: domain
        implicit none
        nx = domain%pic_nx + 2  ! Including ghost cells
        ny = domain%pic_ny + 2
        nz = domain%pic_nz + 2

        allocate(ex(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(ey(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(ez(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(bx(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(by(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(bz(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(ex1(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(ey1(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(ez1(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(bx1(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(by1(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(bz1(0:nx-1, 0:ny-1, 0:nz-1))

        ex = 0.0; ey = 0.0; ez = 0.0
        bx = 0.0; by = 0.0; bz = 0.0
        ex1 = 0.0; ey1 = 0.0; ez1 = 0.0
        bx1 = 0.0; by1 = 0.0; bz1 = 0.0
    end subroutine init_emfields

    !---------------------------------------------------------------------------
    ! Initialize the derivatives of the magnitude of the magnetic field
    !---------------------------------------------------------------------------
    subroutine init_absb_derivatives_single
        use picinfo, only: domain
        implicit none
        nx = domain%pic_nx + 2  ! Including ghost cells
        ny = domain%pic_ny + 2
        nz = domain%pic_nz + 2

        allocate(dbdx_s(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(dbdy_s(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(dbdz_s(0:nx-1, 0:ny-1, 0:nz-1))

        dbdx_s = 0.0; dbdy_s = 0.0; dbdz_s = 0.0
    end subroutine init_absb_derivatives_single

    !---------------------------------------------------------------------------
    ! Initialize the derivatives of the electromagnetic fields
    !---------------------------------------------------------------------------
    subroutine init_emfields_derivatives
        use picinfo, only: domain
        implicit none

        nx = domain%pic_nx + 2  ! Including ghost cells
        ny = domain%pic_ny + 2
        nz = domain%pic_nz + 2

        allocate(dbxdx(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(dbxdy(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(dbxdz(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(dbydx(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(dbydy(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(dbydz(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(dbzdx(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(dbzdy(0:nx-1, 0:ny-1, 0:nz-1))
        allocate(dbzdz(0:nx-1, 0:ny-1, 0:nz-1))

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
        deallocate(ex1, ey1, ez1)
        deallocate(bx1, by1, bz1)
    end subroutine free_emfields

    !---------------------------------------------------------------------------
    ! Free the derivatives of the magnitude of the magnetic field
    !---------------------------------------------------------------------------
    subroutine free_absb_derivatives_single
        implicit none
        deallocate(dbdx_s, dbdy_s, dbdz_s)
    end subroutine free_absb_derivatives_single

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
    ! Open the fields file for a single MPI process of PIC simulation.
    ! Inputs:
    !   fh: file handler.
    !   tindex0: the time step index.
    !   pic_mpi_id: MPI id for the PIC simulation to identify the file.
    !---------------------------------------------------------------------------
    subroutine open_emfields_file(fh, tindex0, pic_mpi_id)
        use path_info, only: rootpath
        use file_header, only: read_boilerplate, read_fields_header
        implicit none
        integer, intent(in) :: fh, tindex0, pic_mpi_id
        character(len=150) :: fname
        logical :: is_exist
        integer :: tindex
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
            open(unit=fh, file=trim(fname), access='stream', status='unknown', &
                 form='unformatted', action='read')
        else
            print *, "Can't find file:", fname
            print *
            print *, " ***  Terminating ***"
            stop
        endif

        call read_boilerplate(fh)
        call read_fields_header(fh)

    end subroutine open_emfields_file

    !---------------------------------------------------------------------------
    ! Read the fields for a neighbor.
    ! Inputs:
    !   fh: file handler.
    !---------------------------------------------------------------------------
    subroutine read_neighbor_fields(fh)
        implicit none
        integer, intent(in) :: fh
        read(fh) ex1
        read(fh) ey1
        read(fh) ez1
        read(fh) bx1
        read(fh) by1
        read(fh) bz1
    end subroutine read_neighbor_fields

    !---------------------------------------------------------------------------
    ! Read the fields for a single MPI process of PIC simulation.
    ! Inputs:
    !   tindex0: the time step index.
    !   pic_mpi_id: MPI id for the PIC simulation to identify the file.
    !---------------------------------------------------------------------------
    subroutine read_emfields_single(tindex0, pic_mpi_id)
        use constants, only: fp
        use neighbors_module, only: get_mpi_neighbors
        use picinfo, only: domain
        implicit none
        integer, intent(in) :: tindex0, pic_mpi_id
        integer :: nxl, nxh, nyl, nyh, nzl, nzh
        integer :: fh

        fh = 10
        call open_emfields_file(fh, tindex0, pic_mpi_id)

        read(fh) ex
        read(fh) ey
        read(fh) ez
        read(fh) bx
        read(fh) by
        read(fh) bz
        close(fh)

        ! Get the neighbors.
        call get_mpi_neighbors(pic_mpi_id, nxl, nxh, nyl, nyh, nzl, nzh)

        if (nxl > 0) then
            call open_emfields_file(fh, tindex0, nxl)
            call read_neighbor_fields(fh)
            bx(0, :, :) = bx1(nx-2, :, :)
            by(0, :, :) = by1(nx-2, :, :)
            bz(0, :, :) = bz1(nx-2, :, :)
            ex(0, :, :) = ex1(nx-2, :, :)
            ey(0, :, :) = ey1(nx-2, :, :)
            ez(0, :, :) = ez1(nx-2, :, :)
            close(fh)
        endif

        if (nxh > 0) then
            call open_emfields_file(fh, tindex0, nxh)
            call read_neighbor_fields(fh)
            bx(nx-1, :, :) = bx1(1, :, :)
            by(nx-1, :, :) = by1(1, :, :)
            bz(nx-1, :, :) = bz1(1, :, :)
            ex(nx-1, :, :) = ex1(1, :, :)
            ey(nx-1, :, :) = ey1(1, :, :)
            ez(nx-1, :, :) = ez1(1, :, :)
            close(fh)
        endif

        if (nyl > 0) then
            call open_emfields_file(fh, tindex0, nyl)
            call read_neighbor_fields(fh)
            bx(:, 0, :) = bx1(:, ny-2, :)
            by(:, 0, :) = by1(:, ny-2, :)
            bz(:, 0, :) = bz1(:, ny-2, :)
            ex(:, 0, :) = ex1(:, ny-2, :)
            ey(:, 0, :) = ey1(:, ny-2, :)
            ez(:, 0, :) = ez1(:, ny-2, :)
            close(fh)
        endif

        if (nyh > 0) then
            call open_emfields_file(fh, tindex0, nyh)
            call read_neighbor_fields(fh)
            bx(:, ny-1, :) = bx1(:, 1, :)
            by(:, ny-1, :) = by1(:, 1, :)
            bz(:, ny-1, :) = bz1(:, 1, :)
            ex(:, ny-1, :) = ex1(:, 1, :)
            ey(:, ny-1, :) = ey1(:, 1, :)
            ez(:, ny-1, :) = ez1(:, 1, :)
            close(fh)
        endif

        if (nzl > 0) then
            call open_emfields_file(fh, tindex0, nzl)
            call read_neighbor_fields(fh)
            bx(:, :, 0) = bx1(:, :, nz-2)
            by(:, :, 0) = by1(:, :, nz-2)
            bz(:, :, 0) = bz1(:, :, nz-2)
            ex(:, :, 0) = ex1(:, :, nz-2)
            ey(:, :, 0) = ey1(:, :, nz-2)
            ez(:, :, 0) = ez1(:, :, nz-2)
            close(fh)
        endif

        if (nzh > 0) then
            call open_emfields_file(fh, tindex0, nzh)
            call read_neighbor_fields(fh)
            bx(:, :, nz-1) = bx1(:, :, 1)
            by(:, :, nz-1) = by1(:, :, 1)
            bz(:, :, nz-1) = bz1(:, :, 1)
            ex(:, :, nz-1) = ex1(:, :, 1)
            ey(:, :, nz-1) = ey1(:, :, 1)
            ez(:, :, nz-1) = ez1(:, :, 1)
            close(fh)
        endif

    end subroutine read_emfields_single

    !---------------------------------------------------------------------------
    ! Calculate the derivatives of the electromagnetic fields.
    !---------------------------------------------------------------------------
    subroutine calc_emfields_derivatives
        use picinfo, only: domain
        use neighbors_module, only: ixl, iyl, izl, ixh, iyh, izh, idx, idy, idz
        implicit none
        integer :: ix, iy, iz

        do ix = 0, nx - 1
            dbxdx(ix, :, :) = (bx(ixh(ix+1)-1, :, :) - bx(ixl(ix+1)-1, :, :)) * idx(ix+1)
            dbydx(ix, :, :) = (by(ixh(ix+1)-1, :, :) - by(ixl(ix+1)-1, :, :)) * idx(ix+1)
            dbzdx(ix, :, :) = (bz(ixh(ix+1)-1, :, :) - bz(ixl(ix+1)-1, :, :)) * idx(ix+1)
        enddo

        if (ny > 3) then
            do iy = 0, ny - 1
                dbxdy(:, iy, :) = (bx(:, iyh(iy+1)-1, :) - bx(:, iyl(iy+1)-1, :)) * idy(iy+1)
                dbydy(:, iy, :) = (by(:, iyh(iy+1)-1, :) - by(:, iyl(iy+1)-1, :)) * idy(iy+1)
                dbzdy(:, iy, :) = (bz(:, iyh(iy+1)-1, :) - bz(:, iyl(iy+1)-1, :)) * idy(iy+1)
            enddo
        else
            ! 2D simulation
            dbxdy = 0.0
            dbydy = 0.0
            dbzdy = 0.0
        endif

        do iz = 0, nz - 1
            dbxdz(:, :, iz) = (bx(:, :, izh(iz+1)-1) - bx(:, :, izl(iz+1)-1)) * idz(iz+1)
            dbydz(:, :, iz) = (by(:, :, izh(iz+1)-1) - by(:, :, izl(iz+1)-1)) * idz(iz+1)
            dbzdz(:, :, iz) = (bz(:, :, izh(iz+1)-1) - bz(:, :, izl(iz+1)-1)) * idz(iz+1)
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
    ! Trilinear interpolation for Bx, By, Bz.
    ! Input:
    !   ix0, iy0, iz0: the indices of the lower-left corner.
    !   dx, dy, dz: the distance to the lower-left corner, [0, 1]
    !---------------------------------------------------------------------------
    subroutine trilinear_interp_bx(ix0, iy0, iz0, dx, dy, dz)
        implicit none
        integer, intent(in) :: ix0, iy0, iz0
        real(fp), intent(in) :: dx, dy, dz
        call calc_interp_weights(dx, dy, dz)
        bx0 = sum(bx(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        dbxdx0 = sum(dbxdx(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        dbxdy0 = sum(dbxdy(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        dbxdz0 = sum(dbxdz(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
    end subroutine trilinear_interp_bx

    subroutine trilinear_interp_by(ix0, iy0, iz0, dx, dy, dz)
        implicit none
        integer, intent(in) :: ix0, iy0, iz0
        real(fp), intent(in) :: dx, dy, dz
        call calc_interp_weights(dx, dy, dz)
        by0 = sum(by(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        dbydx0 = sum(dbydx(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        dbydy0 = sum(dbydy(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        dbydz0 = sum(dbydz(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
    end subroutine trilinear_interp_by

    subroutine trilinear_interp_bz(ix0, iy0, iz0, dx, dy, dz)
        implicit none
        integer, intent(in) :: ix0, iy0, iz0
        real(fp), intent(in) :: dx, dy, dz
        call calc_interp_weights(dx, dy, dz)
        bz0 = sum(bz(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        dbzdx0 = sum(dbzdx(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        dbzdy0 = sum(dbzdy(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        dbzdz0 = sum(dbzdz(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
    end subroutine trilinear_interp_bz

    subroutine trilinear_interp_only_bx(ix0, iy0, iz0, dx, dy, dz)
        implicit none
        integer, intent(in) :: ix0, iy0, iz0
        real(fp), intent(in) :: dx, dy, dz
        call calc_interp_weights(dx, dy, dz)
        bx0 = sum(bx(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
    end subroutine trilinear_interp_only_bx

    subroutine trilinear_interp_only_by(ix0, iy0, iz0, dx, dy, dz)
        implicit none
        integer, intent(in) :: ix0, iy0, iz0
        real(fp), intent(in) :: dx, dy, dz
        call calc_interp_weights(dx, dy, dz)
        by0 = sum(by(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
    end subroutine trilinear_interp_only_by

    subroutine trilinear_interp_only_bz(ix0, iy0, iz0, dx, dy, dz)
        implicit none
        integer, intent(in) :: ix0, iy0, iz0
        real(fp), intent(in) :: dx, dy, dz
        call calc_interp_weights(dx, dy, dz)
        bz0 = sum(bz(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
    end subroutine trilinear_interp_only_bz

    !---------------------------------------------------------------------------
    ! Trilinear interpolation for Ex, Ey, Ez.
    ! Input:
    !   ix0, iy0, iz0: the indices of the lower-left corner.
    !   dx, dy, dz: the distance to the lower-left corner, [0, 1]
    !---------------------------------------------------------------------------
    subroutine trilinear_interp_ex(ix0, iy0, iz0, dx, dy, dz)
        implicit none
        integer, intent(in) :: ix0, iy0, iz0
        real(fp), intent(in) :: dx, dy, dz
        call calc_interp_weights(dx, dy, dz)
        ex0 = sum(ex(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
    end subroutine trilinear_interp_ex

    subroutine trilinear_interp_ey(ix0, iy0, iz0, dx, dy, dz)
        implicit none
        integer, intent(in) :: ix0, iy0, iz0
        real(fp), intent(in) :: dx, dy, dz
        call calc_interp_weights(dx, dy, dz)
        ey0 = sum(ey(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
    end subroutine trilinear_interp_ey

    subroutine trilinear_interp_ez(ix0, iy0, iz0, dx, dy, dz)
        implicit none
        integer, intent(in) :: ix0, iy0, iz0
        real(fp), intent(in) :: dx, dy, dz
        call calc_interp_weights(dx, dy, dz)
        ez0 = sum(ez(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
    end subroutine trilinear_interp_ez

    !---------------------------------------------------------------------------
    ! Trilinear interpolation for the derivatives of the magnitude of absB
    ! Input:
    !   ix0, iy0, iz0: the indices of the lower-left corner.
    !   dx, dy, dz: the distance to the lower-left corner, [0, 1]
    !---------------------------------------------------------------------------
    subroutine trilinear_interp_absb_derivatives(ix0, iy0, iz0, dx, dy, dz)
        implicit none
        integer, intent(in) :: ix0, iy0, iz0
        real(fp), intent(in) :: dx, dy, dz
        call calc_interp_weights(dx, dy, dz)
        dbdx0 = sum(dbdx_s(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        dbdy0 = sum(dbdy_s(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
        dbdz0 = sum(dbdz_s(ix0:ix0+1, iy0:iy0+1, iz0:iz0+1) * weights)
    end subroutine trilinear_interp_absb_derivatives

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

    !<--------------------------------------------------------------------------
    !< Set electromagnetic fields, which is read from translated files rather
    !< than directly from the PIC simulations
    !< Input:
    !<  i, j, k: MPI rank along each direction for the PIC simulation
    !<  tx, ty, tz: MPI sizes along each direction in the PIC simulation
    !<  sx, sy, sz: starting MPI rank along each direction in the PIC simulation
    !               for the MPI rank of current analysis
    !<--------------------------------------------------------------------------
    subroutine set_emf(i, j, k, tx, ty, tz, sx, sy, sz)
        use pic_fields, only: ext => ex, eyt => ey, ezt => ez, &
                              bxt => bx, byt => by, bzt => bz
        use picinfo, only: domain
        implicit none
        integer, intent(in) :: i, j, k, tx, ty, tz, sx, sy, sz
        integer :: ixs_lo, ixe_lo, ixs_gl, ixe_gl
        integer :: iys_lo, iye_lo, iys_gl, iye_gl
        integer :: izs_lo, ize_lo, izs_gl, ize_gl
        integer :: pnx, pny, pnz
        pnx = domain%pic_nx
        pny = domain%pic_ny
        pnz = domain%pic_nz
        call bounding_indcies(i, pnx, tx, sx, ixs_lo, ixe_lo, ixs_gl, ixe_gl)
        call bounding_indcies(j, pny, ty, sy, iys_lo, iye_lo, iys_gl, iye_gl)
        call bounding_indcies(k, pnz, tz, sz, izs_lo, ize_lo, izs_gl, ize_gl)
        ex = 0.0; ey = 0.0; ez = 0.0
        bx = 0.0; by = 0.0; bz = 0.0
        ex(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            ext(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        ey(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            eyt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        ez(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            ezt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        bx(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            bxt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        by(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            byt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        bz(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            bzt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        if (ixs_lo == 1) then
            ex(0, :, :) = ex(1, :, :)
            ey(0, :, :) = ey(1, :, :)
            ez(0, :, :) = ez(1, :, :)
            bx(0, :, :) = bx(1, :, :)
            by(0, :, :) = by(1, :, :)
            bz(0, :, :) = bz(1, :, :)
        endif
        if (ixe_lo == pnx) then
            ex(pnx+1, :, :) = ex(pnx, :, :)
            ey(pnx+1, :, :) = ey(pnx, :, :)
            ez(pnx+1, :, :) = ez(pnx, :, :)
            bx(pnx+1, :, :) = bx(pnx, :, :)
            by(pnx+1, :, :) = by(pnx, :, :)
            bz(pnx+1, :, :) = bz(pnx, :, :)
        endif
        if (iys_lo == 1) then
            ex(:, 0, :) = ex(:, 1, :)
            ey(:, 0, :) = ey(:, 1, :)
            ez(:, 0, :) = ez(:, 1, :)
            bx(:, 0, :) = bx(:, 1, :)
            by(:, 0, :) = by(:, 1, :)
            bz(:, 0, :) = bz(:, 1, :)
        endif
        if (iye_lo == pny) then
            ex(:, pny+1, :) = ex(:, pny, :)
            ey(:, pny+1, :) = ey(:, pny, :)
            ez(:, pny+1, :) = ez(:, pny, :)
            bx(:, pny+1, :) = bx(:, pny, :)
            by(:, pny+1, :) = by(:, pny, :)
            bz(:, pny+1, :) = bz(:, pny, :)
        endif
        if (izs_lo == 1) then
            ex(:, :, 0) = ex(:, :, 1)
            ey(:, :, 0) = ey(:, :, 1)
            ez(:, :, 0) = ez(:, :, 1)
            bx(:, :, 0) = bx(:, :, 1)
            by(:, :, 0) = by(:, :, 1)
            bz(:, :, 0) = bz(:, :, 1)
        endif
        if (ize_lo == pnz) then
            ex(:, :, pnz+1) = ex(:, :, pnz)
            ey(:, :, pnz+1) = ey(:, :, pnz)
            ez(:, :, pnz+1) = ez(:, :, pnz)
            bx(:, :, pnz+1) = bx(:, :, pnz)
            by(:, :, pnz+1) = by(:, :, pnz)
            bz(:, :, pnz+1) = bz(:, :, pnz)
        endif
    end subroutine set_emf

    !<--------------------------------------------------------------------------
    !< Set the derivatives of the electromagnetic fields, which is read from
    !< translated files rather than directly from the PIC simulations
    !<--------------------------------------------------------------------------
    subroutine set_emf_derivatives(i, j, k, tx, ty, tz, sx, sy, sz)
        use emf_derivatives, only: dbxdxt => dbxdx, dbxdyt => dbxdy, dbxdzt => dbxdz, &
            dbydxt => dbydx, dbydyt => dbydy, dbydzt => dbydz, &
            dbzdxt => dbzdx, dbzdyt => dbzdy, dbzdzt => dbzdz
        use picinfo, only: domain
        implicit none
        integer, intent(in) :: i, j, k, tx, ty, tz, sx, sy, sz
        integer :: ixs_lo, ixe_lo, ixs_gl, ixe_gl
        integer :: iys_lo, iye_lo, iys_gl, iye_gl
        integer :: izs_lo, ize_lo, izs_gl, ize_gl
        integer :: pnx, pny, pnz
        pnx = domain%pic_nx
        pny = domain%pic_ny
        pnz = domain%pic_nz
        call bounding_indcies(i, pnx, tx, sx, ixs_lo, ixe_lo, ixs_gl, ixe_gl)
        call bounding_indcies(j, pny, ty, sy, iys_lo, iye_lo, iys_gl, iye_gl)
        call bounding_indcies(k, pnz, tz, sz, izs_lo, ize_lo, izs_gl, ize_gl)
        dbxdx = 0.0; dbxdy = 0.0; dbxdz = 0.0
        dbydx = 0.0; dbydy = 0.0; dbydz = 0.0
        dbzdx = 0.0; dbzdy = 0.0; dbzdz = 0.0
        dbxdx(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            dbxdxt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        dbxdy(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            dbxdyt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        dbxdz(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            dbxdzt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        dbydx(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            dbydxt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        dbydy(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            dbydyt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        dbydz(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            dbydzt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        dbzdx(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            dbzdxt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        dbzdy(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            dbzdyt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        dbzdz(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            dbzdzt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        if (ixs_lo == 1) then
            dbxdx(0, :, :) = dbxdx(1, :, :)
            dbxdy(0, :, :) = dbxdy(1, :, :)
            dbxdz(0, :, :) = dbxdz(1, :, :)
            dbydx(0, :, :) = dbydx(1, :, :)
            dbydy(0, :, :) = dbydy(1, :, :)
            dbydz(0, :, :) = dbydz(1, :, :)
            dbzdx(0, :, :) = dbzdx(1, :, :)
            dbzdy(0, :, :) = dbzdy(1, :, :)
            dbzdz(0, :, :) = dbzdz(1, :, :)
        endif
        if (ixe_lo == pnx) then
            dbxdx(pnx+1, :, :) = dbxdx(pnx, :, :)
            dbxdy(pnx+1, :, :) = dbxdy(pnx, :, :)
            dbxdz(pnx+1, :, :) = dbxdz(pnx, :, :)
            dbydx(pnx+1, :, :) = dbydx(pnx, :, :)
            dbydy(pnx+1, :, :) = dbydy(pnx, :, :)
            dbydz(pnx+1, :, :) = dbydz(pnx, :, :)
            dbzdx(pnx+1, :, :) = dbzdx(pnx, :, :)
            dbzdy(pnx+1, :, :) = dbzdy(pnx, :, :)
            dbzdz(pnx+1, :, :) = dbzdz(pnx, :, :)
        endif
        if (iys_lo == 1) then
            dbxdx(:, 0, :) = dbxdx(:, 1, :)
            dbxdy(:, 0, :) = dbxdy(:, 1, :)
            dbxdz(:, 0, :) = dbxdz(:, 1, :)
            dbydx(:, 0, :) = dbydx(:, 1, :)
            dbydy(:, 0, :) = dbydy(:, 1, :)
            dbydz(:, 0, :) = dbydz(:, 1, :)
            dbzdx(:, 0, :) = dbzdx(:, 1, :)
            dbzdy(:, 0, :) = dbzdy(:, 1, :)
            dbzdz(:, 0, :) = dbzdz(:, 1, :)
        endif
        if (iye_lo == pny) then
            dbxdx(:, pny+1, :) = dbxdx(:, pny, :)
            dbxdy(:, pny+1, :) = dbxdy(:, pny, :)
            dbxdz(:, pny+1, :) = dbxdz(:, pny, :)
            dbydx(:, pny+1, :) = dbydx(:, pny, :)
            dbydy(:, pny+1, :) = dbydy(:, pny, :)
            dbydz(:, pny+1, :) = dbydz(:, pny, :)
            dbzdx(:, pny+1, :) = dbzdx(:, pny, :)
            dbzdy(:, pny+1, :) = dbzdy(:, pny, :)
            dbzdz(:, pny+1, :) = dbzdz(:, pny, :)
        endif
        if (izs_lo == 1) then
            dbxdx(:, :, 0) = dbxdx(:, :, 1)
            dbxdy(:, :, 0) = dbxdy(:, :, 1)
            dbxdz(:, :, 0) = dbxdz(:, :, 1)
            dbydx(:, :, 0) = dbydx(:, :, 1)
            dbydy(:, :, 0) = dbydy(:, :, 1)
            dbydz(:, :, 0) = dbydz(:, :, 1)
            dbzdx(:, :, 0) = dbzdx(:, :, 1)
            dbzdy(:, :, 0) = dbzdy(:, :, 1)
            dbzdz(:, :, 0) = dbzdz(:, :, 1)
        endif
        if (ize_lo == pnz) then
            dbxdx(:, :, pnz+1) = dbxdx(:, :, pnz)
            dbxdy(:, :, pnz+1) = dbxdy(:, :, pnz)
            dbxdz(:, :, pnz+1) = dbxdz(:, :, pnz)
            dbydx(:, :, pnz+1) = dbydx(:, :, pnz)
            dbydy(:, :, pnz+1) = dbydy(:, :, pnz)
            dbydz(:, :, pnz+1) = dbydz(:, :, pnz)
            dbzdx(:, :, pnz+1) = dbzdx(:, :, pnz)
            dbzdy(:, :, pnz+1) = dbzdy(:, :, pnz)
            dbzdz(:, :, pnz+1) = dbzdz(:, :, pnz)
        endif
    end subroutine set_emf_derivatives

    !<--------------------------------------------------------------------------
    !< Set the derivatives of the magnitude of the magnetic field
    !<--------------------------------------------------------------------------
    subroutine set_absb_derivatives(i, j, k, tx, ty, tz, sx, sy, sz)
        use emf_derivatives, only: dbdxt => dbdx, dbdyt => dbdy, dbdzt => dbdz
        use picinfo, only: domain
        implicit none
        integer, intent(in) :: i, j, k, tx, ty, tz, sx, sy, sz
        integer :: ixs_lo, ixe_lo, ixs_gl, ixe_gl
        integer :: iys_lo, iye_lo, iys_gl, iye_gl
        integer :: izs_lo, ize_lo, izs_gl, ize_gl
        integer :: pnx, pny, pnz
        pnx = domain%pic_nx
        pny = domain%pic_ny
        pnz = domain%pic_nz
        call bounding_indcies(i, pnx, tx, sx, ixs_lo, ixe_lo, ixs_gl, ixe_gl)
        call bounding_indcies(j, pny, ty, sy, iys_lo, iye_lo, iys_gl, iye_gl)
        call bounding_indcies(k, pnz, tz, sz, izs_lo, ize_lo, izs_gl, ize_gl)
        dbdx_s = 0.0; dbdy_s = 0.0; dbdz_s = 0.0
        dbdx_s(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            dbdxt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        dbdy_s(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            dbdyt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        dbdz_s(ixs_lo:ixe_lo, iys_lo:iye_lo, izs_lo:ize_lo) = &
            dbdzt(ixs_gl:ixe_gl, iys_gl:iye_gl, izs_gl:ize_gl)
        if (ixs_lo == 1) then
            dbdx_s(0, :, :) = dbdx_s(1, :, :)
            dbdy_s(0, :, :) = dbdy_s(1, :, :)
            dbdz_s(0, :, :) = dbdz_s(1, :, :)
        endif
        if (ixe_lo == pnx) then
            dbdx_s(pnx+1, :, :) = dbdx_s(pnx, :, :)
            dbdy_s(pnx+1, :, :) = dbdy_s(pnx, :, :)
            dbdz_s(pnx+1, :, :) = dbdz_s(pnx, :, :)
        endif
        if (iys_lo == 1) then
            dbdx_s(:, 0, :) = dbdx_s(:, 1, :)
            dbdy_s(:, 0, :) = dbdy_s(:, 1, :)
            dbdz_s(:, 0, :) = dbdz_s(:, 1, :)
        endif
        if (iye_lo == pny) then
            dbdx_s(:, pny+1, :) = dbdx_s(:, pny, :)
            dbdy_s(:, pny+1, :) = dbdy_s(:, pny, :)
            dbdz_s(:, pny+1, :) = dbdz_s(:, pny, :)
        endif
        if (izs_lo == 1) then
            dbdx_s(:, :, 0) = dbdx_s(:, :, 1)
            dbdy_s(:, :, 0) = dbdy_s(:, :, 1)
            dbdz_s(:, :, 0) = dbdz_s(:, :, 1)
        endif
        if (ize_lo == pnz) then
            dbdx_s(:, :, pnz+1) = dbdx_s(:, :, pnz)
            dbdy_s(:, :, pnz+1) = dbdy_s(:, :, pnz)
            dbdz_s(:, :, pnz+1) = dbdz_s(:, :, pnz)
        endif
    end subroutine set_absb_derivatives
end module interpolation_emf
