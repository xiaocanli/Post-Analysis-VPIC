!*******************************************************************************
! Module dealing with compressional heating.
!*******************************************************************************
module compression_shear
    use constants, only: fp
    use mpi_topology, only: htg
    implicit none
    private
    public pdiv_u, pshear, udot_div_ptensor, div_u, bbsigma
    public init_compression_shear, free_compression_shear, &
           calc_compression_shear, save_compression_shear, &
           save_tot_compression_shear, init_div_udot_ptensor, &
           free_div_udot_ptensor, calc_div_udot_ptensor, &
           save_div_udot_ptensor, save_tot_div_udot_ptensor

    real(fp), allocatable, dimension(:, :, :) :: pdiv_u, pshear
    real(fp), allocatable, dimension(:, :, :) :: udot_div_ptensor
    real(fp), allocatable, dimension(:, :, :) :: div_u, bbsigma
    real(fp), allocatable, dimension(:, :, :) :: udot_ptensor_x, &
            udot_ptensor_y, udot_ptensor_z, div_udot_ptensor

    contains

    !---------------------------------------------------------------------------
    ! Initialize the data arrays.
    !---------------------------------------------------------------------------
    subroutine init_compression_shear
        implicit none
        call init_compression
        call init_shear
        call init_udot_div_ptensor
        call init_div_udot_ptensor
    end subroutine init_compression_shear

    !---------------------------------------------------------------------------
    ! Initialize div_u, pdiv_u. 
    !---------------------------------------------------------------------------
    subroutine init_compression
        implicit none
        allocate(div_u(htg%nx, htg%ny, htg%nz))
        allocate(pdiv_u(htg%nx, htg%ny, htg%nz))
        div_u = 0.0
        pdiv_u = 0.0
    end subroutine init_compression

    !---------------------------------------------------------------------------
    ! Initialize pshear, bbsigma.
    !---------------------------------------------------------------------------
    subroutine init_shear
        implicit none
        allocate(pshear(htg%nx, htg%ny, htg%nz))
        allocate(bbsigma(htg%nx, htg%ny, htg%nz))
        pshear = 0.0
        bbsigma = 0.0
    end subroutine init_shear

    !---------------------------------------------------------------------------
    ! Initialize udot_div_ptensor
    !---------------------------------------------------------------------------
    subroutine init_udot_div_ptensor
        implicit none
        allocate(udot_div_ptensor(htg%nx, htg%ny, htg%nz))
        udot_div_ptensor = 0.0
    end subroutine init_udot_div_ptensor

    !---------------------------------------------------------------------------
    ! Initialize div_udot_ptensor and the 3 components of udot_ptensor.
    !---------------------------------------------------------------------------
    subroutine init_div_udot_ptensor
        implicit none
        allocate(div_udot_ptensor(htg%nx, htg%ny, htg%nz))
        allocate(udot_ptensor_x(htg%nx, htg%ny, htg%nz))
        allocate(udot_ptensor_y(htg%nx, htg%ny, htg%nz))
        allocate(udot_ptensor_z(htg%nx, htg%ny, htg%nz))
        div_udot_ptensor = 0.0
        udot_ptensor_x = 0.0
        udot_ptensor_y = 0.0
        udot_ptensor_z = 0.0
    end subroutine init_div_udot_ptensor

    !---------------------------------------------------------------------------
    ! Free the data arrays.
    !---------------------------------------------------------------------------
    subroutine free_compression_shear
        implicit none
        call free_compression
        call free_shear
        call free_udot_div_ptensor
        call free_div_udot_ptensor
    end subroutine free_compression_shear

    !---------------------------------------------------------------------------
    ! Free div_u, pdiv_u.
    !---------------------------------------------------------------------------
    subroutine free_compression
        implicit none
        deallocate(div_u, pdiv_u)
    end subroutine free_compression

    !---------------------------------------------------------------------------
    ! Free pshear, bbsigma.
    !---------------------------------------------------------------------------
    subroutine free_shear
        implicit none
        deallocate(pshear, bbsigma)
    end subroutine free_shear

    !---------------------------------------------------------------------------
    ! Free udot_div_ptensor.
    !---------------------------------------------------------------------------
    subroutine free_udot_div_ptensor
        implicit none
        deallocate(udot_div_ptensor)
    end subroutine free_udot_div_ptensor

    !---------------------------------------------------------------------------
    ! Free div_udot_ptensor and the 3 components of udot_ptensor.
    !---------------------------------------------------------------------------
    subroutine free_div_udot_ptensor
        implicit none
        deallocate(div_udot_ptensor)
        deallocate(udot_ptensor_x, udot_ptensor_y, udot_ptensor_z)
    end subroutine free_div_udot_ptensor

    !---------------------------------------------------------------------------
    ! Calculate the divergence of u.
    !---------------------------------------------------------------------------
    subroutine calc_div_u
        use pic_fields, only: ux, uy, uz
        use neighbors_module, only: ixl, iyl, izl, ixh, iyh, izh, idx, idy, idz
        implicit none
        integer :: nx, ny, nz, ix, iy, iz
        nx = htg%nx
        ny = htg%ny
        nz = htg%nz

        do ix = 1, nx
            div_u(ix, :, :) = (ux(ixh(ix), :, :) - ux(ixl(ix), :, :)) * idx(ix)
        enddo

        do iy = 1, ny
            div_u(:, iy, :) = div_u(:, iy, :) + &
                (uy(:, iyh(iy), :) - uy(:, iyl(iy), :)) * idy(iy)
        enddo

        do iz = 1, nz
            div_u(:, :, iz) = div_u(:, :, iz) + &
                (uz(:, :, izh(iz)) - uz(:, :, izl(iz))) * idz(iz)
        enddo
    end subroutine calc_div_u

    !---------------------------------------------------------------------------
    ! Calculate p\nabla\cdot\vec{u}. Here, p is the scalar pressure.
    !---------------------------------------------------------------------------
    subroutine calc_pdiv_u
        use pressure_tensor, only: pscalar
        implicit none

        pdiv_u = - pscalar * div_u
    end subroutine calc_pdiv_u

    !---------------------------------------------------------------------------
    ! Calculate the compression related variables.
    !---------------------------------------------------------------------------
    subroutine calc_compression
        implicit none
        call calc_div_u
        call calc_pdiv_u
    end subroutine calc_compression

    !---------------------------------------------------------------------------
    ! Calculate bbsigma = b_ib_j\sigma_{ij}.
    !---------------------------------------------------------------------------
    subroutine calc_bbsigma
        use pic_fields, only: bx, by, bz, ux, uy, uz, absB
        use neighbors_module, only: ixl, iyl, izl, ixh, iyh, izh, idx, idy, idz
        implicit none
        real(fp) :: sigma_xx, sigma_xy, sigma_xz, sigma_yy, sigma_yz, sigma_zz
        real(fp) :: bxc, byc, bzc
        integer :: nx, ny, nz, ix, iy, iz
        nx = htg%nx
        ny = htg%ny
        nz = htg%nz

        do iz = 1, nz
            do iy = 1, ny
                 do ix = 1, nx
                    sigma_xx = (ux(ixh(ix), iy, iz) - ux(ixl(ix), iy, iz)) * &
                               idx(ix) - div_u(ix, iy, iz) / 3.0
                    sigma_yy = (uy(ix, iyh(iy), iz) - uy(ix, iyl(iy), iz)) * &
                               idy(iy) - div_u(ix, iy, iz) / 3.0
                    sigma_zz = (uz(ix, iy, izh(iz)) - uz(ix, iy, izl(iz))) * &
                               idz(iz) - div_u(ix, iy, iz) / 3.0
                    sigma_xy = 0.5 * (ux(ix, iyh(iy), iz) - &
                                      ux(ix, iyl(iy), iz)) * idy(iy) + &
                               0.5 * (uy(ixh(ix), iy, iz) - &
                                      uy(ixl(ix), iy, iz)) * idx(ix)
                    sigma_xz = 0.5 * (ux(ix, iy, izh(iz)) - &
                                      ux(ix, iy, izl(iz))) * idz(iz) + &
                               0.5 * (uz(ixh(ix), iy, iz) - &
                                      uz(ixl(ix), iy, iz)) * idx(ix)
                    sigma_yz = 0.5 * (uy(ix, iy, izh(iz)) - &
                                      uy(ix, iy, izl(iz))) * idz(iz) + &
                               0.5 * (uz(ix, iyh(iy), iz) - &
                                      uz(ix, iyl(iy), iz)) * idy(iy)
                    bxc = bx(ix, iy, iz)
                    byc = by(ix, iy, iz)
                    bzc = bz(ix, iy, iz)
                    bbsigma(ix, iy, iz) = bxc**2 * sigma_xx + &
                        byc**2 * sigma_yy + bzc**2 * sigma_zz + &
                        2.0 * bxc * byc * sigma_xy + &
                        2.0 * bxc * bzc * sigma_xz + &
                        2.0 * byc * bzc * sigma_yz
                 enddo
            enddo
        enddo
        bbsigma = bbsigma / absB**2
    end subroutine calc_bbsigma

    !---------------------------------------------------------------------------
    ! Calculate (p_\parallel - p_\perp)b_ib_j\sigma_{ij}, where \sigma_{ij}
    ! is the shear tensor.
    !---------------------------------------------------------------------------
    subroutine calc_pshear
        use para_perp_pressure, only: ppara, pperp
        implicit none
        pshear = (pperp - ppara) * bbsigma
    end subroutine calc_pshear

    !---------------------------------------------------------------------------
    ! Calculate shear related variables.
    !---------------------------------------------------------------------------
    subroutine calc_shear
        implicit none
        call calc_bbsigma
        call calc_pshear
    end subroutine calc_shear

    !---------------------------------------------------------------------------
    ! Calculate \vec{u}\cdot(\nalba\cdot\tensor(P))
    !---------------------------------------------------------------------------
    subroutine calc_udot_div_ptensor
        use pressure_tensor, only: divp_x, divp_y, divp_z
        use pic_fields, only: ux, uy, uz
        implicit none
        udot_div_ptensor = ux * divp_x + uy * divp_y + uz * divp_z
    end subroutine calc_udot_div_ptensor


    !---------------------------------------------------------------------------
    ! Calculate the compressional and shear heating terms.
    !---------------------------------------------------------------------------
    subroutine calc_compression_shear
        implicit none
        call calc_compression
        call calc_shear
        call calc_udot_div_ptensor
        call calc_div_udot_ptensor
    end subroutine calc_compression_shear

    !---------------------------------------------------------------------------
    ! Calculate div_udot_ptensor.
    !---------------------------------------------------------------------------
    subroutine calc_div_udot_ptensor
        use pic_fields, only: ux, uy, uz, pxx, pxy, pxz, pyy, pyz, pzz
        use neighbors_module, only: ixl, iyl, izl, ixh, iyh, izh, idx, idy, idz
        implicit none
        integer :: nx, ny, nz, ix, iy, iz

        udot_ptensor_x = ux * pxx + uy * pxy + uz * pxz
        udot_ptensor_y = ux * pxy + uy * pyy + uz * pyz
        udot_ptensor_z = ux * pxz + uy * pyz + uz * pzz

        nx = htg%nx
        ny = htg%ny
        nz = htg%nz

        do ix = 1, nx
            div_udot_ptensor(ix, :, :) = (udot_ptensor_x(ixh(ix), :, :) - &
                    udot_ptensor_x(ixl(ix), :, :)) * idx(ix)
        enddo

        do iy = 1, ny
            div_udot_ptensor(:, iy, :) = div_udot_ptensor(:, iy, :) + &
                    (udot_ptensor_y(:, iyh(iy), :) - &
                     udot_ptensor_y(:, iyl(iy), :)) * idy(iy)
        enddo

        do iz = 1, nz
            div_udot_ptensor(:, :, iz) = div_udot_ptensor(:, :, iz) + &
                    (udot_ptensor_z(:, :, izh(iz)) - &
                     udot_ptensor_z(:, :, izl(iz))) * idz(iz)
        enddo
    end subroutine calc_div_udot_ptensor

    !---------------------------------------------------------------------------
    ! Save div_u.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_div_u(ct)
        use mpi_module
        use constants, only: fp
        use mpi_io_fields, only: save_field
        implicit none
        integer, intent(in) :: ct
        if (myid == master) then
            print*, 'Saving the divergence of the velocity field', ct
        endif

        call save_field(div_u, 'div_u', ct)
    end subroutine save_div_u

    !---------------------------------------------------------------------------
    ! Save pdiv_u.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_pdiv_u(ct)
        use mpi_module
        use constants, only: fp
        use mpi_io_fields, only: save_field
        implicit none
        integer, intent(in) :: ct
        if (myid == master) then
            print*, 'Saving pdiv_u', ct
        endif

        call save_field(pdiv_u, 'pdiv_u', ct)
    end subroutine save_pdiv_u

    !---------------------------------------------------------------------------
    ! Save compression related variables.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_compression(ct)
        implicit none
        integer, intent(in) :: ct
        call save_div_u(ct)
        call save_pdiv_u(ct)
    end subroutine save_compression

    !---------------------------------------------------------------------------
    ! Save bbsigma.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_bbsigma(ct)
        use mpi_module
        use constants, only: fp
        use mpi_io_fields, only: save_field
        implicit none
        integer, intent(in) :: ct
        if (myid == master) then
            print*, 'Saving b_ib_j\sigma_{ij}', ct
        endif

        call save_field(bbsigma, 'bbsigma', ct)
    end subroutine save_bbsigma

    !---------------------------------------------------------------------------
    ! Save pshear
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_pshear(ct)
        use mpi_module
        use constants, only: fp
        use mpi_io_fields, only: save_field
        implicit none
        integer, intent(in) :: ct
        if (myid == master) then
            print*, 'Saving pshear', ct
        endif

        call save_field(pshear, 'pshear', ct)
    end subroutine save_pshear

    !---------------------------------------------------------------------------
    ! Save shear related variables.
    !---------------------------------------------------------------------------
    subroutine save_shear(ct)
        implicit none
        integer, intent(in) :: ct
        call save_bbsigma(ct)
        call save_pshear(ct)
    end subroutine save_shear

    !---------------------------------------------------------------------------
    ! Save udot_div_ptensor
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_udot_div_ptensor(ct)
        use mpi_module
        use constants, only: fp
        use mpi_io_fields, only: save_field
        implicit none
        integer, intent(in) :: ct
        if (myid == master) then
            print*, 'Saving udot_div_ptensor', ct
        endif

        call save_field(udot_div_ptensor, 'udot_div_ptensor', ct)
    end subroutine save_udot_div_ptensor

    !---------------------------------------------------------------------------
    ! Save compressional and shear heating terms.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_compression_shear(ct)
        implicit none
        integer, intent(in) :: ct
        call save_compression(ct)
        call save_shear(ct)
        call save_udot_div_ptensor(ct)
        call save_div_udot_ptensor(ct)
    end subroutine save_compression_shear

    !---------------------------------------------------------------------------
    ! Save div_udot_ptensor.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_div_udot_ptensor(ct)
        use mpi_module
        use constants, only: fp
        use mpi_io_fields, only: save_field
        implicit none
        integer, intent(in) :: ct
        if (myid == master) then
            print*, 'Saving div_udot_ptensor', ct
        endif

        call save_field(div_udot_ptensor, 'div_udot_ptensor', ct)
    end subroutine save_div_udot_ptensor

    !---------------------------------------------------------------------------
    ! Save the total of the compressional terms.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_tot_compression(ct)
        use mpi_module
        use constants, only: fp
        use particle_info, only: ibtag, species
        use statistics, only: get_average_and_total
        use parameters, only: tp1
        implicit none
        integer, intent(in) :: ct
        real(fp) :: div_u_tot, pdiv_u_tot, avg
        character(len=100) :: fname
        integer :: current_pos, output_record
        logical :: dir_e
    
        if (myid == master) then
            inquire(file='./data/.', exist=dir_e)
            if (.not. dir_e) then
                call system('mkdir ./data')
            endif
        endif

        call get_average_and_total(div_u, avg, div_u_tot)
        call get_average_and_total(pdiv_u, avg, pdiv_u_tot)
        if (myid == master) then
            fname = 'data/compression'//ibtag//'_'//species//'.gda'
            open(unit=51, file=trim(adjustl(fname)), access='stream',&
                 status='unknown', form='unformatted', action='write')
            output_record = ct - tp1 + 1
            current_pos = 2 * sizeof(fp) * (output_record-1) + 1
            write(51, pos=current_pos) div_u_tot, pdiv_u_tot
            close(51)
        endif
    end subroutine save_tot_compression

    !---------------------------------------------------------------------------
    ! Save the total of the shear terms.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_tot_shear(ct)
        use mpi_module
        use constants, only: fp
        use particle_info, only: ibtag, species
        use statistics, only: get_average_and_total
        use parameters, only: tp1
        implicit none
        integer, intent(in) :: ct
        real(fp) :: bbsigma_tot, pshear_tot, avg
        character(len=100) :: fname
        integer :: current_pos, output_record
        logical :: dir_e
    
        if (myid == master) then
            inquire(file='./data/.', exist=dir_e)
            if (.not. dir_e) then
                call system('mkdir ./data')
            endif
        endif

        call get_average_and_total(bbsigma, avg, bbsigma_tot)
        call get_average_and_total(pshear, avg, pshear_tot)
        if (myid == master) then
            fname = 'data/shear'//ibtag//'_'//species//'.gda'
            open(unit=51, file=trim(adjustl(fname)), access='stream',&
                 status='unknown', form='unformatted', action='write')
            output_record = ct - tp1 + 1
            current_pos = 2 * sizeof(fp) * (output_record-1) + 1
            write(51, pos=current_pos) bbsigma_tot, pshear_tot
            close(51)
        endif
    end subroutine save_tot_shear

    !---------------------------------------------------------------------------
    ! Save the total of udot_div_ptensor.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_tot_udot_div_ptensor(ct)
        use mpi_module
        use constants, only: fp
        use particle_info, only: ibtag, species
        use statistics, only: get_average_and_total
        use parameters, only: tp1
        implicit none
        integer, intent(in) :: ct
        real(fp) :: udot_div_ptensor_tot, avg
        character(len=100) :: fname
        integer :: current_pos, output_record
        logical :: dir_e
    
        if (myid == master) then
            inquire(file='./data/.', exist=dir_e)
            if (.not. dir_e) then
                call system('mkdir ./data')
            endif
        endif

        call get_average_and_total(udot_div_ptensor, avg, udot_div_ptensor_tot)
        if (myid == master) then
            fname = 'data/udot_div_ptensor'//ibtag//'_'//species//'.gda'
            open(unit=51, file=trim(adjustl(fname)), access='stream',&
                 status='unknown', form='unformatted', action='write')
            output_record = ct - tp1 + 1
            current_pos = sizeof(fp) * (output_record-1) + 1
            write(51, pos=current_pos) udot_div_ptensor_tot
            close(51)
        endif
    end subroutine save_tot_udot_div_ptensor

    !---------------------------------------------------------------------------
    ! Save the total of the compressional and shear heating terms.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_tot_compression_shear(ct)
        implicit none
        integer, intent(in) :: ct
        call save_tot_compression(ct)
        call save_tot_shear(ct)
        call save_tot_udot_div_ptensor(ct)
        call save_tot_div_udot_ptensor(ct)
    end subroutine save_tot_compression_shear

    !---------------------------------------------------------------------------
    ! Save the total of div_udot_ptensor.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_tot_div_udot_ptensor(ct)
        use mpi_module
        use constants, only: fp
        use particle_info, only: ibtag, species
        use statistics, only: get_average_and_total
        use parameters, only: tp1
        implicit none
        integer, intent(in) :: ct
        real(fp) :: div_udot_ptensor_tot, avg
        character(len=100) :: fname
        integer :: current_pos, output_record
        logical :: dir_e
    
        if (myid == master) then
            inquire(file='./data/.', exist=dir_e)
            if (.not. dir_e) then
                call system('mkdir ./data')
            endif
        endif

        call get_average_and_total(div_udot_ptensor, avg, div_udot_ptensor_tot)
        if (myid == master) then
            fname = 'data/div_udot_ptensor'//ibtag//'_'//species//'.gda'
            open(unit=51, file=trim(adjustl(fname)), access='stream',&
                 status='unknown', form='unformatted', action='write')
            output_record = ct - tp1 + 1
            current_pos = sizeof(fp) * (output_record-1) + 1
            write(51, pos=current_pos) div_udot_ptensor_tot
            close(51)
        endif
    end subroutine save_tot_div_udot_ptensor

end module compression_shear
