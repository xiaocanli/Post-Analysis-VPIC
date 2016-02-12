!*******************************************************************************
! Module dealing with compressional heating.
!*******************************************************************************
module compression_shear
    use constants, only: fp
    use mpi_topology, only: htg
    implicit none
    private
    public pdiv_v, pshear, vdot_div_ptensor, div_v, bbsigma
    public init_compression_shear, free_compression_shear, &
           calc_compression_shear, save_compression_shear, &
           save_tot_compression_shear, init_div_vdot_ptensor, &
           free_div_vdot_ptensor, calc_div_vdot_ptensor, &
           save_div_vdot_ptensor, save_tot_div_vdot_ptensor, &
           init_div_v, free_div_v, calc_div_v

    real(fp), allocatable, dimension(:, :, :) :: pdiv_v, pshear
    real(fp), allocatable, dimension(:, :, :) :: vdot_div_ptensor
    real(fp), allocatable, dimension(:, :, :) :: div_v, bbsigma
    real(fp), allocatable, dimension(:, :, :) :: vdot_ptensor_x, &
            vdot_ptensor_y, vdot_ptensor_z, div_vdot_ptensor

    contains

    !---------------------------------------------------------------------------
    ! Initialize the data arrays.
    !---------------------------------------------------------------------------
    subroutine init_compression_shear
        implicit none
        call init_compression
        call init_shear
        call init_vdot_div_ptensor
        call init_div_vdot_ptensor
    end subroutine init_compression_shear

    !---------------------------------------------------------------------------
    ! Initialize div_v.
    !---------------------------------------------------------------------------
    subroutine init_div_v
        implicit none
        allocate(div_v(htg%nx, htg%ny, htg%nz))
        div_v = 0.0
    end subroutine init_div_v

    !---------------------------------------------------------------------------
    ! Initialize div_v, pdiv_v. 
    !---------------------------------------------------------------------------
    subroutine init_compression
        implicit none
        allocate(div_v(htg%nx, htg%ny, htg%nz))
        allocate(pdiv_v(htg%nx, htg%ny, htg%nz))
        div_v = 0.0
        pdiv_v = 0.0
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
    ! Initialize vdot_div_ptensor
    !---------------------------------------------------------------------------
    subroutine init_vdot_div_ptensor
        implicit none
        allocate(vdot_div_ptensor(htg%nx, htg%ny, htg%nz))
        vdot_div_ptensor = 0.0
    end subroutine init_vdot_div_ptensor

    !---------------------------------------------------------------------------
    ! Initialize div_vdot_ptensor and the 3 components of vdot_ptensor.
    !---------------------------------------------------------------------------
    subroutine init_div_vdot_ptensor
        implicit none
        allocate(div_vdot_ptensor(htg%nx, htg%ny, htg%nz))
        allocate(vdot_ptensor_x(htg%nx, htg%ny, htg%nz))
        allocate(vdot_ptensor_y(htg%nx, htg%ny, htg%nz))
        allocate(vdot_ptensor_z(htg%nx, htg%ny, htg%nz))
        div_vdot_ptensor = 0.0
        vdot_ptensor_x = 0.0
        vdot_ptensor_y = 0.0
        vdot_ptensor_z = 0.0
    end subroutine init_div_vdot_ptensor

    !---------------------------------------------------------------------------
    ! Free the data arrays.
    !---------------------------------------------------------------------------
    subroutine free_compression_shear
        implicit none
        call free_compression
        call free_shear
        call free_vdot_div_ptensor
        call free_div_vdot_ptensor
    end subroutine free_compression_shear

    !---------------------------------------------------------------------------
    ! Free div_v.
    !---------------------------------------------------------------------------
    subroutine free_div_v
        implicit none
        deallocate(div_v)
    end subroutine free_div_v

    !---------------------------------------------------------------------------
    ! Free div_v, pdiv_v.
    !---------------------------------------------------------------------------
    subroutine free_compression
        implicit none
        deallocate(div_v, pdiv_v)
    end subroutine free_compression

    !---------------------------------------------------------------------------
    ! Free pshear, bbsigma.
    !---------------------------------------------------------------------------
    subroutine free_shear
        implicit none
        deallocate(pshear, bbsigma)
    end subroutine free_shear

    !---------------------------------------------------------------------------
    ! Free vdot_div_ptensor.
    !---------------------------------------------------------------------------
    subroutine free_vdot_div_ptensor
        implicit none
        deallocate(vdot_div_ptensor)
    end subroutine free_vdot_div_ptensor

    !---------------------------------------------------------------------------
    ! Free div_vdot_ptensor and the 3 components of vdot_ptensor.
    !---------------------------------------------------------------------------
    subroutine free_div_vdot_ptensor
        implicit none
        deallocate(div_vdot_ptensor)
        deallocate(vdot_ptensor_x, vdot_ptensor_y, vdot_ptensor_z)
    end subroutine free_div_vdot_ptensor

    !---------------------------------------------------------------------------
    ! Calculate the divergence of v.
    !---------------------------------------------------------------------------
    subroutine calc_div_v
        use pic_fields, only: vx, vy, vz
        use neighbors_module, only: ixl, iyl, izl, ixh, iyh, izh, idx, idy, idz
        implicit none
        integer :: nx, ny, nz, ix, iy, iz
        nx = htg%nx
        ny = htg%ny
        nz = htg%nz

        do ix = 1, nx
            div_v(ix, :, :) = (vx(ixh(ix), :, :) - vx(ixl(ix), :, :)) * idx(ix)
        enddo

        do iy = 1, ny
            div_v(:, iy, :) = div_v(:, iy, :) + &
                (vy(:, iyh(iy), :) - vy(:, iyl(iy), :)) * idy(iy)
        enddo

        do iz = 1, nz
            div_v(:, :, iz) = div_v(:, :, iz) + &
                (vz(:, :, izh(iz)) - vz(:, :, izl(iz))) * idz(iz)
        enddo
    end subroutine calc_div_v

    !---------------------------------------------------------------------------
    ! Calculate p\nabla\cdot\vec{v}. Here, p is the scalar pressure.
    !---------------------------------------------------------------------------
    subroutine calc_pdiv_v
        use pressure_tensor, only: pscalar
        implicit none

        pdiv_v = - pscalar * div_v
    end subroutine calc_pdiv_v

    !---------------------------------------------------------------------------
    ! Calculate the compression related variables.
    !---------------------------------------------------------------------------
    subroutine calc_compression
        implicit none
        call calc_div_v
        call calc_pdiv_v
    end subroutine calc_compression

    !---------------------------------------------------------------------------
    ! Calculate bbsigma = b_ib_j\sigma_{ij}.
    !---------------------------------------------------------------------------
    subroutine calc_bbsigma
        use pic_fields, only: bx, by, bz, vx, vy, vz, absB
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
                    sigma_xx = (vx(ixh(ix), iy, iz) - vx(ixl(ix), iy, iz)) * &
                               idx(ix) - div_v(ix, iy, iz) / 3.0
                    sigma_yy = (vy(ix, iyh(iy), iz) - vy(ix, iyl(iy), iz)) * &
                               idy(iy) - div_v(ix, iy, iz) / 3.0
                    sigma_zz = (vz(ix, iy, izh(iz)) - vz(ix, iy, izl(iz))) * &
                               idz(iz) - div_v(ix, iy, iz) / 3.0
                    sigma_xy = 0.5 * (vx(ix, iyh(iy), iz) - &
                                      vx(ix, iyl(iy), iz)) * idy(iy) + &
                               0.5 * (vy(ixh(ix), iy, iz) - &
                                      vy(ixl(ix), iy, iz)) * idx(ix)
                    sigma_xz = 0.5 * (vx(ix, iy, izh(iz)) - &
                                      vx(ix, iy, izl(iz))) * idz(iz) + &
                               0.5 * (vz(ixh(ix), iy, iz) - &
                                      vz(ixl(ix), iy, iz)) * idx(ix)
                    sigma_yz = 0.5 * (vy(ix, iy, izh(iz)) - &
                                      vy(ix, iy, izl(iz))) * idz(iz) + &
                               0.5 * (vz(ix, iyh(iy), iz) - &
                                      vz(ix, iyl(iy), iz)) * idy(iy)
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
    subroutine calc_vdot_div_ptensor
        use pressure_tensor, only: divp_x, divp_y, divp_z
        use pic_fields, only: vx, vy, vz
        implicit none
        vdot_div_ptensor = vx * divp_x + vy * divp_y + vz * divp_z
    end subroutine calc_vdot_div_ptensor


    !---------------------------------------------------------------------------
    ! Calculate the compressional and shear heating terms.
    !---------------------------------------------------------------------------
    subroutine calc_compression_shear
        implicit none
        call calc_compression
        call calc_shear
        call calc_vdot_div_ptensor
        call calc_div_vdot_ptensor
    end subroutine calc_compression_shear

    !---------------------------------------------------------------------------
    ! Calculate div_vdot_ptensor.
    !---------------------------------------------------------------------------
    subroutine calc_div_vdot_ptensor
        use pic_fields, only: vx, vy, vz, pxx, pxy, pxz, pyy, pyz, pzz
        use neighbors_module, only: ixl, iyl, izl, ixh, iyh, izh, idx, idy, idz
        implicit none
        integer :: nx, ny, nz, ix, iy, iz

        vdot_ptensor_x = vx * pxx + vy * pxy + vz * pxz
        vdot_ptensor_y = vx * pxy + vy * pyy + vz * pyz
        vdot_ptensor_z = vx * pxz + vy * pyz + vz * pzz

        nx = htg%nx
        ny = htg%ny
        nz = htg%nz

        do ix = 1, nx
            div_vdot_ptensor(ix, :, :) = (vdot_ptensor_x(ixh(ix), :, :) - &
                    vdot_ptensor_x(ixl(ix), :, :)) * idx(ix)
        enddo

        do iy = 1, ny
            div_vdot_ptensor(:, iy, :) = div_vdot_ptensor(:, iy, :) + &
                    (vdot_ptensor_y(:, iyh(iy), :) - &
                     vdot_ptensor_y(:, iyl(iy), :)) * idy(iy)
        enddo

        do iz = 1, nz
            div_vdot_ptensor(:, :, iz) = div_vdot_ptensor(:, :, iz) + &
                    (vdot_ptensor_z(:, :, izh(iz)) - &
                     vdot_ptensor_z(:, :, izl(iz))) * idz(iz)
        enddo
    end subroutine calc_div_vdot_ptensor

    !---------------------------------------------------------------------------
    ! Save div_v.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_div_v(ct)
        use mpi_module
        use constants, only: fp
        use mpi_io_fields, only: save_field
        implicit none
        integer, intent(in) :: ct
        if (myid == master) then
            print*, 'Saving the divergence of the velocity field', ct
        endif

        call save_field(div_v, 'div_v', ct)
    end subroutine save_div_v

    !---------------------------------------------------------------------------
    ! Save pdiv_v.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_pdiv_v(ct)
        use mpi_module
        use constants, only: fp
        use mpi_io_fields, only: save_field
        implicit none
        integer, intent(in) :: ct
        if (myid == master) then
            print*, 'Saving pdiv_v', ct
        endif

        call save_field(pdiv_v, 'pdiv_v', ct)
    end subroutine save_pdiv_v

    !---------------------------------------------------------------------------
    ! Save compression related variables.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_compression(ct)
        implicit none
        integer, intent(in) :: ct
        call save_div_v(ct)
        call save_pdiv_v(ct)
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
    ! Save vdot_div_ptensor
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_vdot_div_ptensor(ct)
        use mpi_module
        use constants, only: fp
        use mpi_io_fields, only: save_field
        implicit none
        integer, intent(in) :: ct
        if (myid == master) then
            print*, 'Saving vdot_div_ptensor', ct
        endif

        call save_field(vdot_div_ptensor, 'vdot_div_ptensor', ct)
    end subroutine save_vdot_div_ptensor

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
        call save_vdot_div_ptensor(ct)
        call save_div_vdot_ptensor(ct)
    end subroutine save_compression_shear

    !---------------------------------------------------------------------------
    ! Save div_vdot_ptensor.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_div_vdot_ptensor(ct)
        use mpi_module
        use constants, only: fp
        use mpi_io_fields, only: save_field
        implicit none
        integer, intent(in) :: ct
        if (myid == master) then
            print*, 'Saving div_vdot_ptensor', ct
        endif

        call save_field(div_vdot_ptensor, 'div_vdot_ptensor', ct)
    end subroutine save_div_vdot_ptensor

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
        real(fp) :: div_v_tot, pdiv_v_tot, avg
        character(len=100) :: fname
        integer :: current_pos, output_record
        logical :: dir_e
    
        if (myid == master) then
            inquire(file='./data/.', exist=dir_e)
            if (.not. dir_e) then
                call system('mkdir ./data')
            endif
        endif

        call get_average_and_total(div_v, avg, div_v_tot)
        call get_average_and_total(pdiv_v, avg, pdiv_v_tot)
        if (myid == master) then
            fname = 'data/compression'//ibtag//'_'//species//'.gda'
            open(unit=51, file=trim(adjustl(fname)), access='stream',&
                 status='unknown', form='unformatted', action='write')
            output_record = ct - tp1 + 1
            current_pos = 2 * sizeof(fp) * (output_record-1) + 1
            write(51, pos=current_pos) div_v_tot, pdiv_v_tot
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
    ! Save the total of vdot_div_ptensor.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_tot_vdot_div_ptensor(ct)
        use mpi_module
        use constants, only: fp
        use particle_info, only: ibtag, species
        use statistics, only: get_average_and_total
        use parameters, only: tp1
        implicit none
        integer, intent(in) :: ct
        real(fp) :: vdot_div_ptensor_tot, avg
        character(len=100) :: fname
        integer :: current_pos, output_record
        logical :: dir_e
    
        if (myid == master) then
            inquire(file='./data/.', exist=dir_e)
            if (.not. dir_e) then
                call system('mkdir ./data')
            endif
        endif

        call get_average_and_total(vdot_div_ptensor, avg, vdot_div_ptensor_tot)
        if (myid == master) then
            fname = 'data/vdot_div_ptensor'//ibtag//'_'//species//'.gda'
            open(unit=51, file=trim(adjustl(fname)), access='stream',&
                 status='unknown', form='unformatted', action='write')
            output_record = ct - tp1 + 1
            current_pos = sizeof(fp) * (output_record-1) + 1
            write(51, pos=current_pos) vdot_div_ptensor_tot
            close(51)
        endif
    end subroutine save_tot_vdot_div_ptensor

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
        call save_tot_vdot_div_ptensor(ct)
        call save_tot_div_vdot_ptensor(ct)
    end subroutine save_tot_compression_shear

    !---------------------------------------------------------------------------
    ! Save the total of div_vdot_ptensor.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_tot_div_vdot_ptensor(ct)
        use mpi_module
        use constants, only: fp
        use particle_info, only: ibtag, species
        use statistics, only: get_average_and_total
        use parameters, only: tp1
        implicit none
        integer, intent(in) :: ct
        real(fp) :: div_vdot_ptensor_tot, avg
        character(len=100) :: fname
        integer :: current_pos, output_record
        logical :: dir_e
    
        if (myid == master) then
            inquire(file='./data/.', exist=dir_e)
            if (.not. dir_e) then
                call system('mkdir ./data')
            endif
        endif

        call get_average_and_total(div_vdot_ptensor, avg, div_vdot_ptensor_tot)
        if (myid == master) then
            fname = 'data/div_vdot_ptensor'//ibtag//'_'//species//'.gda'
            open(unit=51, file=trim(adjustl(fname)), access='stream',&
                 status='unknown', form='unformatted', action='write')
            output_record = ct - tp1 + 1
            current_pos = sizeof(fp) * (output_record-1) + 1
            write(51, pos=current_pos) div_vdot_ptensor_tot
            close(51)
        endif
    end subroutine save_tot_div_vdot_ptensor

end module compression_shear
