!*******************************************************************************
! Module of particle drift. This module will calculate particle curvature drift
! and gradient drift associated particle energy change. This module will also
! calculate the parallel acceleration. Note the particles are grouped into
! different energy band.
!*******************************************************************************
module particle_drift
    use constants, only: fp, dp
    use picinfo, only: domain 
    use path_info, only: rootpath
    use interpolation_emf, only: bx0, by0, bz0, ex0, ey0, ez0, bxn, byn, bzn
    use mpi_module
    implicit none
    private
    public init_drift_fields, init_para_perp_fields, init_jdote_sum, &
           set_drift_fields_zero, set_para_perp_fields_zero, &
           free_drift_fields, free_para_perp_fields, free_jdote_sum, &
           calc_particle_energy_change_rate, save_data_arrays, &
           sum_data_arrays, save_jdote_sum

    real(fp), allocatable, dimension(:, :, :, :) :: jcpara_dote, jgrad_dote
    real(fp), allocatable, dimension(:, :, :, :) :: jpara_dote, jperp_dote
    real(fp), allocatable, dimension(:, :, :) :: jdote_sum_local, jdote_sum_global
    integer, parameter :: nvar = 4  ! kinds of fieldsk
    real(fp) :: gyrof  ! Gyro frequency
    integer :: nx, ny, nz, nband
    integer :: ntp  ! Number of particle output frames.

    contains

    !---------------------------------------------------------------------------
    ! Initialize the fields due to particle drift motions
    !---------------------------------------------------------------------------
    subroutine init_drift_fields
        use topology_translate, only: ht
        use picinfo, only: nbands
        implicit none
        nx = ht%nx
        ny = ht%ny
        nz = ht%nz
        nband = nbands
        allocate(jcpara_dote(nx, ny, nz, nband))
        allocate(jgrad_dote(nx, ny, nz, nband))
        call set_drift_fields_zero
    end subroutine init_drift_fields

    !---------------------------------------------------------------------------
    ! Set drift fields to be zero
    !---------------------------------------------------------------------------
    subroutine set_drift_fields_zero
        implicit none
        jcpara_dote = 0.0
        jgrad_dote = 0.0
    end subroutine set_drift_fields_zero

    !---------------------------------------------------------------------------
    ! Initialize the fields due to particle parallel and perpendicular motions
    ! with respect to the local magnetic field.
    !---------------------------------------------------------------------------
    subroutine init_para_perp_fields
        use topology_translate, only: ht
        use picinfo, only: nbands
        implicit none
        nx = ht%nx
        ny = ht%ny
        nz = ht%nz
        nband = nbands
        allocate(jpara_dote(nx, ny, nz, nband))
        allocate(jperp_dote(nx, ny, nz, nband))
        call set_para_perp_fields_zero
    end subroutine init_para_perp_fields

    !---------------------------------------------------------------------------
    ! Set parallel and perpendicular fields to be zero
    !---------------------------------------------------------------------------
    subroutine set_para_perp_fields_zero
        implicit none
        jpara_dote = 0.0
        jperp_dote = 0.0
    end subroutine set_para_perp_fields_zero

    !---------------------------------------------------------------------------
    ! Initialize the summation of the calculated jdote over the whole box.
    !---------------------------------------------------------------------------
    subroutine init_jdote_sum
        use picinfo, only: nbands, nt, domain
        implicit none
        ntp = (nt * domain%fields_interval) / domain%particle_interval + 1
        allocate(jdote_sum_local(ntp, nbands, nvar))
        if (myid == master) then
            allocate(jdote_sum_global(ntp, nbands, nvar))
            jdote_sum_global = 0.0
        endif
        jdote_sum_local = 0.0
    end subroutine init_jdote_sum

    !---------------------------------------------------------------------------
    ! Free the fields due to particle drift motions
    !---------------------------------------------------------------------------
    subroutine free_drift_fields
        implicit none
        deallocate(jcpara_dote, jgrad_dote)
    end subroutine free_drift_fields

    !---------------------------------------------------------------------------
    ! Free the fields due to particle drift motions
    !---------------------------------------------------------------------------
    subroutine free_para_perp_fields
        implicit none
        deallocate(jpara_dote, jperp_dote)
    end subroutine free_para_perp_fields

    !---------------------------------------------------------------------------
    ! Free the summation of the calculated jdote over the whole box.
    !---------------------------------------------------------------------------
    subroutine free_jdote_sum
        implicit none
        deallocate(jdote_sum_local)
        if (myid == master) then
            deallocate(jdote_sum_global)
        endif
    end subroutine free_jdote_sum

    !---------------------------------------------------------------------------
    ! Calculate the particle energy change rate of all the particles in one
    ! VPIC simulation MPI process.
    ! Input:
    !   tindex: the time index, indicating the time step numbers in PIC simulation.
    !   species: 'e' for electron. 'h' for others.
    !   np: the VPIC simulation MPI process.
    !   sx, sy, sz: the offset from the starting cell of current MPI process.
    !---------------------------------------------------------------------------
    subroutine calc_particle_energy_change_rate(tindex, species, np, sx, sy, sz)
        use particle_file, only: open_particle_file, close_particle_file, fh
        use particle_module, only: ptl, calc_particle_energy, calc_interp_param, &
                calc_para_perp_velocity_3d, calc_gyrofrequency, &
                calc_gradient_drift_velocity, calc_curvature_drift_velocity, &
                iex, jex, kex, iey, jey, key, iez, jez, kez, ibx, jbx, kbx, &
                iby, jby, kby, ibz, jbz, kbz, dx_ex, dy_ex, dz_ex, &
                dx_ey, dy_ey, dz_ey, dx_ez, dy_ez, dz_ez, dx_bx, dx_by, dx_bz, &
                dy_bx, dy_by, dy_bz, dz_bx, dz_by, dz_bz
        use interpolation_emf, only: trilinear_interp_bx, trilinear_interp_by, &
                trilinear_interp_bz, trilinear_interp_ex, trilinear_interp_ey, &
                trilinear_interp_ez, calc_b_norm, calc_gradient_B, &
                calc_curvature
        use file_header, only: pheader
        implicit none
        character(len=1), intent(in) :: species
        integer, intent(in) :: tindex, np, sx, sy, sz
        character(len=8) :: cid
        integer :: iptl

        write(cid, "(I0)") np
        call open_particle_file(tindex, species, cid)
        ! Loop over particles
        do iptl = 1, pheader%dim, 1
            read(fh) ptl
            call calc_particle_energy
            call calc_interp_param
            call trilinear_interp_bx(ibx, jbx, kbx, dx_bx, dy_bx, dz_bx)
            call trilinear_interp_by(iby, jby, kby, dx_by, dy_by, dz_by)
            call trilinear_interp_bz(ibz, jbz, kbz, dx_bz, dy_bz, dz_bz)
            call trilinear_interp_ex(iex, jex, kex, dx_ex, dy_ex, dz_ex)
            call trilinear_interp_ey(iey, jey, key, dx_ey, dy_ey, dz_ey)
            call trilinear_interp_ez(iez, jez, kez, dx_ez, dy_ez, dz_ez)
            call calc_b_norm
            call calc_gradient_B
            call calc_curvature
            call calc_para_perp_velocity_3d
            call calc_gyrofrequency
            call calc_gradient_drift_velocity
            call calc_curvature_drift_velocity
            call update_data_arrays(sx, sy, sz)
        enddo
        call close_particle_file
    end subroutine calc_particle_energy_change_rate

    !---------------------------------------------------------------------------
    ! Update the particle energy change rate arrays. jcpara_dote, jgrad_dote,
    ! jpara_dote, jperp_dote.
    ! Input:
    !   sx, sy, sz: the offset from the starting cell of current MPI process.
    !---------------------------------------------------------------------------
    subroutine update_data_arrays(sx, sy, sz)
        use particle_module, only: gama, igama, vparax, vparay, vparaz, &
                vperpx, vperpy, vperpz, gyrof, vgx, vgy, vgz, vcx, vcy, vcz
        use interpolation_emf, only: ex0, ey0, ez0
        use particle_info, only: ptl_charge, species
        use picinfo, only: einterval_e, einterval_i, nbands
        use particle_module, only: ci, cj, ck
        implicit none
        integer, intent(in) :: sx, sy, sz
        real(fp) :: einterval
        integer :: ibin, ix, iy, iz
        if (species == 'e') then
            einterval = einterval_e
        else
            einterval = einterval_i
        endif
        ibin = (gama - 1) / einterval + 1
        if (ibin > nbands) ibin = nbands
        ix = ci + sx
        iy = cj + sy
        iz = ck + sz
        jcpara_dote(ix, iy, iz, ibin) = jcpara_dote(ix, iy, iz, ibin) + &
            ptl_charge * (vcx*ex0 + vcy*ey0 + vcz*ez0)
        jgrad_dote(ix, iy, iz, ibin) = jgrad_dote(ix, iy, iz, ibin) + &
            ptl_charge * (vgx*ex0 + vgy*ey0 + vgz*ez0)
        jpara_dote(ix, iy, iz, ibin) = jpara_dote(ix, iy, iz, ibin) + &
            ptl_charge * (vparax*ex0 + vparay*ey0 + vparaz*ez0)
        jperp_dote(ix, iy, iz, ibin) = jperp_dote(ix, iy, iz, ibin) + &
            ptl_charge * (vperpx*ex0 + vperpy*ey0 + vperpz*ez0)
    end subroutine update_data_arrays

    !---------------------------------------------------------------------------
    ! Save the calculated data arrays.
    ! Input:
    !   tindex: the time step index.
    !   output_record: it decides the offset from the file head.
    !---------------------------------------------------------------------------
    subroutine save_data_arrays(tindex, output_record)
        use path_info, only: opath => outputpath
        use mpi_io_translate, only: write_data
        use picinfo, only: nbands
        use particle_info, only: species, ptl_charge
        use particle_fields, only: nrho, eb
        implicit none
        integer, intent(in) :: tindex, output_record
        character(len=150) :: fname
        character(len=2) :: band_tag
        integer :: i 
        integer :: ix, iy, iz

        ! The original nrho is charge density.
        nrho = nrho / ptl_charge

        do i = 1, nbands
            write(band_tag, '(I2.2)') i
            where (eb(:,:,:,i) > 0)
                jcpara_dote(:,:,:,i) = jcpara_dote(:,:,:,i) / (nrho*eb(:,:,:,i))
                jgrad_dote(:,:,:,i) = jgrad_dote(:,:,:,i) / (nrho*eb(:,:,:,i))
                jpara_dote(:,:,:,i) = jpara_dote(:,:,:,i) / (nrho*eb(:,:,:,i))
                jperp_dote(:,:,:,i) = jperp_dote(:,:,:,i) / (nrho*eb(:,:,:,i))
            elsewhere
                jcpara_dote(:,:,:,i) = 0.0
                jgrad_dote(:,:,:,i) = 0.0
                jpara_dote(:,:,:,i) = 0.0
                jperp_dote(:,:,:,i) = 0.0
            end where

            fname = trim(adjustl(opath))//'jcpara_dote_'//species//'_'//band_tag
            call write_data(fname, jcpara_dote(:,:,:,i), tindex, output_record)
            fname = trim(adjustl(opath))//'jgrad_dote_'//species//'_'//band_tag
            call write_data(fname, jgrad_dote(:,:,:,i), tindex, output_record)
            fname = trim(adjustl(opath))//'jpara_dote_'//species//'_'//band_tag
            call write_data(fname, jpara_dote(:,:,:,i), tindex, output_record)
            fname = trim(adjustl(opath))//'jperp_dote_'//species//'_'//band_tag
            call write_data(fname, jperp_dote(:,:,:,i), tindex, output_record)
        enddo

    end subroutine save_data_arrays

    !---------------------------------------------------------------------------
    ! Sum the data arrays over the simulation box.
    ! Input:
    !   ct: current time frame
    !---------------------------------------------------------------------------
    subroutine sum_data_arrays(ct)
        use picinfo, only: nbands, domain
        implicit none
        integer, intent(in) :: ct
        integer :: iband
        real(fp) :: dv
        dv = domain%dx * domain%dy * domain%dz
        where (ISNAN(jcpara_dote))
            jcpara_dote = 0.0
        end where
        where (ISNAN(jgrad_dote))
            jgrad_dote = 0.0
        end where
        where (ISNAN(jpara_dote))
            jpara_dote = 0.0
        end where
        where (ISNAN(jperp_dote))
            jperp_dote = 0.0
        end where
        do iband = 1, nbands
            jdote_sum_local(ct, iband, 1) = sum(jcpara_dote(:, :, :, iband))
            jdote_sum_local(ct, iband, 2) = sum(jgrad_dote(:, :, :, iband))
            jdote_sum_local(ct, iband, 3) = sum(jpara_dote(:, :, :, iband))
            jdote_sum_local(ct, iband, 4) = sum(jperp_dote(:, :, :, iband))
        enddo
    end subroutine sum_data_arrays

    !---------------------------------------------------------------------------
    ! Save the summed data
    !---------------------------------------------------------------------------
    subroutine save_jdote_sum
        use particle_info, only: species
        use picinfo, only: nbands, nt
        implicit none
        integer :: ct, fh
        logical :: dir_e
        call MPI_REDUCE(jdote_sum_local, jdote_sum_global, ntp*nbands*nvar, &
                MPI_REAL, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
        if (myid == master) then
            inquire(file='./data/.', exist=dir_e)
            if (.not. dir_e) then
                call system('mkdir ./data')
            endif
            print*, "Saving particle based drift analysis resutls..."

            fh = 61

            open(unit=fh, file='data/jcpara_dote_'//species//'_p.gda', &
                access='stream', status='unknown', form='unformatted', &
                action='write')
            write(61, pos=1) jdote_sum_global(:, :, 1)
            close(fh)

            open(unit=fh, file='data/jgrad_dote_'//species//'_p.gda', &
                access='stream', status='unknown', form='unformatted', &
                action='write')
            write(61, pos=1) jdote_sum_global(:, :, 2)
            close(fh)

            open(unit=fh, file='data/jpara_dote_'//species//'_p.gda', &
                access='stream', status='unknown', form='unformatted', &
                action='write')
            write(61, pos=1) jdote_sum_global(:, :, 3)
            close(fh)

            open(unit=fh, file='data/jperp_dote_'//species//'_p.gda', &
                access='stream', status='unknown', form='unformatted', &
                action='write')
            write(61, pos=1) jdote_sum_global(:, :, 4)
            close(fh)
        endif
    end subroutine save_jdote_sum

end module particle_drift
