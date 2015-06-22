!*******************************************************************************
! Module for parallel and perpendicular pressure. It includes the data and
! the methods to calculate the pressures.
!*******************************************************************************
module para_perp_pressure
    use constants, only: fp
    implicit none
    private
    public ppara, pperp, init_para_perp_pressure, free_para_perp_pressure
    public calc_para_perp_pressure, calc_real_para_perp_pressure
    public save_para_perp_pressure, save_averaged_para_perp_pressure
    real(fp), allocatable, dimension(:,:,:) :: pperp, ppara

    contains

    !---------------------------------------------------------------------------
    ! Initialization of the parallel and perpendicular pressure.
    !---------------------------------------------------------------------------
    subroutine init_para_perp_pressure
        use mpi_topology, only: htg
        implicit none
        integer :: nx, ny, nz

        nx = htg%nx
        ny = htg%ny
        nz = htg%nz

        ! Allocate storage space for fields and moments
        allocate(ppara(nx,ny,nz))
        allocate(pperp(nx,ny,nz))

        ppara = 0.0
        pperp = 0.0
    end subroutine init_para_perp_pressure

    !---------------------------------------------------------------------------
    ! Free the memory used by parallel and perpendicular pressure.
    !---------------------------------------------------------------------------
    subroutine free_para_perp_pressure
        implicit none
        deallocate(ppara, pperp)
    end subroutine free_para_perp_pressure

    !---------------------------------------------------------------------------
    ! Calculate parallel and perpendicular pressure from the pressure tensor
    ! and magnetic field. P is the pressure tensor below. 
    !       P_\parallel = \hat{b}\cdot P \cdot\hat{b}
    !       P_\perp = (tr(P)-P_\parallel)*0.5
    ! The bulk flow is NOT subtracted from the pressure.
    ! Inputs:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine calc_para_perp_pressure(ct)
        use constants, only: fp
        use pic_fields, only: bx, by, bz, absB, ux, uy, uz, num_rho, &
                              pxx, pxy, pxz, pyy, pyz, pzz
        use particle_info, only: ptl_mass
        use mpi_topology, only: htg
        use saving_flags, only: save_pre
        implicit none
        integer, intent(in) :: ct
        real(fp), allocatable, dimension(:,:,:) :: bsquare, prexx, preyy, prezz
        integer :: nx, ny, nz

        nx = htg%nx
        ny = htg%ny
        nz = htg%nz

        allocate(bsquare(nx, ny, nz))
        allocate(prexx(nx, ny, nz))
        allocate(preyy(nx, ny, nz))
        allocate(prezz(nx, ny, nz))

        bsquare = bx*bx + by*by + bz*bz
        prexx = pxx + ux*ux*num_rho*ptl_mass
        preyy = pyy + uy*uy*num_rho*ptl_mass
        prezz = pzz + uz*uz*num_rho*ptl_mass
        ppara = prexx * bx * bx + &
                preyy * by * by + &
                prezz * bz * bz + &
                (pxy + ux*uy*num_rho*ptl_mass) * bx * by * 2.0 + &
                (pxz + ux*uz*num_rho*ptl_mass) * bx * bz * 2.0 + &
                (pyz + uy*uz*num_rho*ptl_mass) * by * bz * 2.0
        ppara = ppara / bsquare
        pperp = 0.5 * (prexx + preyy + prezz - ppara)

        if (save_pre==1) then
            ! Save calculated parallel and perpendicular pressure.
            call save_para_perp_pressure(ct, is_subtract_bulkflow=.false.)
        endif
        deallocate(bsquare, prexx, preyy, prezz)
        call save_averaged_para_perp_pressure(ct, is_subtract_bulkflow=.false.)
    end subroutine calc_para_perp_pressure

    !---------------------------------------------------------------------------
    ! Calculate parallel and perpendicular pressure from the pressure tensor
    ! and magnetic field. The bulk flow is subtracted from the pressure.
    ! Inputs:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine calc_real_para_perp_pressure(ct)
        use constants, only: fp
        use pic_fields, only: bx, by, bz, absB, ux, uy, uz, num_rho, &
                              pxx, pxy, pxz, pyy, pyz, pzz
        use mpi_topology, only: htg
        use saving_flags, only: save_pre
        implicit none
        integer, intent(in) :: ct
        real(fp), allocatable, dimension(:,:,:) :: bsquare
        integer :: nx, ny, nz

        nx = htg%nx
        ny = htg%ny
        nz = htg%nz

        allocate(bsquare(nx, ny, nz))

        bsquare = bx*bx + by*by + bz*bz
        ppara = pxx*bx*bx + pyy*by*by + pzz*bz*bz + &
                (pxy * bx * by * 2.0) + (pxz * bx * bz * 2.0) + &
                (pyz * by * bz * 2.0)
        ppara = ppara / bsquare
        pperp = 0.5 * (pxx + pyy + pzz - ppara)

        if (save_pre==1) then
            ! Save calculated parallel and perpendicular pressure.
            call save_para_perp_pressure(ct, is_subtract_bulkflow=.true.)
        endif
        deallocate(bsquare)
        call save_averaged_para_perp_pressure(ct, is_subtract_bulkflow=.true.)
    end subroutine calc_real_para_perp_pressure

    !---------------------------------------------------------------------------
    ! Save calculated parallel and perpendicular pressure.
    ! Input:
    !   ct: current time frame.
    !   is_subtract_bulkflow: flag for whether bulk flow is subtracted.
    !---------------------------------------------------------------------------
    subroutine save_para_perp_pressure(ct, is_subtract_bulkflow)
        use mpi_module
        use constants, only: fp, delta
        use mpi_io_module, only: save_field
        implicit none
        integer, intent(in) :: ct
        logical, intent(in) :: is_subtract_bulkflow
        if (myid == master) then
            print*, 'Saving parallel and perpendicular pressure', ct
        endif

        if (is_subtract_bulkflow) then
            call save_field(ppara, 'ppara_real', ct)
            call save_field(pperp, 'pperp_real', ct)
            call save_field(ppara/pperp, 'aniso_real', ct)
        else
            call save_field(ppara, 'ppara', ct)
            call save_field(pperp, 'pperp', ct)
            call save_field(ppara/pperp, 'aniso', ct)
        endif
    end subroutine save_para_perp_pressure

    !---------------------------------------------------------------------------
    ! Save averaged parallel and perpendicular pressure.
    ! Input:
    !   ct: current time frame.
    !   is_subtract_bulkflow: flag for whether bulk flow is subtracted.
    !---------------------------------------------------------------------------
    subroutine save_averaged_para_perp_pressure(ct, is_subtract_bulkflow)
        use mpi_module
        use constants, only: fp
        use particle_info, only: ibtag, species
        use statistics, only: get_average_and_total
        use parameters, only: it1
        implicit none
        integer, intent(in) :: ct
        logical, intent(in) :: is_subtract_bulkflow
        real(fp) :: ppara_avg, pperp_avg, tot
        character(len=100) :: fname
        integer :: current_pos, output_record
        call get_average_and_total(ppara, ppara_avg, tot)
        call get_average_and_total(pperp, pperp_avg, tot)
        if (myid == master) then
            if (is_subtract_bulkflow) then
                fname = 'data/pre_real'//ibtag//'_'//species//'.gda'
            else
                fname = 'data/pre'//ibtag//'_'//species//'.gda'
            endif
            open(unit=51, file=trim(adjustl(fname)), access='stream',&
                 status='unknown', form='unformatted', action='write')
            output_record = ct - it1 + 1
            current_pos = 2 * sizeof(fp) * (output_record-1) + 1
            write(51, pos=current_pos) ppara_avg, pperp_avg
            !PRINT*, input_record, 'Averaged pressure for ', & 
            !    species, ppara_avg, pperp_avg
            close(51)
        endif
    end subroutine save_averaged_para_perp_pressure

end module para_perp_pressure
