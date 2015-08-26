!*******************************************************************************
! Module for parallel and perpendicular pressure. It includes the data and
! the methods to calculate the pressures.
!*******************************************************************************
module para_perp_pressure
    use constants, only: fp
    use parameters, only: is_rel
    implicit none
    private
    public ppara, pperp, init_para_perp_pressure, free_para_perp_pressure
    public calc_para_perp_pressure, calc_real_para_perp_pressure
    public save_para_perp_pressure, save_averaged_para_perp_pressure
    public init_avg_para_perp_pressure, free_avg_para_perp_pressure
    real(fp), allocatable, dimension(:,:,:) :: pperp, ppara
    real(fp), allocatable, dimension(:) :: pperp_avg, ppara_avg
    logical :: is_subtract_bulkflow

    contains

    !---------------------------------------------------------------------------
    ! Initialize the averaged parallel and perpendicular pressure.
    !---------------------------------------------------------------------------
    subroutine init_avg_para_perp_pressure
        use parameters, only: tp1, tp2
        implicit none
        allocate(ppara_avg(tp2-tp1+1))
        allocate(pperp_avg(tp2-tp1+1))

        ppara_avg = 0.0
        pperp_avg = 0.0
    end subroutine init_avg_para_perp_pressure

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

        call init_avg_para_perp_pressure
    end subroutine init_para_perp_pressure

    !---------------------------------------------------------------------------
    ! Free the averaged parallel and perpendicular pressure.
    !---------------------------------------------------------------------------
    subroutine free_avg_para_perp_pressure
        implicit none
        deallocate(ppara_avg, pperp_avg)
    end subroutine free_avg_para_perp_pressure

    !---------------------------------------------------------------------------
    ! Free the memory used by parallel and perpendicular pressure.
    !---------------------------------------------------------------------------
    subroutine free_para_perp_pressure
        implicit none
        deallocate(ppara, pperp)
        call free_avg_para_perp_pressure
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
        use parameters, only: tp1
        use pic_fields, only: bx, by, bz, absB, vx, vy, vz, ux, uy, uz, &
                num_rho, pxx, pxy, pxz, pyy, pyz, pzz, pyx, pzx, pzy
        use statistics, only: get_average_and_total
        use particle_info, only: ptl_mass
        use mpi_topology, only: htg
        use saving_flags, only: save_pre
        implicit none
        integer, intent(in) :: ct
        real(fp), allocatable, dimension(:,:,:) :: bsquare, prexx, preyy, prezz
        real(fp) :: tot
        integer :: nx, ny, nz

        nx = htg%nx
        ny = htg%ny
        nz = htg%nz

        allocate(bsquare(nx, ny, nz))
        allocate(prexx(nx, ny, nz))
        allocate(preyy(nx, ny, nz))
        allocate(prezz(nx, ny, nz))

        bsquare = bx*bx + by*by + bz*bz
        if (is_rel == 1) then
            prexx = pxx + vx*ux*num_rho*ptl_mass
            preyy = pyy + vy*uy*num_rho*ptl_mass
            prezz = pzz + vz*uz*num_rho*ptl_mass
            ppara = prexx * bx * bx + &
                    preyy * by * by + &
                    prezz * bz * bz + &
                    (pxy + vx*uy*num_rho*ptl_mass) * bx * by + &
                    (pyx + vy*ux*num_rho*ptl_mass) * bx * by + &
                    (pxz + vx*uz*num_rho*ptl_mass) * bx * bz + &
                    (pzx + vz*ux*num_rho*ptl_mass) * bx * bz + &
                    (pyz + vy*uz*num_rho*ptl_mass) * by * bz + &
                    (pzy + vz*uy*num_rho*ptl_mass) * by * bz
        else
            prexx = pxx + vx*vx*num_rho*ptl_mass
            preyy = pyy + vy*vy*num_rho*ptl_mass
            prezz = pzz + vz*vz*num_rho*ptl_mass
            ppara = prexx * bx * bx + &
                    preyy * by * by + &
                    prezz * bz * bz + &
                    (pxy + vx*vy*num_rho*ptl_mass) * bx * by * 2.0 + &
                    (pxz + vx*vz*num_rho*ptl_mass) * bx * bz * 2.0 + &
                    (pyz + vy*vz*num_rho*ptl_mass) * by * bz * 2.0
        endif
        ppara = ppara / bsquare
        pperp = 0.5 * (prexx + preyy + prezz - ppara)

        is_subtract_bulkflow = .false.
        if (save_pre==1) then
            call save_para_perp_pressure(ct)
        endif
        deallocate(bsquare, prexx, preyy, prezz)
        call get_average_and_total(ppara, ppara_avg(ct-tp1+1), tot)
        call get_average_and_total(pperp, pperp_avg(ct-tp1+1), tot)
    end subroutine calc_para_perp_pressure

    !---------------------------------------------------------------------------
    ! Calculate parallel and perpendicular pressure from the pressure tensor
    ! and magnetic field. The bulk flow is subtracted from the pressure.
    ! Inputs:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine calc_real_para_perp_pressure(ct)
        use constants, only: fp
        use parameters, only: tp1
        use pic_fields, only: bx, by, bz, absB, num_rho, &
                pxx, pxy, pxz, pyy, pyz, pzz, pyx, pzx, pzy
        use statistics, only: get_average_and_total
        use mpi_topology, only: htg
        use saving_flags, only: save_pre
        implicit none
        integer, intent(in) :: ct
        real(fp), allocatable, dimension(:,:,:) :: bsquare
        real(fp) :: tot
        integer :: nx, ny, nz

        nx = htg%nx
        ny = htg%ny
        nz = htg%nz

        allocate(bsquare(nx, ny, nz))

        bsquare = bx*bx + by*by + bz*bz
        if (is_rel == 1) then
            ppara = pxx*bx*bx + pyy*by*by + pzz*bz*bz + &
                    (pxy + pyx) * bx * by + (pxz + pzx) * bx * bz + &
                    (pyz + pzy) * by * bz
        else
            ppara = pxx*bx*bx + pyy*by*by + pzz*bz*bz + &
                    (pxy * bx * by * 2.0) + (pxz * bx * bz * 2.0) + &
                    (pyz * by * bz * 2.0)
        endif
        ppara = ppara / bsquare
        pperp = 0.5 * (pxx + pyy + pzz - ppara)

        is_subtract_bulkflow = .true.
        if (save_pre==1) then
            call save_para_perp_pressure(ct)
        endif
        deallocate(bsquare)
        call get_average_and_total(ppara, ppara_avg(ct-tp1+1), tot)
        call get_average_and_total(pperp, pperp_avg(ct-tp1+1), tot)
    end subroutine calc_real_para_perp_pressure

    !---------------------------------------------------------------------------
    ! Save calculated parallel and perpendicular pressure.
    ! Input:
    !   ct: current time frame.
    !---------------------------------------------------------------------------
    subroutine save_para_perp_pressure(ct)
        use mpi_module
        use constants, only: fp, delta
        use mpi_io_fields, only: save_field
        implicit none
        integer, intent(in) :: ct
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
    !---------------------------------------------------------------------------
    subroutine save_averaged_para_perp_pressure
        use constants, only: fp
        use particle_info, only: ibtag, species
        use parameters, only: tp1, tp2
        implicit none
        character(len=100) :: fname
        integer :: current_pos, output_record
        logical :: dir_e
        integer :: ct
    
        inquire(file='./data/.', exist=dir_e)
        if (.not. dir_e) then
            call system('mkdir ./data')
        endif

        if (is_subtract_bulkflow) then
            fname = 'data/pre_real'//ibtag//'_'//species//'.gda'
        else
            fname = 'data/pre'//ibtag//'_'//species//'.gda'
        endif
        open(unit=51, file=trim(adjustl(fname)), access='stream',&
             status='unknown', form='unformatted', action='write')
        do ct = tp1, tp2
            output_record = ct - tp1 + 1
            current_pos = 2 * sizeof(fp) * (output_record-1) + 1
            write(51, pos=current_pos) ppara_avg(output_record), &
                    pperp_avg(output_record)
        enddo
        close(51)
    end subroutine save_averaged_para_perp_pressure

end module para_perp_pressure
