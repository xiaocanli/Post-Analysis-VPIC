!*******************************************************************************
! This model includes the methods to calculate j \cdot E, where j is the current
! density, E is the electric field.
!*******************************************************************************
module jdote_module
    use constants, only: fp
    implicit none
    private
    real(fp), allocatable, dimension(:, :, :) :: jdote
    public jdote, save_jdote_total, calc_jdote, init_jdote, free_jdote

    contains

    !---------------------------------------------------------------------------
    ! Initialize the jdote data array.
    !---------------------------------------------------------------------------
    subroutine init_jdote
        use mpi_topology, only: htg
        implicit none
        allocate(jdote(htg%nx, htg%ny, htg%nz))
        jdote = 0.0
    end subroutine init_jdote

    !---------------------------------------------------------------------------
    ! Free jdote data array.
    !---------------------------------------------------------------------------
    subroutine free_jdote
        implicit none
        deallocate(jdote)
    end subroutine free_jdote

    !---------------------------------------------------------------------------
    ! Save the total jdote summed in the whole simulation box.
    ! Inputs:
    !   ct: current time frame.
    !   jdtote_tot: the total jdote summed over the simulation box.
    !---------------------------------------------------------------------------
    subroutine save_jdote_total(ct, ncurrents, jdote_tot)
        use constants, only: fp
        use parameters, only: tp1
        use parameters, only: inductive
        use particle_info, only: species, ibtag
        implicit none
        integer, intent(in) :: ct, ncurrents
        real(fp), dimension(ncurrents+1), intent(in) :: jdote_tot
        integer :: pos1, output_record
        logical :: dir_e

        inquire(file='./data/.', exist=dir_e)
        if (.not. dir_e) then
            call system('mkdir ./data')
        endif

        print*, "Saving j.E", ct
        if (inductive == 0) then
            open(unit=41, &
                file='data/jdote'//ibtag//'_'//species//'.gda',access='stream',&
                status='unknown',form='unformatted',action='write')     
        else
            open(unit=41, &
                file='data/jdote_in'//ibtag//'_'//species//'.gda',access='stream',&
                status='unknown',form='unformatted',action='write')     
        endif
        output_record = ct - tp1 + 1
        pos1 = (output_record-1)*sizeof(fp)*(ncurrents+1) + 1
        write(41, pos=pos1) jdote_tot(1:13), jdote_tot(15), jdote_tot(14)
        close(41)
    end subroutine save_jdote_total

    !---------------------------------------------------------------------------
    ! Calculate J dot E from the fields.
    ! Input:
    !   jx, jy, jz: 3 components of the current density.
    ! Output:
    !   jdote_tot: the total jdote in the simulation box.
    !---------------------------------------------------------------------------
    subroutine calc_jdote(jx, jy, jz, jdote_tot)
        use mpi_module
        use constants, only: fp
        use parameters, only: inductive
        use pic_fields, only: ex, ey, ez
        use inductive_electric_field, only: exin, eyin, ezin
        use statistics, only: get_average_and_total
        implicit none
        real(fp), intent(out) :: jdote_tot
        real(fp), dimension(:, :, :), intent(in) :: jx, jy, jz
        real(fp) :: avg

        if (inductive == 0) then
            jdote = jx*ex + jy*ey + jz*ez
        else
            jdote = jx*exin + jy*eyin + jz*ezin
        endif
        jdote_tot = 0.0
        call get_average_and_total(jdote, avg, jdote_tot)
    end subroutine calc_jdote

end module jdote_module
