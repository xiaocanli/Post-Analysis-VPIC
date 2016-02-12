!*******************************************************************************
! This model includes the methods to calculate j \cdot E, where j is the current
! density, E is the electric field.
!*******************************************************************************
module jdote_module
    use constants, only: fp
    implicit none
    private
    public jdote, jdote_tot
    public init_jdote_total, free_jdote_total, save_jdote_total, &
           calc_jdote, init_jdote, free_jdote
    real(fp), allocatable, dimension(:, :, :) :: jdote
    real(fp), allocatable, dimension(:, :) :: jdote_tot
    integer, parameter :: njdote = 16

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
    ! Initialize the total of jdote.
    !---------------------------------------------------------------------------
    subroutine init_jdote_total
        use parameters, only: tp1, tp2
        implicit none
        allocate(jdote_tot(njdote, tp2-tp1+1))
        jdote_tot = 0.0
    end subroutine init_jdote_total

    !---------------------------------------------------------------------------
    ! Free jdote data array.
    !---------------------------------------------------------------------------
    subroutine free_jdote
        implicit none
        deallocate(jdote)
    end subroutine free_jdote

    !---------------------------------------------------------------------------
    ! Free jdote_tot
    !---------------------------------------------------------------------------
    subroutine free_jdote_total
        implicit none
        deallocate(jdote_tot)
    end subroutine free_jdote_total

    !---------------------------------------------------------------------------
    ! Save the total jdote summed in the whole simulation box.
    !---------------------------------------------------------------------------
    subroutine save_jdote_total
        use constants, only: fp
        use parameters, only: tp1, tp2
        use parameters, only: inductive
        use particle_info, only: species, ibtag
        implicit none
        integer :: pos1, output_record
        logical :: dir_e
        integer :: ct

        inquire(file='./data/.', exist=dir_e)
        if (.not. dir_e) then
            call system('mkdir ./data')
        endif

        print*, "Saving j.E"
        if (inductive == 0) then
            open(unit=41, &
                file='data/jdote'//ibtag//'_'//species//'.gda',access='stream',&
                status='unknown',form='unformatted',action='write')     
        else
            open(unit=41, &
                file='data/jdote_in'//ibtag//'_'//species//'.gda',access='stream',&
                status='unknown',form='unformatted',action='write')     
        endif
        do ct = tp1, tp2
            output_record = ct - tp1 + 1
            pos1 = (output_record-1)*sizeof(fp)*njdote + 1
            write(41, pos=pos1) jdote_tot(1:13, output_record), &
                    jdote_tot(15, output_record), jdote_tot(14, output_record), &
                    jdote_tot(16, output_record)
        enddo
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
