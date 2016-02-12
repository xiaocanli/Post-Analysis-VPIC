!*******************************************************************************
! Module of particle output frames.
!*******************************************************************************
module particle_frames
    implicit none
    private
    public get_particle_frames_hydro, get_particle_frames
    public nt, tinterval, is_frame0

    integer :: nt, tinterval ! Number of time frames and time interval
    logical :: is_frame0     ! Whether time 0 is saved

    contains

    !---------------------------------------------------------------------------
    ! Get the number of particle frames and the time step interval. This is
    ! based on particle dump.
    !---------------------------------------------------------------------------
    subroutine get_particle_frames
        implicit none
        character(len=64) :: fpath

        fpath = '../particle/'
        call get_particle_frames_general(trim(fpath))
    end subroutine get_particle_frames

    !---------------------------------------------------------------------------
    ! Get the number of particle frames and the time step interval. This is
    ! based on spectra dump for each MPI process. The spectra are calculated
    ! for each MPI process of the PIC simulation and saved in "hydro" directory.
    !---------------------------------------------------------------------------
    subroutine get_particle_frames_hydro
        implicit none
        character(len=64) :: fpath

        fpath = '../hydro/'
        call get_particle_frames_general(trim(fpath))
    end subroutine get_particle_frames_hydro

    !---------------------------------------------------------------------------
    ! Get the number of particle frames and the time step interval.
    ! Input:
    !   fpath: the file path which contains the data
    !---------------------------------------------------------------------------
    subroutine get_particle_frames_general(fpath)
        implicit none
        character(*), intent(in) :: fpath
        character(len=64) :: fname, fname_full
        integer :: reason, tmp, tmax, stat, access
        integer :: nt0
        call system('ls '//fpath//' > fileContents.txt')
        open(31, file='fileContents.txt', action='read')
        is_frame0 = .False.
        ! How many
        nt = 0
        tinterval = 1000000 ! some large number
        tmax = 1            ! Maximum time
        do
            read(31, '(A)', iostat=reason) fname
            read(fname(index(fname, '.')+1:), *) tmp
            if (reason /= 0) exit
            if (tmp < tinterval .and. tmp > 0) tinterval = tmp
            if (tmp > tmax) tmax = tmp
            nt = nt + 1
        enddo
        close(31)

        ! In case there is time gap
        nt0 = tmax / tinterval
        fname_full = trim(fpath)//'T.0'  ! Time 0 frame
        stat = access(fname_full, 'r')
        if (stat .eq. 0) then
            nt0 = nt0 + 1
            is_frame0 = .True.
        endif
        nt = nt0

        ! Echo this information
        print *, "---------------------------------------------------"
        write(*, '(A,I0)') ' Number of time frames: ', nt
        write(*, '(A,I0)') ' The time interval is: ', tinterval
        print *, "---------------------------------------------------"
    end subroutine get_particle_frames_general

end module particle_frames
