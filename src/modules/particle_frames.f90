!*******************************************************************************
! Module of particle output frames.
!*******************************************************************************
module particle_frames
    implicit none
    integer :: nt, tinterval ! Number of time frames and time interval

    contains

    subroutine get_particle_frames
        implicit none
        character(len=20) :: fname
        integer :: reason, tmp, tmax

        ! Get the files
        call system('ls ../particle > fileContents.txt')
        open(31, file='fileContents.txt', action='read')
        ! How many
        nt = 0
        tinterval = 1000000 ! some large number
        tmax = 1            ! Maximum time
        do
            read(31, '(A)', iostat=reason) fname
            read(fname(index(fname, '.')+1:), *) tmp
            if (reason /= 0) exit
            if (tmp < tinterval) tinterval = tmp
            if (tmp > tmax)      tmax = tmp
            nt = nt + 1
        enddo
        close(31)

       nt = tmax / tinterval ! Special treatment for data with time gap in it.

       ! Echo this information
       print *, "---------------------------------------------------"
       write(*, '(A,I0)') ' Number of time frames: ', nt
       write(*, '(A,I0)') ' The time interval is: ', tinterval
       print *, "---------------------------------------------------"
    end subroutine get_particle_frames

end module particle_frames
