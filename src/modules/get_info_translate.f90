!*******************************************************************************
! Module of time information. It includes subroutines to determine the number
! iterations between output files.
!*******************************************************************************
module time_info
    use mpi_module
    use path_info, only: rootpath
    implicit none
    private
    public nout, output_record, get_nout, adjust_tindex_start, set_output_record
    public nout_fd
    integer :: nout, tindex, tindex_first, tindex_next, output_record
    integer :: nout_fd  ! Time frame interval between frequent dump

    contains

    !---------------------------------------------------------------------------
    ! Get PIC the number of iterations between output files.
    !
    ! Args:
    !   frequent_dump: whether VPIC saves fields from previous and next time steps
    !---------------------------------------------------------------------------
    subroutine get_nout(frequent_dump)
        implicit none
        logical :: dfile, dfile1, dfile2
        logical, intent(in) :: frequent_dump
        character(len=256) :: fname1, fname2

        if (myid == master) then
            dfile = .false.
            tindex = 0
            do while(.not.dfile .and. tindex < 1000000)
                write(fname1, "(A, I0, A, I0, A)") &
                    trim(adjustl(rootpath))//"fields/T.", tindex, &
                    "/fields.", tindex, ".0"
                write(fname2, "(A, I0, A, I0, A)") &
                    trim(adjustl(rootpath))//"fields/0/T.", tindex, &
                    "/fields.", tindex, ".0"
                if (tindex .ne. 1) then
                    inquire(file=trim(fname1), exist=dfile1)
                    inquire(file=trim(fname2), exist=dfile2)
                    dfile = dfile1 .or. dfile2
                endif
                tindex = tindex + 1
            enddo
            tindex_first = tindex - 1
            dfile = .false.

            if (frequent_dump) then
                tindex = tindex_first + 3 ! Skip three steps
            endif

            do while(.not.dfile)
                write(fname1, "(A, I0, A, I0, A)") &
                    trim(adjustl(rootpath))//"fields/T.", tindex, &
                    "/fields.", tindex, ".0"
                write(fname2, "(A, I0, A, I0, A)") &
                    trim(adjustl(rootpath))//"fields/0/T.", tindex, &
                    "/fields.", tindex, ".0"
                if (tindex .ne. 1) then
                    inquire(file=trim(fname1), exist=dfile1)
                    inquire(file=trim(fname2), exist=dfile2)
                    dfile = dfile1 .or. dfile2
                endif
                tindex = tindex + 1
            enddo

            dfile = .false.
            nout_fd = 0
            if (frequent_dump) then
                do while(.not.dfile)
                    write(fname1, "(A, I0, A, I0, A)") &
                        trim(adjustl(rootpath))//"fields/T.", tindex, &
                        "/fields.", tindex, ".0"
                    write(fname2, "(A, I0, A, I0, A)") &
                        trim(adjustl(rootpath))//"fields/0/T.", tindex, &
                        "/fields.", tindex, ".0"
                    if (tindex .ne. 1) then
                        inquire(file=trim(fname1), exist=dfile1)
                        inquire(file=trim(fname2), exist=dfile2)
                        dfile = dfile1 .or. dfile2
                    endif
                    tindex = tindex + 1
                    nout_fd = nout_fd + 1
                enddo
            endif
            tindex_next = tindex - 1
            nout = tindex_next - tindex_first

            ! Total size of domain
            print *,"---------------------------------------------------"
            print *,"Iterations between output = ", nout
            if (nout_fd > 0) then
                print *,"Iterations between frequent dump = ", nout_fd
            endif
            print *,"---------------------------------------------------"
        endif

        call MPI_BCAST(nout, 1, MPI_INTEGER, master, MPI_COMM_WORLD, ierr)
        call MPI_BCAST(nout_fd, 1, MPI_INTEGER, master, MPI_COMM_WORLD, ierr)
        call MPI_BCAST(tindex_first, 1, MPI_INTEGER, master, MPI_COMM_WORLD, ierr)
    end subroutine get_nout

    !---------------------------------------------------------------------------
    ! Adjust tindex_start in case it is smaller than the minimal time step
    ! output that is available. This occurs when some of the earlier outputs
    ! are deleted.
    !---------------------------------------------------------------------------
    subroutine adjust_tindex_start
        use mpi_module
        use configuration_translate, only: tindex_start
        implicit none
        character(len=150) :: fname
        logical :: dfile, dfile1, dfile2
        if (tindex_start < tindex_first) then
            dfile = .false.
            write(fname, "(A,I0,A,I0,A)") &
                trim(adjustl(rootpath))//"fields/T.0/fields.0.0"  ! 1st frame
            inquire(file=trim(fname), exist=dfile1)
            write(fname, "(A,I0,A,I0,A)") &
                trim(adjustl(rootpath))//"fields/0/T.0/fields.0.0"
            inquire(file=trim(fname), exist=dfile2)
            dfile = dfile1 .or. dfile2
            if (.not. dfile) then
                tindex_start = tindex_first
                if (myid == master) then
                    print *,"---------------------------------------------------"
                    write(*, '(A, I0)') ' tindex_start is updated to: ', tindex_start
                    print *,"---------------------------------------------------"
                endif
            endif
        endif
    end subroutine adjust_tindex_start

    !---------------------------------------------------------------------------
    ! Set output_record, which determines the offset from the file beginning
    ! when writing to a file. It depends on whether to append to previous
    ! output. It needs to determine the last record written, so we know which
    ! time slice to process next.
    !---------------------------------------------------------------------------
    subroutine set_output_record
        use configuration_translate, only: tindex_start, append_to_files
        implicit none
        if (append_to_files==1) then
            output_record = (tindex_start/nout) + 1
        else
            output_record = 1
        endif
    end subroutine set_output_record

end module time_info
