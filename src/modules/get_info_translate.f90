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
    integer :: nout, tindex, tindex_first, tindex_next, output_record

    contains

    !---------------------------------------------------------------------------
    ! Get PIC the number of iterations between output files.
    !---------------------------------------------------------------------------
    subroutine get_nout
        implicit none
        logical :: dfile
        character(len=150) :: fname

        if (myid == master) then
            dfile = .false.
            tindex = 1
            do while(.not.dfile)
                tindex = tindex + 1
                write(fname, "(A, I0, A, I0, A)") &
                    trim(adjustl(rootpath))//"fields/T.", tindex, &
                    "/fields.", tindex, ".0"
                if (tindex .ne. 1) inquire(file=trim(fname), exist=dfile)
            enddo
            tindex_first = tindex
            dfile = .false.

            do while(.not.dfile)
                tindex = tindex + 1
                write(fname, "(A, I0, A, I0, A)") &
                    trim(adjustl(rootpath))//"fields/T.", tindex, &
                    "/fields.", tindex, ".0"
                if (tindex .ne. 1) inquire(file=trim(fname), exist=dfile)
            enddo
            tindex_next = tindex
            nout = tindex_next - tindex_first

            ! Total size of domain
            print *,"---------------------------------------------------"
            print *,"Iterations between output = ", nout
            print *,"---------------------------------------------------"
        endif

        call MPI_BCAST(nout, 1, MPI_INTEGER, master, MPI_COMM_WORLD, ierr)
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
        logical :: dfile
        if (tindex_start < tindex_first) then
            dfile = .false.
            write(fname, "(A,I0,A,I0,A)") &
                trim(adjustl(rootpath))//"fields/T.0/fields.0.0"  ! 1st frame
            inquire(file=trim(fname), exist=dfile)
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
