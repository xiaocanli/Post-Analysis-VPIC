!*******************************************************************************
! Module of file path information, including the root path of current PIC run,
! the filepath of the data files (bx, by, bz ...) and the outputpath of current
! analysis.
!*******************************************************************************
module path_info
    implicit none
    save
    private
    public rootpath, filepath, outputpath, get_file_paths
    character(len=150) :: rootpath, filepath, outputpath

    contains

    subroutine get_file_paths(rpath)
        use mpi_module
        implicit none
        character(*), intent(in), optional :: rpath
        integer :: status1, getcwd, index1
        if (present(rpath)) then
            rootpath = trim(adjustl(rpath))
        else
            status1 = getcwd(rootpath)
            index1 = index(rootpath, '/', back=.true.)
            rootpath = rootpath(1:index1)
        endif
        filepath = trim(rootpath)//'data/'
        outputpath = trim(rootpath)//'data1/'
        if (myid == master) then
            call create_directories
        endif
    end subroutine get_file_paths

    subroutine create_directories
        implicit none
        logical :: dir_e
        character(len=256) :: fname
        dir_e = .false.
        inquire(file=filepath, exist=dir_e)
        if (.not. dir_e) then
            fname = 'mkdir '//filepath
            call system(fname)
        endif

        inquire(file=outputpath, exist=dir_e)
        if (.not. dir_e) then
            fname = 'mkdir '//outputpath
            call system(fname)
        endif
    end subroutine create_directories
end module path_info
