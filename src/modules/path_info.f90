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

    subroutine get_file_paths
        implicit none
        integer :: status1, getcwd, index1
        status1 = getcwd(rootpath)
        index1 = index(rootpath, '/', back=.true.)
        rootpath = rootpath(1:index1)
        filepath = trim(rootpath)//'data/'
        outputpath = trim(rootpath)//'data1/'
    end subroutine get_file_paths
end module path_info
