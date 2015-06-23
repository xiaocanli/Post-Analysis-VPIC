!*******************************************************************************
! Module of particle file. It includes the routines to check the existence of
! a particle file, to open a particle file and read its headers, to check if the
! particles are in the required spatial range.
!*******************************************************************************
module particle_file
    use path_info, only: rootpath
    implicit none
    private
    public check_existence, open_particle_file, close_particle_file
    public check_particle_in_range, fh, ratio_interval, get_ratio_interval, &
           check_both_particle_fields_exist
    ! The ratio of the particle and field output intervals
    integer :: ratio_interval
    integer :: fh

    contains

    !---------------------------------------------------------------------------
    ! Check the existence of the dataset. This is for the case that there is
    ! time gaps in the output files.
    ! Inputs:
    !   tindex: the time index, indicating the time step numbers in PIC simulation.
    !   species: 'e' for electron. 'h' for others.
    !---------------------------------------------------------------------------
    subroutine check_existence(tindex, species, existFlag)
        implicit none
        character(len=1), intent(in) :: species
        integer, intent(in) :: tindex
        logical, intent(out) :: existFlag
        character(len=20) :: ctindex
        character(len=150) :: dataset, fname

        write(ctindex, "(I0)") tindex
        dataset = trim(adjustl(rootpath))//"particle/T."//trim(ctindex)
        dataset = trim(adjustl(dataset))//"/"//species//"particle."
        fname = trim(dataset)//trim(ctindex)//".0"
        inquire(file=fname, exist=existFlag)
        if (.not. existFlag) then
            print*, fname, " doesn't exist." 
            print*, "There is probably a gap in the output."
        endif
    end subroutine check_existence

    !---------------------------------------------------------------------------
    ! Open a particle file and read its headers.
    ! Input:
    !   tindex: the time index, indicating the time step numbers in PIC simulation.
    !   species: 'e' for electron. 'h' for others.
    !   cid: the MPI ID for the PIC simulation.
    !---------------------------------------------------------------------------
    subroutine open_particle_file(tindex, species, cid)
        use file_header, only: read_boilerplate, read_particle_header
        implicit none
        character(len=1), intent(in) :: species
        integer, intent(in) :: tindex
        character(*), intent(in) :: cid
        character(len=50) :: ctindex
        character(len=150) :: dataset, fname

        fh = 10
        write(ctindex, "(I0)") tindex
        dataset = trim(adjustl(rootpath))//"particle/T."//trim(ctindex)
        dataset = trim(adjustl(dataset))//"/"//species//"particle."
        fname = trim(dataset)//trim(ctindex)//"."//trim(cid)

        open(unit=fh, file=trim(fname), status='unknown', &
             form='unformatted', access='stream', action='read')
        write(*, '(A,A)') "Reading --> ", trim(fname)

        call read_boilerplate(fh)
        call read_particle_header(fh)
    end subroutine open_particle_file

    !---------------------------------------------------------------------------
    ! Check if the particles in this particle file are in the required spatial
    ! range. The method is to check the bottom-left and top-right corners.
    !---------------------------------------------------------------------------
    function check_particle_in_range(spatial_range) result(isrange)
        use constants, only: fp
        use picinfo, only: domain
        use file_header, only: v0
        implicit none
        real(fp), dimension(2,3), intent(in) :: spatial_range
        real(fp) :: x0, y0, z0, x1, y1, z1
        logical :: isrange, isrange1, isrange2

        ! Corners of this MPI process's domain
        x0 = v0%x0
        y0 = v0%y0
        z0 = v0%z0
        x1 = v0%x0 + domain%pic_nx * domain%dx
        y1 = v0%y0 + domain%pic_ny * domain%dy
        z1 = v0%z0 + domain%pic_nz * domain%dz

        ! Only if the corners are within the box.
        ! Shift one grid to cover boundary.
        isrange1 = x1 >= (spatial_range(1,1) - domain%dx) &
             .and. x1 <= (spatial_range(2,1) + domain%dx) &
             .and. y1 >= (spatial_range(1,2) - domain%dy) &
             .and. y1 <= (spatial_range(2,2) + domain%dy) &
             .and. z1 >= (spatial_range(1,3) - domain%dz) &
             .and. z1 <= (spatial_range(2,3) + domain%dz)
        isrange2 = x0 >= (spatial_range(1,1) - domain%dx) &
             .and. x0 <= (spatial_range(2,1) + domain%dx) &
             .and. y0 >= (spatial_range(1,2) - domain%dy) &
             .and. y0 <= (spatial_range(2,2) + domain%dy) &
             .and. z0 >= (spatial_range(1,3) - domain%dz) &
             .and. z0 <= (spatial_range(2,3) + domain%dz)

        isrange = isrange1 .or. isrange2
    end function check_particle_in_range

    !---------------------------------------------------------------------------
    ! Close the particle file.
    !---------------------------------------------------------------------------
    subroutine close_particle_file
        implicit none
        close(fh)
    end subroutine close_particle_file

    !---------------------------------------------------------------------------
    ! To get the ratio of particle output interval and the fields_interval.
    ! These two are different for these two. And the ratio is given in sigma.cxx
    !---------------------------------------------------------------------------
    subroutine get_ratio_interval
        implicit none
        character(len=150) :: buff
        character(len=50) :: buff1, format1
        integer :: len1, len2, len3
        ratio_interval = 0
        open(unit=40, file=trim(adjustl(rootpath))//'sigma.cxx', status='old')
        read(40,'(A)') buff
        do while (index(buff, 'int Hhydro_interval = ') == 0)
            read(40,'(A)') buff
        enddo
        read(40, '(A)') buff
        len1 = len(trim(buff))
        ! "int eparticle_interval = " has 25 characters
        len2 = index(buff, 'int') + 24
        ! The last 10 characters are "*interval;"
        len3 = len1 - len2 - 10
        write(format1, "(A,I2.2,A,I1.1,A,I1.1,A)") &
                "(A", len2, ",I", len3, ".", len3, ")'"
        read(buff, trim(adjustl(format1))) buff1, ratio_interval
        close(40)
        print*, "The ratio of particle and field output intervals: ", &
            ratio_interval
    end subroutine get_ratio_interval

    !---------------------------------------------------------------------------
    ! Check if the time frame has both particle and fields data.
    ! Input:
    !   ct: current time frame for the fields output.
    !---------------------------------------------------------------------------
    function check_both_particle_fields_exist(ct) result(is_time_valid)
        implicit none
        integer, intent(in) :: ct
        logical :: is_time_valid
        is_time_valid = .true.
        if (mod(ct, ratio_interval) /= 0) then
            is_time_valid = .false.
        endif
    end function check_both_particle_fields_exist

end module particle_file
