!*******************************************************************************
! Flags for whether save one kind of calculated field. 0 for not. 1 for yes.
! This is for saving large 2D or 3D data.
!*******************************************************************************
module saving_flags
    implicit none
    private
    public save_jcpara, save_jcperp, save_jmag, save_jgrad, save_jdiagm, &
           save_jpolar, save_jexb, save_jpara, save_jperp, save_jperp1, &
           save_jperp2, save_jqnupara, save_jqnuperp, save_jagy, save_jtot, &
           save_jdivu, save_pre
    public get_saving_flags

    integer :: save_jcpara, save_jcperp, save_jmag
    integer :: save_jgrad, save_jdiagm, save_jpolar
    integer :: save_jexb, save_jpara, save_jperp
    integer :: save_jperp1, save_jperp2, save_jqnupara
    integer :: save_jqnuperp, save_jagy, save_jtot, save_jdivu
    integer :: save_pre

    contains

    !---------------------------------------------------------------------------
    ! Read the saving flags from configuration file.
    !---------------------------------------------------------------------------
    subroutine get_saving_flags
        use read_config, only: get_variable_int
        implicit none
        integer :: fh
        fh = 15
        open(unit=fh, file='config_files/saving_flags.dat', status='old')
        save_jcpara = get_variable_int(fh, 'save_jcpara', '=')
        save_jcperp = get_variable_int(fh, 'save_jcperp', '=')
        save_jmag = get_variable_int(fh, 'save_jmag', '=')
        save_jgrad = get_variable_int(fh, 'save_jgrad', '=')
        save_jdiagm = get_variable_int(fh, 'save_jdiagm', '=')
        save_jpolar = get_variable_int(fh, 'save_jpolar', '=')
        save_jexb = get_variable_int(fh, 'save_jexb', '=')
        save_jpara = get_variable_int(fh, 'save_jpara', '=')
        save_jperp = get_variable_int(fh, 'save_jperp', '=')
        save_jperp1 = get_variable_int(fh, 'save_jperp1', '=')
        save_jperp2 = get_variable_int(fh, 'save_jperp2', '=')
        save_jqnupara = get_variable_int(fh, 'save_jqnupara', '=')
        save_jqnuperp = get_variable_int(fh, 'save_jqnuperp', '=')
        save_jagy = get_variable_int(fh, 'save_jagy', '=')
        save_jtot = get_variable_int(fh, 'save_jtot', '=')
        save_jtot = get_variable_int(fh, 'save_jdivu', '=')
        save_pre = get_variable_int(fh, 'save_pre', '=')
        close(fh)
    end subroutine get_saving_flags

end module saving_flags
