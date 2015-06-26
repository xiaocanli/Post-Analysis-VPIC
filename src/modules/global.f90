!*******************************************************************************
! Module of the constants used in this code.
!*******************************************************************************
module constants
    implicit none
    private
    public dp, fp, delta, qi, qe
    integer, parameter :: dp=kind(0.d0)
    integer, parameter :: fp=kind(0.0)
    real(dp), parameter :: delta = 1.0E-15      ! Tiny number
    real(fp), parameter :: qi = 1.0, qe = -1.0  ! Charges
end module constants


!*******************************************************************************
! Module for MPI info
!*******************************************************************************
module mpi_module
    implicit none
    include "mpif.h"
    integer :: myid, numprocs, ierr
    integer, parameter :: master = 0

    integer :: ierror, ierror2, err_length
    integer :: status(MPI_STATUS_SIZE)
    character(len=256) :: err_msg
end module mpi_module


!*******************************************************************************
! Module of the variables that may need to be changed for different run.
!*******************************************************************************
module parameters
    implicit none
    private
    public tp1, tp2, inductive
    integer :: tp1 = 1, tp2 = 10 ! Starting and ending time output.
    integer :: inductive = 0
end module parameters

!*******************************************************************************
! Flags for whether save one kind of calculated field. 0 for not. 1 for yes.
!*******************************************************************************
module saving_flags
    implicit none
    private
    public save_jcpara, save_jcperp, save_jmag, save_jgrad, save_jdiagm, &
        save_jpolar, save_jexb, save_jpara, save_jperp, save_jperp1, save_jperp2, &
        save_jqnupara, save_jqnuperp, save_jagy, save_jtot, &
        save_pre
    integer, parameter :: save_jcpara=0, save_jcperp=0, save_jmag=0
    integer, parameter :: save_jgrad=0, save_jdiagm=0, save_jpolar=0
    integer, parameter :: save_jexb=0, save_jpara=0, save_jperp=0
    integer, parameter :: save_jperp1=0, save_jperp2=0, save_jqnupara=0
    integer, parameter :: save_jqnuperp=0, save_jagy=0, save_jtot=0
    integer, parameter :: save_pre=0
end module saving_flags
