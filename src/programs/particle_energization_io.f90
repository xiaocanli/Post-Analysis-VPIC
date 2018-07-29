!<******************************************************************************
!< Program for calculating particle energization.
!< This version tries to minimize the IO operations.
!<******************************************************************************
program particle_energization_io
    use constants, only: fp, dp
    use mpi_module
    use path_info, only: set_filepath
    use particle_info, only: species, ptl_mass, ptl_charge
    use parameters, only: tp1, tp2
    use configuration_translate, only: output_format
    use particle_module, only: particle
    use hdf5
    implicit none
    character(len=256) :: rootpath
    character(len=16) :: dir_emf, dir_hydro
    real(dp) :: start, finish, step1, step2, time1, time2
    integer :: tstart, tend, tinterval, tframe, fields_interval, fd_tinterval
    integer, parameter :: nbins = 60
    real(fp), parameter :: emin = 1E-4
    real(fp), parameter :: emax = 1E2
    integer :: nvar, separated_pre_post
    real(fp), allocatable, dimension(:) :: ebins
    real(fp), allocatable, dimension(:, :) :: fbins, fbins_sum
    real(fp) :: de_log, emin_log
    integer :: i, tp_emf, tp_hydro
    logical :: is_translated_file
    type(particle), allocatable, dimension(:) :: ptls
    integer :: ptl_rm_local, ptl_rm_global  ! Number of particles got removed
                                            ! when local electric field is too large

    ! Particles in HDF5 format
    integer, allocatable, dimension(:) :: np_local, offset_local
    logical :: particle_hdf5, parallel_read, collective_io
    integer, parameter :: num_dset = 8
    integer(hid_t), dimension(num_dset) :: dset_ids
    integer(hid_t) :: file_id, group_id
    integer(hid_t) :: filespace
    integer(hsize_t), dimension(1) :: dset_dims, dset_dims_max

    call MPI_INIT(ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)

    call cpu_time(start)

    call get_cmd_args

    call init_analysis

    nvar = 18
    call calc_particle_energization

    call end_analysis

    call cpu_time(finish)
    if (myid == master) then
        print '("Time = ",f9.4," seconds.")',finish-start
    endif

    call MPI_FINALIZE(ierr)

    contains

    !<--------------------------------------------------------------------------
    !< Initialize energy bins and distributions
    !<--------------------------------------------------------------------------
    subroutine init_dists
        implicit none
        de_log = (log10(emax/ptl_mass) - log10(emin/ptl_mass)) / nbins
        emin_log = log10(emin/ptl_mass)
        do i = 1, nbins + 1
            ebins(i) = emin * 10**(de_log * (i - 1))
        enddo
        ebins = ebins / ptl_mass
        call set_dists_zero
    end subroutine init_dists

    !<--------------------------------------------------------------------------
    !< Set distributions to be 0
    !<--------------------------------------------------------------------------
    subroutine set_dists_zero
        implicit none
        fbins = 0.0
        fbins_sum = 0.0
    end subroutine set_dists_zero

    !<--------------------------------------------------------------------------
    !< Calculate particle energization due to parallel and perpendicular
    !< electric field, compression and shear, curvature drift, gradient drift,
    !< initial drift, polarization drift, and the conservation magnetic moment.
    !< It requires 86 field components:
    !<  * electric field components (3)
    !<  * magnetic fields components and magnitude (4)
    !<  * velocity field (3)
    !<  * momentum field (3)
    !<  * electric and magnetic fields at previous and latter time steps (16)
    !<  * momentum field at previous and latter time steps (6)
    !<  * velocity field at previous and latter time steps (6)
    !<  * ExB drift velocity (3)
    !<  * the perpendicular components of the velocity field (3)
    !<  * the magnitude of the magnetic field (3)
    !<  * gradients of magnetic field (9)
    !<  * gradients of momentum field (9)
    !<  * gradients of the perpendicular components of the velocity field (9)
    !<  * gradients of ExB drift (9)
    !< We need to read 36 field components:
    !<  * electric field components (3)
    !<  * magnetic fields components (3)
    !<  * velocity field (3)
    !<  * momentum field (3)
    !<  * electric and magnetic fields at previous and latter time steps (12)
    !<  * momentum field at previous and latter time steps (6)
    !<  * velocity field at previous and latter time steps (6)
    !<--------------------------------------------------------------------------
    subroutine calc_particle_energization
        use picinfo, only: domain
        use topology_translate, only: ht
        use mpi_topology, only: htg
        use pic_fields, only: init_electric_fields, init_magnetic_fields, &
            init_velocity_fields, free_electric_fields, free_magnetic_fields, &
            free_velocity_fields, open_electric_field_files, open_magnetic_field_files, &
            open_velocity_field_files, read_electric_fields, read_magnetic_fields, &
            read_velocity_fields, close_electric_field_files, &
            close_magnetic_field_files, close_velocity_field_files, &
            interp_emf_node_ghost
        use pre_post_emf, only: init_pre_post_bfield, init_pre_post_efield, &
            free_pre_post_bfield, free_pre_post_efield, open_bfield_pre_post, &
            open_efield_pre_post, close_bfield_pre_post, close_efield_pre_post, &
            read_pre_post_bfield, read_pre_post_efield, interp_bfield_node_ghost, &
            interp_efield_node_ghost
        use pre_post_hydro, only: init_pre_post_u, init_pre_post_v, &
            free_pre_post_u, free_pre_post_v, open_ufield_pre_post, &
            open_vfield_pre_post, close_ufield_pre_post, close_vfield_pre_post, &
            read_pre_post_u, read_pre_post_v
        use emf_derivatives, only: init_bfield_derivatives, &
            free_bfield_derivatives, calc_bfield_derivatives, &
            init_absb_derivatives, free_absb_derivatives, &
            calc_absb_derivatives
        use hydro_derivatives, only: init_ufield_derivatives, &
            free_ufield_derivatives, calc_ufield_derivatives
        use vperp_derivatives, only: init_vperp_derivatives, &
            init_vperp_components, free_vperp_derivatives, free_vperp_components, &
            calc_vperp_derivatives, calc_vperp_components
        use exb_drift, only: init_exb_drift, free_exb_drift, calc_exb_drift, &
            init_exb_derivatives, free_exb_derivatives, calc_exb_derivatives
        use interpolation_emf, only: init_emfields, init_emfields_derivatives, &
            free_emfields, free_emfields_derivatives, &
            init_absb_derivatives_single, free_absb_derivatives_single
        use interpolation_vel_mom, only: init_vel_mom, free_vel_mom
        use interpolation_pre_post_bfield, only: init_bfield_magnitude, &
            init_bfield_components, free_bfield_magnitude, free_bfield_components
        use interpolation_pre_post_efield, only: init_efield_components, &
            free_efield_components
        use interpolation_pre_post_ufield, only: init_ufield_components, &
            free_ufield_components
        use interpolation_pre_post_vfield, only: init_vfield_components, &
            free_vfield_components
        use interpolation_vexb, only: init_exb_derivatives_single, &
            free_exb_derivatives_single
        use interpolation_ufield_derivatives, only: init_ufields_derivatives_single, &
            free_ufields_derivatives_single
        use interpolation_vperp_derivatives, only: init_vperp_derivatives_single, &
            free_vperp_derivatives_single
        implicit none
        integer :: dom_x, dom_y, dom_z
        integer :: tindex, tindex_pre, tindex_pos
        real(fp) :: dx_domain, dy_domain, dz_domain
        real(fp) :: dt_fields

        call init_emfields
        call init_emfields_derivatives
        call init_vel_mom
        call init_bfield_components
        call init_bfield_magnitude
        call init_efield_components
        call init_ufield_components
        call init_ufields_derivatives_single
        call init_exb_derivatives_single
        call init_vfield_components
        call init_vperp_derivatives_single
        call init_absb_derivatives_single
        if (is_translated_file) then
            call init_electric_fields(htg%nx, htg%ny, htg%nz)    ! 3 components
            call init_magnetic_fields(htg%nx, htg%ny, htg%nz)    ! 3 components + magnitude
            call init_velocity_fields(htg%nx, htg%ny, htg%nz)    ! 3 v + 3 u
            call init_pre_post_bfield(htg%nx, htg%ny, htg%nz)    ! 6 components + 2 magnitude
            call init_pre_post_efield(htg%nx, htg%ny, htg%nz)    ! 6 components + 2 magnitude
            call init_pre_post_u(htg%nx, htg%ny, htg%nz)         ! 6 components
            call init_pre_post_v(htg%nx, htg%ny, htg%nz)         ! 6 components
            call init_exb_drift(htg%nx, htg%ny, htg%nz)          ! 3 components
            call init_bfield_derivatives(htg%nx, htg%ny, htg%nz) ! 9 components
            call init_exb_derivatives(htg%nx, htg%ny, htg%nz)    ! 9 components
            call init_ufield_derivatives(htg%nx, htg%ny, htg%nz) ! 9 components
            call init_vperp_components(htg%nx, htg%ny, htg%nz)   ! 3 components
            call init_vperp_derivatives(htg%nx, htg%ny, htg%nz)  ! 9 components
            call init_absb_derivatives(htg%nx, htg%ny, htg%nz)   ! 3 components
        endif

        allocate(ebins(nbins + 1))
        allocate(fbins(nbins + 1, nvar))
        allocate(fbins_sum(nbins + 1, nvar))

        call init_dists

        if (myid == master) then
            print '(A)', 'Finished initializing the analysis'
        endif

        if (is_translated_file .and. output_format == 1) then
            call set_filepath(dir_emf)
            call open_electric_field_files
            call open_magnetic_field_files
            call open_velocity_field_files(species)
            call open_bfield_pre_post(separated_pre_post)
            call open_efield_pre_post(separated_pre_post)
            call open_ufield_pre_post(species, separated_pre_post)
            call open_vfield_pre_post(species, separated_pre_post)
        endif

        call cpu_time(step1)
        do tframe = tstart, tend, tinterval
            if (myid == master) print*, tframe
            tp_emf = tframe / fields_interval + 1
            call set_dists_zero
            call cpu_time(time1)

            ! Time frame and interval
            tindex = domain%fields_interval * (tp_emf - tp1)
            if (separated_pre_post) then
                if (tp_emf == tp1) then
                    tindex_pre = tindex
                    tindex_pos = 1
                else if (tp_emf == tp2) then
                    tindex_pre = tindex - fd_tinterval
                    tindex_pos = tindex
                else
                    tindex_pre = tindex - fd_tinterval
                    tindex_pos = tindex + fd_tinterval
                endif
            else
                ! Not well tested now
                ! if (tp_emf == tp1 .or. tp_emf == tp2) then
                !     dt_fields = domain%dt
                ! else
                !     dt_fields = domain%dt * 2.0
                ! endif
            endif
            dt_fields = domain%dtwpe * (tindex_pos - tindex_pre)

            if (is_translated_file) then
                if (output_format /= 1) then
                    call open_electric_field_files(tindex)
                    call read_electric_fields(tp1)
                    call close_electric_field_files
                    call open_magnetic_field_files(tindex)
                    call read_magnetic_fields(tp1)
                    call close_magnetic_field_files
                    call open_velocity_field_files(species, tindex)
                    call read_velocity_fields(tp1)
                    call close_velocity_field_files
                    call open_bfield_pre_post(separated_pre_post, tindex, &
                                              tindex_pre, tindex_pos)
                    call read_pre_post_bfield(tp1, output_format, separated_pre_post)
                    call close_bfield_pre_post
                    call open_efield_pre_post(separated_pre_post, tindex, &
                                              tindex_pre, tindex_pos)
                    call read_pre_post_efield(tp1, output_format, separated_pre_post)
                    call close_efield_pre_post
                    call open_ufield_pre_post(species, separated_pre_post, &
                                              tindex, tindex_pre, tindex_pos)
                    call read_pre_post_u(tp1, output_format, separated_pre_post)
                    call close_ufield_pre_post
                    call open_vfield_pre_post(species, separated_pre_post, &
                                              tindex, tindex_pre, tindex_pos)
                    call read_pre_post_v(tp1, output_format, separated_pre_post)
                    call close_vfield_pre_post
                else
                    ! Fields at all time steps are saved in the same file
                    call read_electric_fields(tp_emf)
                    call read_magnetic_fields(tp_emf)
                    call read_velocity_fields(tp_emf)
                    call read_pre_post_bfield(tp_emf, output_format, separated_pre_post)
                    call read_pre_post_efield(tp_emf, output_format, separated_pre_post)
                    call read_pre_post_u(tp_emf, output_format, separated_pre_post)
                    call read_pre_post_v(tp_emf, output_format, separated_pre_post)
                endif
            endif
            call cpu_time(time2)
            if (myid == master) then
                print '("Time for reading fields = ",f8.3," seconds.")', &
                    time2 - time1
            endif

            ! Interpolate EMF to node position
            call interp_emf_node_ghost
            call interp_bfield_node_ghost
            call interp_efield_node_ghost
            call calc_exb_drift
            call calc_exb_derivatives(htg%nx, htg%ny, htg%nz)
            call calc_bfield_derivatives(htg%nx, htg%ny, htg%nz)
            call calc_ufield_derivatives(htg%nx, htg%ny, htg%nz)
            call calc_vperp_components
            call calc_vperp_derivatives(htg%nx, htg%ny, htg%nz)
            call calc_absb_derivatives(htg%nx, htg%ny, htg%nz)

            dx_domain = domain%lx_de / domain%pic_tx
            dy_domain = domain%ly_de / domain%pic_ty
            dz_domain = domain%lz_de / domain%pic_tz

            ! Particles are saved in HDF5
            call cpu_time(time1)
            if (particle_hdf5) then
                call get_np_local_vpic(tframe, species)
                call open_particle_file_h5(tframe, species)
            endif
            call cpu_time(time2)
            if (myid == master) then
                print '("Time for openning HDF5 = ",f8.3," seconds.")', &
                    time2 - time1
            endif

            call cpu_time(time1)
            do dom_z = ht%start_z, ht%stop_z
                do dom_y = ht%start_y, ht%stop_y
                    do dom_x = ht%start_x, ht%stop_x
                        call calc_particle_energization_single(tframe, &
                            dom_x, dom_y, dom_z, dx_domain, dy_domain, &
                            dz_domain, dt_fields)
                    enddo ! x
                enddo ! y
            enddo ! z
            call cpu_time(time2)
            if (myid == master) then
                print '("Time for computing = ",f8.3," seconds.")', &
                    time2 - time1
            endif

            ! Particles are saved in HDF5
            call cpu_time(time1)
            if (particle_hdf5) then
                call free_np_offset_local
                call close_particle_file_h5
            endif
            call cpu_time(time2)
            if (myid == master) then
                print '("Time for closing HDF5 = ",f8.3," seconds.")', &
                    time2 - time1
            endif

            call cpu_time(time1)
            call save_particle_energization(tframe, "particle_energization")
            call cpu_time(time2)
            if (myid == master) then
                print '("Time saveing data = ",f8.3," seconds.")', &
                    time2 - time1
            endif
            call cpu_time(step2)
            if (myid == master) then
                print '("Time for this step = ",f8.3," seconds.")', step2 - step1
            endif
            step1 = step2
        enddo  ! Time loop

        if (is_translated_file .and. output_format == 1) then
            call close_electric_field_files
            call close_magnetic_field_files
            call close_velocity_field_files
            call close_bfield_pre_post
            call close_efield_pre_post
            call close_ufield_pre_post
            call close_vfield_pre_post
        endif

        deallocate(ebins, fbins, fbins_sum)

        if (is_translated_file) then
            call free_electric_fields
            call free_magnetic_fields
            call free_velocity_fields
            call free_pre_post_bfield
            call free_pre_post_efield
            call free_pre_post_u
            call free_pre_post_v
            call free_exb_drift
            call free_bfield_derivatives
            call free_exb_derivatives
            call free_ufield_derivatives
            call free_vperp_components
            call free_vperp_derivatives
            call free_absb_derivatives
        endif
        call free_emfields
        call free_emfields_derivatives
        call free_vel_mom
        call free_bfield_components
        call free_bfield_magnitude
        call free_efield_components
        call free_ufield_components
        call free_ufields_derivatives_single
        call free_exb_derivatives_single
        call free_vfield_components
        call free_vperp_derivatives_single
        call free_absb_derivatives_single
    end subroutine calc_particle_energization

    !<--------------------------------------------------------------------------
    !< Calculate particle energization for particles in a single PIC MPI rank
    !<--------------------------------------------------------------------------
    subroutine calc_particle_energization_single(tindex, dom_x, dom_y, dom_z, &
            dx_domain, dy_domain, dz_domain, dt_fields)
        use picinfo, only: domain
        use topology_translate, only: ht
        use rank_index_mapping, only: index_to_rank
        use interpolation_emf, only: trilinear_interp_bx, trilinear_interp_by, &
            trilinear_interp_bz, trilinear_interp_ex, trilinear_interp_ey, &
            trilinear_interp_ez, trilinear_interp_absb_derivatives, &
            set_emf, set_emf_derivatives, set_absb_derivatives, &
            bx0, by0, bz0, ex0, ey0, ez0, dbxdx0, dbxdy0, dbxdz0, &
            dbydx0, dbydy0, dbydz0, dbzdx0, dbzdy0, dbzdz0, &
            dbdx0, dbdy0, dbdz0
        use interpolation_vel_mom, only: trilinear_interp_vel_mom, &
            set_vel_mom, vx0, vy0, vz0, ux0, uy0, uz0
        use interpolation_pre_post_bfield, only: set_bfield_components, &
            trilinear_interp_bfield_components, bx1_0, by1_0, bz1_0, &
            bx2_0, by2_0, bz2_0
        use interpolation_pre_post_efield, only: set_efield_components, &
            trilinear_interp_efield_components, ex1_0, ey1_0, ez1_0, &
            ex2_0, ey2_0, ez2_0
        use interpolation_pre_post_ufield, only: set_ufield_components, &
            trilinear_interp_ufield_components, ux1_0, uy1_0, uz1_0, &
            ux2_0, uy2_0, uz2_0
        use interpolation_vexb, only: set_exb_derivatives, &
            trilinear_interp_exb_derivatives, dvxdx0, dvxdy0, dvxdz0, &
            dvydx0, dvydy0, dvydz0, dvzdx0, dvzdy0, dvzdz0
        use interpolation_ufield_derivatives, only: set_ufield_derivatives, &
            trilinear_interp_ufield_derivatives, duxdx0, duxdy0, duxdz0, &
            duydx0, duydy0, duydz0, duzdx0, duzdy0, duzdz0
        use interpolation_pre_post_vfield, only: set_vfield_components, &
            trilinear_interp_vfield_components, vx1_0, vy1_0, vz1_0, &
            vx2_0, vy2_0, vz2_0
        use interpolation_vperp_derivatives, only: set_vperp_derivatives, &
            trilinear_interp_vperp_derivatives, dvperpx_dx0, dvperpx_dy0, &
            dvperpx_dz0, dvperpy_dx0, dvperpy_dy0, dvperpy_dz0, &
            dvperpz_dx0, dvperpz_dy0, dvperpz_dz0
        use file_header, only: pheader
        implicit none
        integer, intent(in) :: tindex, dom_x, dom_y, dom_z
        real(fp), intent(in) :: dx_domain, dy_domain, dz_domain, dt_fields
        type(particle) :: ptl
        integer :: ino, jno, kno   ! w.r.t the node
        real(fp) :: dnx, dny, dnz
        integer :: nxg, nyg, nzg, icell
        integer :: ibin, n, iptl, nptl
        real(fp) :: gama, igama, iene, ux, uy, uz, vx, vy, vz
        real(fp) :: vpara, vperp, vparax, vparay, vparaz
        real(fp) :: vperpx, vperpy, vperpz, tmp
        real(fp) :: bxn, byn, bzn, absB0, ib2, ib, edotb
        real(fp) :: bxx, byy, bzz, bxy, bxz, byz
        real(fp) :: dke_para, dke_perp, weight
        real(fp) :: pxx, pxy, pxz, pyx, pyy, pyz, pzx, pzy, pzz
        real(fp) :: pscalar, ppara, pperp, bbsigma, divv0, divv0_3
        real(fp) :: pdivv, pshear, ptensor_ene
        real(fp) :: curv_ene, grad_ene, parad_ene
        real(fp) :: vcx, vcy, vcz, kappax, kappay, kappaz
        real(fp) :: mag_moment, vgx, vgy, vgz, dBdx, dBdy, dBdz
        real(fp) :: vdotB, vperpx0, vperpy0, vperpz0
        real(fp) :: udotB, uperpx0, uperpy0, uperpz0
        real(fp) :: vexb_x, vexb_y, vexb_z
        real(fp) :: vpara_dx, vpara_dy, vpara_dz, vpara_d  ! Parallel drift
        real(fp) :: idt, dene_m
        real(fp) :: vexbx1, vexby1, vexbz1
        real(fp) :: vexbx2, vexby2, vexbz2
        real(fp) :: vpx, vpy, vpz, polar_ene_time
        real(fp) :: dvxdt, dvydt, dvzdt
        real(fp) :: absB1_0, absB2_0, ib2_1, ib2_2, ib_1, ib_2
        real(fp) :: dbxdt, dbydt, dbzdt, init_ene_time
        real(fp) :: vpx0, vpy0, vpz0
        real(fp) :: polar_ene_spatial, init_ene_spatial
        real(fp) :: polar_fluid_time, polar_fluid_spatial
        real(fp) :: vperpx1, vperpy1, vperpz1
        real(fp) :: vperpx2, vperpy2, vperpz2
        real(fp) :: polar_ene_time_v, polar_ene_spatial_v
        real(fp) :: param
        character(len=16) :: cid

        call index_to_rank(dom_x, dom_y, dom_z, domain%pic_tx, &
            domain%pic_ty, domain%pic_tz, n)
        write(cid, "(I0)") n - 1

        if (particle_hdf5) then
            if (parallel_read) then
                call read_particle_h5_parallel(n - 1)
            else
                call read_particle_h5(n - 1)
            endif
            nptl = np_local(n)
        else
            call read_particle_binary(tindex, species, cid)
            nptl = pheader%dim
        endif

        if (is_translated_file) then
            call set_emf(dom_x, dom_y, dom_z, &
                domain%pic_tx, domain%pic_ty, &
                domain%pic_tz, ht%start_x, ht%start_y, ht%start_z)
            call set_vel_mom(dom_x, dom_y, dom_z, &
                domain%pic_tx, domain%pic_ty, &
                domain%pic_tz, ht%start_x, ht%start_y, ht%start_z)
            call set_emf_derivatives(dom_x, dom_y, dom_z, &
                domain%pic_tx, domain%pic_ty, domain%pic_tz, &
                ht%start_x, ht%start_y, ht%start_z)
            call set_bfield_components(dom_x, dom_y, dom_z, &
                domain%pic_tx, domain%pic_ty, &
                domain%pic_tz, ht%start_x, ht%start_y, ht%start_z)
            call set_efield_components(dom_x, dom_y, dom_z, &
                domain%pic_tx, domain%pic_ty, &
                domain%pic_tz, ht%start_x, ht%start_y, ht%start_z)
            call set_ufield_components(dom_x, dom_y, dom_z, &
                domain%pic_tx, domain%pic_ty, &
                domain%pic_tz, ht%start_x, ht%start_y, ht%start_z)
            call set_exb_derivatives(dom_x, dom_y, dom_z, &
                domain%pic_tx, domain%pic_ty, &
                domain%pic_tz, ht%start_x, ht%start_y, ht%start_z)
            call set_ufield_derivatives(dom_x, dom_y, dom_z, &
                domain%pic_tx, domain%pic_ty, &
                domain%pic_tz, ht%start_x, ht%start_y, ht%start_z)
            call set_vfield_components(dom_x, dom_y, dom_z, &
                domain%pic_tx, domain%pic_ty, &
                domain%pic_tz, ht%start_x, ht%start_y, ht%start_z)
            call set_vperp_derivatives(dom_x, dom_y, dom_z, &
                domain%pic_tx, domain%pic_ty, &
                domain%pic_tz, ht%start_x, ht%start_y, ht%start_z)
            call set_absb_derivatives(dom_x, dom_y, dom_z, &
                domain%pic_tx, domain%pic_ty, &
                domain%pic_tz, ht%start_x, ht%start_y, ht%start_z)
        endif

        nxg = domain%pic_nx + 2  ! Including ghost cells
        nyg = domain%pic_ny + 2
        nzg = domain%pic_nz + 2

        idt = 1.0 / dt_fields
        do iptl = 1, nptl, 1
            ptl = ptls(iptl)
            icell = ptl%icell
            kno = icell / (nxg*nyg)          ! [1,nzg-2]
            jno = mod(icell, nxg*nyg) / nxg  ! [1,nyg-2]
            ino = mod(icell, nxg)            ! [1,nxg-2]
            dnx = (1 + ptl%dx) * 0.5
            dny = (1 + ptl%dy) * 0.5
            dnz = (1 + ptl%dz) * 0.5
            call trilinear_interp_bx(ino, jno, kno, dnx, dny, dnz)
            call trilinear_interp_by(ino, jno, kno, dnx, dny, dnz)
            call trilinear_interp_bz(ino, jno, kno, dnx, dny, dnz)
            call trilinear_interp_ex(ino, jno, kno, dnx, dny, dnz)
            call trilinear_interp_ey(ino, jno, kno, dnx, dny, dnz)
            call trilinear_interp_ez(ino, jno, kno, dnx, dny, dnz)
            call trilinear_interp_vel_mom(ino, jno, kno, dnx, dny, dnz)
            call trilinear_interp_bfield_components(ino, jno, kno, dnx, dny, dnz)
            call trilinear_interp_efield_components(ino, jno, kno, dnx, dny, dnz)
            call trilinear_interp_ufield_components(ino, jno, kno, dnx, dny, dnz)
            call trilinear_interp_exb_derivatives(ino, jno, kno, dnx, dny, dnz)
            call trilinear_interp_ufield_derivatives(ino, jno, kno, dnx, dny, dnz)
            call trilinear_interp_vfield_components(ino, jno, kno, dnx, dny, dnz)
            call trilinear_interp_vperp_derivatives(ino, jno, kno, dnx, dny, dnz)
            call trilinear_interp_absb_derivatives(ino, jno, kno, dnx, dny, dnz)
            ux = ptl%vx  ! v in ptl is actually gamma*v
            uy = ptl%vy
            uz = ptl%vz
            gama = sqrt(1.0 + ux**2 + uy**2 + uz**2)
            igama = 1.0 / gama
            iene = 1.0 / (gama - 1.0)
            vx = ux * igama
            vy = uy * igama
            vz = uz * igama

            bxx = bx0**2
            byy = by0**2
            bzz = bz0**2
            bxy = bx0 * by0
            bxz = bx0 * bz0
            byz = by0 * bz0
            absB0 = sqrt(bxx + byy + bzz)
            ib = 1.0 / absB0
            ib2 = ib * ib
            bxn = bx0 * ib
            byn = by0 * ib
            bzn = bz0 * ib

            ! Particle parallel and perpendicular velocity
            vpara = vx * bxn + vy * byn + vz * bzn
            vparax = vpara * bxn
            vparay = vpara * byn
            vparaz = vpara * bzn
            tmp = vx**2 + vy**2 + vz**2 - vpara**2
            if (tmp < 0) then
                vperp = 0.0
            else
                vperp = sqrt(tmp)
            endif
            vperpx = vx - vparax
            vperpy = vy - vparay
            vperpz = vz - vparaz

            ! Local ExB drift velocity
            vexb_x = (ey0 * bz0 - ez0 * by0) * ib2
            vexb_y = (ez0 * bx0 - ex0 * bz0) * ib2
            vexb_z = (ex0 * by0 - ey0 * bx0) * ib2

            weight = abs(ptl%q)

            ! Energization due to parallel and perpendicular electric field
            edotb = ex0 * bxn + ey0 * byn + ez0 * bzn
            dke_para = (bxn * vx + byn * vy + bzn * vz) * &
                weight * ptl_charge * edotb
            dke_perp = (ex0 * vx + ey0 * vy + ez0 * vz) * &
                weight * ptl_charge - dke_para

            ! Energization due to compression and shear
            pxx = (vx - vx0) * (ux - ux0) * ptl_mass * weight
            pxy = (vx - vx0) * (uy - uy0) * ptl_mass * weight
            pxz = (vx - vx0) * (uz - uz0) * ptl_mass * weight
            pyx = (vy - vy0) * (ux - ux0) * ptl_mass * weight
            pyy = (vy - vy0) * (uy - uy0) * ptl_mass * weight
            pyz = (vy - vy0) * (uz - uz0) * ptl_mass * weight
            pzx = (vz - vz0) * (ux - ux0) * ptl_mass * weight
            pzy = (vz - vz0) * (uy - uy0) * ptl_mass * weight
            pzz = (vz - vz0) * (uz - uz0) * ptl_mass * weight
            pscalar = (pxx + pyy + pzz) / 3.0
            ppara = pxx * bxx + pyy * byy + pzz * bzz + &
                (pxy + pyx) * bxy + (pxz + pzx) * bxz + (pyz + pzy) * byz
            ppara = ppara * ib2
            pperp = 0.5 * (pscalar * 3 - ppara)
            divv0 = dvxdx0 + dvydy0 + dvzdz0
            divv0_3 = divv0 / 3.0
            bbsigma = (dvxdx0 - divv0_3) * bxx + &
                      (dvydy0 - divv0_3) * byy + &
                      (dvzdz0 - divv0_3) * bzz + &
                      (dvxdy0 + dvydx0) * bxy + &
                      (dvxdz0 + dvzdx0) * bxz + &
                      (dvydz0 + dvzdy0) * byz
            bbsigma = bbsigma * ib2
            pdivv = -pscalar * divv0
            pshear = (pperp - ppara) * bbsigma
            ptensor_ene = -(pxx * dvxdx0 + pxy * dvxdy0 + pxz * dvxdz0 + &
                            pyx * dvydx0 + pyy * dvydy0 + pyz * dvydz0 + &
                            pzx * dvzdx0 + pzy * dvzdy0 + pzz * dvzdz0)

            ! Energization due to gradient drift
            dBdx = bxn * dbxdx0 + byn * dbydx0 + bzn * dbzdx0
            dBdy = bxn * dbxdy0 + byn * dbydy0 + bzn * dbzdy0
            dBdz = bxn * dbxdz0 + byn * dbydz0 + bzn * dbzdz0
            ! mag_moment = ((vperpx - vexb_x)**2 + &
            !               (vperpy - vexb_y)**2 + &
            !               (vperpz - vexb_z)**2) * &
            !               gama * ptl_mass * ib * 0.5
            vdotB = vx0 * bxn + vy0 * byn + vz0 * bzn
            vperpx0 = vx0 - vdotB * bxn
            vperpy0 = vy0 - vdotB * byn
            vperpz0 = vz0 - vdotB * bzn
            udotB = ux0 * bxn + uy0 * byn + uz0 * bzn
            uperpx0 = ux0 - udotB * bxn
            uperpy0 = uy0 - udotB * byn
            uperpz0 = uz0 - udotB * bzn
            ! mag_moment = ((vperpx - vperpx0)**2 + &
            !               (vperpy - vperpy0)**2 + &
            !               (vperpz - vperpz0)**2) * &
            !               gama * ptl_mass * ib * 0.5
            mag_moment = ((vperpx - vperpx0) * (gama * vperpx - uperpx0) + &
                          (vperpy - vperpy0) * (gama * vperpy - uperpy0) + &
                          (vperpz - vperpz0) * (gama * vperpz - uperpz0)) * &
                          ptl_mass * ib * 0.5
            param = mag_moment * ib / ptl_charge
            vgx = (byn*dbdz0 - bzn*dbdy0) * param
            vgy = (bzn*dbdx0 - bxn*dbdz0) * param
            vgz = (bxn*dbdy0 - byn*dbdx0) * param
            grad_ene = weight * ptl_charge * (ex0 * vgx + ey0 * vgy + ez0 * vgz)

            ! Energization due to parallel drift
            vpara_d = ((dbzdy0 - dbydz0) * bxn + &
                       (dbxdz0 - dbzdx0) * byn + &
                       (dbydx0 - dbxdy0) * bzn) * ib
            ! parad_ene = weight * vpara_d * mag_moment * &
            !     (ex0 * bxn + ey0 * byn + ez0 * bzn)
            vpara_dx = vpara_d * mag_moment * bxn / ptl_charge
            vpara_dy = vpara_d * mag_moment * byn / ptl_charge
            vpara_dz = vpara_d * mag_moment * bzn / ptl_charge
            parad_ene = weight * ptl_charge * &
                (vpara_dx * ex0 + vpara_dy * ey0 + vpara_dz * ez0)

            ! Energization due to curvature drift. Another term is always 0.
            kappax = (bxn*dbxdx0 + byn*dbxdy0 + bzn*dbxdz0)*ib
            kappay = (bxn*dbydx0 + byn*dbydy0 + bzn*dbydz0)*ib
            kappaz = (bxn*dbzdx0 + byn*dbzdy0 + bzn*dbzdz0)*ib
            ! vcx = byn*kappaz - bzn*kappay
            ! vcy = bzn*kappax - bxn*kappaz
            ! vcz = bxn*kappay - byn*kappax
            ! curv_ene = weight * ptl_mass * gama * vpara**2 * ib * &
            !     (ex0 * vcx + ey0 * vcy + ez0 * vcz)
            param = ptl_mass * gama * vpara**2 * ib / ptl_charge
            vcx = (byn*kappaz - bzn*kappay) * param
            vcy = (bzn*kappax - bxn*kappaz) * param
            vcz = (bxn*kappay - byn*kappax) * param
            curv_ene = weight * ptl_charge * (ex0 * vcx + ey0 * vcy + ez0 * vcz)

            ! Energization due to the conservation of magnetic moment
            absB1_0 = sqrt(bx1_0**2 + by1_0**2 + bz1_0**2)
            absB2_0 = sqrt(bx2_0**2 + by2_0**2 + bz2_0**2)
            dene_m = mag_moment * (absB2_0 - absB1_0) * idt * weight

            ! Energization due to polarization drift (time varying field)
            ib_1 = 1.0 / absB1_0
            ib_2 = 1.0 / absB2_0
            ib2_1 = ib_1 * ib_1
            ib2_2 = ib_2 * ib_2

            vexbx1 = (ey1_0 * bz1_0 - ez1_0 * by1_0) * ib2_1
            vexby1 = (ez1_0 * bx1_0 - ex1_0 * bz1_0) * ib2_1
            vexbz1 = (ex1_0 * by1_0 - ey1_0 * bx1_0) * ib2_1

            vexbx2 = (ey2_0 * bz2_0 - ez2_0 * by2_0) * ib2_2
            vexby2 = (ez2_0 * bx2_0 - ex2_0 * bz2_0) * ib2_2
            vexbz2 = (ex2_0 * by2_0 - ey2_0 * bx2_0) * ib2_2

            dvxdt = (vexbx2 - vexbx1) * idt
            dvydt = (vexby2 - vexby1) * idt
            dvzdt = (vexbz2 - vexbz1) * idt
            vpx = by0 * dvzdt - bz0 * dvydt
            vpy = bz0 * dvxdt - bx0 * dvzdt
            vpz = bx0 * dvydt - by0 * dvxdt
            polar_ene_time = weight * gama * ptl_mass * ib2 * &
                (ex0 * vpx + ey0 * vpy + ez0 * vpz)

            ! Inertial drift (time varying field)
            dbxdt = (bx2_0 * ib_2 - bx1_0 * ib_1) * idt
            dbydt = (by2_0 * ib_2 - by1_0 * ib_1) * idt
            dbzdt = (bz2_0 * ib_2 - bz1_0 * ib_1) * idt
            init_ene_time = gama * ptl_mass * vpara * weight * &
                            (vexb_x * dbxdt + vexb_y * dbydt + vexb_z * dbzdt)

            ! Energization due to polarization drift (spatially varying field)
            vpx0 = vparax + vexb_x
            vpy0 = vparay + vexb_y
            vpz0 = vparaz + vexb_z

            dvxdt = vpx0 * dvxdx0 + vpy0 * dvxdy0 + vpz0 * dvxdz0
            dvydt = vpx0 * dvydx0 + vpy0 * dvydy0 + vpz0 * dvydz0
            dvzdt = vpx0 * dvzdx0 + vpy0 * dvzdy0 + vpz0 * dvzdz0

            polar_ene_spatial = weight * gama * ptl_mass * &
                (vexb_x * dvxdt + vexb_y * dvydt + vexb_z * dvzdt)

            ! Inertial drift (spatially varying field)
            dbxdt = (vexb_x*dbxdx0 + vexb_y*dbxdy0 + vexb_z*dbxdz0) * ib
            dbydt = (vexb_x*dbydx0 + vexb_y*dbydy0 + vexb_z*dbydz0) * ib
            dbzdt = (vexb_x*dbzdx0 + vexb_y*dbzdy0 + vexb_z*dbzdz0) * ib
            init_ene_spatial = gama * ptl_mass * vpara * weight * &
                (vexb_x * dbxdt + vexb_y * dbydt + vexb_z * dbzdt)

            ! Fluid polarization (time varying field)
            dvxdt = (ux2_0 - ux1_0) * idt
            dvydt = (uy2_0 - uy1_0) * idt
            dvzdt = (uz2_0 - uz1_0) * idt

            polar_fluid_time = weight * ptl_mass * &
                (vexb_x * dvxdt + vexb_y * dvydt + vexb_z * dvzdt)

            ! Fluid polarization (spatially varying field)
            ! vpx0 = vparax + vexb_x + vcx + vgx
            ! vpy0 = vparay + vexb_y + vcy + vgy
            ! vpz0 = vparaz + vexb_z + vcz + vgz
            vpx0 = vparax + vexb_x
            vpy0 = vparay + vexb_y
            vpz0 = vparaz + vexb_z
            dvxdt = vpx0 * duxdx0 + vpy0 * duxdy0 + vpz0 * duxdz0
            dvydt = vpx0 * duydx0 + vpy0 * duydy0 + vpz0 * duydz0
            dvzdt = vpx0 * duzdx0 + vpy0 * duzdy0 + vpz0 * duzdz0
            ! dvxdt = vx * duxdx0 + vy * duxdy0 + vz * duxdz0
            ! dvydt = vx * duydx0 + vy * duydy0 + vz * duydz0
            ! dvzdt = vx * duzdx0 + vy * duzdy0 + vz * duzdz0

            polar_fluid_spatial = weight * ptl_mass * &
                (vexb_x * dvxdt + vexb_y * dvydt + vexb_z * dvzdt)

            ! Energization due to polarization drift, using vperp instead of vexb
            vdotB = (vx1_0 * bx1_0 + vy1_0 * by1_0 + vz1_0 * bz1_0) * ib2_1
            vperpx1 = vx1_0 - vdotB * bx1_0
            vperpy1 = vy1_0 - vdotB * by1_0
            vperpz1 = vz1_0 - vdotB * bz1_0
            vdotB = (vx2_0 * bx2_0 + vy2_0 * by2_0 + vz2_0 * bz2_0) * ib2_2
            vperpx2 = vx2_0 - vdotB * bx2_0
            vperpy2 = vy2_0 - vdotB * by2_0
            vperpz2 = vz2_0 - vdotB * bz2_0

            dvxdt = (vperpx2 - vperpx1) * idt
            dvydt = (vperpy2 - vperpy1) * idt
            dvzdt = (vperpz2 - vperpz1) * idt
            polar_ene_time_v = weight * gama * ptl_mass * &
                (dvxdt * vexb_x + dvydt * vexb_y + dvzdt * vexb_z)

            vpx0 = vparax + vexb_x
            vpy0 = vparay + vexb_y
            vpz0 = vparaz + vexb_z

            dvxdt = vpx0 * dvperpx_dx0 + vpy0 * dvperpx_dy0 + vpz0 * dvperpx_dz0
            dvydt = vpx0 * dvperpy_dx0 + vpy0 * dvperpy_dy0 + vpz0 * dvperpy_dz0
            dvzdt = vpx0 * dvperpz_dx0 + vpy0 * dvperpz_dy0 + vpz0 * dvperpz_dz0
            polar_ene_spatial_v = weight * gama * ptl_mass * &
                (vexb_x * dvxdt + vexb_y * dvydt + vexb_z * dvzdt)

            ibin = floor((log10(gama - 1) - emin_log) / de_log)
            if (ibin > 0 .and. ibin < nbins + 1) then
                fbins(ibin+1, 1) = fbins(ibin+1, 1) + weight
                fbins(ibin+1, 2) = fbins(ibin+1, 2) + dke_para * iene
                fbins(ibin+1, 3) = fbins(ibin+1, 3) + dke_perp * iene
                fbins(ibin+1, 4) = fbins(ibin+1, 4) + pdivv * iene
                fbins(ibin+1, 5) = fbins(ibin+1, 5) + pshear * iene
                fbins(ibin+1, 6) = fbins(ibin+1, 6) + curv_ene * iene
                fbins(ibin+1, 7) = fbins(ibin+1, 7) + grad_ene * iene
                fbins(ibin+1, 8) = fbins(ibin+1, 8) + parad_ene * iene
                fbins(ibin+1, 9) = fbins(ibin+1, 9) + dene_m * iene
                fbins(ibin+1, 10) = fbins(ibin+1, 10) + polar_ene_time * iene
                fbins(ibin+1, 11) = fbins(ibin+1, 11) + polar_ene_spatial * iene
                fbins(ibin+1, 12) = fbins(ibin+1, 12) + init_ene_time * iene
                fbins(ibin+1, 13) = fbins(ibin+1, 13) + init_ene_spatial * iene
                fbins(ibin+1, 14) = fbins(ibin+1, 14) + polar_fluid_time * iene
                fbins(ibin+1, 15) = fbins(ibin+1, 15) + polar_fluid_spatial * iene
                fbins(ibin+1, 16) = fbins(ibin+1, 16) + polar_ene_time_v * iene
                fbins(ibin+1, 17) = fbins(ibin+1, 17) + polar_ene_spatial_v * iene
                fbins(ibin+1, 18) = fbins(ibin+1, 18) + ptensor_ene * iene
            endif
        enddo
        deallocate(ptls)
    end subroutine calc_particle_energization_single

    !<--------------------------------------------------------------------------
    !< Save particle energization due to parallel and perpendicular electric field.
    !<--------------------------------------------------------------------------
    subroutine save_particle_energization(tindex, var_name)
        implicit none
        integer, intent(in) :: tindex
        character(*), intent(in) :: var_name
        integer :: fh1, posf
        character(len=16) :: tindex_str
        character(len=256) :: fname
        logical :: dir_e
        call MPI_REDUCE(fbins, fbins_sum, (nbins+1)*nvar, &
                MPI_REAL, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
        if (myid == master) then
            inquire(file='./data/particle_interp/.', exist=dir_e)
            if (.not. dir_e) then
                call system('mkdir -p ./data/particle_interp/')
            endif
            print*, "Saving particle based analysis resutls..."

            fh1 = 66

            write(tindex_str, "(I0)") tindex
            fname = 'data/particle_interp/'//trim(var_name)//'_'//species
            fname = trim(fname)//"_"//trim(tindex_str)//'.gda'
            open(unit=fh1, file=fname, access='stream', status='unknown', &
                form='unformatted', action='write')
            posf = 1
            write(fh1, pos=posf) nbins + 1.0
            posf = posf + 4
            write(fh1, pos=posf) nvar + 0.0
            posf = posf + 4
            write(fh1, pos=posf) ebins
            posf = posf + (nbins + 1) * 4
            write(fh1, pos=posf) fbins_sum
            close(fh1)
        endif
    end subroutine save_particle_energization

    !<--------------------------------------------------------------------------
    !< Initialize the analysis.
    !<--------------------------------------------------------------------------
    subroutine init_analysis
        use mpi_topology, only: set_mpi_topology, htg
        use mpi_datatype_fields, only: set_mpi_datatype_fields
        use mpi_info_module, only: set_mpi_info
        use particle_info, only: get_ptl_mass_charge
        use path_info, only: get_file_paths
        use picinfo, only: read_domain, broadcast_pic_info, &
                get_total_time_frames, get_energy_band_number, &
                read_thermal_params, calc_energy_interval, nbands, &
                write_pic_info, domain
        use configuration_translate, only: read_configuration
        use topology_translate, only: set_topology, set_start_stop_cells
        use mpi_io_translate, only: set_mpi_io
        use parameters, only: get_relativistic_flag, get_start_end_time_points, tp2
        use neighbors_module, only: init_neighbors, get_neighbors
        implicit none
        integer :: nx, ny, nz

        call get_file_paths(rootpath)
        if (myid == master) then
            call read_domain
        endif
        call broadcast_pic_info
        call get_ptl_mass_charge(species)
        call get_start_end_time_points
        call get_relativistic_flag
        ! call get_energy_band_number
        call read_thermal_params
        if (nbands > 0) then
            call calc_energy_interval
        endif
        call read_configuration
        call get_total_time_frames(tp2)
        call set_topology
        call set_start_stop_cells
        call set_mpi_io

        call set_mpi_topology(1)   ! MPI topology
        call set_mpi_datatype_fields
        call set_mpi_info

        call init_neighbors(htg%nx, htg%ny, htg%nz)
        call get_neighbors

    end subroutine init_analysis

    !!--------------------------------------------------------------------------
    !! End the analysis by free the memory.
    !!--------------------------------------------------------------------------
    subroutine end_analysis
        use topology_translate, only: free_start_stop_cells
        use mpi_io_translate, only: datatype
        use mpi_info_module, only: fileinfo
        use neighbors_module, only: free_neighbors
        use mpi_datatype_fields, only: filetype_ghost, filetype_nghost
        implicit none
        call free_neighbors
        call free_start_stop_cells
        call MPI_TYPE_FREE(datatype, ierror)
        call MPI_INFO_FREE(fileinfo, ierror)
        call MPI_TYPE_FREE(filetype_ghost, ierror)
        call MPI_TYPE_FREE(filetype_nghost, ierror)
    end subroutine end_analysis

    !<--------------------------------------------------------------------------
    !< Read particle data in binary format
    !<--------------------------------------------------------------------------
    subroutine read_particle_binary(tindex, species, cid)
        use particle_file, only: open_particle_file, close_particle_file, fh
        use file_header, only: pheader
        implicit none
        integer, intent(in) :: tindex
        character(*), intent(in) :: species
        character(*), intent(in) :: cid
        integer :: IOstatus
        ! Read particle data
        if (species == 'e') then
            call open_particle_file(tindex, species, cid)
        else
            call open_particle_file(tindex, 'h', cid)
        endif
        allocate(ptls(pheader%dim))
        read(fh, IOSTAT=IOstatus) ptls
        call close_particle_file
    end subroutine read_particle_binary

    !<--------------------------------------------------------------------------
    !< Get commandline arguments
    !<--------------------------------------------------------------------------
    subroutine get_cmd_args
        use flap                                !< FLAP package
        use penf
        implicit none
        type(command_line_interface) :: cli     !< Command Line Interface (CLI).
        integer(I4P)                 :: error   !< Error trapping flag.
        call cli%init(progname = 'interpolation', &
            authors     = 'Xiaocan Li', &
            help        = 'Usage: ', &
            description = 'Interpolate fields at particle positions', &
            examples    = ['interpolation -rp rootpath'])
        call cli%add(switch='--rootpath', switch_ab='-rp', &
            help='simulation root path', required=.true., act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--translated_file', switch_ab='-tf', &
            help='whether using translated fields file', required=.false., &
            act='store_true', def='.false.', error=error)
        if (error/=0) stop
        call cli%add(switch='--tstart', switch_ab='-ts', &
            help='Starting time frame', required=.false., act='store', &
            def='0', error=error)
        if (error/=0) stop
        call cli%add(switch='--tend', switch_ab='-te', help='Last time frame', &
            required=.true., act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--tinterval', switch_ab='-ti', help='Time interval', &
            required=.true., act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--fields_interval', switch_ab='-fi', &
            help='Time interval for PIC fields', &
            required=.true., act='store', error=error)
        if (error/=0) stop
        call cli%add(switch='--species', switch_ab='-sp', &
            help="Particle species: 'e' or 'h'", required=.false., &
            act='store', def='e', error=error)
        if (error/=0) stop
        call cli%add(switch='--dir_emf', switch_ab='-de', &
            help='EMF data directory', required=.false., &
            act='store', def='data', error=error)
        if (error/=0) stop
        call cli%add(switch='--dir_hydro', switch_ab='-dh', &
            help='Hydro data directory', required=.false., &
            act='store', def='data', error=error)
        if (error/=0) stop
        call cli%add(switch='--separated_pre_post', switch_ab='-pp', &
            help='separated pre and post fields', required=.false., act='store', &
            def='1', error=error)
        if (error/=0) stop
        call cli%add(switch='--particle_hdf5', switch_ab='-ph', &
            help='Whether particles are saved in HDF5', &
            required=.false., act='store_true', def='.false.', error=error)
        if (error/=0) stop
        call cli%add(switch='--parallel_read', switch_ab='-pr', &
            help='Whether to read HDF5 partile file in parallel', &
            required=.false., act='store_true', def='.false.', error=error)
        if (error/=0) stop
        call cli%add(switch='--collective_io', switch_ab='-ci', &
            help='Whether to use collective IO to read HDF5 partile file', &
            required=.false., act='store_true', def='.false.', error=error)
        if (error/=0) stop
        call cli%add(switch='--fd_tinterval', switch_ab='-ft', &
            help='Frame interval when dumping 3 continuous frames', &
            required=.false., def='1', act='store', error=error)
        if (error/=0) stop
        call cli%get(switch='-rp', val=rootpath, error=error)
        if (error/=0) stop
        call cli%get(switch='-tf', val=is_translated_file, error=error)
        if (error/=0) stop
        call cli%get(switch='-ts', val=tstart, error=error)
        if (error/=0) stop
        call cli%get(switch='-te', val=tend, error=error)
        if (error/=0) stop
        call cli%get(switch='-ti', val=tinterval, error=error)
        if (error/=0) stop
        call cli%get(switch='-fi', val=fields_interval, error=error)
        if (error/=0) stop
        call cli%get(switch='-sp', val=species, error=error)
        if (error/=0) stop
        call cli%get(switch='-de', val=dir_emf, error=error)
        if (error/=0) stop
        call cli%get(switch='-dh', val=dir_hydro, error=error)
        if (error/=0) stop
        call cli%get(switch='-pp', val=separated_pre_post, error=error)
        if (error/=0) stop
        call cli%get(switch='-ph', val=particle_hdf5, error=error)
        if (error/=0) stop
        call cli%get(switch='-pr', val=parallel_read, error=error)
        if (error/=0) stop
        call cli%get(switch='-ci', val=collective_io, error=error)
        if (error/=0) stop
        call cli%get(switch='-ft', val=fd_tinterval, error=error)
        if (error/=0) stop

        if (myid == 0) then
            print '(A,A)', ' The simulation rootpath: ', trim(adjustl(rootpath))
            print '(A,L1)', ' Whether using translated fields file: ', is_translated_file
            print '(A,I0,A,I0,A,I0)', ' Min, max and interval: ', &
                tstart, ' ', tend, ' ', tinterval
            print '(A,I0)', ' Time interval for electric and magnetic fields: ', &
                fields_interval
            if (species == 'e') then
                print '(A,A)', ' Particle: electron'
            else if (species == 'h' .or. species == 'i') then
                print '(A,A)', ' Particle: ion'
            endif
            print '(A,A)', ' EMF data directory: ', trim(dir_emf)
            print '(A,A)', ' Hydro data directory: ', trim(dir_hydro)
            if (separated_pre_post) then
                print '(A)', ' Fields at previous and next time steps are saved separately'
                print '(A, I0)', ' Frame interval between previous and current step is: ', &
                    fd_tinterval
            endif
            if (particle_hdf5) then
                print '(A)', ' Particles are saved in HDF5 format'
                if (parallel_read) then
                    print '(A)', ' Read HDF5 particle file in parallel'
                    if (collective_io) then
                        print '(A)', ' Using colletive IO to read HDF5 particle file'
                    endif
                endif
            endif
        endif
    end subroutine get_cmd_args

    !<--------------------------------------------------------------------------
    !< Initialize the np_local and offset_local array
    !<--------------------------------------------------------------------------
    subroutine init_np_offset_local(dset_dims)
        implicit none
        integer(hsize_t), dimension(1), intent(in) :: dset_dims
        allocate(np_local(dset_dims(1)))
        allocate(offset_local(dset_dims(1)))
        np_local = 0
        offset_local = 0
    end subroutine init_np_offset_local

    !<--------------------------------------------------------------------------
    !< Free the np_local and offset_local array
    !<--------------------------------------------------------------------------
    subroutine free_np_offset_local
        implicit none
        deallocate(np_local)
        deallocate(offset_local)
    end subroutine free_np_offset_local

    !<--------------------------------------------------------------------------
    !< Open metadata file and dataset of "np_local"
    !<--------------------------------------------------------------------------
    subroutine open_metadata_dset(fname_metadata, groupname, file_id, &
            group_id, dataset_id, dset_dims, dset_dims_max, filespace)
        implicit none
        character(*), intent(in) :: fname_metadata, groupname
        integer(hid_t), intent(out) :: file_id, group_id, dataset_id
        integer(hsize_t), dimension(1), intent(out) :: dset_dims, dset_dims_max
        integer(hid_t), intent(out) :: filespace
        call open_hdf5_serial(fname_metadata, groupname, file_id, group_id)
        call open_hdf5_dataset("np_local", group_id, dataset_id, &
            dset_dims, dset_dims_max, filespace)
    end subroutine open_metadata_dset

    !<--------------------------------------------------------------------------
    !< Close dataset, filespace, group and file of metadata
    !<--------------------------------------------------------------------------
    subroutine close_metadata_dset(file_id, group_id, dataset_id, filespace)
        implicit none
        integer(hid_t), intent(in) :: file_id, group_id, dataset_id, filespace
        integer :: error
        call h5sclose_f(filespace, error)
        call h5dclose_f(dataset_id, error)
        call h5gclose_f(group_id, error)
        call h5fclose_f(file_id, error)
    end subroutine close_metadata_dset

    !<--------------------------------------------------------------------------
    !< Get the number of particles for each MPI process of PIC simulations
    !<--------------------------------------------------------------------------
    subroutine get_np_local_vpic(tframe, species)
        implicit none
        integer, intent(in) :: tframe
        character(*), intent(in) :: species
        character(len=256) :: fname_meta
        character(len=16) :: groupname
        integer(hid_t) :: file_id, group_id, dataset_id
        integer(hsize_t), dimension(1) :: dset_dims, dset_dims_max
        integer(hid_t) :: filespace
        integer :: i, error
        character(len=8) :: tframe_char
        write(tframe_char, "(I0)") tframe
        fname_meta = trim(adjustl(rootpath))//"/particle/T."//trim(tframe_char)
        if (species == 'e') then
            fname_meta = trim(fname_meta)//"/grid_metadata_electron_"
        else if (species == 'H' .or. species == 'h' .or. species == 'i') then
            fname_meta = trim(fname_meta)//"/grid_metadata_ion_"
        endif
        fname_meta = trim(fname_meta)//trim(tframe_char)//".h5part"
        groupname = "Step#"//trim(tframe_char)
        if (myid == master) then
            call open_metadata_dset(fname_meta, groupname, file_id, &
                group_id, dataset_id, dset_dims, dset_dims_max, filespace)
        endif
        call MPI_BCAST(dset_dims, 1, MPI_INTEGER, master, MPI_COMM_WORLD, &
            ierror)

        call init_np_offset_local(dset_dims)

        if (myid == master) then
            call h5dread_f(dataset_id, H5T_NATIVE_INTEGER, np_local, &
                dset_dims, error)
        endif
        call MPI_BCAST(np_local, dset_dims(1), MPI_INTEGER, master, &
            MPI_COMM_WORLD, ierror)
        offset_local = 0
        do i = 2, dset_dims(1)
            offset_local(i) = offset_local(i-1) + np_local(i-1)
        enddo
        if (myid == master) then
            call h5sclose_f(filespace, error)
            call h5dclose_f(dataset_id, error)
            call h5gclose_f(group_id, error)
            call h5fclose_f(file_id, error)
        endif
    end subroutine get_np_local_vpic

    !<--------------------------------------------------------------------------
    !< Open hdf5 file using one process
    !<--------------------------------------------------------------------------
    subroutine open_hdf5_serial(filename, groupname, file_id, group_id)
        implicit none
        character(*), intent(in) :: filename, groupname
        integer(hid_t), intent(out) :: file_id, group_id
        integer(size_t) :: obj_count_g, obj_count_d
        integer :: error
        call h5open_f(error)
        call h5fopen_f(filename, H5F_ACC_RDWR_F, file_id, error, &
            access_prp=h5p_default_f)
        call h5gopen_f(file_id, groupname, group_id, error)
    end subroutine open_hdf5_serial

    !<--------------------------------------------------------------------------
    !< Open hdf5 file in parallel
    !<--------------------------------------------------------------------------
    subroutine open_hdf5_parallel(filename, groupname, file_id, group_id)
        implicit none
        character(*), intent(in) :: filename, groupname
        integer(hid_t), intent(out) :: file_id, group_id
        integer(hid_t) :: plist_id
        integer :: storage_type, max_corder
        integer(size_t) :: obj_count_g, obj_count_d
        integer :: fileinfo, error
        call MPI_INFO_CREATE(fileinfo, ierror)
        call h5open_f(error)
        call h5pcreate_f(H5P_FILE_ACCESS_F, plist_id, error)
        if (collective_io) then
            call MPI_INFO_SET(fileinfo, "romio_cb_read", "automatic", ierror)
            call MPI_INFO_SET(fileinfo, "romio_ds_read", "automatic", ierror)
            ! call MPI_INFO_SET(fileinfo, "romio_cb_read", "enable", ierror)
            ! call MPI_INFO_SET(fileinfo, "romio_ds_read", "disable", ierror)
            call MPI_INFO_SET(fileinfo, "striping_factor", "4", ierror)
            call MPI_INFO_SET(fileinfo, "striping_unit", "1048576", ierror)
            ! call MPI_INFO_SET(fileinfo, "cb_buffer_size", "8388608", ierror)
            call MPI_INFO_SET(fileinfo, "cb_nodes", "4", ierror)
        else
            call MPI_INFO_SET(fileinfo, "romio_ds_read", "automatic", ierror)
        endif
        call h5pset_fapl_mpio_f(plist_id, MPI_COMM_WORLD, fileinfo, error)
        call MPI_INFO_FREE(fileinfo, ierror)
        call h5fopen_f(filename, H5F_ACC_RDWR_F, file_id, error, &
            access_prp=plist_id)
        call h5pclose_f(plist_id, error)
        call h5gopen_f(file_id, groupname, group_id, error)
    end subroutine open_hdf5_parallel

    !<--------------------------------------------------------------------------
    !< Open hdf5 dataset and get the dataset dimensions
    !<--------------------------------------------------------------------------
    subroutine open_hdf5_dataset(dataset_name, group_id, dataset_id, &
            dset_dims, dset_dims_max, filespace)
        implicit none
        character(*), intent(in) :: dataset_name
        integer(hid_t), intent(in) :: group_id
        integer(hid_t), intent(out) :: dataset_id, filespace
        integer(hsize_t), dimension(1), intent(out) :: dset_dims, &
            dset_dims_max
        integer(hid_t) :: datatype_id
        integer :: error
        call h5dopen_f(group_id, dataset_name, dataset_id, error)
        call h5dget_type_f(dataset_id, datatype_id, error)
        call h5dget_space_f(dataset_id, filespace, error)
        call h5Sget_simple_extent_dims_f(filespace, dset_dims, &
            dset_dims_max, error)
    end subroutine open_hdf5_dataset

    !<--------------------------------------------------------------------------
    !< Open particle file, group, and datasets in HDF5 format
    !<--------------------------------------------------------------------------
    subroutine open_particle_file_h5(tframe, species)
        implicit none
        integer, intent(in) :: tframe
        character(*), intent(in) :: species
        character(len=256) :: fname
        character(len=16) :: groupname
        character(len=8) :: tframe_char
        write(tframe_char, "(I0)") tframe
        fname = trim(adjustl(rootpath))//"/particle/T."//trim(tframe_char)
        if (species == 'e') then
            fname = trim(fname)//"/electron_"
        else if (species == 'H' .or. species == 'h' .or. species == 'i') then
            fname = trim(fname)//"/ion_"
        endif
        fname = trim(fname)//trim(tframe_char)//".h5part"
        groupname = "Step#"//trim(tframe_char)

        if (parallel_read) then
            call open_hdf5_parallel(fname, groupname, file_id, group_id)
        else
            call open_hdf5_serial(fname, groupname, file_id, group_id)
        endif
        call open_hdf5_dataset("Ux", group_id, dset_ids(1), &
            dset_dims, dset_dims_max, filespace)
        call open_hdf5_dataset("Uy", group_id, dset_ids(2), &
            dset_dims, dset_dims_max, filespace)
        call open_hdf5_dataset("Uz", group_id, dset_ids(3), &
            dset_dims, dset_dims_max, filespace)
        call open_hdf5_dataset("dX", group_id, dset_ids(4), &
            dset_dims, dset_dims_max, filespace)
        call open_hdf5_dataset("dY", group_id, dset_ids(5), &
            dset_dims, dset_dims_max, filespace)
        call open_hdf5_dataset("dZ", group_id, dset_ids(6), &
            dset_dims, dset_dims_max, filespace)
        call open_hdf5_dataset("i", group_id, dset_ids(7), &
            dset_dims, dset_dims_max, filespace)
        call open_hdf5_dataset("q", group_id, dset_ids(8), &
            dset_dims, dset_dims_max, filespace)
    end subroutine open_particle_file_h5

    !<--------------------------------------------------------------------------
    !< Close particle file, group, and datasets in HDF5 format
    !<--------------------------------------------------------------------------
    subroutine close_particle_file_h5
        implicit none
        integer :: i, error
        call h5sclose_f(filespace, error)
        do i = 1, num_dset
            call h5dclose_f(dset_ids(i), error)
        enddo
        call h5gclose_f(group_id, error)
        call h5fclose_f(file_id, error)
    end subroutine close_particle_file_h5

    !<--------------------------------------------------------------------------
    !< Initial setup for reading hdf5 file
    !<--------------------------------------------------------------------------
    subroutine init_read_hdf5(dset_id, dcount, doffset, dset_dims, &
            filespace, memspace)
        implicit none
        integer(hid_t), intent(in) :: dset_id
        integer(hsize_t), dimension(1), intent(in) :: dcount, doffset, dset_dims
        integer(hid_t), intent(out) :: filespace, memspace
        integer :: error
        call h5screate_simple_f(1, dcount, memspace, error)
        call h5dget_space_f(dset_id, filespace, error)
        call h5sselect_hyperslab_f(filespace, H5S_SELECT_SET_F, doffset, &
            dcount, error)
    end subroutine init_read_hdf5

    !<--------------------------------------------------------------------------
    !< Finalize reading hdf5 file
    !<--------------------------------------------------------------------------
    subroutine final_read_hdf5(filespace, memspace)
        implicit none
        integer(hid_t), intent(in) :: filespace, memspace
        integer :: error
        call h5sclose_f(filespace, error)
        call h5sclose_f(memspace, error)
    end subroutine final_read_hdf5

    !---------------------------------------------------------------------------
    ! Read hdf5 dataset for integer data
    !---------------------------------------------------------------------------
    subroutine read_hdf5_integer(dset_id, dcount, doffset, dset_dims, fdata)
        implicit none
        integer(hid_t), intent(in) :: dset_id
        integer(hsize_t), dimension(1), intent(in) :: dcount, doffset, dset_dims
        integer, dimension(*), intent(out) :: fdata
        integer(hid_t) :: filespace, memspace
        integer :: error
        call init_read_hdf5(dset_id, dcount, doffset, dset_dims, filespace, memspace)
        call h5dread_f(dset_id, H5T_NATIVE_INTEGER, fdata, dset_dims, error, &
            file_space_id=filespace, mem_space_id=memspace)
        call final_read_hdf5(filespace, memspace)
    end subroutine read_hdf5_integer

    !---------------------------------------------------------------------------
    ! Read hdf5 dataset for real data
    !---------------------------------------------------------------------------
    subroutine read_hdf5_real(dset_id, dcount, doffset, dset_dims, fdata)
        implicit none
        integer(hid_t), intent(in) :: dset_id
        integer(hsize_t), dimension(1), intent(in) :: dcount, doffset, dset_dims
        real(fp), dimension(*), intent(out) :: fdata
        integer(hid_t) :: filespace, memspace
        integer :: error
        call init_read_hdf5(dset_id, dcount, doffset, dset_dims, filespace, memspace)
        call h5dread_f(dset_id, H5T_NATIVE_REAL, fdata, dset_dims, error, &
            file_space_id=filespace, mem_space_id=memspace)
        call final_read_hdf5(filespace, memspace)
    end subroutine read_hdf5_real

    !<--------------------------------------------------------------------------
    !< Read particle data in HDF5 format
    !<--------------------------------------------------------------------------
    subroutine read_particle_h5(pic_mpi_rank)
        implicit none
        integer, intent(in) :: pic_mpi_rank
        integer(hsize_t), dimension(1) :: dcount, doffset
        allocate(ptls(np_local(pic_mpi_rank + 1)))
        dcount(1) = np_local(pic_mpi_rank + 1)
        doffset(1) = offset_local(pic_mpi_rank + 1)
        call read_hdf5_real(dset_ids(1), dcount, doffset, dset_dims, ptls%vx)
        call read_hdf5_real(dset_ids(2), dcount, doffset, dset_dims, ptls%vy)
        call read_hdf5_real(dset_ids(3), dcount, doffset, dset_dims, ptls%vz)
        call read_hdf5_real(dset_ids(4), dcount, doffset, dset_dims, ptls%dx)
        call read_hdf5_real(dset_ids(5), dcount, doffset, dset_dims, ptls%dy)
        call read_hdf5_real(dset_ids(6), dcount, doffset, dset_dims, ptls%dz)
        call read_hdf5_integer(dset_ids(7), dcount, doffset, dset_dims, ptls%icell)
        call read_hdf5_real(dset_ids(8), dcount, doffset, dset_dims, ptls%q)
    end subroutine read_particle_h5

    !<--------------------------------------------------------------------------
    !< Initial setup for reading hdf5 file in parallel
    !<--------------------------------------------------------------------------
    subroutine init_read_hdf5_parallel(dset_id, dcount, doffset, dset_dims, &
            filespace, memspace, plist_id)
        implicit none
        integer(hid_t), intent(in) :: dset_id
        integer(hsize_t), dimension(1), intent(in) :: dcount, doffset, dset_dims
        integer(hid_t), intent(out) :: filespace, memspace, plist_id
        integer :: error
        ! Create property list for collective dataset write
        call h5pcreate_f(H5P_DATASET_XFER_F, plist_id, error)
        if (collective_io) then
            call h5pset_dxpl_mpio_f(plist_id, H5FD_MPIO_COLLECTIVE_F, error)
        else
            call h5pset_dxpl_mpio_f(plist_id, H5FD_MPIO_INDEPENDENT_F, error)
        endif

        call h5screate_simple_f(1, dcount, memspace, error)
        call h5dget_space_f(dset_id, filespace, error)
        call h5sselect_hyperslab_f(filespace, H5S_SELECT_SET_F, doffset, &
            dcount, error)
    end subroutine init_read_hdf5_parallel

    !<--------------------------------------------------------------------------
    !< Finalize reading hdf5 file in parallel
    !<--------------------------------------------------------------------------
    subroutine final_read_hdf5_parallel(filespace, memspace, plist_id)
        implicit none
        integer(hid_t), intent(in) :: filespace, memspace, plist_id
        integer :: error
        call h5sclose_f(filespace, error)
        call h5sclose_f(memspace, error)
        call h5pclose_f(plist_id, error)
    end subroutine final_read_hdf5_parallel

    !---------------------------------------------------------------------------
    ! Read hdf5 dataset for integer data in parallel
    !---------------------------------------------------------------------------
    subroutine read_hdf5_integer_parallel(dset_id, dcount, doffset, dset_dims, fdata)
        implicit none
        integer(hid_t), intent(in) :: dset_id
        integer(hsize_t), dimension(1), intent(in) :: dcount, doffset, dset_dims
        integer, dimension(*), intent(out) :: fdata
        integer(hid_t) :: filespace, memspace, plist_id
        integer :: error, actual_io_mode
        call init_read_hdf5_parallel(dset_id, dcount, doffset, dset_dims, &
            filespace, memspace, plist_id)
        call h5dread_f(dset_id, H5T_NATIVE_INTEGER, fdata, dset_dims, error, &
            file_space_id=filespace, mem_space_id=memspace, xfer_prp=plist_id)
        call final_read_hdf5_parallel(filespace, memspace, plist_id)
    end subroutine read_hdf5_integer_parallel

    !---------------------------------------------------------------------------
    ! Read hdf5 dataset for real data in parallel
    !---------------------------------------------------------------------------
    subroutine read_hdf5_real_parallel(dset_id, dcount, doffset, dset_dims, fdata)
        implicit none
        integer(hid_t), intent(in) :: dset_id
        integer(hsize_t), dimension(1), intent(in) :: dcount, doffset, dset_dims
        real(fp), dimension(*), intent(out) :: fdata
        integer(hid_t) :: filespace, memspace, plist_id
        integer :: error
        call init_read_hdf5_parallel(dset_id, dcount, doffset, dset_dims, &
            filespace, memspace, plist_id)
        call h5dread_f(dset_id, H5T_NATIVE_REAL, fdata, dset_dims, error, &
            file_space_id=filespace, mem_space_id=memspace, xfer_prp=plist_id)
        call final_read_hdf5_parallel(filespace, memspace, plist_id)
    end subroutine read_hdf5_real_parallel

    !<--------------------------------------------------------------------------
    !< Read particle data in HDF5 format in parallel
    !<--------------------------------------------------------------------------
    subroutine read_particle_h5_parallel(pic_mpi_rank)
        implicit none
        integer, intent(in) :: pic_mpi_rank
        integer(hsize_t), dimension(1) :: dcount, doffset
        allocate(ptls(np_local(pic_mpi_rank + 1)))
        dcount(1) = np_local(pic_mpi_rank + 1)
        doffset(1) = offset_local(pic_mpi_rank + 1)
        call read_hdf5_real_parallel(dset_ids(1), dcount, doffset, dset_dims, ptls%vx)
        call read_hdf5_real_parallel(dset_ids(2), dcount, doffset, dset_dims, ptls%vy)
        call read_hdf5_real_parallel(dset_ids(3), dcount, doffset, dset_dims, ptls%vz)
        call read_hdf5_real_parallel(dset_ids(4), dcount, doffset, dset_dims, ptls%dx)
        call read_hdf5_real_parallel(dset_ids(5), dcount, doffset, dset_dims, ptls%dy)
        call read_hdf5_real_parallel(dset_ids(6), dcount, doffset, dset_dims, ptls%dz)
        call read_hdf5_integer_parallel(dset_ids(7), dcount, doffset, dset_dims, ptls%icell)
        call read_hdf5_real_parallel(dset_ids(8), dcount, doffset, dset_dims, ptls%q)
    end subroutine read_particle_h5_parallel

end program particle_energization_io
