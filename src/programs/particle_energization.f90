!!******************************************************************************
!! Module for calculating particle energization
!!******************************************************************************
program particle_energization
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
    real :: start, finish, step1, step2
    integer :: tstart, tend, tinterval, tframe, fields_interval
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
    logical :: calc_para_perp, calc_comp_shear, calc_curv_grad_para_drifts
    logical :: calc_magnetic_moment, calc_polar_initial_time, calc_polar_initial_spatial

    ! Particles in HDF5 format
    integer, allocatable, dimension(:) :: np_local, offset_local
    logical :: particle_hdf5
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

    if (calc_para_perp) then
        nvar = 3
        call para_perp
    endif
    if (calc_comp_shear) then
        nvar = 3
        call comp_shear
    endif
    if (calc_curv_grad_para_drifts) then
        nvar = 4
        call curv_grad_para_drifts
    endif
    if (calc_magnetic_moment) then
        nvar = 2
        call magnetic_moment
    endif
    if (calc_polar_initial_time) then
        nvar = 3
        call polarization_initial_time
    endif
    if (calc_polar_initial_spatial) then
        nvar = 3
        call polarization_initial_spatial
    endif

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
        de_log = (log10(emax) - log10(emin)) / nbins
        emin_log = log10(emin)
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
    !< electric field
    !<--------------------------------------------------------------------------
    subroutine para_perp
        use picinfo, only: domain
        use topology_translate, only: ht
        use mpi_topology, only: htg
        use pic_fields, only: init_electric_fields, init_magnetic_fields, &
            free_electric_fields, free_magnetic_fields, &
            open_electric_field_files, open_magnetic_field_files, &
            read_electric_fields, read_magnetic_fields, &
            close_electric_field_files, close_magnetic_field_files
        use interpolation_emf, only: init_emfields, free_emfields
        implicit none
        integer :: dom_x, dom_y, dom_z
        real(fp) :: dx_domain, dy_domain, dz_domain

        call init_emfields
        if (is_translated_file) then
            call init_electric_fields(htg%nx, htg%ny, htg%nz)
            call init_magnetic_fields(htg%nx, htg%ny, htg%nz)
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
        endif

        call cpu_time(step1)
        do tframe = tstart, tend, tinterval
            if (myid == master) print*, tframe
            tp_emf = tframe / fields_interval
            call set_dists_zero
            if (is_translated_file) then
                if (output_format /= 1) then
                    ! Fields at each time step are saved in different files
                    call set_filepath(dir_emf)
                    call open_electric_field_files(tframe)
                    call open_magnetic_field_files(tframe)
                    call read_electric_fields(tp1)
                    call read_magnetic_fields(tp1)
                    call close_magnetic_field_files
                    call close_electric_field_files
                else
                    ! Fields at all time steps are saved in the same file
                    call read_electric_fields(tp_emf + 1)
                    call read_magnetic_fields(tp_emf + 1)
                endif
            endif  ! is_translated_file
            dx_domain = domain%lx_de / domain%pic_tx
            dy_domain = domain%ly_de / domain%pic_ty
            dz_domain = domain%lz_de / domain%pic_tz

            ! Particles are saved in HDF5
            if (particle_hdf5) then
                call get_np_local_vpic(tframe, species)
                call open_particle_file_h5(tframe, species)
            endif

            do dom_z = ht%start_z, ht%stop_z
                do dom_y = ht%start_y, ht%stop_y
                    do dom_x = ht%start_x, ht%stop_x
                        call para_perp_single(tframe, &
                            dom_x, dom_y, dom_z, dx_domain, dy_domain, dz_domain)
                    enddo ! x
                enddo ! y
            enddo ! z

            ! Particles are saved in HDF5
            if (particle_hdf5) then
                call free_np_offset_local
                call close_particle_file_h5
            endif

            call save_particle_energization(tframe, "para_perp")
            call cpu_time(step2)
            if (myid == master) then
                print '("Time for this step = ",f6.3," seconds.")', step2 - step1
            endif
            step1 = step2
        enddo  ! Time loop

        if (is_translated_file .and. output_format == 1) then
            call close_electric_field_files
            call close_magnetic_field_files
        endif

        deallocate(ebins, fbins, fbins_sum)

        if (is_translated_file) then
            call free_electric_fields
            call free_magnetic_fields
        endif
        call free_emfields
    end subroutine para_perp

    !<--------------------------------------------------------------------------
    !< Calculate particle energization due to parallel and perpendicular
    !< electric field for particles in a single PIC MPI rank
    !<--------------------------------------------------------------------------
    subroutine para_perp_single(tindex, dom_x, dom_y, dom_z, &
            dx_domain, dy_domain, dz_domain)
        use picinfo, only: domain
        use topology_translate, only: ht
        use rank_index_mapping, only: index_to_rank
        use interpolation_emf, only: read_emfields_single
        use particle_module, only: ptl, calc_interp_param, &
            iex, jex, kex, iey, jey, key, iez, jez, kez, ibx, jbx, kbx, &
            iby, jby, kby, ibz, jbz, kbz, dx_ex, dy_ex, dz_ex, &
            dx_ey, dy_ey, dz_ey, dx_ez, dy_ez, dz_ez, dx_bx, dx_by, dx_bz, &
            dy_bx, dy_by, dy_bz, dz_bx, dz_by, dz_bz
        use interpolation_emf, only: trilinear_interp_only_bx, &
            trilinear_interp_only_by, trilinear_interp_only_bz, &
            trilinear_interp_ex, trilinear_interp_ey, trilinear_interp_ez, &
            set_emf, bx0, by0, bz0, ex0, ey0, ez0
        use file_header, only: set_v0header, pheader
        implicit none
        integer, intent(in) :: tindex, dom_x, dom_y, dom_z
        real(fp), intent(in) :: dx_domain, dy_domain, dz_domain
        integer :: ibin, n, iptl, nptl
        real(fp) :: x0, y0, z0
        real(fp) :: ex_para, ey_para, ez_para, edotb, ib2
        real(fp) :: gama, igama, dke_para, dke_perp, weight
        real(fp) :: ux, uy, uz
        character(len=16) :: cid

        call index_to_rank(dom_x, dom_y, dom_z, domain%pic_tx, &
            domain%pic_ty, domain%pic_tz, n)
        write(cid, "(I0)") n - 1

        if (particle_hdf5) then
            call read_particle_h5(n - 1)
            nptl = np_local(n)
        else
            call read_particle_binary(tindex, species, cid)
            nptl = pheader%dim
        endif

        if (is_translated_file) then
            x0 = dx_domain * dom_x
            y0 = dy_domain * dom_y
            z0 = dz_domain * dom_z
            call set_v0header(domain%pic_nx, domain%pic_ny, domain%pic_nz, &
                x0, y0, z0, real(domain%dx), real(domain%dy), real(domain%dz))
            call set_emf(dom_x, dom_y, dom_z, domain%pic_tx, domain%pic_ty, &
                domain%pic_tz, ht%start_x, ht%start_y, ht%start_z)
        else
            call read_emfields_single(tindex, n - 1)
        endif
        do iptl = 1, nptl, 1
            ptl = ptls(iptl)
            call calc_interp_param
            call trilinear_interp_only_bx(ibx, jbx, kbx, dx_bx, dy_bx, dz_bx)
            call trilinear_interp_only_by(iby, jby, kby, dx_by, dy_by, dz_by)
            call trilinear_interp_only_bz(ibz, jbz, kbz, dx_bz, dy_bz, dz_bz)
            call trilinear_interp_ex(iex, jex, kex, dx_ex, dy_ex, dz_ex)
            call trilinear_interp_ey(iey, jey, key, dx_ey, dy_ey, dz_ey)
            call trilinear_interp_ez(iez, jez, kez, dx_ez, dy_ez, dz_ez)
            ib2 = 1.0 / (bx0**2 + by0**2 + bz0**2)
            edotb = ex0 * bx0 + ey0 * by0 + ez0 * bz0
            ux = ptl%vx  ! v in ptl is actually gamma*v
            uy = ptl%vy
            uz = ptl%vz
            gama = sqrt(1.0 + ux**2 + uy**2 + uz**2)
            igama = 1.0 / gama
            ex_para = edotb * bx0 * ib2
            ey_para = edotb * by0 * ib2
            ez_para = edotb * bz0 * ib2
            weight = abs(ptl%q)
            dke_para = (ex_para * ux + ey_para * uy + ez_para * uz) * &
                weight * igama * ptl_charge
            dke_perp = (ex0 * ux + ey0 * uy + ez0 * uz) * &
                weight * igama * ptl_charge - dke_para

            ibin = floor((log10(gama - 1) - emin_log) / de_log)
            if (ibin > 0 .and. ibin < nbins + 1) then
                fbins(ibin+1, 1) = fbins(ibin+1, 1) + weight
                fbins(ibin+1, 2) = fbins(ibin+1, 2) + dke_para
                fbins(ibin+1, 3) = fbins(ibin+1, 3) + dke_perp
            endif
        enddo
        deallocate(ptls)
    end subroutine para_perp_single

    !<--------------------------------------------------------------------------
    !< Calculate particle energization due to compression and shear
    !<--------------------------------------------------------------------------
    subroutine comp_shear
        use picinfo, only: domain
        use topology_translate, only: ht
        use mpi_topology, only: htg
        use pic_fields, only: init_electric_fields, init_magnetic_fields, &
            init_velocity_fields, free_electric_fields, free_magnetic_fields, &
            free_velocity_fields, open_electric_field_files, &
            open_magnetic_field_files, open_velocity_field_files, &
            read_electric_fields, read_magnetic_fields, read_velocity_fields, &
            close_electric_field_files, close_magnetic_field_files, &
            close_velocity_field_files, interp_emf_node_ghost
        use interpolation_emf, only: init_emfields, free_emfields
        use interpolation_vel_mom, only: init_vel_mom, free_vel_mom
        use interpolation_comp_shear, only: init_exb_drift, free_exb_drift, &
            calc_exb_drift, init_comp_shear, free_comp_shear, calc_comp_shear, &
            init_comp_shear_single, free_comp_shear_single
        implicit none
        integer :: dom_x, dom_y, dom_z
        real(fp) :: dx_domain, dy_domain, dz_domain

        call init_emfields
        call init_vel_mom
        call init_comp_shear_single
        if (is_translated_file) then
            call init_electric_fields(htg%nx, htg%ny, htg%nz)
            call init_magnetic_fields(htg%nx, htg%ny, htg%nz)
            call init_velocity_fields(htg%nx, htg%ny, htg%nz)
            call init_exb_drift(htg%nx, htg%ny, htg%nz)
            call init_comp_shear(htg%nx, htg%ny, htg%nz)
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
        endif

        call cpu_time(step1)
        do tframe = tstart, tend, tinterval
            if (myid == master) print*, tframe
            tp_emf = tframe / fields_interval
            call set_dists_zero
            if (is_translated_file) then
                if (output_format /= 1) then
                    ! Fields at each time step are saved in different files
                    call set_filepath(dir_emf)
                    call open_electric_field_files(tframe)
                    call open_magnetic_field_files(tframe)
                    call open_velocity_field_files(species, tframe)
                    call read_electric_fields(tp1)
                    call read_magnetic_fields(tp1)
                    call read_velocity_fields(tp1)
                    call close_magnetic_field_files
                    call close_electric_field_files
                    call close_velocity_field_files
                else
                    ! Fields at all time steps are saved in the same file
                    call read_electric_fields(tp_emf + 1)
                    call read_magnetic_fields(tp_emf + 1)
                    call read_velocity_fields(tp_emf + 1)
                endif
                call interp_emf_node_ghost
                call calc_exb_drift
                call calc_comp_shear(htg%nx, htg%ny, htg%nz)
            endif  ! is_translated_file
            dx_domain = domain%lx_de / domain%pic_tx
            dy_domain = domain%ly_de / domain%pic_ty
            dz_domain = domain%lz_de / domain%pic_tz

            ! Particles are saved in HDF5
            if (particle_hdf5) then
                call get_np_local_vpic(tframe, species)
                call open_particle_file_h5(tframe, species)
            endif

            do dom_z = ht%start_z, ht%stop_z
                do dom_y = ht%start_y, ht%stop_y
                    do dom_x = ht%start_x, ht%stop_x
                        call comp_shear_single(tframe, &
                            dom_x, dom_y, dom_z, dx_domain, dy_domain, dz_domain)
                    enddo ! x
                enddo ! y
            enddo ! z

            ! Particles are saved in HDF5
            if (particle_hdf5) then
                call free_np_offset_local
                call close_particle_file_h5
            endif

            call save_particle_energization(tframe, "comp_shear")
            call cpu_time(step2)
            if (myid == master) then
                print '("Time for this step = ",f6.3," seconds.")', step2 - step1
            endif
            step1 = step2
        enddo  ! Time loop

        if (is_translated_file .and. output_format == 1) then
            call close_electric_field_files
            call close_magnetic_field_files
            call close_velocity_field_files
        endif

        deallocate(ebins, fbins, fbins_sum)

        if (is_translated_file) then
            call free_electric_fields
            call free_magnetic_fields
            call free_velocity_fields
            call free_exb_drift
            call free_comp_shear
        endif
        call free_emfields
        call free_vel_mom
        call free_comp_shear_single
    end subroutine comp_shear

    !<--------------------------------------------------------------------------
    !< Calculate particle energization due to compression and shear for
    !< particles in a single PIC MPI rank
    !<--------------------------------------------------------------------------
    subroutine comp_shear_single(tindex, dom_x, dom_y, dom_z, &
            dx_domain, dy_domain, dz_domain)
        use picinfo, only: domain
        use topology_translate, only: ht
        use rank_index_mapping, only: index_to_rank
        use interpolation_emf, only: read_emfields_single
        use particle_module, only: ptl, calc_interp_param, &
            ino, jno, kno, dnx, dny, dnz
        use interpolation_emf, only: trilinear_interp_only_bx, &
            trilinear_interp_only_by, trilinear_interp_only_bz, &
            trilinear_interp_ex, trilinear_interp_ey, trilinear_interp_ez, &
            set_emf, bx0, by0, bz0, ex0, ey0, ez0
        use interpolation_vel_mom, only: trilinear_interp_vel_mom, &
            set_vel_mom, vx0, vy0, vz0, ux0, uy0, uz0
        use interpolation_comp_shear, only: trilinear_interp_comp_shear, &
            set_comp_shear_single, divv0, sigmaxx0, sigmaxy0, sigmaxz0, &
            sigmayy0, sigmayz0, sigmazz0
        use file_header, only: set_v0header, pheader
        implicit none
        integer, intent(in) :: tindex, dom_x, dom_y, dom_z
        real(fp), intent(in) :: dx_domain, dy_domain, dz_domain
        integer :: ibin, n, iptl, nptl
        real(fp) :: x0, y0, z0, ib2
        real(fp) :: gama, igama, dke_para, dke_perp
        real(fp) :: vx, vy, vz, ux, uy, uz
        real(fp) :: pxx, pxy, pxz, pyx, pyy, pyz, pzx, pzy, pzz
        real(fp) :: bxx, byy, bzz, bxy, bxz, byz
        real(fp) :: weight, pscalar, ppara, pperp, bbsigma
        real(fp) :: pdivv, pshear, ptensor_divv
        character(len=16) :: cid

        call index_to_rank(dom_x, dom_y, dom_z, domain%pic_tx, &
            domain%pic_ty, domain%pic_tz, n)
        write(cid, "(I0)") n - 1

        if (particle_hdf5) then
            call read_particle_h5(n - 1)
            nptl = np_local(n)
        else
            call read_particle_binary(tindex, species, cid)
            nptl = pheader%dim
        endif

        if (is_translated_file) then
            x0 = dx_domain * dom_x
            y0 = dy_domain * dom_y
            z0 = dz_domain * dom_z
            call set_v0header(domain%pic_nx, domain%pic_ny, domain%pic_nz, &
                x0, y0, z0, real(domain%dx), real(domain%dy), real(domain%dz))
            call set_emf(dom_x, dom_y, dom_z, domain%pic_tx, domain%pic_ty, &
                domain%pic_tz, ht%start_x, ht%start_y, ht%start_z)
            call set_vel_mom(dom_x, dom_y, dom_z, domain%pic_tx, domain%pic_ty, &
                domain%pic_tz, ht%start_x, ht%start_y, ht%start_z)
            call set_comp_shear_single(dom_x, dom_y, dom_z, domain%pic_tx, &
                domain%pic_ty, domain%pic_tz, ht%start_x, ht%start_y, ht%start_z)
        else
            call read_emfields_single(tindex, n - 1)
        endif
        do iptl = 1, nptl, 1
            ptl = ptls(iptl)
            call calc_interp_param
            call trilinear_interp_only_bx(ino, jno, kno, dnx, dny, dnz)
            call trilinear_interp_only_by(ino, jno, kno, dnx, dny, dnz)
            call trilinear_interp_only_bz(ino, jno, kno, dnx, dny, dnz)
            call trilinear_interp_ex(ino, jno, kno, dnx, dny, dnz)
            call trilinear_interp_ey(ino, jno, kno, dnx, dny, dnz)
            call trilinear_interp_ez(ino, jno, kno, dnx, dny, dnz)
            call trilinear_interp_vel_mom(ino, jno, kno, dnx, dny, dnz)
            call trilinear_interp_comp_shear(ino, jno, kno, dnx, dny, dnz)
            bxx = bx0**2
            byy = by0**2
            bzz = bz0**2
            bxy = bx0 * by0
            bxz = bx0 * bz0
            byz = by0 * bz0
            ib2 = 1.0 / (bxx + byy + bzz)
            ux = ptl%vx  ! v in ptl is actually gamma*v
            uy = ptl%vy
            uz = ptl%vz
            gama = sqrt(1.0 + ux**2 + uy**2 + uz**2)
            igama = 1.0 / gama
            vx = ux * igama
            vy = uy * igama
            vz = uz * igama
            weight = abs(ptl%q)
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
            bbsigma = sigmaxx0 * bxx + sigmayy0 * byy + sigmazz0 * bzz + &
                2.0 * (sigmaxy0 * bxy + sigmaxz0 * bxz + sigmayz0 * byz)
            bbsigma = bbsigma * ib2
            pdivv = -pscalar * divv0
            pshear = (pperp - ppara) * bbsigma

            ibin = floor((log10(gama - 1) - emin_log) / de_log)
            if (ibin > 0 .and. ibin < nbins + 1) then
                fbins(ibin+1, 1) = fbins(ibin+1, 1) + weight
                fbins(ibin+1, 2) = fbins(ibin+1, 2) + pdivv
                fbins(ibin+1, 3) = fbins(ibin+1, 3) + pshear
            endif
        enddo
        deallocate(ptls)
    end subroutine comp_shear_single

    !<--------------------------------------------------------------------------
    !< Calculate particle energization due to parallel and perpendicular
    !< electric field, compression and shear
    !<--------------------------------------------------------------------------
    subroutine para_perp_comp_shear
        use picinfo, only: domain
        use topology_translate, only: ht
        use mpi_topology, only: htg
        use pic_fields, only: init_electric_fields, init_magnetic_fields, &
            init_velocity_fields, free_electric_fields, free_magnetic_fields, &
            free_velocity_fields, open_electric_field_files, &
            open_magnetic_field_files, open_velocity_field_files, &
            read_electric_fields, read_magnetic_fields, read_velocity_fields, &
            close_electric_field_files, close_magnetic_field_files, &
            close_velocity_field_files
        use interpolation_emf, only: init_emfields, free_emfields
        use interpolation_vel_mom, only: init_vel_mom, free_vel_mom
        use interpolation_comp_shear, only: init_exb_drift, free_exb_drift, &
            read_exb_drift, init_comp_shear, free_comp_shear, calc_comp_shear, &
            init_comp_shear_single, free_comp_shear_single, &
            open_exb_drift_files, close_exb_drift_files
        implicit none
        integer :: dom_x, dom_y, dom_z
        real(fp) :: dx_domain, dy_domain, dz_domain

        call init_emfields
        call init_vel_mom
        call init_comp_shear_single
        if (is_translated_file) then
            call init_electric_fields(htg%nx, htg%ny, htg%nz)
            call init_magnetic_fields(htg%nx, htg%ny, htg%nz)
            call init_velocity_fields(htg%nx, htg%ny, htg%nz)
            call init_exb_drift(htg%nx, htg%ny, htg%nz)
            call init_comp_shear(htg%nx, htg%ny, htg%nz)
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
            call open_exb_drift_files
        endif

        call cpu_time(step1)
        do tframe = tstart, tend, tinterval
            if (myid == master) print*, tframe
            tp_emf = tframe / fields_interval
            call set_dists_zero
            if (is_translated_file) then
                if (output_format /= 1) then
                    ! Fields at each time step are saved in different files
                    call set_filepath(dir_emf)
                    call open_electric_field_files(tframe)
                    call open_magnetic_field_files(tframe)
                    call open_velocity_field_files(species, tframe)
                    call open_exb_drift_files(tframe)
                    call read_electric_fields(tp1)
                    call read_magnetic_fields(tp1)
                    call read_velocity_fields(tp1)
                    call read_exb_drift(tp1)
                    call close_magnetic_field_files
                    call close_electric_field_files
                    call close_velocity_field_files
                    call close_exb_drift_files
                else
                    ! Fields at all time steps are saved in the same file
                    call read_electric_fields(tp_emf + 1)
                    call read_magnetic_fields(tp_emf + 1)
                    call read_velocity_fields(tp_emf + 1)
                    call read_exb_drift(tp_emf + 1)
                endif
                call calc_comp_shear(htg%nx, htg%ny, htg%nz)
            endif  ! is_translated_file
            dx_domain = domain%lx_de / domain%pic_tx
            dy_domain = domain%ly_de / domain%pic_ty
            dz_domain = domain%lz_de / domain%pic_tz
            do dom_z = ht%start_z, ht%stop_z
                do dom_y = ht%start_y, ht%stop_y
                    do dom_x = ht%start_x, ht%stop_x
                        call para_perp_comp_shear_single(tframe, &
                            dom_x, dom_y, dom_z, dx_domain, dy_domain, dz_domain)
                    enddo ! x
                enddo ! y
            enddo ! z
            call save_particle_energization(tframe, "para_perp_comp_shear")
            call cpu_time(step2)
            if (myid == master) then
                print '("Time for this step = ",f6.3," seconds.")', step2 - step1
            endif
            step1 = step2
        enddo  ! Time loop

        if (is_translated_file .and. output_format == 1) then
            call close_electric_field_files
            call close_magnetic_field_files
            call close_velocity_field_files
            call close_exb_drift_files
        endif

        deallocate(ebins, fbins, fbins_sum)

        if (is_translated_file) then
            call free_electric_fields
            call free_magnetic_fields
            call free_velocity_fields
            call free_exb_drift
            call free_comp_shear
        endif
        call free_emfields
        call free_vel_mom
        call free_comp_shear_single
    end subroutine para_perp_comp_shear

    !<--------------------------------------------------------------------------
    !< Calculate particle energization due to parallel and perpendicular
    !< electric field, compression and shear for particles in a single PIC MPI rank
    !<--------------------------------------------------------------------------
    subroutine para_perp_comp_shear_single(tindex, dom_x, dom_y, dom_z, &
            dx_domain, dy_domain, dz_domain)
        use picinfo, only: domain
        use topology_translate, only: ht
        use rank_index_mapping, only: index_to_rank
        use interpolation_emf, only: read_emfields_single
        use particle_module, only: ptl, calc_interp_param, &
            iex, jex, kex, iey, jey, key, iez, jez, kez, ibx, jbx, kbx, &
            iby, jby, kby, ibz, jbz, kbz, dx_ex, dy_ex, dz_ex, &
            dx_ey, dy_ey, dz_ey, dx_ez, dy_ez, dz_ez, dx_bx, dx_by, dx_bz, &
            dy_bx, dy_by, dy_bz, dz_bx, dz_by, dz_bz, ino, jno, kno, &
            dnx, dny, dnz
        use interpolation_emf, only: trilinear_interp_only_bx, &
            trilinear_interp_only_by, trilinear_interp_only_bz, &
            trilinear_interp_ex, trilinear_interp_ey, trilinear_interp_ez, &
            set_emf, bx0, by0, bz0, ex0, ey0, ez0
        use interpolation_vel_mom, only: trilinear_interp_vel_mom, &
            set_vel_mom, vx0, vy0, vz0, ux0, uy0, uz0
        use interpolation_comp_shear, only: trilinear_interp_comp_shear, &
            set_comp_shear_single, divv0, sigmaxx0, sigmaxy0, sigmaxz0, &
            sigmayy0, sigmayz0, sigmazz0
        use file_header, only: set_v0header, pheader
        implicit none
        integer, intent(in) :: tindex, dom_x, dom_y, dom_z
        real(fp), intent(in) :: dx_domain, dy_domain, dz_domain
        integer :: ibin, n, iptl, nptl
        real(fp) :: x0, y0, z0
        real(fp) :: ex_para, ey_para, ez_para, edotb, ib2
        real(fp) :: gama, igama, dke_para, dke_perp
        real(fp) :: vx, vy, vz, ux, uy, uz
        real(fp) :: pxx, pxy, pxz, pyx, pyy, pyz, pzx, pzy, pzz
        real(fp) :: bxx, byy, bzz, bxy, bxz, byz
        real(fp) :: weight, pscalar, ppara, pperp, bbsigma
        real(fp) :: pdivv, pshear, ptensor_divv
        character(len=16) :: cid

        call index_to_rank(dom_x, dom_y, dom_z, domain%pic_tx, &
            domain%pic_ty, domain%pic_tz, n)
        write(cid, "(I0)") n - 1

        call read_particle_binary(tindex, species, cid)

        if (is_translated_file) then
            x0 = dx_domain * dom_x
            y0 = dy_domain * dom_y
            z0 = dz_domain * dom_z
            call set_v0header(domain%pic_nx, domain%pic_ny, domain%pic_nz, &
                x0, y0, z0, real(domain%dx), real(domain%dy), real(domain%dz))
            call set_emf(dom_x, dom_y, dom_z, domain%pic_tx, domain%pic_ty, &
                domain%pic_tz, ht%start_x, ht%start_y, ht%start_z)
            call set_vel_mom(dom_x, dom_y, dom_z, domain%pic_tx, domain%pic_ty, &
                domain%pic_tz, ht%start_x, ht%start_y, ht%start_z)
            call set_comp_shear_single(dom_x, dom_y, dom_z, domain%pic_tx, &
                domain%pic_ty, domain%pic_tz, ht%start_x, ht%start_y, ht%start_z)
        else
            call read_emfields_single(tindex, n - 1)
        endif
        do iptl = 1, pheader%dim, 1
            ptl = ptls(iptl)
            call calc_interp_param
            call trilinear_interp_only_bx(ibx, jbx, kbx, dx_bx, dy_bx, dz_bx)
            call trilinear_interp_only_by(iby, jby, kby, dx_by, dy_by, dz_by)
            call trilinear_interp_only_bz(ibz, jbz, kbz, dx_bz, dy_bz, dz_bz)
            call trilinear_interp_ex(iex, jex, kex, dx_ex, dy_ex, dz_ex)
            call trilinear_interp_ey(iey, jey, key, dx_ey, dy_ey, dz_ey)
            call trilinear_interp_ez(iez, jez, kez, dx_ez, dy_ez, dz_ez)
            call trilinear_interp_vel_mom(ino, jno, kno, dnx, dny, dnz)
            call trilinear_interp_comp_shear(ino, jno, kno, dnx, dny, dnz)
            bxx = bx0**2
            byy = by0**2
            bzz = bz0**2
            bxy = bx0 * by0
            bxz = bx0 * bz0
            byz = by0 * bz0
            ib2 = 1.0 / (bxx + byy + bzz)
            edotb = ex0 * bx0 + ey0 * by0 + ez0 * bz0
            ux = ptl%vx  ! v in ptl is actually gamma*v
            uy = ptl%vy
            uz = ptl%vz
            gama = sqrt(1.0 + ux**2 + uy**2 + uz**2)
            igama = 1.0 / gama
            vx = ux * igama
            vy = uy * igama
            vz = uz * igama
            ex_para = edotb * bx0 * ib2
            ey_para = edotb * by0 * ib2
            ez_para = edotb * bz0 * ib2
            dke_para = (ex_para * vx + ey_para * vy + ez_para * vz) * ptl%q
            dke_perp = (ex0 * vx + ey0 * vy + ez0 * vz) * ptl%q - dke_para
            weight = abs(ptl%q)
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
            bbsigma = sigmaxx0 * bxx + sigmayy0 * byy + sigmazz0 * bzz + &
                2.0 * (sigmaxy0 * bxy + sigmaxz0 * bxz + sigmayz0 * byz)
            bbsigma = bbsigma * ib2
            pdivv = -pscalar * divv0
            pshear = (pperp - ppara) * bbsigma

            ibin = floor((log10(gama - 1) - emin_log) / de_log)
            if (ibin > 0 .and. ibin < nbins + 1) then
                fbins(ibin+1, 1) = fbins(ibin+1, 1) + weight
                fbins(ibin+1, 2) = fbins(ibin+1, 2) + dke_para
                fbins(ibin+1, 3) = fbins(ibin+1, 3) + dke_perp
                fbins(ibin+1, 4) = fbins(ibin+1, 4) + pdivv
                fbins(ibin+1, 5) = fbins(ibin+1, 5) + pshear
            endif
        enddo
        deallocate(ptls)
    end subroutine para_perp_comp_shear_single

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

    !!--------------------------------------------------------------------------
    !! Initialize the analysis.
    !!--------------------------------------------------------------------------
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
    !< Calculate particle energization curvature drift, gradient drift, and
    !< parallel drift.
    !<--------------------------------------------------------------------------
    subroutine curv_grad_para_drifts
        use picinfo, only: domain
        use topology_translate, only: ht
        use mpi_topology, only: htg
        use pic_fields, only: init_electric_fields, init_magnetic_fields, &
            free_electric_fields, free_magnetic_fields, open_electric_field_files, &
            open_magnetic_field_files, read_electric_fields, read_magnetic_fields, &
            close_electric_field_files, close_magnetic_field_files
        use interpolation_emf, only: init_emfields, init_emfields_derivatives, &
            free_emfields, free_emfields_derivatives
        use emf_derivatives, only: init_bfield_derivatives, &
            free_bfield_derivatives, calc_bfield_derivatives
        implicit none
        integer :: dom_x, dom_y, dom_z
        real(fp) :: dx_domain, dy_domain, dz_domain

        call init_emfields
        call init_emfields_derivatives
        if (is_translated_file) then
            call init_electric_fields(htg%nx, htg%ny, htg%nz)
            call init_magnetic_fields(htg%nx, htg%ny, htg%nz)
            call init_bfield_derivatives(htg%nx, htg%ny, htg%nz)
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
        endif

        call cpu_time(step1)
        do tframe = tstart, tend, tinterval
            if (myid == master) print*, tframe
            tp_emf = tframe / fields_interval
            call set_dists_zero
            if (is_translated_file) then
                if (output_format /= 1) then
                    ! Fields at each time step are saved in different files
                    call set_filepath(dir_emf)
                    call open_electric_field_files(tframe)
                    call open_magnetic_field_files(tframe)
                    call read_electric_fields(tp1)
                    call read_magnetic_fields(tp1)
                    call close_magnetic_field_files
                    call close_electric_field_files
                else
                    ! Fields at all time steps are saved in the same file
                    call read_electric_fields(tp_emf + 1)
                    call read_magnetic_fields(tp_emf + 1)
                endif
                call calc_bfield_derivatives(htg%nx, htg%ny, htg%nz)
            endif  ! is_translated_file
            dx_domain = domain%lx_de / domain%pic_tx
            dy_domain = domain%ly_de / domain%pic_ty
            dz_domain = domain%lz_de / domain%pic_tz
            ptl_rm_local = 0

            ! Particles are saved in HDF5
            if (particle_hdf5) then
                call get_np_local_vpic(tframe, species)
                call open_particle_file_h5(tframe, species)
            endif

            do dom_z = ht%start_z, ht%stop_z
                do dom_y = ht%start_y, ht%stop_y
                    do dom_x = ht%start_x, ht%stop_x
                        call curv_grad_para_drifts_single(tframe, &
                            dom_x, dom_y, dom_z, dx_domain, dy_domain, dz_domain)
                    enddo ! x
                enddo ! y
            enddo ! z

            ! Particles are saved in HDF5
            if (particle_hdf5) then
                call free_np_offset_local
                call close_particle_file_h5
            endif

            ! Check how many particles are removed
            call MPI_REDUCE(ptl_rm_local, ptl_rm_global, 1, MPI_INTEGER, &
                MPI_SUM, master, MPI_COMM_WORLD, ierr)
            if (myid == master) then
                print '("Number of removed particles is "I)', ptl_rm_global
                print '("Fraction of removed particles is "E10.3)', &
                    ptl_rm_global*1.0 / (domain%nx * domain%ny * domain%nz * domain%nppc)
            endif

            call save_particle_energization(tframe, "curv_grad_para")
            call cpu_time(step2)
            if (myid == master) then
                print '("Time for this step = ",f6.3," seconds.")', step2 - step1
            endif
            step1 = step2
        enddo  ! Time loop

        if (is_translated_file .and. output_format == 1) then
            call close_electric_field_files
            call close_magnetic_field_files
        endif

        deallocate(ebins, fbins, fbins_sum)

        if (is_translated_file) then
            call free_electric_fields
            call free_magnetic_fields
            call free_bfield_derivatives
        endif
        call free_emfields
        call free_emfields_derivatives
    end subroutine curv_grad_para_drifts

    !<--------------------------------------------------------------------------
    !< Calculate particle energization due to parallel and perpendicular
    !< electric field, compression and shear for particles in a single PIC MPI rank
    !<--------------------------------------------------------------------------
    subroutine curv_grad_para_drifts_single(tindex, dom_x, dom_y, dom_z, &
            dx_domain, dy_domain, dz_domain)
        use picinfo, only: domain
        use topology_translate, only: ht
        use rank_index_mapping, only: index_to_rank
        use particle_module, only: ptl, calc_interp_param, calc_gyrofrequency, &
            calc_particle_energy, calc_para_perp_velocity_3d, &
            iex, jex, kex, iey, jey, key, iez, jez, kez, ibx, jbx, kbx, &
            iby, jby, kby, ibz, jbz, kbz, dx_ex, dy_ex, dz_ex, &
            dx_ey, dy_ey, dz_ey, dx_ez, dy_ez, dz_ez, dx_bx, dx_by, dx_bz, &
            dy_bx, dy_by, dy_bz, dz_bx, dz_by, dz_bz, ino, jno, kno, &
            dnx, dny, dnz, gyrof, vperpx, vperpy, vperpz, vpara, vperp
        use interpolation_emf, only: read_emfields_single, &
            trilinear_interp_bx, trilinear_interp_by, trilinear_interp_bz, &
            trilinear_interp_ex, trilinear_interp_ey, trilinear_interp_ez, &
            set_emf, set_emf_derivatives, calc_b_norm, calc_gradient_B, &
            calc_curvature, ex0, ey0, ez0, bx0, by0, bz0, bxn, byn, bzn, &
            dbxdy0, dbxdz0, dbydx0, dbydz0, dbzdx0, dbzdy0, &
            dBdx, dBdy, dBdz, kappax, kappay, kappaz, absB0
        use file_header, only: set_v0header, pheader
        implicit none
        integer, intent(in) :: tindex, dom_x, dom_y, dom_z
        real(fp), intent(in) :: dx_domain, dy_domain, dz_domain
        integer :: ibin, n, iptl, nptl
        real(fp) :: x0, y0, z0, ux, uy, uz
        real(fp) :: weight, gama, curv_ene, grad_ene, parad_ene, ib2
        real(fp) :: param, vcx, vcy, vcz, vgx, vgy, vgz
        real(fp) :: vexb_x, vexb_y, vexb_z
        real(fp) :: vpara_dx, vpara_dy, vpara_dz, vpara_d  ! Parallel drift
        character(len=16) :: cid

        call index_to_rank(dom_x, dom_y, dom_z, domain%pic_tx, &
            domain%pic_ty, domain%pic_tz, n)
        write(cid, "(I0)") n - 1

        if (particle_hdf5) then
            call read_particle_h5(n - 1)
            nptl = np_local(n)
        else
            call read_particle_binary(tindex, species, cid)
            nptl = pheader%dim
        endif

        if (is_translated_file) then
            x0 = dx_domain * dom_x
            y0 = dy_domain * dom_y
            z0 = dz_domain * dom_z
            call set_v0header(domain%pic_nx, domain%pic_ny, domain%pic_nz, &
                x0, y0, z0, real(domain%dx), real(domain%dy), real(domain%dz))
            call set_emf(dom_x, dom_y, dom_z, domain%pic_tx, domain%pic_ty, &
                domain%pic_tz, ht%start_x, ht%start_y, ht%start_z)
            call set_emf_derivatives(dom_x, dom_y, dom_z, domain%pic_tx, &
                domain%pic_ty, domain%pic_tz, ht%start_x, ht%start_y, ht%start_z)
        else
            call read_emfields_single(tindex, n - 1)
        endif
        do iptl = 1, nptl, 1
            ptl = ptls(iptl)
            call calc_interp_param
            call trilinear_interp_bx(ibx, jbx, kbx, dx_bx, dy_bx, dz_bx)
            call trilinear_interp_by(iby, jby, kby, dx_by, dy_by, dz_by)
            call trilinear_interp_bz(ibz, jbz, kbz, dx_bz, dy_bz, dz_bz)
            call trilinear_interp_ex(iex, jex, kex, dx_ex, dy_ex, dz_ex)
            call trilinear_interp_ey(iey, jey, key, dx_ey, dy_ey, dz_ey)
            call trilinear_interp_ez(iez, jez, kez, dx_ez, dy_ez, dz_ez)
            call calc_b_norm
            call calc_gradient_B
            call calc_curvature
            call calc_particle_energy
            call calc_para_perp_velocity_3d
            call calc_gyrofrequency

            ux = ptl%vx  ! v in ptl is actually gamma*v
            uy = ptl%vy
            uz = ptl%vz
            gama = sqrt(1.0 + ux**2 + uy**2 + uz**2)

            ib2 = 1.0 / (bx0**2 + by0**2 + bz0**2)
            vexb_x = (ey0 * bz0 - ez0 * by0) * ib2
            vexb_y = (ez0 * bx0 - ex0 * bz0) * ib2
            vexb_z = (ex0 * by0 - ey0 * bx0) * ib2

            ! if ((ex0**2 + ey0**2 + ez0**2) > 0.25*(bx0**2 + by0**2 + bz0**2)) then
            !     ptl_rm_local = ptl_rm_local + 1
            !     cycle
            ! endif

            weight = abs(ptl%q)

            ! Gradient drift
            param = ((vperpx - vexb_x)**2 + &
                     (vperpy - vexb_y)**2 + &
                     (vperpz - vexb_z)**2) / (2 * gyrof * absB0)
            ! param = vperp**2 / (2 * gyrof * absB0)
            vgx = param * (byn*dBdz - bzn*dBdy)
            vgy = param * (bzn*dBdx - bxn*dBdz)
            vgz = param * (bxn*dBdy - byn*dBdx)
            grad_ene = weight * ptl_charge * (ex0 * vgx + ey0 * vgy + ez0 * vgz)

            ! Curvature drift
            param = vpara * vpara / gyrof
            vcx = param * (byn*kappaz - bzn*kappay)
            vcy = param * (bzn*kappax - bxn*kappaz)
            vcz = param * (bxn*kappay - byn*kappax)
            curv_ene = weight * ptl_charge * (ex0 * vcx + ey0 * vcy + ez0 * vcz)

            ! Parallel drift
            param = ((vperpx - vexb_x)**2 + &
                     (vperpy - vexb_y)**2 + &
                     (vperpz - vexb_z)**2) / gyrof
            ! param = vperp**2 / gyrof
            vpara_d = ((dbzdy0 - dbydz0) * bxn + &
                       (dbxdz0 - dbzdx0) * byn + &
                       (dbydx0 - dbxdy0) * bzn) / absB0
            parad_ene = weight * ptl_charge * vpara_d * param * &
                (ex0 * bxn + ey0 * byn + ez0 * bzn)

            ibin = floor((log10(gama - 1) - emin_log) / de_log)
            if (ibin > 0 .and. ibin < nbins + 1) then
                fbins(ibin+1, 1) = fbins(ibin+1, 1) + weight
                fbins(ibin+1, 2) = fbins(ibin+1, 2) + curv_ene
                fbins(ibin+1, 3) = fbins(ibin+1, 3) + grad_ene
                fbins(ibin+1, 4) = fbins(ibin+1, 4) + parad_ene
            endif
        enddo
        deallocate(ptls)
    end subroutine curv_grad_para_drifts_single

    !<--------------------------------------------------------------------------
    !< Calculate particle energy change due to the conservation of magnetic
    !< moment.
    !<--------------------------------------------------------------------------
    subroutine magnetic_moment
        use picinfo, only: domain
        use topology_translate, only: ht
        use mpi_topology, only: htg
        use pic_fields, only: init_electric_fields, init_magnetic_fields, &
            free_electric_fields, free_magnetic_fields, open_electric_field_files, &
            open_magnetic_field_files, read_electric_fields, read_magnetic_fields, &
            close_electric_field_files, close_magnetic_field_files
        use interpolation_emf, only: init_emfields, free_emfields
        use pre_post_emf, only: init_pre_post_bfield, free_pre_post_bfield, &
            open_bfield_pre_post, close_bfield_pre_post, read_pre_post_bfield, &
            interp_bfield_node_ghost
        use interpolation_pre_post_bfield, only: init_bfield_magnitude, &
            free_bfield_magnitude
        implicit none
        integer :: dom_x, dom_y, dom_z
        integer :: tframe_pre, tframe_post
        real(fp) :: dx_domain, dy_domain, dz_domain
        real(fp) :: dt_fields

        call init_emfields
        call init_bfield_magnitude
        if (is_translated_file) then
            call init_electric_fields(htg%nx, htg%ny, htg%nz)
            call init_magnetic_fields(htg%nx, htg%ny, htg%nz)
            call init_pre_post_bfield(htg%nx, htg%ny, htg%nz)
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
            call open_bfield_pre_post(separated_pre_post)
        endif

        call cpu_time(step1)
        do tframe = tstart, tend, tinterval
            if (myid == master) print*, tframe
            tp_emf = tframe / fields_interval
            call set_dists_zero
            if (is_translated_file) then
                if (output_format /= 1) then
                    ! Fields at each time step are saved in different files
                    if (tp_emf == tp1) then
                        tframe_pre = tframe
                    else
                        tframe_pre = tframe - 1
                    endif
                    if (tp_emf == tp2) then
                        tframe_post = tframe
                    else
                        tframe_post = tframe + 1
                    endif
                    call set_filepath(dir_emf)
                    call open_electric_field_files(tframe)
                    call open_magnetic_field_files(tframe)
                    call open_bfield_pre_post(separated_pre_post, &
                        tframe, tframe_pre, tframe_post)
                    call read_electric_fields(tp1)
                    call read_magnetic_fields(tp1)
                    call read_pre_post_bfield(tp1, output_format, separated_pre_post)
                    call close_magnetic_field_files
                    call close_electric_field_files
                    call close_bfield_pre_post
                else
                    ! Fields at all time steps are saved in the same file
                    call read_electric_fields(tp_emf + 1)
                    call read_magnetic_fields(tp_emf + 1)
                    call read_pre_post_bfield(tp_emf + 1, output_format, separated_pre_post)
                endif
            endif  ! is_translated_file
            call interp_bfield_node_ghost
            dx_domain = domain%lx_de / domain%pic_tx
            dy_domain = domain%ly_de / domain%pic_ty
            dz_domain = domain%lz_de / domain%pic_tz
            ptl_rm_local = 0

            ! Particles are saved in HDF5
            if (particle_hdf5) then
                call get_np_local_vpic(tframe, species)
                call open_particle_file_h5(tframe, species)
            endif

            ! Time interval
            if (separated_pre_post) then
                if (tframe == tp1 .or. tframe == tp2) then
                    dt_fields = domain%dtwpe
                else
                    dt_fields = domain%dtwpe * 2.0
                endif
            else
                if (tframe == tp1 .or. tframe == tp2) then
                    dt_fields = domain%dt
                else
                    dt_fields = domain%dt * 2.0
                endif
            endif
            do dom_z = ht%start_z, ht%stop_z
                do dom_y = ht%start_y, ht%stop_y
                    do dom_x = ht%start_x, ht%stop_x
                        call magnetic_moment_single(tframe, dom_x, dom_y, dom_z, &
                            dx_domain, dy_domain, dz_domain, dt_fields)
                    enddo ! x
                enddo ! y
            enddo ! z

            ! Particles are saved in HDF5
            if (particle_hdf5) then
                call free_np_offset_local
                call close_particle_file_h5
            endif

            ! Check how many particles are removed
            call MPI_REDUCE(ptl_rm_local, ptl_rm_global, 1, MPI_INTEGER, &
                MPI_SUM, master, MPI_COMM_WORLD, ierr)
            if (myid == master) then
                print '("Number of removed particles is "I)', ptl_rm_global
                print '("Fraction of removed particles is "E10.3)', &
                    ptl_rm_global*1.0 / (domain%nx * domain%ny * domain%nz * domain%nppc)
            endif

            call save_particle_energization(tframe, "magnetic_moment")
            call cpu_time(step2)
            if (myid == master) then
                print '("Time for this step = ",f6.3," seconds.")', step2 - step1
            endif
            step1 = step2
        enddo  ! Time loop

        if (is_translated_file .and. output_format == 1) then
            call close_electric_field_files
            call close_magnetic_field_files
            call close_bfield_pre_post
        endif

        deallocate(ebins, fbins, fbins_sum)

        if (is_translated_file) then
            call free_electric_fields
            call free_magnetic_fields
            call free_pre_post_bfield
        endif
        call free_emfields
        call free_bfield_magnitude
    end subroutine magnetic_moment

    !<--------------------------------------------------------------------------
    !< Calculate particle energy change due to the conservation of magnetic
    !< moment for particles in a single PIC MPI rank
    !<--------------------------------------------------------------------------
    subroutine magnetic_moment_single(tindex, dom_x, dom_y, dom_z, &
            dx_domain, dy_domain, dz_domain, dt_fields)
        use picinfo, only: domain
        use topology_translate, only: ht
        use rank_index_mapping, only: index_to_rank
        use particle_module, only: ptl, calc_interp_param, calc_gyrofrequency, &
            calc_particle_energy, calc_para_perp_velocity_3d, &
            iex, jex, kex, iey, jey, key, iez, jez, kez, ibx, jbx, kbx, &
            iby, jby, kby, ibz, jbz, kbz, dx_ex, dy_ex, dz_ex, &
            dx_ey, dy_ey, dz_ey, dx_ez, dy_ez, dz_ez, dx_bx, dx_by, dx_bz, &
            dy_bx, dy_by, dy_bz, dz_bx, dz_by, dz_bz, ino, jno, kno, &
            dnx, dny, dnz, gyrof, vperpx, vperpy, vperpz, vpara, vperp
        use interpolation_emf, only: read_emfields_single, &
            trilinear_interp_only_bx, trilinear_interp_only_by, trilinear_interp_only_bz, &
            trilinear_interp_ex, trilinear_interp_ey, trilinear_interp_ez, &
            set_emf, calc_b_norm, ex0, ey0, ez0, bx0, by0, bz0, bxn, byn, bzn,&
            absB0
        use interpolation_pre_post_bfield, only: set_bfield_magnitude, &
            trilinear_interp_bfield_magnitude, absB1_0, absB2_0
        use file_header, only: set_v0header, pheader
        implicit none
        integer, intent(in) :: tindex, dom_x, dom_y, dom_z
        real(fp), intent(in) :: dx_domain, dy_domain, dz_domain, dt_fields
        integer :: ibin, n, iptl, nptl
        real(fp) :: x0, y0, z0, ux, uy, uz
        real(fp) :: weight, gama, ib2, mag_moment, idt, dene_m
        real(fp) :: vexb_x, vexb_y, vexb_z
        character(len=16) :: cid

        call index_to_rank(dom_x, dom_y, dom_z, domain%pic_tx, &
            domain%pic_ty, domain%pic_tz, n)
        write(cid, "(I0)") n - 1

        if (particle_hdf5) then
            call read_particle_h5(n - 1)
            nptl = np_local(n)
        else
            call read_particle_binary(tindex, species, cid)
            nptl = pheader%dim
        endif

        if (is_translated_file) then
            x0 = dx_domain * dom_x
            y0 = dy_domain * dom_y
            z0 = dz_domain * dom_z
            call set_v0header(domain%pic_nx, domain%pic_ny, domain%pic_nz, &
                x0, y0, z0, real(domain%dx), real(domain%dy), real(domain%dz))
            call set_emf(dom_x, dom_y, dom_z, domain%pic_tx, domain%pic_ty, &
                domain%pic_tz, ht%start_x, ht%start_y, ht%start_z)
            call set_bfield_magnitude(dom_x, dom_y, dom_z, domain%pic_tx, &
                domain%pic_ty, domain%pic_tz, ht%start_x, ht%start_y, ht%start_z)
        else
            call read_emfields_single(tindex, n - 1)
        endif

        idt = 1.0 / dt_fields
        do iptl = 1, nptl, 1
            ptl = ptls(iptl)
            call calc_interp_param
            call trilinear_interp_only_bx(ibx, jbx, kbx, dx_bx, dy_bx, dz_bx)
            call trilinear_interp_only_by(iby, jby, kby, dx_by, dy_by, dz_by)
            call trilinear_interp_only_bz(ibz, jbz, kbz, dx_bz, dy_bz, dz_bz)
            call trilinear_interp_ex(iex, jex, kex, dx_ex, dy_ex, dz_ex)
            call trilinear_interp_ey(iey, jey, key, dx_ey, dy_ey, dz_ey)
            call trilinear_interp_ez(iez, jez, kez, dx_ez, dy_ez, dz_ez)
            call trilinear_interp_bfield_magnitude(ino, jno, kno, dnx, dny, dnz)
            call calc_b_norm
            call calc_particle_energy
            call calc_para_perp_velocity_3d
            call calc_gyrofrequency
            ib2 = 1.0 / (bx0**2 + by0**2 + bz0**2)
            vexb_x = (ey0 * bz0 - ez0 * by0) * ib2
            vexb_y = (ez0 * bx0 - ex0 * bz0) * ib2
            vexb_z = (ex0 * by0 - ey0 * bx0) * ib2

            weight = abs(ptl%q)
            gama = sqrt(1.0 + ptl%vx**2 + ptl%vy**2 + ptl%vz**2)

            mag_moment = ((vperpx - vexb_x)**2 + &
                          (vperpy - vexb_y)**2 + &
                          (vperpz - vexb_z)**2) * ptl_mass * gama / (2 * absB0)
            ! mag_moment = vperp**2 * ptl_mass * gama / (2 * absB0)
            dene_m = mag_moment * (absB2_0 - absB1_0) * idt * weight

            ibin = floor((log10(gama - 1) - emin_log) / de_log)
            if (ibin > 0 .and. ibin < nbins + 1) then
                fbins(ibin+1, 1) = fbins(ibin+1, 1) + weight
                fbins(ibin+1, 2) = fbins(ibin+1, 2) + dene_m
            endif
        enddo
        deallocate(ptls)
    end subroutine magnetic_moment_single

    !<--------------------------------------------------------------------------
    !< Calculate particle energy change due to the time-dependent part of
    !< polarization drift and initial drift
    !<--------------------------------------------------------------------------
    subroutine polarization_initial_time
        use picinfo, only: domain
        use topology_translate, only: ht
        use mpi_topology, only: htg
        use pic_fields, only: init_electric_fields, init_magnetic_fields, &
            free_electric_fields, free_magnetic_fields, open_electric_field_files, &
            open_magnetic_field_files, read_electric_fields, read_magnetic_fields, &
            close_electric_field_files, close_magnetic_field_files
        use interpolation_emf, only: init_emfields, free_emfields
        use pre_post_emf, only: init_pre_post_bfield, init_pre_post_efield, &
            free_pre_post_bfield, free_pre_post_efield, open_bfield_pre_post, &
            open_efield_pre_post, close_bfield_pre_post, close_efield_pre_post, &
            read_pre_post_bfield, read_pre_post_efield, interp_bfield_node_ghost, &
            interp_efield_node_ghost
        use interpolation_pre_post_bfield, only: init_bfield_magnitude, &
            init_bfield_components, free_bfield_magnitude, free_bfield_components
        use interpolation_pre_post_efield, only: init_efield_components, &
            free_efield_components
        implicit none
        integer :: dom_x, dom_y, dom_z
        integer :: tframe_pre, tframe_post
        real(fp) :: dx_domain, dy_domain, dz_domain
        real(fp) :: dt_fields

        call init_emfields
        call init_bfield_magnitude
        call init_bfield_components
        call init_efield_components
        if (is_translated_file) then
            call init_electric_fields(htg%nx, htg%ny, htg%nz)
            call init_magnetic_fields(htg%nx, htg%ny, htg%nz)
            call init_pre_post_bfield(htg%nx, htg%ny, htg%nz)
            call init_pre_post_efield(htg%nx, htg%ny, htg%nz)
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
            call open_bfield_pre_post(separated_pre_post)
            call open_efield_pre_post(separated_pre_post)
        endif

        call cpu_time(step1)
        do tframe = tstart, tend, tinterval
            if (myid == master) print*, tframe
            tp_emf = tframe / fields_interval
            call set_dists_zero
            if (is_translated_file) then
                if (output_format /= 1) then
                    ! Fields at each time step are saved in different files
                    if (tp_emf == tp1) then
                        tframe_pre = tframe
                    else
                        tframe_pre = tframe - 1
                    endif
                    if (tp_emf == tp2) then
                        tframe_post = tframe
                    else
                        tframe_post = tframe + 1
                    endif
                    call set_filepath(dir_emf)
                    call open_electric_field_files(tframe)
                    call open_magnetic_field_files(tframe)
                    call open_bfield_pre_post(separated_pre_post, &
                        tframe, tframe_pre, tframe_post)
                    call open_efield_pre_post(separated_pre_post, &
                        tframe, tframe_pre, tframe_post)
                    call read_electric_fields(tp1)
                    call read_magnetic_fields(tp1)
                    call read_pre_post_bfield(tp1, output_format, separated_pre_post)
                    call read_pre_post_efield(tp1, output_format, separated_pre_post)
                    call close_magnetic_field_files
                    call close_electric_field_files
                    call close_bfield_pre_post
                    call close_efield_pre_post
                else
                    ! Fields at all time steps are saved in the same file
                    call read_electric_fields(tp_emf + 1)
                    call read_magnetic_fields(tp_emf + 1)
                    call read_pre_post_bfield(tp_emf + 1, output_format, separated_pre_post)
                    call read_pre_post_efield(tp_emf + 1, output_format, separated_pre_post)
                endif
            endif  ! is_translated_file
            call interp_bfield_node_ghost
            call interp_efield_node_ghost
            dx_domain = domain%lx_de / domain%pic_tx
            dy_domain = domain%ly_de / domain%pic_ty
            dz_domain = domain%lz_de / domain%pic_tz
            ptl_rm_local = 0

            ! Particles are saved in HDF5
            if (particle_hdf5) then
                call get_np_local_vpic(tframe, species)
                call open_particle_file_h5(tframe, species)
            endif

            ! Time interval
            if (separated_pre_post) then
                if (tframe == tp1 .or. tframe == tp2) then
                    dt_fields = domain%dtwpe
                else
                    dt_fields = domain%dtwpe * 2.0
                endif
            else
                if (tframe == tp1 .or. tframe == tp2) then
                    dt_fields = domain%dt
                else
                    dt_fields = domain%dt * 2.0
                endif
            endif
            do dom_z = ht%start_z, ht%stop_z
                do dom_y = ht%start_y, ht%stop_y
                    do dom_x = ht%start_x, ht%stop_x
                        call polarization_initial_time_single(tframe, &
                            dom_x, dom_y, dom_z, dx_domain, dy_domain, &
                            dz_domain, dt_fields)
                    enddo ! x
                enddo ! y
            enddo ! z

            ! Particles are saved in HDF5
            if (particle_hdf5) then
                call free_np_offset_local
                call close_particle_file_h5
            endif

            ! Check how many particles are removed
            call MPI_REDUCE(ptl_rm_local, ptl_rm_global, 1, MPI_INTEGER, &
                MPI_SUM, master, MPI_COMM_WORLD, ierr)
            if (myid == master) then
                print '("Number of removed particles is "I)', ptl_rm_global
                print '("Fraction of removed particles is "E10.3)', &
                    ptl_rm_global*1.0 / (domain%nx * domain%ny * domain%nz * domain%nppc)
            endif

            call save_particle_energization(tframe, "polarization_initial_time")
            call cpu_time(step2)
            if (myid == master) then
                print '("Time for this step = ",f6.3," seconds.")', step2 - step1
            endif
            step1 = step2
        enddo  ! Time loop

        if (is_translated_file .and. output_format == 1) then
            call close_electric_field_files
            call close_magnetic_field_files
            call close_bfield_pre_post
            call close_efield_pre_post
        endif

        deallocate(ebins, fbins, fbins_sum)

        if (is_translated_file) then
            call free_electric_fields
            call free_magnetic_fields
            call free_pre_post_bfield
            call free_pre_post_efield
        endif
        call free_emfields
        call free_bfield_magnitude
        call free_bfield_components
        call free_efield_components
    end subroutine polarization_initial_time

    !<--------------------------------------------------------------------------
    !< Calculate particle energy change due to the time-dependent part polarization
    !< drift and initial drift for particles in a single PIC MPI rank
    !<--------------------------------------------------------------------------
    subroutine polarization_initial_time_single(tindex, dom_x, dom_y, dom_z, &
            dx_domain, dy_domain, dz_domain, dt_fields)
        use picinfo, only: domain
        use topology_translate, only: ht
        use rank_index_mapping, only: index_to_rank
        use particle_module, only: ptl, calc_interp_param, calc_gyrofrequency, &
            calc_particle_energy, calc_para_perp_velocity_3d, &
            iex, jex, kex, iey, jey, key, iez, jez, kez, ibx, jbx, kbx, &
            iby, jby, kby, ibz, jbz, kbz, dx_ex, dy_ex, dz_ex, &
            dx_ey, dy_ey, dz_ey, dx_ez, dy_ez, dz_ez, dx_bx, dx_by, dx_bz, &
            dy_bx, dy_by, dy_bz, dz_bx, dz_by, dz_bz, ino, jno, kno, &
            dnx, dny, dnz, gyrof, vpara
        use interpolation_emf, only: read_emfields_single, &
            trilinear_interp_only_bx, trilinear_interp_only_by, trilinear_interp_only_bz, &
            trilinear_interp_ex, trilinear_interp_ey, trilinear_interp_ez, &
            set_emf, calc_b_norm, ex0, ey0, ez0, bx0, by0, bz0, bxn, byn, bzn,&
            absB0
        use interpolation_pre_post_bfield, only: set_bfield_magnitude, &
            set_bfield_components, trilinear_interp_bfield_components, &
            bx1_0, by1_0, bz1_0, bx2_0, by2_0, bz2_0
        use interpolation_pre_post_efield, only: set_efield_components, &
            trilinear_interp_efield_components, ex1_0, ey1_0, ez1_0, &
            ex2_0, ey2_0, ez2_0
        use file_header, only: set_v0header, pheader
        implicit none
        integer, intent(in) :: tindex, dom_x, dom_y, dom_z
        real(fp), intent(in) :: dx_domain, dy_domain, dz_domain, dt_fields
        integer :: ibin, n, iptl, nptl
        real(fp) :: x0, y0, z0, ux, uy, uz
        real(fp) :: weight, gama, ib2, idt, param
        real(fp) :: vexbx1, vexby1, vexbz1
        real(fp) :: vexbx2, vexby2, vexbz2
        real(fp) :: vpx, vpy, vpz, polar_ene
        real(fp) :: dvxdt, dvydt, dvzdt, ib2_1, ib2_2, ib_1, ib_2
        real(fp) :: dbxdt, dbydt, dbzdt, init_ene
        real(fp) :: vexbx, vexby, vexbz
        character(len=16) :: cid

        call index_to_rank(dom_x, dom_y, dom_z, domain%pic_tx, &
            domain%pic_ty, domain%pic_tz, n)
        write(cid, "(I0)") n - 1

        if (particle_hdf5) then
            call read_particle_h5(n - 1)
            nptl = np_local(n)
        else
            call read_particle_binary(tindex, species, cid)
            nptl = pheader%dim
        endif

        if (is_translated_file) then
            x0 = dx_domain * dom_x
            y0 = dy_domain * dom_y
            z0 = dz_domain * dom_z
            call set_v0header(domain%pic_nx, domain%pic_ny, domain%pic_nz, &
                x0, y0, z0, real(domain%dx), real(domain%dy), real(domain%dz))
            call set_emf(dom_x, dom_y, dom_z, domain%pic_tx, domain%pic_ty, &
                domain%pic_tz, ht%start_x, ht%start_y, ht%start_z)
            call set_bfield_magnitude(dom_x, dom_y, dom_z, domain%pic_tx, &
                domain%pic_ty, domain%pic_tz, ht%start_x, ht%start_y, ht%start_z)
            call set_bfield_components(dom_x, dom_y, dom_z, domain%pic_tx, &
                domain%pic_ty, domain%pic_tz, ht%start_x, ht%start_y, ht%start_z)
            call set_efield_components(dom_x, dom_y, dom_z, domain%pic_tx, &
                domain%pic_ty, domain%pic_tz, ht%start_x, ht%start_y, ht%start_z)
        else
            call read_emfields_single(tindex, n - 1)
        endif

        idt = 1.0 / dt_fields
        do iptl = 1, nptl, 1
            ptl = ptls(iptl)
            call calc_interp_param
            call trilinear_interp_only_bx(ibx, jbx, kbx, dx_bx, dy_bx, dz_bx)
            call trilinear_interp_only_by(iby, jby, kby, dx_by, dy_by, dz_by)
            call trilinear_interp_only_bz(ibz, jbz, kbz, dx_bz, dy_bz, dz_bz)
            call trilinear_interp_ex(iex, jex, kex, dx_ex, dy_ex, dz_ex)
            call trilinear_interp_ey(iey, jey, key, dx_ey, dy_ey, dz_ey)
            call trilinear_interp_ez(iez, jez, kez, dx_ez, dy_ez, dz_ez)
            call trilinear_interp_bfield_components(ino, jno, kno, dnx, dny, dnz)
            call trilinear_interp_efield_components(ino, jno, kno, dnx, dny, dnz)
            call calc_b_norm
            call calc_particle_energy
            call calc_para_perp_velocity_3d
            call calc_gyrofrequency

            param = 1.0 / (gyrof * absB0)
            weight = abs(ptl%q)

            ! Polarization drift
            ib_1 = 1.0 / sqrt(bx1_0**2 + by1_0**2 + bz1_0**2)
            ib_2 = 1.0 / sqrt(bx2_0**2 + by2_0**2 + bz2_0**2)
            ib2_1 = ib_1 * ib_1
            ib2_2 = ib_2 * ib_2

            ib2 = 1.0 / absB0**2

            vexbx = (ey0 * bz0 - ez0 * by0) * ib2
            vexby = (ez0 * bx0 - ex0 * bz0) * ib2
            vexbz = (ex0 * by0 - ey0 * bx0) * ib2

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
            polar_ene = weight * ptl_charge * param * (ex0 * vpx + ey0 * vpy + ez0 * vpz)

            ! Initial drift
            dbxdt = (bx2_0 * ib_2 - bx1_0 * ib_1) * idt
            dbydt = (by2_0 * ib_2 - by1_0 * ib_1) * idt
            dbzdt = (bz2_0 * ib_2 - bz1_0 * ib_1) * idt
            init_ene = ((ey0 * bz0 - ez0 * by0) * dbxdt + &
                        (ez0 * bx0 - ex0 * bz0) * dbydt + &
                        (ex0 * by0 - ey0 * bx0) * dbzdt) * ib2

            init_ene = init_ene * ptl_mass * vpara * weight

            gama = sqrt(1.0 + ptl%vx**2 + ptl%vy**2 + ptl%vz**2)

            ibin = floor((log10(gama - 1) - emin_log) / de_log)
            if (ibin > 0 .and. ibin < nbins + 1) then
                fbins(ibin+1, 1) = fbins(ibin+1, 1) + weight
                fbins(ibin+1, 2) = fbins(ibin+1, 2) + polar_ene
                fbins(ibin+1, 3) = fbins(ibin+1, 3) + init_ene
            endif
        enddo
        deallocate(ptls)
    end subroutine polarization_initial_time_single

    !<--------------------------------------------------------------------------
    !< Calculate particle energy change due to the spatial part of
    !< polarization drift and initial drift
    !<--------------------------------------------------------------------------
    subroutine polarization_initial_spatial
        use picinfo, only: domain
        use topology_translate, only: ht
        use mpi_topology, only: htg
        use pic_fields, only: init_electric_fields, init_magnetic_fields, &
            free_electric_fields, free_magnetic_fields, open_electric_field_files, &
            open_magnetic_field_files, read_electric_fields, read_magnetic_fields, &
            close_electric_field_files, close_magnetic_field_files, &
            interp_emf_node_ghost
        use interpolation_emf, only: init_emfields, free_emfields
        use interpolation_vexb, only: init_exb_drift, free_exb_drift, &
            calc_exb_drift, init_exb_derivatives, free_exb_derivatives, &
            calc_exb_derivatives, init_exb_derivatives_single, free_exb_derivatives_single
        implicit none
        integer :: dom_x, dom_y, dom_z
        integer :: tframe_pre, tframe_post
        real(fp) :: dx_domain, dy_domain, dz_domain
        real(fp) :: dt_fields

        call init_emfields
        call init_exb_derivatives_single
        if (is_translated_file) then
            call init_electric_fields(htg%nx, htg%ny, htg%nz)
            call init_magnetic_fields(htg%nx, htg%ny, htg%nz)
            call init_exb_drift(htg%nx, htg%ny, htg%nz)
            call init_exb_derivatives(htg%nx, htg%ny, htg%nz)
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
        endif

        call cpu_time(step1)
        do tframe = tstart, tend, tinterval
            if (myid == master) print*, tframe
            tp_emf = tframe / fields_interval
            call set_dists_zero
            if (is_translated_file) then
                if (output_format /= 1) then
                    ! Fields at each time step are saved in different files
                    if (tp_emf == tp1) then
                        tframe_pre = tframe
                    else
                        tframe_pre = tframe - 1
                    endif
                    if (tp_emf == tp2) then
                        tframe_post = tframe
                    else
                        tframe_post = tframe + 1
                    endif
                    call set_filepath(dir_emf)
                    call open_electric_field_files(tframe)
                    call open_magnetic_field_files(tframe)
                    call read_electric_fields(tp1)
                    call read_magnetic_fields(tp1)
                    call close_magnetic_field_files
                    call close_electric_field_files
                else
                    ! Fields at all time steps are saved in the same file
                    call read_electric_fields(tp_emf + 1)
                    call read_magnetic_fields(tp_emf + 1)
                endif
            endif  ! is_translated_file
            call interp_emf_node_ghost
            call calc_exb_drift
            call calc_exb_derivatives(htg%nx, htg%ny, htg%nz)
            dx_domain = domain%lx_de / domain%pic_tx
            dy_domain = domain%ly_de / domain%pic_ty
            dz_domain = domain%lz_de / domain%pic_tz
            ptl_rm_local = 0

            ! Particles are saved in HDF5
            if (particle_hdf5) then
                call get_np_local_vpic(tframe, species)
                call open_particle_file_h5(tframe, species)
            endif

            do dom_z = ht%start_z, ht%stop_z
                do dom_y = ht%start_y, ht%stop_y
                    do dom_x = ht%start_x, ht%stop_x
                        call polarization_initial_spatial_single(tframe, &
                            dom_x, dom_y, dom_z, dx_domain, dy_domain, &
                            dz_domain)
                    enddo ! x
                enddo ! y
            enddo ! z

            ! Particles are saved in HDF5
            if (particle_hdf5) then
                call free_np_offset_local
                call close_particle_file_h5
            endif

            ! Check how many particles are removed
            call MPI_REDUCE(ptl_rm_local, ptl_rm_global, 1, MPI_INTEGER, &
                MPI_SUM, master, MPI_COMM_WORLD, ierr)
            if (myid == master) then
                print '("Number of removed particles is "I)', ptl_rm_global
                print '("Fraction of removed particles is "E10.3)', &
                    ptl_rm_global*1.0 / (domain%nx * domain%ny * domain%nz * domain%nppc)
            endif

            call save_particle_energization(tframe, "polarization_initial_spatial")
            call cpu_time(step2)
            if (myid == master) then
                print '("Time for this step = ",f6.3," seconds.")', step2 - step1
            endif
            step1 = step2
        enddo  ! Time loop

        if (is_translated_file .and. output_format == 1) then
            call close_electric_field_files
            call close_magnetic_field_files
        endif

        deallocate(ebins, fbins, fbins_sum)

        if (is_translated_file) then
            call free_electric_fields
            call free_magnetic_fields
            call free_exb_drift
            call free_exb_derivatives
        endif
        call free_emfields
        call free_exb_derivatives_single
    end subroutine polarization_initial_spatial

    !<--------------------------------------------------------------------------
    !< Calculate particle energy change due to the spatial part polarization
    !< drift and initial drift for particles in a single PIC MPI rank
    !<--------------------------------------------------------------------------
    subroutine polarization_initial_spatial_single(tindex, dom_x, dom_y, dom_z, &
            dx_domain, dy_domain, dz_domain)
        use picinfo, only: domain
        use topology_translate, only: ht
        use rank_index_mapping, only: index_to_rank
        use particle_module, only: ptl, calc_interp_param, calc_gyrofrequency, &
            calc_particle_energy, calc_para_perp_velocity_3d, ino, jno, kno, &
            dnx, dny, dnz, gyrof, vparax, vparay, vparaz, vpara
        use interpolation_emf, only: read_emfields_single, &
            trilinear_interp_only_bx, trilinear_interp_only_by, &
            trilinear_interp_only_bz, trilinear_interp_ex, trilinear_interp_ey, &
            trilinear_interp_ez, set_emf, calc_b_norm, &
            ex0, ey0, ez0, bx0, by0, bz0, bxn, byn, bzn, absB0
        use interpolation_vexb, only: set_exb_derivatives, trilinear_interp_exb_derivatives
        use interpolation_vexb, only: dvxdx0, dvxdy0, dvxdz0, &
            dvydx0, dvydy0, dvydz0, dvzdx0, dvzdy0, dvzdz0
        use file_header, only: set_v0header, pheader
        implicit none
        integer, intent(in) :: tindex, dom_x, dom_y, dom_z
        real(fp), intent(in) :: dx_domain, dy_domain, dz_domain
        integer :: ibin, n, iptl, nptl
        real(fp) :: x0, y0, z0, ux, uy, uz
        real(fp) :: weight, gama, ib2, param
        real(fp) :: vx0, vy0, vz0
        real(fp) :: vpx, vpy, vpz, polar_ene, init_ene
        real(fp) :: dvxdt, dvydt, dvzdt
        character(len=16) :: cid

        call index_to_rank(dom_x, dom_y, dom_z, domain%pic_tx, &
            domain%pic_ty, domain%pic_tz, n)
        write(cid, "(I0)") n - 1

        if (particle_hdf5) then
            call read_particle_h5(n - 1)
            nptl = np_local(n)
        else
            call read_particle_binary(tindex, species, cid)
            nptl = pheader%dim
        endif

        if (is_translated_file) then
            x0 = dx_domain * dom_x
            y0 = dy_domain * dom_y
            z0 = dz_domain * dom_z
            call set_v0header(domain%pic_nx, domain%pic_ny, domain%pic_nz, &
                x0, y0, z0, real(domain%dx), real(domain%dy), real(domain%dz))
            call set_emf(dom_x, dom_y, dom_z, domain%pic_tx, domain%pic_ty, &
                domain%pic_tz, ht%start_x, ht%start_y, ht%start_z)
            call set_exb_derivatives(dom_x, dom_y, dom_z, domain%pic_tx, domain%pic_ty, &
                domain%pic_tz, ht%start_x, ht%start_y, ht%start_z)
        else
            call read_emfields_single(tindex, n - 1)
        endif

        do iptl = 1, nptl, 1
            ptl = ptls(iptl)
            call calc_interp_param
            call trilinear_interp_only_bx(ino, jno, kno, dnx, dny, dnz)
            call trilinear_interp_only_by(ino, jno, kno, dnx, dny, dnz)
            call trilinear_interp_only_bz(ino, jno, kno, dnx, dny, dnz)
            call trilinear_interp_ex(ino, jno, kno, dnx, dny, dnz)
            call trilinear_interp_ey(ino, jno, kno, dnx, dny, dnz)
            call trilinear_interp_ez(ino, jno, kno, dnx, dny, dnz)
            call trilinear_interp_exb_derivatives(ino, jno, kno, dnx, dny, dnz)
            call calc_b_norm
            call calc_particle_energy
            call calc_para_perp_velocity_3d
            call calc_gyrofrequency

            gama = sqrt(1.0 + ptl%vx**2 + ptl%vy**2 + ptl%vz**2)

            ! Polarization drift
            ib2 = 1.0 / absB0**2
            vx0 = vparax + (ey0 * bz0 - ez0 * by0) * ib2
            vy0 = vparay + (ez0 * bx0 - ex0 * bz0) * ib2
            vz0 = vparaz + (ex0 * by0 - ey0 * bx0) * ib2

            dvxdt = vx0 * dvxdx0 + vy0 * dvxdy0 + vz0 * dvxdz0
            dvydt = vx0 * dvydx0 + vy0 * dvydy0 + vz0 * dvydz0
            dvzdt = vx0 * dvzdx0 + vy0 * dvzdy0 + vz0 * dvzdz0

            vpx = by0 * dvzdt - bz0 * dvydt
            vpy = bz0 * dvxdt - bx0 * dvzdt
            vpz = bx0 * dvydt - by0 * dvxdt

            param = 1.0 / (gyrof * absB0)
            weight = abs(ptl%q)
            polar_ene = weight * ptl_charge * param * (ex0 * vpx + ey0 * vpy + ez0 * vpz)

            ! Initial drift
            vx0 = vx0 - vparax
            vy0 = vy0 - vparay
            vz0 = vz0 - vparaz
            dvxdt = vx0 * dvxdx0 + vy0 * dvxdy0 + vz0 * dvxdz0
            dvydt = vx0 * dvydx0 + vy0 * dvydy0 + vz0 * dvydz0
            dvzdt = vx0 * dvzdx0 + vy0 * dvzdy0 + vz0 * dvzdz0
            init_ene = -ptl_mass * vpara * weight * &
                (bxn * dvxdt + byn * dvydt + bzn * dvzdt)

            ibin = floor((log10(gama - 1) - emin_log) / de_log)
            if (ibin > 0 .and. ibin < nbins + 1) then
                fbins(ibin+1, 1) = fbins(ibin+1, 1) + weight
                fbins(ibin+1, 2) = fbins(ibin+1, 2) + polar_ene
                fbins(ibin+1, 3) = fbins(ibin+1, 3) + init_ene
            endif
        enddo
        deallocate(ptls)
    end subroutine polarization_initial_spatial_single

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
        call cli%add(switch='--para_perp', switch_ab='-pa', &
            help='Calculate energization due to parallel and perpendicular electric field', &
            required=.false., act='store_true', def='.false.', error=error)
        if (error/=0) stop
        call cli%add(switch='--comp_shear', switch_ab='-cs', &
            help='Calculate energization due to compression and shear', &
            required=.false., act='store_true', def='.false.', error=error)
        if (error/=0) stop
        call cli%add(switch='--curv_grad_para_drifts', switch_ab='-cg', &
            help='Calculate energization due to curvature, gradient and parallel drifts', &
            required=.false., act='store_true', def='.false.', error=error)
        if (error/=0) stop
        call cli%add(switch='--magnetic_moment', switch_ab='-mm', &
            help='Calculate energization due to conservation of magnetic moment', &
            required=.false., act='store_true', def='.false.', error=error)
        if (error/=0) stop
        call cli%add(switch='--polarization_initial_time', switch_ab='-pt', &
            help='Calculate energization due to polarization and initial drifts (time)', &
            required=.false., act='store_true', def='.false.', error=error)
        if (error/=0) stop
        call cli%add(switch='--polarization_initial_spatial', switch_ab='-ps', &
            help='Calculate energization due to polarization and initial drifts (spatial)', &
            required=.false., act='store_true', def='.false.', error=error)
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
        call cli%get(switch='-pa', val=calc_para_perp, error=error)
        if (error/=0) stop
        call cli%get(switch='-cs', val=calc_comp_shear, error=error)
        if (error/=0) stop
        call cli%get(switch='-cg', val=calc_curv_grad_para_drifts, error=error)
        if (error/=0) stop
        call cli%get(switch='-mm', val=calc_magnetic_moment, error=error)
        if (error/=0) stop
        call cli%get(switch='-pt', val=calc_polar_initial_time, error=error)
        if (error/=0) stop
        call cli%get(switch='-ps', val=calc_polar_initial_spatial, error=error)
        if (error/=0) stop

        if (myid == 0) then
            print '(A,A)', 'The simulation rootpath: ', trim(adjustl(rootpath))
            print '(A,L1)', 'Whether using translated fields file: ', is_translated_file
            print '(A,I0,A,I0,A,I0)', 'Min, max and interval: ', &
                tstart, ' ', tend, ' ', tinterval
            print '(A,I0)', 'Time interval for electric and magnetic fields: ', &
                fields_interval
            if (species == 'e') then
                print '(A,A)', 'Particle: electron'
            else if (species == 'h' .or. species == 'i') then
                print '(A,A)', 'Particle: ion'
            endif
            print '(A,A)', 'EMF data directory: ', trim(dir_emf)
            print '(A,A)', 'Hydro data directory: ', trim(dir_hydro)
            if (separated_pre_post) then
                print '(A)', 'Fields at previous and next time steps are saved separately'
            endif
            if (particle_hdf5) then
                print '(A)', 'Particles are saved in HDF5 format'
            endif
            if (calc_para_perp) then
                print '(A)', 'Calculate energization due to parallel and perpendicular electric field'
            endif
            if (calc_comp_shear) then
                print '(A)', 'Calculate energization due to compression and shear'
            endif
            if (calc_curv_grad_para_drifts) then
                print '(A)', 'Calculate energization due to curvature, gradient, and parallel drifts'
            endif
            if (calc_magnetic_moment) then
                print '(A)', 'Calculate energization due to conservation of magnetic moment'
            endif
            if (calc_polar_initial_time) then
                print '(A)', 'Calculate energization due to polarization and inital drifts (time)'
            endif
            if (calc_polar_initial_spatial) then
                print '(A)', 'Calculate energization due to polarization and inital drifts (spatial)'
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
        integer :: datatype_id, error
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

        call open_hdf5_serial(fname, groupname, file_id, group_id)
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

end program particle_energization
