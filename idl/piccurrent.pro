pro piccurrent
    common picinfo
    common jdote_data
    print, 'Start plotting currents'
    PICSimulationInformation
;    setConstants
;    jqnudote, 'e'
;    jqnudote, 'i'
;    econversion, 'e'
;    econversion, 'i'
;    econversionDrifts, 'e'
;    econversionDrifts, 'i'
;    energiespic
;    EnergyChangeRate

    ;ReadJdotE, ntf, dtf, 0, 'e', dtwpe, dtwci, 0
    ;maxEc_e = max(jcpara_dote, tindexMaxEc2_e)
    ;maxEc_e = max(jcpara_dote[0:tindexMaxEc2_e-10], tindexMaxEc1_e)
    ;print, "Two peaks of electron energy gain (1/omega_ci)", $
    ;    tindexMaxEc1_e, tindexMaxEc2_e, $
    ;    tf(tindexMaxEc1_e), tf(tindexMaxEc2_e)
    ;print, tf(40), tf(480)
;    maxEc_i = max(dkei, tindexMaxEc2_i)
;    maxEc_i = max(dkei[0:tindexMaxEc2_i-10], tindexMaxEc1_i)
;    print, "Two peaks of ion energy gain (1/omega_ci)", $
;        te(tindexMaxEc1_i), te(tindexMaxEc2_i)
;
;    FieldsImage, 'data', 'bx', 1, $
;        0, 0, 1, 0, 0, [-1.0, 1.0], im1, 1, 50, 0, 0
;    FieldsImage, 'data1', 'jcparay00_i', tindexMaxEc1_e, $
;        1, 0, 1, 1, 0, [-0.2, 0.2], im1, 0, 50, 0, 0
;    FieldsImage, 'data1', 'jagy_dote00_i', tindexMaxEc1_i, $
;        1, 0, 0, 0, 0, [-0.001, 0.001], im1, 50, 0, 0
;    FieldsImage, 'data', 'jy', tindexMaxEc1_e, $
;        0, 0, 0, 0, 0, [-0.5, 0.5], im1, 1, 50, 0, 0
;    FieldsImage, 'data', 'jy', tindexMaxEc1_e, $
;        0, 0, 0, 1, 0, [-0.3, 0.3], im1, 1, 50, 0, 0
;    FieldsImage, 'data1', 'jexbz00_e', tindexMaxEc1_e, $
;        0, 0, 0, 1, 0, [-0.3, 0.3], im1, 1, 50, 0, 0
;    FieldsImage, 'data3', 'ene_curv00', tindexMaxEc1_e+45, $
;        0, 39, 1, 1, 1, [-2, 1], im1, 1, 50, 0, 0
;    FieldsImage, 'data1', 'phi_para', 39, $
;        0, 0, 1, 1, 0, [-0.2, 0.2], im1, 0, 50, 0, 0
;
;    for i = 10, 200 do begin
;        FieldsImage, 'data', 'ne', i, $
;            0, 33, 0, 0, 0, [-1, 1], im1, 1, 20, 1, 1
;    endfor
;    FieldsImage, 'data', 'ne', 40, $
;        0, 33, 1, 0, 1, [-2, 1], im1, 0, 20, 0, 0

    FieldsImage, 'data1', 'phi_para', 11, $
        0, 33, 1, 0, 1, [-0.1, 0.1], im1, 0, 80, 0, 0

;    ReadDataSavePart, 'data', 'ne', 27, 2
;    for i = 100, 800, 10 do begin
;        ReadDataSavePart, 'data1', 'agyrotropy00_e', i, 2
;    endfor
;    ReadDataSavePart, 'data1', 'jcpara', 27, 2

;    FieldsImage, 'data1', 'agyrotropy00_e', 40, $
;        0, 33, 0, 0, 0, [0, 1.7], im1, 0, 30, 0, 0
;    FieldsImage, 'data1', 'curvRadius00_e', 40, $
;        0, 33, 1, 0, 0, [0, 1.7], im1, 0, 30, 0, 0

;    for i = 1, 42 do begin
;        print, i
;        ;FieldsImage, 'data', 'ey', tindexMaxEc1_e, $
;        ;    0, i, 0, 0, 0, [-0.1, 0.1], im1, 1, 50, 0, 0
;        ;FieldsImage, 'data', 'eEB05', tindexMaxEc1_e, $
;        ;    0, i, 0, 0, 0, [0, 0.8], im1, 0, 50, 0, 0
;        FieldsImage, 'data', 'absB', tindexMaxEc1_e, $
;            0, i, 0, 0, 0, [0, 2.5], im1, 0, 50, 0, 0
;    endfor
;
;    for it = 1, 200, 10 do begin
;        ;FieldsImage, 'data1', 'jcpara_dote00_e', it, $
;        ;    1, 0, 0, 1, [-0.0001, 0.0001], im1, 1, 50, 0, 0
;        FieldsImage, 'data1', 'jperp_dote00_e', it, $
;            1, 0, 0, 1, [-0.0001, 0.0001], im1, 1, 50, 0, 0
;    endfor
;
;    for ix = 1, 199 do begin
;        print, ix
;        PerpCurrent, tindexMaxEc1_e, ix, 'e', 0, 1, 1, 1, 0
;        PerpCurrent, tindexMaxEc1_e, ix, 'i', 0, 1, 1, 1, 0
;    endfor
;    PerpCurrent, tindexMaxEc2_e, 70, 'e', 0, 1, 1, 1, 0, 1
;    PerpCurrent, 50, 70, 'e', 0, 0, 0, 0, 0, 0

;        jDriftsCurrent, tindexMaxEc1_e, 70, 'e', 1, 1, 1, 1

;    plotcomp, 'e'
;    compression, 'e'
;    compression, 'i'
;    current2d, 'jdiagm', 30, 0, 1, [], 'e', im1
;    jdote2d, tindexMaxEc2, 0, [], 'e', im1
;
;    jDriftsDote2d, tindexMaxEc1_e, 0, 0, 1, 0, [], 'e', im1
;    jDriftsDote2d, tindexMaxEc2_e, 1, 0, 1, 0, [], 'e', im1
;    jDriftsDote2d, tindexMaxEc1_i, 1, 0, 1, 0, [], 'i', im1
;    jDriftsDote2d, tindexMaxEc2_i, 1, 0, 1, 0, [], 'i', im1

;    for i = 1, 20 do begin
;        jcmDote2d, i*10, 1, 1, 1, 1, [], 'e', im1
;    endfor
;    jcmDote2d, 40, 1, 0, 1, 0, [], 'e', im1

;    jParaPerpDote2d, tindexMaxEc1_e, 1, [], 'e', im1
;    jParaPerpDote2d, tindexMaxEc2_e, 1, [], 'e', im1
;    jParaPerpDote2d, tindexMaxEc1_i, 1, [], 'i', im1
;    jParaPerpDote2d, tindexMaxEc2_i, 1, [], 'i', im1

;    jDriftsDote1d, tindexMaxEc1_e, [40, 160], [50], 'e', im1

;    EnergyEfieldRho, tindexMaxEc2_e, 1, [], 'e', im1
;    ParticleDensity2dPlot, tindexMaxEc2_e, 1, [], 'e', im1
;    ParticleDensity2dPlot, 40, 1, [], 'e', im1

;    plotAgyrotropy, 40, 1, 0, 1, 0, [], 'e', im1

;    anisotropy2d, 30, [], 1, im1
;    Video_current, 'ene_curv', 481, 0
;    Video_current, 'jDriftsDotE', 481, 0
;    Video_current, 'ne', 481, 0
end

;;*******************************************************************************
;; Constants used in this script.
;; ene_norm: normalized energy.
;; ene_rate_norm: normalized energy change rate.
;;*******************************************************************************
;pro setConstants
;    common consts, ene_norm, ene_rate_norm
;    common picinfo
;    ene_norm = e_bx(0)
;    ene_rate_norm = de_bx(0)
;end

;*******************************************************************************
; Read the information for PIC simulations and calculate relavent variables or
; arrays that would be used for further data analysis.
; Output (picinfo common block):
;   mime: mass ratio of ions and electrons.
;   lx, ly, lz: the physical sizes of the simulation box in ion skin length di.
;   nx, ny, nz: the grid dimensions of the simulation box.
;   dx, dy, dz: the grid sizes of the simulation box.
;   x, y, z: the grid points of the simulation box (arrays).
;   dtwpe: dt in the inverse of the electron plasma frequency.
;   dtwce: dt in the inverse of the electron gyro-frequency.
;   dtwci: dt in the inverse of the ion gyro-frequency
;   e_interval: the energy output interval.
;   ntf: the number of fields output time points.
;   dtf: the time interval for fields output.
;   tf: time points for fields output.
;   nte: the number of energy output time points.
;   dte: the time interval for energy ouput.
;   te: time points for energy output.
;   kee: the time series of kinetic energy of electrons.
;   kei: the time series of kinetic energy of ions.
;   e_magnetic: the time series of magnetic energy.
;   e_electric: the time series of electric energy.
;   dkee: the change of electron kinetic energy in each step.
;   dkei: the change of ion kinetic energy in each step.
;   de_magnetic: the change of magnetic energy in each step.
;   de_electric: the change of electric energy in each step.
;   e_bx, e_by, e_bz: the magnetic energies in each component.
;   de_bx, de_by, de_bz: the change of magnetic energies in each step.
;*******************************************************************************
pro PICSimulationInformation
    common picinfo, mime, lx, ly, lz, nx, ny, nz, dx, dy, dz, x, y, z, $
        dtwpe, dtwce, dtwci, e_interval, ntf, dtf, tf, nte, dte, te, $
        kee, kei, e_magnetic, e_electric, dkee, dkei, de_magnetic, $
        de_electric, e_bx, e_by, e_bz, de_bx, de_by, de_bz
    ReadPICInfo ; Read PIC simulation information
    info_struct = file_info("../data/ex.gda")
    ntf = fix(info_struct.size/nx/nz/4) ; 4 for single precision float.
;    result = file_test('../fields/T.2', /directory)
;    i = 3
;    while (result eq 0) do begin
;        fname = '../fields/T.' + string(i, format='(I0)')
;        result = file_test(fname, /directory)
;        i += 1
;    endwhile
;    nsteps_out = i - 1
;    ntf = 1
;    result = 1
;    i = long(nsteps_out)
;    while (result eq 1) do begin
;        fname = '../fields/T.' + string(i, format='(I0)')
;        result = file_test(fname, /directory)
;        i += nsteps_out
;        ntf += 1
;    endwhile
;    ntf -= 1
    dte = e_interval * dtwci
    dtf = FIX(2.5/dtwci) * dtwci ; Formula from PIC code
    tf = FINDGEN(ntf) * dtf
    dx = lx / nx
    dy = ly / ny
    dz = lz / nz
    x = findgen(nx) * dx
    y = findgen(ny) * dy
    z = findgen(nz) * dz
    ReadEnergies ; Read PIC energies evolution output.
    return
end

;*******************************************************************************
; Read PIC simulation information from file.
; Output (part of picinfo common block):
;   mime: mass ratio of ions and electrons.
;   lx, ly, lz: the physical sizes of the simulation box in ion skin length di.
;   nx, ny, nz: the grid sizes of the simulation box.
;   wpe: the electron plasma frequency.
;   wce: the electron gyro-frequency.
;   wci: the ion gyro-frequency
;   e_interval: the energy output interval.
;*******************************************************************************
pro ReadPICInfo
    common picinfo
    header = strarr(1)
    fname = '../info'
    openr, lun, fname, /get_lun
    i = 1
    while (i le 6) do begin
        readf, lun, header
        ;print, header
        i = i + 1
    endwhile
    junk = "" ; A string variable
    reads, header, junk, mime, format='(A8, F12)'
    print, 'Mass ratio: ', mime
    while (i le 13) do begin
        readf, lun, header
        ;print, header
        i = i + 1
    endwhile
    reads, header, junk, lx, format='(A9, F12.6)'
    readf, lun, header
    reads, header, junk, ly, format='(A9, F12.6)'
    readf, lun, header
    reads, header, junk, lz, format='(A8, F12.6)'
    readf, lun, header
    reads, header, junk, nx, format='(A6, F12.6)'
    readf, lun, header
    reads, header, junk, ny, format='(A6, F12.6)'
    readf, lun, header
    reads, header, junk, nz, format='(A5, F12.6)'

    nx = fix(nx)
    ny = fix(ny)
    nz = fix(nz)

    i = 19
    while (i le 27) do begin
        READF, lun, header
        ;print, header
        i = i + 1
    endwhile
    reads, header, junk, dtwpe, format='(A10, F12)'
    readf, lun, header
    reads, header, junk, dtwce, format='(A10, F12)'
    readf, lun, header
    reads, header, junk, dtwci, format='(A10, F12)'
    readf, lun, header
    reads, header, junk, e_interval, format='(A20, I3)'

    free_lun, lun
    return
end

;*******************************************************************************
; Reading energies evolution results from PIC simulation.
; Output (part of picinfo common block):
;   nte: number of energy output time points.
;   te: the time array when energy output.
;   kee: the time series of kinetic energy of electrons.
;   kei: the time series of kinetic energy of ions.
;   e_magnetic: the time series of magnetic energy.
;   e_electric: the time series of electric energy.
;   dkee: the change of electron kinetic energy in each step.
;   dkei: the change of ion kinetic energy in each step.
;   de_magnetic: the change of magnetic energy in each step.
;   de_electric: the change of electric energy in each step.
;   e_bx, e_by, e_bz: the magnetic energies in each component.
;   de_bx, de_by, de_bz: the change of magnetic energies in each step.
; common block consts:
;   ene_norm: normalized energy.
;   ene_rate_norm: normalized energy change rate.
;*******************************************************************************
pro ReadEnergies
    common picinfo
    common consts, ene_norm, ene_rate_norm
    header = strarr(3)
    fname = '../energies'
    nlines = FILE_LINES(fname)
    openr, lun, fname, /get_lun
    readf, lun, header
    junk = "" ; A string variable
    dt = 0.0  ; A float variable
    reads, header(2), junk, dt, format='(A13, F12)'
    nte = nlines-3
    te = findgen(nte) * dte
    data = fltarr(9,nte)
    readf, lun, data
    free_lun, lun

    ; Energies evolution
    kei = fltarr(nte)
    kee = fltarr(nte)
    e_magnetic = fltarr(nte)
    e_electric = fltarr(nte)
    kei = reform(data(7,*)) ; kinetic energy of ions
    kee = reform(data(8,*)) ; kinetic energy of electrons
    e_electric = total(data(1:3,*), 1)
    e_magnetic = total(data(4:6,*), 1)
    e_bx = reform(data(4,*))
    e_by = reform(data(5,*))
    e_bz = reform(data(6,*))

    ; Energy changes in every output time step.
    dkei = fltarr(nte) ; ion kinetic energy change of each step
    dkee = fltarr(nte) ; electron kinetic energy change of each step
    de_magnetic = fltarr(nte)
    de_electric = fltarr(nte)
    de_bx = fltarr(nte)
    de_by = fltarr(nte)
    de_bz = fltarr(nte)
    dkee(1:nte-1) = (kee(1:nte-1)-kee(0:nte-2))/dte
    dkei(1:nte-1) = (kei(1:nte-1)-kei(0:nte-2))/dte
    de_magnetic(1:nte-1) = (e_magnetic(1:nte-1)-e_magnetic(0:nte-2))/dte
    de_electric(1:nte-1) = (e_electric(1:nte-1)-e_electric(0:nte-2))/dte
    de_bx(1:nte-1) = (e_bx(1:nte-1)-e_bx(0:nte-2))/dte
    de_by(1:nte-1) = (e_by(1:nte-1)-e_by(0:nte-2))/dte
    de_bz(1:nte-1) = (e_bz(1:nte-1)-e_bz(0:nte-2))/dte
    dkee(0) = 0.0
    dkei(0) = 0.0
    de_magnetic(0) = 0.0
    de_electric(0) = 0.0
    de_bx(0) = 0.0
    de_by(0) = 0.0
    de_bz(0) = 0.0

    ene_norm = e_bx(0)
    ene_rate_norm = max(abs(de_bx))
    print, ene_norm, ene_rate_norm

    kei = kei / ene_norm
    kee = kee / ene_norm
    e_electric = e_electric / ene_norm
    e_magnetic = e_magnetic / ene_norm
    e_bx = e_bx / ene_norm
    e_by = e_by / ene_norm
    e_bz = e_bz / ene_norm

    dkei = dkei / ene_rate_norm
    dkee = dkee / ene_rate_norm
    de_electric = de_electric / ene_rate_norm
    de_magnetic = de_magnetic / ene_rate_norm
    de_bx = de_bx / ene_rate_norm
    de_by = de_by / ene_rate_norm
    de_bz = de_bz / ene_rate_norm

    openw, lun, 'energies_pic.dat', /get_lun
    for i = 0, nte - 1 do begin
        printf, lun, format='(15F)', te(i), dkei(i), dkee(i), de_electric(i), $
            de_magnetic(i), de_bx(i), de_by(i), de_bz(i), $
            kei(i), kee(i), e_electric(i), $
            e_magnetic(i), e_bx(i), e_by(i), e_bz(i)
    endfor
    free_lun, lun

    return
end

;*******************************************************************************
; Read jdote data from file and integrate them in time.
; Input:
;   ntf: Totoal number of time points for fields output.
;   dtf: time interval for fields output.
;   iband: the id of energy band. The total is band 0.
;   species: 'e' for electrons, 'i' for ions
;   dtwpe: time step in the inverse of electron plasma frequency.
;   dtwci: time step in the inverse of ion gyro-frequency.
;   isInductive: whether the jdote is calcuated using inductive electric field.
; Output:
;   jdote_data: common block of j dot E data and their accumulation in time. 
;*******************************************************************************
pro ReadJdotE, ntf, dtf, iband, species, dtwpe, dtwci, isInductive
    common jdote_data, jcpara_dote, jcperp_dote, jmag_dote, jgrad_dote, $
        jdiagm_dote, jpolar_dote, jexb_dote, jpara_dote, jperp_dote, $
        jperp1_dote, jperp2_dote, jqnupara_dote, jqnuperp_dote, $
        jagy_dote, jtot_dote, $
        jcpara_dote_int, jcperp_dote_int, jmag_dote_int, jgrad_dote_int, $
        jdiagm_dote_int, jpolar_dote_int, jexb_dote_int, jpara_dote_int, $
        jperp_dote_int, jperp1_dote_int, jperp2_dote_int, jqnupara_dote_int, $
        jqnuperp_dote_int, jagy_dote_int, jtot_dote_int
    common consts
    nk = 15 ; number of kinds of jdoteE in different forms.
    data = fltarr(nk, ntf)
    data1 = fltarr(nk)
    if (isInductive eq 0) then begin
        openr, lun, 'data/jdote' + string(iband, FORMAT='(I2.2)') + $
            '_' + species + '.gda', /get_lun
    endif else begin
        openr, lun, 'data/jdote_in' + string(iband, FORMAT='(I2.2)') + $
            '_' + species + '.gda', /get_lun
    endelse
    for it = 0, ntf-1 do begin
        field = assoc(lun,data1)
        data1 = field(it)
        data(*,it) = data1
    endfor
    free_lun, lun

    jcpara_dote = reform(data(0,*)) * dtwpe / dtwci
    jcperp_dote = reform(data(1,*)) * dtwpe / dtwci
    jmag_dote = reform(data(2,*)) * dtwpe / dtwci
    jgrad_dote = reform(data(3,*)) * dtwpe / dtwci
    jdiagm_dote = reform(data(4,*)) * dtwpe / dtwci
    jpolar_dote = reform(data(5,*)) * dtwpe / dtwci
    jexb_dote = reform(data(6,*)) * dtwpe / dtwci
    jpara_dote = reform(data(7,*)) * dtwpe / dtwci
    jperp_dote = reform(data(8,*)) * dtwpe / dtwci
    jperp1_dote = reform(data(9,*)) * dtwpe / dtwci
    jperp2_dote = reform(data(10,*)) * dtwpe / dtwci
    jqnupara_dote = reform(data(11,*)) * dtwpe / dtwci
    jqnuperp_dote = reform(data(12,*)) * dtwpe / dtwci
    jagy_dote = reform(data(14,*)) * dtwpe / dtwci
    jtot_dote = reform(data(13,*)) * dtwpe / dtwci

    IntegrateDataInTime, jcpara_dote, ntf, dtf, jcpara_dote_int, ene_norm
    IntegrateDataInTime, jcperp_dote, ntf, dtf, jcperp_dote_int, ene_norm
    IntegrateDataInTime, jmag_dote, ntf, dtf, jmag_dote_int, ene_norm
    IntegrateDataInTime, jgrad_dote, ntf, dtf, jgrad_dote_int, ene_norm
    IntegrateDataInTime, jdiagm_dote, ntf, dtf, jdiagm_dote_int, ene_norm
    IntegrateDataInTime, jpolar_dote, ntf, dtf, jpolar_dote_int, ene_norm
    IntegrateDataInTime, jexb_dote, ntf, dtf, jexb_dote_int, ene_norm
    IntegrateDataInTime, jpara_dote, ntf, dtf, jpara_dote_int, ene_norm
    IntegrateDataInTime, jperp_dote, ntf, dtf, jperp_dote_int, ene_norm
    IntegrateDataInTime, jperp1_dote, ntf, dtf, jperp1_dote_int, ene_norm
    IntegrateDataInTime, jperp2_dote, ntf, dtf, jperp2_dote_int, ene_norm
    IntegrateDataInTime, jqnupara_dote, ntf, dtf, jqnupara_dote_int, ene_norm
    IntegrateDataInTime, jqnuperp_dote, ntf, dtf, jqnuperp_dote_int, ene_norm
    IntegrateDataInTime, jagy_dote, ntf, dtf, jagy_dote_int, ene_norm
    IntegrateDataInTime, jtot_dote, ntf, dtf, jtot_dote_int, ene_norm

    jcpara_dote = jcpara_dote / ene_rate_norm 
    jcperp_dote = jcperp_dote / ene_rate_norm
    jmag_dote   = jmag_dote   / ene_rate_norm
    jgrad_dote  = jgrad_dote  / ene_rate_norm
    jdiagm_dote = jdiagm_dote / ene_rate_norm
    jpolar_dote = jpolar_dote / ene_rate_norm
    jexb_dote   = jexb_dote   / ene_rate_norm
    jpara_dote  = jpara_dote  / ene_rate_norm
    jperp_dote  = jperp_dote  / ene_rate_norm
    jperp1_dote = jperp1_dote / ene_rate_norm
    jperp2_dote = jperp2_dote / ene_rate_norm
    jqnupara_dote = jqnupara_dote / ene_rate_norm 
    jqnuperp_dote = jqnuperp_dote / ene_rate_norm  
    jagy_dote     = jagy_dote     / ene_rate_norm  
    jtot_dote     = jtot_dote     / ene_rate_norm  

    ; Save the data to ascii files for python to use
    tf = FINDGEN(ntf) * dtf
    openw, lun, 'jdote_drifts_' + species + '.dat', /get_lun
    for i = 0, ntf - 1 do begin
        printf, lun, format='(31F)', tf(i), jcpara_dote(i), jcperp_dote(i), $
            jmag_dote(i), jgrad_dote(i), jdiagm_dote(i), jpolar_dote(i), $
            jexb_dote(i), jpara_dote(i), jperp_dote(i), jperp1_dote(i), $
            jperp2_dote(i), jqnupara_dote(i), jqnuperp_dote(i), jagy_dote(i), $
            jtot_dote(i), jcpara_dote_int(i), jcperp_dote_int(i), $
            jmag_dote_int(i), jgrad_dote_int(i), jdiagm_dote_int(i), $
            jpolar_dote_int(i), jexb_dote_int(i), jpara_dote_int(i), $
            jperp_dote_int(i), jperp1_dote_int(i), jperp2_dote_int(i), $
            jqnupara_dote_int(i), jqnuperp_dote_int(i), jagy_dote_int(i), $
            jtot_dote_int(i)
    endfor
    free_lun, lun
end

;*******************************************************************************
; Integrate of 1D time series of data in time.
; Input:
;   nt: number of time points.
;   dt: time interval.
;   data: the time series of data to integrate.
;   norm0: the normalization value for the integral.
; Ouput:
;   data_int: the integrated time series of data.
;*******************************************************************************
pro IntegrateDataInTime, data, nt, dt, data_integral, norm0
    data_integral = fltarr(nt)
    data_integral(0) = 0.0
    for i = 1, nt-1 do begin
        data_integral(i) = data_integral(i-1) + data(i)*dt
    endfor
    data_integral = data_integral / norm0
end

;*******************************************************************************
; Compare the q*n*u dot E and the actural energy change rate directly from 
; PIC simulation, where q is particle charge, n is particle number density,
; u is fluid bulk velocity for single species.
; Input:
;   species: 'e' for electron, 'i' for ion.
;*******************************************************************************
pro jqnudote, species
    common picinfo
    common jdote_data
    tmin = min(tf)
    tmax = max(tf)

    if (species eq 'e') then begin
        ke = kee
        dke = dkee
    endif else begin
        ke = kei
        dke = dkei
    endelse

    ReadJdotE, ntf, dtf, 0, species, dtwpe, dtwci, 0
    jqnu_dote = jqnupara_dote + jqnuperp_dote
    jqnu_dote_int = jqnupara_dote_int + jqnuperp_dote_int
    ymin1 = dblarr(4)
    ymin1(0) = min(jqnupara_dote)
    ymin1(1) = min(jqnuperp_dote)
    ymin1(2) = min(jqnu_dote)
    ymin1(3) = min(dke)
    ymax1 = dblarr(4)
    ymax1(0) = max(jqnupara_dote)
    ymax1(1) = max(jqnuperp_dote)
    ymax1(2) = max(jqnu_dote)
    ymax1(3) = max(dke)
    ymin = min(ymin1)
    ymax = max(ymax1)
    yrange_values, ymax, ymin ; Adjustment

    ys = 0.56
    yint = 0.36
    pos1 = [0.18,ys,0.95,ys+yint]

    fname = '$!8K_' + species + '$'
    p1 = plot(te, dke, 'k2', $
        font_size=16, xshowtext = 0, $
        ytitle='$!8dE_c!3/!8dt$', $
        xrange=[tmin, tmax], yrange=[ymin,ymax], $
        dimensions=[500,350], $
        name = fname, $
        position=pos1)
    fname = '$!16j!8_' + species + '\cdot!16E$'
    p2 = plot(tf, jqnu_dote, 'k2--', name = fname, /overplot)
    fname = '$!16j!8_' + species + '_{!9||}\cdot!16E$'
    p3 = plot(tf, jqnupara_dote, 'r2', name = fname, /overplot)
    fname = '$!16j!8_' + species + '_{!9\perp}\cdot!16E$'
    p4 = plot(tf, jqnuperp_dote, 'b2', name = fname, /overplot)

    p11 = plot([tmin, tmax], [0,0], '--', /overplot)

    leg1 = legend(target=[p1,p2,p3,p4], /auto_text_color, $
        font_size=16,position=[0.35, 0.57], transparency=100)

    ymin1(0) = min(jqnupara_dote_int)
    ymin1(1) = min(jqnuperp_dote_int)
    ymin1(2) = min(jqnu_dote_int)
    ymin1(3) = min(ke)
    ymax1 = dblarr(4)
    ymax1(0) = max(jqnupara_dote_int)
    ymax1(1) = max(jqnuperp_dote_int)
    ymax1(2) = max(jqnu_dote_int)
    ymax1(3) = max(ke)
    ymin = min(ymin1)
    ymax = max(ymax1)
    yrange_values, ymax, ymin ; Adjustment

    p1 = plot(te, ke-ke(0), 'k2', $
        font_size=16, /current, $
        xtitle='!8t$!9w_{!8ci}$', ytitle='$!8E_c$', $
        xrange=[tmin, tmax], yrange=[ymin,ymax],$
        position=[pos1(0),ys-yint,pos1(2),ys])
    p2 = plot(tf, jqnu_dote_int, 'k2--', /overplot)
    p3 = plot(tf, jqnupara_dote_int, 'r2', /overplot)
    p4 = plot(tf, jqnuperp_dote_int, 'b2', /overplot)

    p11 = plot([tmin, tmax], [0,0], '--', /overplot)

    fname = 'ec_para_perp_' + species + '.eps'
    p1.save, fname
end

;*******************************************************************************
; Plot the time evolution of EM fields, kinetic energies of electrons and ions.
;*******************************************************************************
pro energiespic
    common picinfo
    tmin = min(te)
    tmax = max(te)
    ;ene_mag = e_magnetic
    ene_mag = e_bx      ; reconnection component.
    binit = ene_mag(0)
    p1 = plot(te, e_electric*100/binit, 'g2', font_size=16, $
        xrange=[tmin, tmax], $
        dimension=[500, 350], $
        position=[0.17, 0.17, 0.95, 0.95], $
        yrange=[0, 1.1], $
        xtitle='$!8 t\omega_{ci}$', $
        ytitle = 'Energies normalized by $!8 B_x^2(0)$', $
        name='$!8 100 E^2$')
    p2 = plot(te, ene_mag/binit, 'k2', /overplot, $
        name='$!8 B_x^2$')
    p3 = plot(te, kei/binit, 'r2', /overplot, $
        name='$!8 K_i$')
    p4 = plot(te, kee/binit, 'b2', /overplot, $
        name='$!8 K_e$')
    p5 = plot(te, (e_bz-e_bz(0))/binit, 'k2--', /overplot, $
        name='$!8 B_z^2(t)-B_z^2(0)$')
    p6 = plot(te, (e_by-e_by(0))/binit, 'k2-.', /overplot, $
        name='$!8 B_y^2(t)-B_y^2(0)$')
    leg1 = legend(target=[p1,p5,p6], /auto_text_color, $
        position=[0.45, 0.6], font_size=16, transparency=100)
    leg2 = legend(target=[p2,p3,p4], /auto_text_color, $
        position=[0.55, 0.9], font_size=16, transparency=100,$
        orientation=1)
    ;print, e_magnetic/binit
    ;print, kee/binit
    ;print, kei/binit
    p1.save, 'pic_ene.eps'
    sz = size(te)
    openw, lun, 'ene_pic.dat', /get_lun
    for i = 0, sz(1)-1 do begin
        printf, lun, format='(8F)', te(i), e_electric(i), e_magnetic(i), $
            kei(i), kee(i), e_bx(i), e_by(i), e_bz(i)
    endfor
    free_lun, lun

    ;p5 = plot(te, e2+b2+kei+kee)
    print, e_magnetic(0) + e_electric(0) + kee(0) + kei(0)
    print, e_magnetic(nte-1) + e_electric(nte-1) + kee(nte-1) + kei(nte-1)
    deltaKe = kee(nte-1) - kee(0)
    deltaKi = kei(nte-1) - kei(0)
    deltaEB = e_bx(0) - e_bx(nte-1)
    print, deltaKe/deltaEB, deltaKi/deltaEB
    print, kee(-1) / kee(0), kei(-1) / kei(0)
    print, (e_bx(0) - e_bx(-1)) / e_bx(0)
end

;*******************************************************************************
; Plot the time evolution of energy change rates of EM fields, kinetic energies
; of electrons and ions.
;*******************************************************************************
pro EnergyChangeRate
    common picinfo
    tmin = min(te)
    tmax = max(te)
    p1 = plot(te, de_electric, 'g2', font_size=16, $
        xrange=[tmin, tmax], $
        dimension=[500, 350], $
        position=[0.17, 0.17, 0.95, 0.95], $
        ;yrange=[0, 1.1], $
        xtitle='$!8 t\omega_{ci}$', $
        ytitle = 'Energies normalized by $!8 B_x^2(0)$', $
        name='$!8 100 E^2$')
    p2 = plot(te, de_bx, 'k2', /overplot, $
        name='$!8 B_x^2$')
    ;p3 = plot(te, dkei, 'r2', /overplot, $
    ;    name='$!8 K_i$')
    ;p4 = plot(te, dkee, 'b2', /overplot, $
    ;    name='$!8 K_e$')
    p5 = plot(te, de_bz, 'r2--', /overplot, $
        name='$!8 B_z^2(t)-B_z^2(0)$')
    p6 = plot(te, de_by, 'b2--', /overplot, $
        name='$!8 B_y^2(t)-B_y^2(0)$')
    leg1 = legend(target=[p1,p5,p6], /auto_text_color, $
        position=[0.45, 0.6], font_size=16, transparency=100)
    leg2 = legend(target=[p2], /auto_text_color, $
        position=[0.55, 0.9], font_size=16, transparency=100,$
        orientation=1)
    ;p1.save, 'pic_ene.eps'
end

;*******************************************************************************
; Plot energy conersion and its rate.
;*******************************************************************************
pro econversion, species
    common picinfo
    common jdote_data
    ReadJdotE, ntf, dtf, 0, species, dtwpe, dtwci, 0
    if (species EQ 'e') then begin
        dke = dkee
        ke = kee
    endif else begin
        dke = dkei
        ke = kei
    endelse
    ; Calculated j dot E and its integral over time
    jtot_prime_dote = jqnupara_dote + jdiagm_dote + $
        jcpara_dote + jcperp_dote + jpolar_dote + jagy_dote
    jtot_prime_dote_int = jqnupara_dote_int + jdiagm_dote_int + $
        jcpara_dote_int + jcperp_dote_int + jpolar_dote_int + jagy_dote_int
    ;jtot_prime_dote = jqnupara_dote + jdiagm_dote + $
    ;    jcpara_dote + jcperp_dote
    ;jtot_prime_dote_int = jqnupara_dote_int + jdiagm_dote_int + $
    ;    jcpara_dote_int + jcperp_dote_int
    ; Current density related to the pressure anisotropy
    jap_dote = jcpara_dote + jcperp_dote
    jap_dote_int = jcpara_dote_int + jcperp_dote_int

    names = ['$\bf j\rm_{c}\rm\cdot\bf E$', $
        '$\bf j\rm_{c}\prime\rm\cdot\bf E$', $
        '$\bf j\rm_{g}\rm\cdot\bf E$', $
        '$!16j_{!8d}\cdot!16E$', $
        '!8$K_e$', '$!16j_{!8ap}\cdot!16E$', $
        '$!16j!8_{!9||}\cdot!16E$', '$!16j\cdot!16E$', $
        '$!16j_{!8p}\cdot!16E$', '$!16j_{!8a}\cdot!16E$']

    xmax = max(tf)
    xmin = min(tf)

    ymin1 = fltarr(9)
    ymax1 = fltarr(9)
    ymin1(*) = 0.0
    ymax1(*) = 0.0
;    ymin1(0) = min(jcpara_dote)
;    ymin1(1) = min(jcperp_dote)
;    ymin1(2) = min(jgrad_dote)
    ymin1(3) = min(jdiagm_dote)
    ymin1(4) = min(jqnupara_dote)
    ymin1(5) = min(dke)
    ymin1(6) = min(jap_dote)
    ymin1(7) = min(jtot_prime_dote)
;    ymin1(8) = min(jagy_dote)
;    ymax1(0) = max(jcpara_dote)
;    ymax1(1) = max(jcperp_dote)
;    ymax1(2) = max(jgrad_dote)
    ymax1(3) = max(jdiagm_dote)
    ymax1(4) = max(jqnupara_dote)
    ymax1(5) = max(dke)
    ymax1(6) = max(jap_dote)
    ymax1(7) = max(jtot_prime_dote)
;    ymax1(8) = min(jagy_dote)
    
    ymin = min(ymin1)
    ymax = max(ymax1)

    yrange_values, ymax, ymin

    p1 = objarr(10)

    ys = 0.56
    yint = 0.36

    pos1 = [0.18,ys,0.82,ys+yint]

;  ; Curvature drift
;    p1(0) = plot(tf, jcpara_dote, 'r2--', font_size=20,$
;        xtitle='$t\omega_{ci}$', ytitle='Rate', $
;        xshowtext = 0, $
;        xrange=[xmin, xmax], yrange=[ymin,ymax],$
;        dimensions=[800,400], name=names(0), $
;        position=pos1)
;  ; The term that is similar with curvature drift term except the P_\perp
;    p1(1) = plot(tf, jcperp_dote, /overplot, 'b2--', name=names(1))
;  ; Gradient drift term
;    p1(2) = plot(tf, jgrad_dote, /overplot, 'g2--', name=names(2))
  ; Diamagnetic drift term
    p1(3) = plot(tf, jdiagm_dote, $
        color='blue',thick=2, name=names(3), $
        ;font_name='Hershey 3',$
        font_size=16, xshowtext = 0, $
        xtitle='$t\omega_{ci}$', ytitle='$!8dE_c!3/!8dt$', $
        xrange=[xmin, xmax], yrange=[ymin,ymax],$
        dimensions=[500,350], $
        position=pos1)
;  ; Electron energy change/change rate
    p1(4) = plot(te, dke, 'k2', /overplot, name=names(4))
  ; Curvature drift term associated with anisotropy
    p1(5) = plot(tf, jap_dote, $
        /overplot, color='orange', thick=2, name=names(5))
    p20 = plot([xmin,xmax], [0,0], '--', /overplot)
  ; Parallel heating
    p1(6) = plot(tf, jqnupara_dote, 'r2', /overplot, name=names(6))
  ; Total j jdot E, except for polarization drift
    p1(7) = plot(tf, jtot_prime_dote, 'k2--', /overplot, name=names(7))
;    p1(8) = plot(tf, jpolar_dote, /overplot, 'g2', name=names(8))
;    p1(9) = plot(tf, jagy_dote, /overplot, '--', $
;    thick=2, color='orange', name=names(9))

    leg1 = legend(target=[p1(5),p1(6),p1(3),p1(7),p1(4)], $
        /auto_text_color, position=[0.97, 0.8], $
        font_size=16, transparency=100)

    jcpara_dote_int = jcpara_dote_int
    jcperp_dote_int = jcperp_dote_int
    jap_dote_int = jap_dote_int
    jgrad_dote_int = jgrad_dote_int
    jdiagm_dote_int = jdiagm_dote_int
    jtot_prime_dote_int = jtot_prime_dote_int
    jpolar_dote_int = jpolar_dote_int
    jagy_dote_int = jagy_dote_int
    jqnupara_dote_int = jqnupara_dote_int
    
;    ymin1(0) = min(jcpara_dote_int)
;    ymin1(1) = min(jcperp_dote_int)
;    ymin1(2) = min(jgrad_dote_int)
    ymin1(3) = min(jdiagm_dote_int)
    ymin1(4) = min(jqnupara_dote_int)
    ymin1(5) = min(ke-ke(0))
    ymin1(6) = min(jap_dote_int)
    ymin1(7) = min(jtot_prime_dote_int)
;    ymin1(8) = min(jagy_dote_int)
;    ymax1(0) = max(jcpara_dote_int)
;    ymax1(1) = max(jcperp_dote_int)
;    ymax1(2) = max(jgrad_dote_int)
    ymax1(3) = max(jdiagm_dote_int)
    ymax1(4) = max(jqnupara_dote_int)
    ymax1(5) = max(ke-ke(0))
    ymax1(6) = max(jap_dote_int)
    ymax1(7) = max(jtot_prime_dote_int)
;    ymax1(8) = min(jagy_dote_int)
    
    ymin = min(ymin1)
    ymax = max(ymax1)

    yrange_values, ymax, ymin

    p2 = objarr(10)

;  ; Curvature drift
;    p2(0) = plot(tf, jcpara_dote_int, 'r2--', font_size=20, /current, $
;        xtitle='$t\omega_{ci}$', ytitle='Energy Change', $
;        xrange=[xmin, xmax], yrange=[ymin,ymax],$
;        dimensions=[800,400], name=names(0), $
;        position=[pos1(0),ys-yint,pos1(2),ys])
;  ; The term that is similar with curvature drift term except the P_\perp
;    p2(1) = plot(tf, jcperp_dote_int, /overplot, 'b2--', name=names(1))
;  ; Gradient drift term
;    p2(2) = plot(tf, jgrad_dote_int, /overplot, 'g2--', name=names(2))
  ; Diamagnetic drift term
    p2(3) = plot(tf, jdiagm_dote_int, $
        color='blue',thick=2, name=names(3), $
        ;font_name='Hershey',$
        font_size=16, /current, $
        xtitle='!8t$!9w_{!8ci}$', ytitle='$!8E_c$', $
        xrange=[xmin, xmax], yrange=[ymin,ymax],$
        position=[pos1(0),ys-yint,pos1(2),ys])

;  ; Electron energy change/change rate
    p2(4) = plot(te, ke-ke(0), 'k2', /overplot, name=names(4))
  ; Curvature drift term associated with anisotropy
    p2(5) = plot(tf, jap_dote_int, $
        /overplot, color='orange', thick=2, name=names(5))
    p20 = plot([xmin,xmax], [0,0], '--', /overplot)
  ; Parallel heating
    p2(6) = plot(tf, jqnupara_dote_int, 'r2', /overplot, name=names(6))
  ; Total j jdot E, except for polarization drift
    p2(7) = plot(tf, jtot_prime_dote_int, 'k2--', /overplot, name=names(7))
;    p2(8) = plot(tf, jpolar_dote_int, /overplot, 'g2', name=names(8))
;    p2(9) = plot(tf, jagy_dote_int, /overplot, '--', $
;        thick=2, color='orange', name=names(9))
    ; Portion of energy converation due to parallel electric field
    paraE_Ec = jqnupara_dote_int(-1) / jtot_prime_dote_int(-1)
    print, paraE_Ec, 1.0-paraE_Ec, jtot_prime_dote_int(-1)

;    t1 = TEXT(0.03, 0.9, '(I)', font_size=24)

    fname = 'jdote_kinds_' + species + '.eps'
    p1(3).save, fname
end

;*******************************************************************************
; Plot energy conersion and its rate. The different terms are form different
; drifts, including curvature drift and gradient drift. The magnetization
; term is also considered.
; Input:
;   species: particle species. 'e' for electron, 'i' for ion.
;*******************************************************************************
pro econversionDrifts, species
    common picinfo
    common jdote_data
    ReadJdotE, ntf, dtf, 0, species, dtwpe, dtwci, 0
    if (species EQ 'e') then begin
        dke = dkee
        ke = kee
    endif else begin
        dke = dkei
        ke = kei
    endelse

    ;jtot_dote = jcpara_dote + jgrad_dote + jmag_dote + $
    ;    jqnupara_dote + jpolar_dote
    ;jtot_dote_int = jcpara_dote_int + jgrad_dote_int + $
    ;    jmag_dote_int + jqnupara_dote_int + jpolar_dote_int
    jtot_dote = jcpara_dote + jgrad_dote + jmag_dote + $
        jqnupara_dote + jpolar_dote + jagy_dote
    jtot_dote_int = jcpara_dote_int + jgrad_dote_int + $
        jmag_dote_int + jqnupara_dote_int + jpolar_dote_int + jagy_dote_int
    ;jtot_dote = jcpara_dote + jgrad_dote + jqnupara_dote
    ;jtot_dote_int = jcpara_dote_int + jgrad_dote_int + jqnupara_dote_int

    names = ['$!16j_{!8c}\cdot!16E$', $
        '$!16j_{!8g}\cdot!16E$', $
        '$!16j_{!8m}\cdot!16E$', $
        '$!16j!8_{!9||}\cdot!16E$', $
        '$!16j\cdot!16E$', $
        '!8$K_' + species + '$', $
        '$!16j_{!8p}\cdot!16E$']

    xmax = max(tf)
    xmin = min(tf)

    ymin1 = fltarr(7)
    ymax1 = fltarr(7)
    ymin1(*) = 0.0
    ymax1(*) = 0.0
    ymin1(0) = min(jcpara_dote)
    ymin1(1) = min(jgrad_dote)
    ymin1(2) = min(jmag_dote)
    ymin1(3) = min(jqnupara_dote)
    ymin1(4) = min(jtot_dote)
    ymin1(5) = min(dke)
    ymin1(6) = min(jpolar_dote)
    ymax1(0) = max(jcpara_dote)
    ymax1(1) = max(jgrad_dote)
    ymax1(2) = max(jmag_dote)
    ymax1(3) = max(jqnupara_dote)
    ymax1(4) = max(jtot_dote)
    ymax1(5) = max(dke)
    ymax1(6) = max(jpolar_dote_int)
    
    ymin = min(ymin1)
    ymax = max(ymax1)

    yrange_values, ymax, ymin

    p1 = objarr(10)

    ys = 0.56
    yint = 0.36

    pos1 = [0.18,ys,0.82,ys+yint]


    xmax = 1200
  ; Curvature drift
    p1(0) = plot(tf, jcpara_dote, 'r2', font_size=16,$
        ;/buffer, $
        xtitle='!8t$!9w_{!8ci}$', ytitle='$!8dE_c/dt$', $
        xshowtext = 0, $
        xrange=[xmin, xmax], yrange=[ymin,ymax],$
        dimensions=[500, 350], name=names(0), $
        position=pos1)
  ; Gradient drift term
    p1(1) = plot(tf, jgrad_dote, /overplot, 'b2', name=names(1))
  ; Magnetization term
    p1(2) = plot(tf, jmag_dote, /overplot, 'g2', name=names(2))
  ; Parallel heating
    p1(3) = plot(tf, jqnupara_dote, 'r2--', /overplot, name=names(3))
  ; Total j jdot E, except for polarization drift
    p1(4) = plot(tf, jtot_dote, 'k2--', /overplot, name=names(4))
    p1(5) = plot(te, dke, 'k2', /overplot, name=names(5))
    p1(6) = plot(te, jpolar_dote, 'g2--', /overplot, name=names(6))
    p10 = plot([xmin,xmax], [0,0], '--', /overplot)

    leg1 = legend(target=[p1(0),p1(1),p1(2),p1(6),p1(3),p1(4), p1(5)], $
        /auto_text_color, position=[0.97, 0.8], $
        font_size=16, transparency=100)

    ymin1(0) = min(jcpara_dote_int)
    ymin1(1) = min(jgrad_dote_int)
    ymin1(2) = min(jmag_dote_int)
    ymin1(3) = min(jqnupara_dote_int)
    ymin1(4) = min(jtot_dote_int)
    ymin1(5) = min(ke-ke(0))
    ymax1(0) = max(jcpara_dote_int)
    ymax1(1) = max(jgrad_dote_int)
    ymax1(2) = max(jmag_dote_int)
    ymax1(3) = max(jqnupara_dote_int)
    ymax1(4) = max(jtot_dote_int)
    ymax1(5) = max(ke-ke(0))
    
    ymin = min(ymin1)
    ymax = max(ymax1)

    yrange_values, ymax, ymin

    p2 = objarr(10)

  ; Curvature drift
    p2(0) = plot(tf, jcpara_dote_int, 'r2', /current, font_size=16,$
        xtitle='!8t$!9w_{!8ci}$', ytitle='$!8E_c$', $
        xrange=[xmin, xmax], yrange=[ymin,ymax],$
        dimensions=[500, 350], name=names(0), $
        position=[pos1(0),ys-yint,pos1(2),ys])
  ; Gradient drift term
    p2(1) = plot(tf, jgrad_dote_int, /overplot, 'b2', name=names(1))
  ; Magnetization term
    p2(2) = plot(tf, jmag_dote_int, /overplot, 'g2', name=names(2))
  ; Parallel heating
    p2(3) = plot(tf, jqnupara_dote_int, 'r2--', /overplot, name=names(3))
  ; Total j jdot E, except for polarization drift
    p2(4) = plot(tf, jtot_dote_int, 'k2--', /overplot, name=names(4))
    p2(5) = plot(te, ke-ke(0), 'k2', /overplot, name=names(5))
    p2(6) = plot(te, jpolar_dote_int, 'g2--', /overplot, name=names(6))
    p20 = plot([xmin,xmax], [0,0], '--', /overplot)

;    t1 = TEXT(0.03, 0.9, '(I)', font_size=24)

    fname = 'jdote_drifts_' + species + '.eps'
    p1(0).save, fname
end

;*******************************************************************************
; 2D color image for simulated data.
; Input:
;   dir: the directory where the data is in. It can be data, data1, data, etc.
;   qname: the variable name.
;   it: the time point ID.
;   icumulate: flag for whether to acculate the data for line plots.
;   icolor: 0 for blue-white-red color, others for the default color.
;   ifieldline: flag for whether to overplot magnetic field lines.
;   isLimits: whether to use the input limits.
;   isLog: whether to use log scale of the data.
;   lims: the limits for the data to show.
;   isSaveFigure: whether to save the figure.
;   zLength: the length of the range of z in grids number.
;   isBuffer: whether to buffer the image rather than display it.
;   isClose: whether to close the image after plotting.
; Ouput:
;   im1: the image object.
;*******************************************************************************
pro FieldsImage, dir, qname, it, icumulation, icolor, $
    islimits, ifieldline, isLog, lims, im1, isSaveFigure, $
    zLength, isBuffer, isClose
    common picinfo
    fname = '../' + dir + '/' + qname + '.gda'
    openr, lun1, fname, /get_lun
    data = fltarr(nx, nz)

    field = assoc(lun1,data)
    data = field(it)
    ;print, data(4000, *)
    PRINT, WHERE(FINITE(data, /NAN, SIGN=1))

    data = smooth(data, [15,15])

    data1 = ptr_new(data, /no_copy)

    plot2d, data1, it, lims, icumulation, icolor, $
        isLimits, ifieldline, isLog, isBuffer, im1, zLength
    fname = 'img/' + qname + '_' + $
        STRING(it, FORMAT='(I3.3)') + '.jpg'
    if (isSaveFigure eq 1) then begin
        im1.save, fname, resolution=300
    endif
    if (isClose eq 1) then begin
        im1.close
    endif
    free_lun, lun1
end

;*******************************************************************************
; Read data and resize it to make it smaller.
; Input:
;   fnames: array of file names, so it can read multiple data sets
;       and sum them together.
;   it: the time point ID.
;   rfactor: resize factor. 
;       nd grids along each dimension will be resized to 1 grid.
; Ouput:
;   rsized_data: resized data set.
;   nx1, nz1: resized data dimensiones.
;   x1, z1: reszied grids points.
;*******************************************************************************
pro readfield, fnames, it, rfactor, resized_data, nx1, nz1, x1, z1
    common picinfo
    sz = size(fnames)
    nx1 = floor(nx/rfactor)+1
    nz1 = floor(nz/rfactor)+1
    idx1 = findgen(nx1)*rfactor
    idz1 = findgen(nz1)*rfactor
    x1 = idx1 * dx
    z1 = idz1 * dz

    data = fltarr(nx, nz)
    resized_data = fltarr(nx1, nz1)
    resized_data(*,*) = 0.0
    for i = 0, sz(1)-1 do begin
        openr, lun1, fnames(i), /get_lun

        field = assoc(lun1,data)
        data = field(it)

        data1 = fltarr(nx1,nz1)
        data1 = interpolate(data, idx1, idz1, cubic=-0.5, /grid)
        resized_data = resized_data + data1
    endfor
;    zs = nz1*2/5
;    ze = nz1*3/5
    FREE_LUN, lun1
end

;*******************************************************************************
; Read the original data without resizing it.
; Input:
;   fnames: array of file names, so it can read multiple data sets
;       and sum them together.
;   it: the time point ID.
; Ouput:
;   data: the read data set.
;   nx1, nz1: resized data dimensiones.
;   x1, z1: reszied grids points.
;   zs, ze: starting and ending poing along z-direction for further plots, so
;       so it can cover ineresting areas.
;*******************************************************************************
pro readOriginalField, fnames, it, data
    common picinfo
    sz = size(fnames)

    data = fltarr(nx, nz)
    data(*,*) = 0.0
    for i = 0, sz(1)-1 do begin
        openr, lun1, fnames(i), /get_lun
        field = assoc(lun1,data)
        data = data + field(it)
    endfor
    free_lun, lun1
end

;*******************************************************************************
; Read the original current data without resizing it.
; Input:
;   cnames: array of different current terms, so it can read multiple data sets
;       and sum them together.
;   species: 'e' for electron, 'i' for ion.
;   it: the time point ID.
; Ouput:
;   data: the data from the files.
;*******************************************************************************
pro ReadOriginalCurrent, cnames, species, it, data
    common picinfo
    sz = size(cnames)
    data = FLTARR(nx, nz)
    data1= FLTARR(nx, nz)
    data(*,*) = 0.0
    for i = 0, sz(1)-1 do begin
        fname = '../data1/' + cnames(i) + '_' + species + '.gda'
        openr, lun1, fname, /get_lun

        field = assoc(lun1,data1)
        data1 = field(it)

        data = data + data1
        free_lun, lun1
    endfor
end

;*******************************************************************************
; Get slices of calculated current data along z-drection. The current due to
; different drifts are plotted separately.
;   it: the time point ID.
;   xslice: the x positions to plot schematic cutting lines.
;   species: 'e' for electron, 'i' for ion
;   isBuffer: whether to buffer the image rather than display it.
;   isSave: whether to save the image after plotting.
;   isClose: whether to close the image after plotting.
;   isMultiple: whether to plot multiple slices at the same time.
;*******************************************************************************
pro jDriftsCurrent, it, xslice, species, isBuffer, isSave, isClose, isMultiple
    common picinfo
    xs = floor(xslice/dx)
    zs = nz*3 / 7
    ze = nz*4 / 7
    if (isMultiple eq 1) then begin
        x0 = findgen(floor(max(x))-1) + 1
    endif else begin
        x0 = [xslice]
    endelse
    sz = size(x0)
    xs = intarr(sz(1))
    for i = 0, sz(1) - 1 do begin
        xs(i) = floor(x0(i)/dx)
    endfor
    ReadCurrentComponents, species, it, 'jcpara', $
        xs, zs, ze, jcparax, jcparay, jcparaz
    ReadCurrentComponents, species, it, 'jgrad', $
        xs, zs, ze, jgradx, jgrady, jgradz
    ReadCurrentComponents, species, it, 'jmag', $
        xs, zs, ze, jmagx, jmagy, jmagz
    ReadCurrentComponents, species, it, 'jpolar', $
        xs, zs, ze, jpolarx, jpolary, jpolarz
    ReadCurrentComponents, species, it, 'jagy', $
        xs, zs, ze, jagyx, jagyy, jagyz
    ReadCurrentComponents, species, it, 'jexb', $
        xs, zs, ze, jexbx, jexby, jexbz
    ReadCurrentComponents, species, it, 'jqnuperp', $
        xs, zs, ze, jqnuperpx, jqnuperpy, jqnuperpz
    jqnuperpx = jqnuperpx - jexbx
    jqnuperpy = jqnuperpy - jexby
    jqnuperpz = jqnuperpz - jexbz

    for xid = 0, sz(1) - 1 do begin
        print, 'x slice: ', xid
        nsmooth = 4
        jcparax1 = smooth(jcparax(xid, *), nsmooth)
        jcparay1 = smooth(jcparay(xid, *), nsmooth)
        jcparaz1 = smooth(jcparaz(xid, *), nsmooth)

        jgradx1 = smooth(jgradx(xid, *), nsmooth)
        jgrady1 = smooth(jgrady(xid, *), nsmooth)
        jgradz1 = smooth(jgradz(xid, *), nsmooth)

        jmagx1 = smooth(jmagx(xid, *), nsmooth)
        jmagy1 = smooth(jmagy(xid, *), nsmooth)
        jmagz1 = smooth(jmagz(xid, *), nsmooth)

        jpolarx1 = smooth(jpolarx(xid, *), nsmooth)
        jpolary1 = smooth(jpolary(xid, *), nsmooth)
        jpolarz1 = smooth(jpolarz(xid, *), nsmooth)

        jagyx1 = smooth(jagyx(xid, *), nsmooth)
        jagyy1 = smooth(jagyy(xid, *), nsmooth)
        jagyz1 = smooth(jagyz(xid, *), nsmooth)

        jqnuperpx1 = smooth(jqnuperpx(xid, *), nsmooth)
        jqnuperpy1 = smooth(jqnuperpy(xid, *), nsmooth)
        jqnuperpz1 = smooth(jqnuperpz(xid, *), nsmooth)
        folderName = 'img_jdrifts_slices/'

        fname = folderName + 'jdrifts' + STRING(it, FORMAT='(I3.3)') + '_' +$
            STRING(x0(xid), FORMAT='(I4.4)') + '_' + species + '.eps'
        PlotJdriftsCurrent, jcparax1, jcparay1, jcparaz1, jgradx1, jgrady1, jgradz1, $
            jmagx1, jmagy1, jmagz1, jpolarx1, jpolary1, jpolarz1, jagyx1, jagyy1, jagyz1, $
            jqnuperpx1, jqnuperpy1, jqnuperpz1, zs, ze, fname, isBuffer, isSave, isClose
    endfor
end

;*******************************************************************************
; Plot slices of perpendicular current from different drifts.
; Args:
;   jperpx, jperpy, jperpz: 3 current components from model.
;   jqnuperpx, jqnuperpy, jqnuperpz: 3 current components from simulation.
;   fname: the filename to save the plot.
;   zs, ze: the starting and stopping indices in z directions.
;   isBuffer: whether to buffer the image rather than display it.
;   isSave: whether to save the image after plotting.
;   isClose: whether to close the image after plotting.
;*******************************************************************************
pro PlotJdriftsCurrent, jcparax, jcparay, jcparaz, jgradx, jgrady, jgradz, $
    jmagx, jmagy, jmagz, jpolarx, jpolary, jpolarz, jagyx, jagyy, jagyz, $
    jqnuperpx, jqnuperpy, jqnuperpz, zs, ze, fname, isBuffer, isSave, isClose

    common picinfo

    xmin = z(zs)
    xmax = z(ze)

    xs = 0.12
    xe = 0.95
    hei = 0.28
    ys = 0.7
    ye = ys + hei

    if (isBuffer eq 1) then begin
        p11 = PLOT(z(zs:ze), jcparax, 'r2', font_size=16, $
            dimensions=[640, 512], /buffer, name='$!8j_c$')
    endif else begin
        p11 = PLOT(z(zs:ze), jcparax, 'r2', font_size=16, $
            dimensions=[640, 512], name='$!8j_c$')
    endelse
    p11.position=[xs, ys, xe, ye]
    p11.xshowtext = 0
    p11.xtitle = '!8z ($d_i$)'
    p11.xrange=[xmin,xmax]
    p12 = PLOT(z(zs:ze), jgradx, 'b2', /overplot, name='$!8j_g$')
    p13 = PLOT(z(zs:ze), jmagx, 'g2', /overplot, name='$!8j_m$')
    p14 = PLOT(z(zs:ze), jpolarx, 'g2--', /overplot, name='$!8j_p$')
    p15 = PLOT(z(zs:ze), jagyx, 'b2', /overplot, name='$!8j_a$')
    p16 = PLOT(z(zs:ze), jqnuperpx, 'k2', /overplot, name='$!8j_\perp$')

    leg1 = legend(target=[p11,p12,p13,p14,p15,p16], font_size=16, $
        position=[xe-0.06,ys+0.06], /auto_text_color)

    ys = ys - hei
    ye = ys + hei
    p21 = PLOT(z(zs:ze), jcparay, 'r2', /current, font_size=16, $
        position=[xs, ys, xe, ye], xshowtext = 0, $
        xtitle = 'z ($d_i$)', $
        xrange=[xmin,xmax])
    p22 = PLOT(z(zs:ze), jgrady, 'b2', /overplot)
    p23 = PLOT(z(zs:ze), jmagy, 'g2', /overplot)
    p24 = PLOT(z(zs:ze), jpolary, 'g2--', /overplot)
    p25 = PLOT(z(zs:ze), jagyy, 'b2', /overplot)
    p26 = PLOT(z(zs:ze), jqnuperpy, 'k2', /overplot)

    ys = ys - hei
    ye = ys + hei
    p31 = PLOT(z(zs:ze), jcparaz, 'r2', /current, font_size=16, $
       position=[xs, ys, xe, ye], $
       xtitle = '!8z !3($!8d_i$!3)', $
       xrange=[xmin,xmax])
    p32 = PLOT(z(zs:ze), jgradz, 'b2', /overplot)
    p33 = PLOT(z(zs:ze), jmagz, 'g2', /overplot)
    p34 = PLOT(z(zs:ze), jpolarz, 'g2--', /overplot)
    p35 = PLOT(z(zs:ze), jagyz, 'b2', /overplot)
    p36 = PLOT(z(zs:ze), jqnuperpz, 'k2', /overplot)

    t1 = text(xs+0.02, ys+hei-0.07, '($!8z$!3)', font_size=20)
    t2 = text(xs+0.02, ys+hei*2-0.07, '($!8y$!3)', font_size=20)
    t3 = text(xs+0.02, ys+hei*3-0.07, '($!8x$!3)', font_size=20)
    if (isSave eq 1) then begin
        p11.save, fname
    endif
    if (isClose eq 1) then begin
        p11.close
    endif
end


;*******************************************************************************
; Read three componets of current due to different terms.
; Args:
;   species: 'e' for electron, 'i' for ion
;   it: the time point ID.
;   varName: variable name.
;   jx, jy, jz: the tree components of current.
;   xpos: the x position in grid number.
;   zs, ze: the starting and stoping grid along z-direction.
;*******************************************************************************
pro ReadCurrentComponents, species, it, varName, xpos, zs, ze, jx, jy, jz
    fname = varName + 'x00'
    sz = size(xpos)
    ReadOriginalCurrent, [fname], species, it, data1
    jx = reform(data1(xpos,zs:ze), sz(1), ze-zs+1)
    fname = varName + 'y00'
    ReadOriginalCurrent, [fname], species, it, data1
    jy = reform(data1(xpos,zs:ze), sz(1), ze-zs+1)
    fname = varName + 'z00'
    ReadOriginalCurrent, [fname], species, it, data1
    jz = reform(data1(xpos,zs:ze), sz(1), ze-zs+1)
end

;*******************************************************************************
; Get slices of calculated current data along z-direction. They are then
; compared with electric current calculated using q*n*u.
; Input:
;   it: the time point ID.
;   xslice: the x positions to plot schematic cutting lines.
;   species: 'e' for electron, 'i' for ion
;   isBuffer: whether to buffer the image rather than display it.
;   isSave: whether to save the image after plotting.
;   isClose: whether to close the image after plotting.
;   isPlot2D: whether to plot 2D image.
;   isJpolar: whether including current due to polarization drift.
;   isMultiple: whether to plot multiple slices at the same time.
;*******************************************************************************
PRO PerpCurrent, it, xslice, species, isJpolar, isBuffer, isSave, isClose, $
        isPlot2D, isMultiple
    common picinfo
    xs = floor(xslice/dx)
    zs = nz*3 / 7
    ze = nz*4 / 7
    if (isMultiple eq 1) then begin
        x0 = findgen(floor(max(x))-1) + 1
    endif else begin
        x0 = [xslice]
    endelse
    sz = size(x0)
    xs = intarr(sz(1))
    for i = 0, sz(1) - 1 do begin
        xs(i) = floor(x0(i)/dx)
    endfor

    ReadCurrentComponents, species, it, 'jperp1', $
        xs, zs, ze, jperpx, jperpy, jperpz
    ReadCurrentComponents, species, it, 'jqnuperp', $
        xs, zs, ze, jqnuperpx, jqnuperpy, jqnuperpz
    ReadCurrentComponents, species, it, 'jexb', $
        xs, zs, ze, jexbx, jexby, jexbz
    ReadCurrentComponents, species, it, 'jagy', $
        xs, zs, ze, jagyx, jagyy, jagyz

    fname = '../data/absB.gda'
    readOriginalField, [fname], it, absB
    btot = reform(absB(xs, zs:ze))

    jperpx = jperpx - jexbx
    jperpy = jperpy - jexby
    jperpz = jperpz - jexbz

    if (isJpolar eq 0) then begin
        ; exclude the current due to polarization drift
        ReadOriginalCurrent, ['jpolarx00'], species, it, data1
        jpolarx = reform(data1(xs,zs:ze))
        ReadOriginalCurrent, ['jpolary00'], species, it, data1
        jpolary = reform(data1(xs,zs:ze))
        ReadOriginalCurrent, ['jpolarz00'], species, it, data1
        jpolarz = reform(data1(xs,zs:ze))
        jperpx = jperpx - jpolarx
        jperpy = jperpy - jpolary
        jperpz = jperpz - jpolarz
    endif

    jqnuperpx = jqnuperpx - jexbx
    jqnuperpy = jqnuperpy - jexby
    jqnuperpz = jqnuperpz - jexbz

    if (isJpolar eq 1) then begin
        folderName = 'img_jperp_no_jpolar/'
    endif else begin
        folderName = 'img_jperp/'
    endelse

    for xid = 0, sz(1) - 1 do begin
        print, 'x slice: ', xid
        nsmooth = 4
        jperpx1 = smooth(jperpx(xid, *), nsmooth)
        jperpy1 = smooth(jperpy(xid, *), nsmooth)
        jperpz1 = smooth(jperpz(xid, *), nsmooth)
        jagyx1 = smooth(jagyx(xid, *), nsmooth)
        jagyy1 = smooth(jagyy(xid, *), nsmooth)
        jagyz1 = smooth(jagyz(xid, *), nsmooth)
        jqnuperpx1 = smooth(jqnuperpx(xid, *), nsmooth)
        jqnuperpy1 = smooth(jqnuperpy(xid, *), nsmooth)
        jqnuperpz1 = smooth(jqnuperpz(xid, *), nsmooth)
        btot1 = smooth(btot(xid, *), nsmooth)

        fname = folderName + 'jperp' + STRING(it, FORMAT='(I3.3)') + '_' +$
            STRING(xid, FORMAT='(I4.4)') + '_' + species + '_1.eps'
        PlotPerpCurrent, jperpx1, jperpy1, jperpz1, $
            jqnuperpx1, jqnuperpy1, jqnuperpz1, btot1, zs, ze, fname, $
            isBuffer, isSave, isClose

        ; Add the current due to agyrotropic pressure.
        jperpx1 = jperpx1 + jagyx1
        jperpy1 = jperpy1 + jagyy1
        jperpz1 = jperpz1 + jagyz1
        fname = folderName + 'jperp' + STRING(it, FORMAT='(I3.3)') + '_' +$
            STRING(xid, FORMAT='(I4.4)') + '_' + species + '_2.eps'
        PlotPerpCurrent, jperpx1, jperpy1, jperpz1, $
            jqnuperpx1, jqnuperpy1, jqnuperpz1, btot1, zs, ze, fname, $
            isBuffer, isSave, isClose

        if (isPlot2D eq 1) then begin
            FieldsImage, 'data', 'jy', it, $
                0, 0, 0, 1, 0, [-0.5, 0.5], im1, 0, 30, isBuffer, 0
            ;im1.yrange=[z(zs), z(ze)]
            yrange = im1.yrange
            p1 = plot([x(xs), x(xs)], [yrange(0), yrange(1)], $
                color='m', linestyle='--', thick=2, /overplot)
            if (isSave eq 1) then begin
                fname = folderName + 'jy' + STRING(it, FORMAT='(I3.3)') + '_' +$
                    STRING(xid, FORMAT='(I4.4)') + '.jpg'
                im1.save, fname
            endif
            if (isClose eq 1) then begin
                im1.close
            endif
        endif
    endfor
END

;*******************************************************************************
; Plot perpendicular current from simulations and model.
; Args:
;   jperpx, jperpy, jperpz: 3 current components from model.
;   jqnuperpx, jqnuperpy, jqnuperpz: 3 current components from simulation.
;   fname: the filename to save the plot.
;   zs, ze: the starting and stopping indices in z directions.
;   isBuffer: whether to buffer the image rather than display it.
;   isSave: whether to save the image after plotting.
;   isClose: whether to close the image after plotting.
;*******************************************************************************
pro PlotPerpCurrent, jperpx, jperpy, jperpz, jqnuperpx, jqnuperpy, jqnuperpz, $
    btot, zs, ze, fname, isBuffer, isSave, isClose

    common picinfo

    ymax1 = fltarr(2)
    ymin1 = fltarr(2)
    xmin = z(zs)
    xmax = z(ze)

    xs = 0.12
    xe = 0.95
    hei = 0.21
    ys = 0.77
    ye = ys + hei
    ymax1(0) = max(jperpx)
    ymax1(1) = max(jqnuperpx)
    ymin1(0) = min(jperpx)
    ymin1(1) = min(jqnuperpx)
    ymax = max(ymax1)
    ymin = min(ymin1)
    yrange_values, ymax, ymin

    if (isBuffer eq 1) then begin
        p11 = PLOT(z(zs:ze), jperpx, 'r2', font_size=16, $
            dimensions=[640, 512], /buffer)
    endif else begin
        p11 = PLOT(z(zs:ze), jperpx, 'r2', font_size=16, $
            dimensions=[640, 512])
    endelse
    p11.position=[xs, ys, xe, ye]
    p11.xshowtext = 0
    p11.xtitle = '!8z ($d_i$)'
    p11.xrange=[xmin,xmax]
    p11.yrange=[ymin,ymax]
    p11.name='Model'
    p21 = PLOT(z(zs:ze), jqnuperpx, 'b2', $
        /overplot, name='Simulation')
    leg1 = legend(target=[p11,p21], font_size=16, $
        position=[xe-0.06,ys+0.06], /auto_text_color)

    ys = ys - hei
    ye = ys + hei
    ymax1(0) = max(jperpy)
    ymax1(1) = max(jqnuperpy)
    ymin1(0) = min(jperpy)
    ymin1(1) = min(jqnuperpy)
    ymax = max(ymax1)
    ymin = min(ymin1)
    yrange_values, ymax, ymin
    p12 = PLOT(z(zs:ze), jperpy, 'r2', /current, font_size=16, $
        position=[xs, ys, xe, ye], $
        xshowtext = 0, $
        xtitle = 'z ($d_i$)', $
        xrange=[xmin,xmax], yrange=[ymin,ymax])
    p22 = PLOT(z(zs:ze), jqnuperpy, 'b2', /overplot)

    ys = ys - hei
    ye = ys + hei
    ymax1(0) = max(jperpz)
    ymax1(1) = max(jqnuperpz)
    ymin1(0) = min(jperpz)
    ymin1(1) = min(jqnuperpz)
    ymax = max(ymax1)
    ymin = min(ymin1)
    yrange_values, ymax, ymin
    p13 = PLOT(z(zs:ze), jperpz, 'r2', /current, font_size=16, $
        position=[xs, ys, xe, ye], $
        xshowtext = 0, $
        xrange=[xmin,xmax], yrange=[ymin,ymax])
    p23 = PLOT(z(zs:ze), jqnuperpz, 'b2', /overplot)

    t1 = text(xs+0.02, ys+hei-0.07, '($!8z$!3)', font_size=20)
    t2 = text(xs+0.02, ys+hei*2-0.07, '($!8y$!3)', font_size=20)
    t3 = text(xs+0.02, ys+hei*3-0.07, '($!8x$!3)', font_size=20)

    ys = ys - hei
    ye = ys + hei
    ymax1(0) = max(btot)
    ymax1(1) = max(btot)
    ymin1(0) = min(btot)
    ymin1(1) = min(btot)
    ymax = max(ymax1)
    ymin = min(ymin1)
    yrange_values, ymax, ymin
    p14 = PLOT(z(zs:ze), btot, 'k2', /current, font_size=16, $
        position=[xs, ys, xe, ye], $
        xtitle = '!8z ($d_i$)', $
        xrange=[xmin,xmax], yrange=[ymin,ymax])
    t4 = text(xs+0.02, ys+0.03, '!3|$!8B$!3|', font_size=20)

    ;p11.save, fname, resolution=300
    if (isSave eq 1) then begin
        p11.save, fname
    endif
    if (isClose eq 1) then begin
        p11.close
    endif
end

;*******************************************************************************
; 2D image plot of magnetic field amplitude, pressure anisotropy and firehose
; parameters.
;*******************************************************************************
PRO anisotropy2d, it, xslice, ifieldline, im1
    common picinfo
    lims1 = [0.15,2.5]
    lims2 = [0.38,7.0]
    lims3 = [0.1,5.0]

    nd = 4.0 ; Interpolate grid size
    iheight = 350
    ys = 0.70
    hei = 0.27
    xs = 0.15
    xe = 0.82
    jname = '$j_{'

    fpath_b = '../data/'
    fpath = '../data1/'
  
  ;  |B|
    qname = ['absB.gda']
    fname = fpath_b + qname
    readfield, fname, it, nd, data1, nx1, nz1, x1, z1
    data1 = smooth(data1, 9)
    ratio = nx1/(ze-zs)/3.6
    im1 = IMAGE(data1(*,zs:ze), x1, z1(zs:ze),  $
        aspect_ratio=ratio,$
        font_size=16, xtitle='!8x ($d_i$)', ytitle='!8z ($d_i$)', $
        position=[xs,ys,xe,ys+hei], $
        dimensions=[500, iheight], $
        xshowtext = 0, $
        ;ytickvalues=[200, 400, 600, 800, 1000], $
        ;max_value=5.0, min_value=0.5, $
        ;position=[0.07,0.53,0.95,0.96],$
        rgb_table=33, axis_style=2, interpolate=1)
    maxdata = max(data1)
    mindata = min(data1)
    print, 'Maximum', maxdata
    print, 'Minimum', mindata
    im1.max_value = lims1(1)
    im1.min_value = lims1(0)
    pos1 = im1.position
    ;print, pos1
    CB1 = COLORBAR(TARGET=im1,ORIENTATION=1,$
        position=[xe+0.01,pos1(1),xe+0.03,pos1(3)])
    CB1.TEXTPOS = 1
    CB1.TICKDIR = 1
    CB1.FONT_SIZE = 12
    CB1.TITLE='$|B|$'

    if (ifieldline eq 1) then begin
        zs1 = nz/4
        ze1 = nz*3/4
        fname = '../data/Ay.gda'
        OPENR, lun1, fname, /get_lun
        data = FLTARR(nx, nz)

        field = assoc(lun1,data)
        data = field(it)
        cn1 = contour(data(*,zs1:ze1), x, z(zs1:ze1), /overplot, $
            c_label_show=0, color='black')
    endif

    yrmax = pos1(3)

  ; electron Pressure anisotropy 
    ys = ys-hei
    qname = ['aniso00_e.gda']
    fname = fpath + qname
    readfield, fname, it, nd, data1, nx1, nz1, x1, z1
    data1 = smooth(data1, 9)
    ;ratio = nx1/(ze-zs)/2.0
    lims2 = [-1.0, 1.0]
    BlueWhiteRed, rgbtable, lims2
    im1 = IMAGE(ALOG10(data1(*,zs:ze)), x1, z1(zs:ze), /current, $
        aspect_ratio=ratio,$
        font_size=16, xtitle='!8x ($d_i$)', ytitle='!8z ($d_i$)', $
        position=[xs,ys,xe,ys+hei], $
        xshowtext = 0, $
        ;ytickvalues=[200, 400, 600, 800, 1000], $
        ;max_value=5.0, min_value=0.5, $
        ;position=[0.07,0.53,0.95,0.96],$
        rgb_table=rgbtable, axis_style=2, interpolate=1)
    pos1 = im1.position
    maxdata = max(data1)
    mindata = min(data1)
    print, 'Maximum', maxdata
    print, 'Minimum', mindata
    im1.max_value = lims2(1)
    im1.min_value = lims2(0)
    CB1 = COLORBAR(TARGET=im1,ORIENTATION=1,$
        position=[xe+0.01,pos1(1),xe+0.03,pos1(3)])
    ticken = cb1.tickname
    CB1.tickvalues = FLOAT(ticken)
    CB1.tickname = STRING(FLOAT(ticken), format='(F4.1)')
    CB1.TEXTPOS = 1
    CB1.TICKDIR = 1
    CB1.FONT_SIZE = 12
    CB1.TITLE='$log(!8P_{!8e\parallel}/!8P_{!8e\perp}!5)$'

    if (ifieldline eq 1) then begin
        cn1 = contour(data(*,zs1:ze1), x, z(zs1:ze1), /overplot, $
            c_label_show=0, color='black')
    endif
;
  ; ion pressure anisotropy
    ys = ys-hei
    qname = ['aniso00_i.gda']
    fname = fpath + qname
    readfield, fname, it, nd, data1, nx1, nz1, x1, z1
    data1 = smooth(data1, 9)
    ;ratio = nx1/(ze-zs)/2.0
    lims3 = [-1.0, 1.0]
    BlueWhiteRed, rgbtable, lims3
    im1 = IMAGE(ALOG10(data1(*,zs:ze)), x1, z1(zs:ze), /current, $
        aspect_ratio=ratio,$
        font_size=16, xtitle ='!8x ($d_i$)', ytitle='!8z ($d_i$)', $
        position=[xs,ys,xe,ys+hei], $
        ;xshowtext = 0, $
        ;ytickvalues=[200, 400, 600, 800, 1000], $
        ;max_value=5.0, min_value=0.5, $
        ;position=[0.07,0.53,0.95,0.96],$
        rgb_table=rgbtable, axis_style=2, interpolate=1)
    maxdata = max(data1)
    mindata = min(data1)
    print, 'Maximum', maxdata
    print, 'Minimum', mindata
    im1.max_value = lims3(1)
    im1.min_value = lims3(0)
    pos1 = im1.position
    ;print, pos1
    CB1 = COLORBAR(TARGET=im1,ORIENTATION=1,$
        position=[xe+0.01,pos1(1),xe+0.03,pos1(3)])
    CB1.TICKVALUES = [-1.0, -0.5, 0.0, 0.5]
    CB1.TEXTPOS = 1
    CB1.TICKDIR = 1
    CB1.FONT_SIZE = 12
    CB1.TITLE='$log(!8P_{!8i\parallel}/!8P_{!8i\perp})$'

    if (ifieldline eq 1) then begin
        cn1 = contour(data(*,zs1:ze1), x, z(zs1:ze1), /overplot, $
            c_label_show=0, color='black')
    endif

    yrmin = pos1(1)

    sz = size(xslice)
    FOR i = 0, sz(1)-1 DO BEGIN
        p1 = PLOT([xslice(i),xslice(i)],[1.0,2.0],'k--',/current,$
            xrange=[x1(0),x1(nx1-1)], yrange=[1.0,2.0],$
            position=[pos1(0),yrmin,pos1(2),yrmax])
        ax = p1.axes
        ax[0].HIDE=1
        ax[1].HIDE=1
        ax[2].HIDE=1
        ax[3].HIDE=1
    ENDFOR

    fname = 'anisotropy2d' + STRING(it, FORMAT='(I2.2)') + '.jpg'
    im1.save, fname, resolution=300
END

;*******************************************************************************
; 2D plot for 3 electric current components.
; Input:
;   qname: variable name.
;   it: time point ID.
;   ipic: flag for whether the original current from pic simulation.
;   ifieldline: flag for whether to plot B field lines.
;   xslice: x positions to plot vertical cut dashed lines.
;   species: 'e' for electron, 'i' for ion.
;   im1: image object for handler.
;*******************************************************************************
PRO current2d, qname, it, ipic, ifieldline, xslice, species, im1
    common picinfo
;    lims1 = [-0.2,0.2]
;    lims2 = [-0.3,0.3]
;    lims3 = [-0.4,0.4]
    lims1 = [-0.1,0.1]
    lims2 = [-0.1,0.1]
    lims3 = [-0.1,0.1]

    nd = 2.0 ; Interpolate grid size
    iheight = 350
    ys = 0.70
    hei = 0.27
    xs = 0.15
    xe = 0.76
    jname = '$!16j_{!8'
  
  ; x-component
    IF (ipic EQ 0) THEN BEGIN
        fname = '../data1/' + qname + 'x00_' + species + '.gda'
    ENDIF ELSE BEGIN
        fname = '../data/' + qname + 'x.gda'
    ENDELSE
    readfield, [fname], it, nd, data1, nx1, nz1, x1, z1
    data1 = smooth(data1, 9)
    ratio = nx1*hei/(ze-zs)/(xe-xs)/1.5
    BlueWhiteRed, rgbtable, lims1
    im1 = IMAGE(data1(*,zs:ze), x1, z1(zs:ze),  $
        aspect_ratio=ratio,$
        font_size=16, xtitle='x ($d_i$)', ytitle='!8z ($d_i$)', $
        position=[xs,ys,xe,ys+hei], $
        dimensions=[500, iheight], $
        xshowtext = 0, $
        ;ytickvalues=[200, 400, 600, 800, 1000], $
        ;max_value=5.0, min_value=0.5, $
        ;position=[0.07,0.53,0.95,0.96],$
        rgb_table=rgbtable, axis_style=2, interpolate=1)
    maxdata = max(data1)
    mindata = min(data1)
    print, 'Maximum', maxdata
    print, 'Minimum', mindata
    im1.max_value = lims1(1)
    im1.min_value = lims1(0)
    pos1 = im1.position
;    ;print, pos1
;    CB1 = COLORBAR(TARGET=im1,ORIENTATION=1,$
;        position=[xe+0.01,pos1(1),xe+0.03,pos1(3)])
;    CB1.TEXTPOS = 1
;    CB1.TICKDIR = 1
;    CB1.FONT_SIZE = 16
;    CB1.TITLE=jname + 'x}$'
;
    if (ifieldline eq 1) then begin
        zs1 = nz/4
        ze1 = nz*3/4
        fname = '../data/Ay.gda'
        OPENR, lun1, fname, /get_lun
        data = FLTARR(nx, nz)

        field = assoc(lun1,data)
        data = field(it)
        cn1 = contour(data(*,zs1:ze1), x, z(zs1:ze1), /overplot, $
            c_label_show=0, color='black')
    endif

    yrmax = pos1(3)

  ; y-component
    ys = ys-hei
    IF (ipic EQ 0) THEN BEGIN
        fname = '../data1/' + qname + 'y00_' + species + '.gda'
    ENDIF ELSE BEGIN
        fname = '../data/' + qname + 'y.gda'
    ENDELSE
    readfield, [fname], it, nd, data1, nx1, nz1, x1, z1
    data1 = smooth(data1, 9)
    ;ratio = nx1/(ze-zs)/2.0
    BlueWhiteRed, rgbtable, lims2
    im1 = IMAGE(data1(*,zs:ze), x1, z1(zs:ze), /current, $
        aspect_ratio=ratio,$
        font_size=16, xtitle='x ($d_i$)', ytitle='!8z ($d_i$)', $
        position=[xs,ys,xe,ys+hei], $
        dimensions=[768, iheight], $
        xshowtext = 0, $
        ;ytickvalues=[200, 400, 600, 800, 1000], $
        ;max_value=5.0, min_value=0.5, $
        ;position=[0.07,0.53,0.95,0.96],$
        rgb_table=rgbtable, axis_style=2, interpolate=1)
    maxdata = max(data1)
    mindata = min(data1)
    print, 'Maximum', maxdata
    print, 'Minimum', mindata
    im1.max_value = lims2(1)
    im1.min_value = lims2(0)
    pos2 = im1.position
;    ;print, pos1
;    CB1 = COLORBAR(TARGET=im1,ORIENTATION=1,$
;        position=[xe+0.01,pos1(1),xe+0.03,pos1(3)])
;    CB1.TEXTPOS = 1
;    CB1.TICKDIR = 1
;    CB1.FONT_SIZE = 16
;    CB1.TITLE=jname + 'y}$'
;    
    if (ifieldline eq 1) then begin
        cn1 = contour(data(*,zs1:ze1), x, z(zs1:ze1), /overplot, $
            c_label_show=0, color='black')
    endif

  ; z-component
    ys = ys-hei
    IF (ipic EQ 0) THEN BEGIN
        fname = '../data1/' + qname + 'z00_' + species + '.gda'
    ENDIF ELSE BEGIN
        fname = '../data/' + qname + 'z.gda'
    ENDELSE
    readfield, [fname], it, nd, data1, nx1, nz1, x1, z1
    data1 = smooth(data1, 9)
    ;ratio = nx1/(ze-zs)/2.0
    BlueWhiteRed, rgbtable, lims3
    im1 = IMAGE(data1(*,zs:ze), x1, z1(zs:ze), /current, $
        aspect_ratio=ratio,$
        font_size=16, xtitle ='!8x ($d_i$)', ytitle='!8z ($d_i$)', $
        position=[xs,ys,xe,ys+hei], $
        ;xshowtext = 0, $
        ;ytickvalues=[200, 400, 600, 800, 1000], $
        ;max_value=5.0, min_value=0.5, $
        ;position=[0.07,0.53,0.95,0.96],$
        rgb_table=rgbtable, axis_style=2, interpolate=1)
    maxdata = max(data1)
    mindata = min(data1)
    print, 'Maximum', maxdata
    print, 'Minimum', mindata
    im1.max_value = lims3(1)
    im1.min_value = lims3(0)
    pos3 = im1.position
    ;print, pos1
    CB1 = COLORBAR(TARGET=im1,ORIENTATION=1,$
        position=[xe+0.01,pos3(1),xe+0.03,pos1(3)])
    CB1.TEXTPOS = 1
    CB1.TICKDIR = 1
    CB1.FONT_SIZE = 16
    ;CB1.TITLE=jname + 'z}$'
    CB1.TITLE='$!16j_{!8z}, !16j_{!8y}, !16j_{!8x}$'

    if (ifieldline eq 1) then begin
        cn1 = contour(data(*,zs1:ze1), x, z(zs1:ze1), /overplot, $
            c_label_show=0, color='black')
    endif

    yrmin = pos1(1)

    sz = size(xslice)
    FOR i = 0, sz(1)-1 DO BEGIN
        p1 = PLOT([xslice(i),xslice(i)],[1.0,2.0],'k--',/current,$
            xrange=[x1(0),x1(nx1-1)], yrange=[1.0,2.0],$
            position=[pos1(0),yrmin,pos1(2),yrmax])
        ax = p1.axes
        ax[0].HIDE=1
        ax[1].HIDE=1
        ax[2].HIDE=1
        ax[3].HIDE=1
    ENDFOR

    fname = 'jperp2d' + STRING(it, FORMAT='(I2.2)') + $
        '_' + species + '.jpg'
    im1.save, fname, resolution=300
END
;
;
;; +
;; 2D plot for jqnuperp_dote, vdot_divp, j'dote
;; -
;PRO jdote2d, it, ipic, lims, xslice, im1
;    ReadDomain, nx, nz, dx, dz, x, z, mime
;    readt, 81, dtf, tf, dte
;
;    lims1 = [-0.002,0.002]
;    lims2 = [-0.002,0.002]
;    lims3 = [-0.002,0.002]
;
;    nd = 1.0 ; Interpolate grid size
;    BlueWhiteRed, rgbtable, lims
;    iheight = 700
;    ys = 0.74
;    hei = 0.22
;    jname = '$j_{'
;
;    dx1 = dx*sqrt(mime) / nd
;    dz1 = dz*sqrt(mime) / nd
;    dxdz1 = dx1*dz1
;
;  ; slice for line plots
;    xs = FLOOR(xslice/dx1)
;
;  ; vdot_divp
;    qname = ['vdot_divp']
;    fname = '../data/' + qname + '00.gda'
;    readfield, fname, it, nd, data1, nx1, nz1, x1, z1
;    data1 = smooth(data1, [9,9])
;    ratio = nx1/(ze-zs)/2.0
;    im1 = IMAGE(data1(*,zs:ze), x1, z1(zs:ze), /buffer, $
;        ;aspect_ratio=ratio,$
;        font_size=16, xtitle='x ($d_i$)', ytitle='z ($d_i$)', $
;        position=[0.10,ys,0.82,ys+hei], $
;        dimensions=[768, iheight], $
;        xshowtext = 0, $
;        ;ytickvalues=[200, 400, 600, 800, 1000], $
;        ;max_value=5.0, min_value=0.5, $
;        ;position=[0.07,0.53,0.95,0.96],$
;        rgb_table=rgbtable, axis_style=2, interpolate=1)
;    maxdata = max(data1)
;    mindata = min(data1)
;    print, 'Maximum', maxdata
;    print, 'Minimum', mindata
;    im1.max_value = lims1(1)
;    im1.min_value = lims1(0)
;    pos1 = im1.position
;    ;print, pos1
;    CB1 = COLORBAR(TARGET=im1,ORIENTATION=1,$
;        position=[0.83,pos1(1),0.85,pos1(3)])
;    CB1.TEXTPOS = 1
;    CB1.TICKDIR = 1
;    CB1.FONT_SIZE = 16
;    CB1.TITLE='Compressional'
;
;    zs1 = nz/4
;    ze1 = nz*3/4
;    fname = '../data1/Ay.gda'
;    OPENR, lun1, fname, /get_lun
;    data = FLTARR(nx, nz)
;
;    field = assoc(lun1,data)
;    data = field(it)
;    cn1 = contour(data(*,zs1:ze1), x, z(zs1:ze1), /overplot, $
;        c_label_show=0, color='black')
;
;    data1_cu = TOTAL(data1,2)
;    data1_cu_int = fltarr(nx1)
;    data1_cu_int(0) = data1_cu(0)
;    FOR i = 1, nx1-1 DO BEGIN
;        data1_cu_int(i) = data1_cu_int(i-1) + data1_cu(i)*dxdz1
;    ENDFOR
;
;    data1_xs = reform(data1(xs,*))
;
;    yrmax = pos1(3)
;
;  ; jqnuperp_dote
;    ys = ys-hei
;    qname = ['jqnuperp_dote']
;    fname = '../data/' + qname + '00.gda'
;    readfield, fname, it, nd, data1, nx1, nz1, x1, z1
;    data1 = smooth(data1, [9,9])
;    ratio = nx1/(ze-zs)/2.0
;    im1 = IMAGE(data1(*,zs:ze), x1, z1(zs:ze), /current, $
;        ;aspect_ratio=ratio,$
;        font_size=16, xtitle='x ($d_i$)', ytitle='z ($d_i$)', $
;        position=[0.10,ys,0.82,ys+hei], $
;        dimensions=[768, iheight], $
;        xshowtext = 0, $
;        ;ytickvalues=[200, 400, 600, 800, 1000], $
;        ;max_value=5.0, min_value=0.5, $
;        ;position=[0.07,0.53,0.95,0.96],$
;        rgb_table=rgbtable, axis_style=2, interpolate=1)
;    maxdata = max(data1)
;    mindata = min(data1)
;    print, 'Maximum', maxdata
;    print, 'Minimum', mindata
;    im1.max_value = lims2(1)
;    im1.min_value = lims2(0)
;    pos1 = im1.position
;    ;print, pos1
;    CB1 = COLORBAR(TARGET=im1,ORIENTATION=1,$
;        position=[0.83,pos1(1),0.85,pos1(3)])
;    CB1.TEXTPOS = 1
;    CB1.TICKDIR = 1
;    CB1.FONT_SIZE = 16
;    CB1.TITLE='$\bf j\rm\prime_\perp\cdot\bf E$'
;
;    cn1 = contour(data(*,zs1:ze1), x, z(zs1:ze1), /overplot, $
;        c_label_show=0, color='black')
;;    ys = ys-hei
;;    qname = ['pepara']
;;    fname1 = '../data/' + qname + '00.gda'
;;    qname = ['peperp']
;;    fname2 = '../data/' + qname + '00.gda'
;;    qname = ['vdot_gradp']
;;    fname3 = '../data/' + qname + '00.gda'
;;    qname = ['pdivv']
;;    fname4 = '../data/' + qname + '00.gda'
;;    readfield, fname1, it+1, nd, pepara2, nx1, nz1, x1, z1
;;    readfield, fname1, it-1, nd, pepara1, nx1, nz1, x1, z1
;;    readfield, fname2, it+1, nd, peperp2, nx1, nz1, x1, z1
;;    readfield, fname2, it-1, nd, peperp1, nx1, nz1, x1, z1
;;    readfield, fname3, it, nd, vdot_gradp, nx1, nz1, x1, z1
;;    readfield, fname4, it, nd, pdivv, nx1, nz1, x1, z1
;;    print, TOTAL((pepara2+2.0*peperp2)*dxdz1/3.0) 
;;    data1 = ((0.5*pepara2+peperp2)-(0.5*pepara1+peperp1))/dtf/2.0
;;    data1 = data1 + vdot_gradp*1.5
;;    data1 = data1 + 2.5*pdivv
;;    ;data1 = data1
;;    ;data1 = vdot_gradp*1.5
;;    data1 = smooth(data1, [9,9])
;;    ratio = nx1/(ze-zs)/2.0
;;    im1 = IMAGE(data1(*,zs:ze), x1, z1(zs:ze), /current, $
;;        ;aspect_ratio=ratio,$
;;        font_size=16, xtitle='x ($d_i$)', ytitle='z ($d_i$)', $
;;        position=[0.10,ys,0.82,ys+hei], $
;;        dimensions=[768, iheight], $
;;        xshowtext = 0, $
;;        ;ytickvalues=[200, 400, 600, 800, 1000], $
;;        ;max_value=5.0, min_value=0.5, $
;;        ;position=[0.07,0.53,0.95,0.96],$
;;        rgb_table=rgbtable, axis_style=2, interpolate=1)
;;    maxdata = max(data1)
;;    mindata = min(data1)
;;    print, 'Maximum', maxdata
;;    print, 'Minimum', mindata
;;    im1.max_value = lims2(1)
;;    im1.min_value = lims2(0)
;;    pos1 = im1.position
;;    ;print, pos1
;;    CB1 = COLORBAR(TARGET=im1,ORIENTATION=1,$
;;        position=[0.83,pos1(1),0.85,pos1(3)])
;;    CB1.TEXTPOS = 1
;;    CB1.TICKDIR = 1
;;    CB1.FONT_SIZE = 16
;;    CB1.TITLE='$\partial (3P_e/2)/\partial t$'
;
;    data2_cu = TOTAL(data1,2)
;    data2_cu_int = fltarr(nx1)
;    data2_cu_int(0) = data2_cu(0)
;    FOR i = 1, nx1-1 DO BEGIN
;        data2_cu_int(i) = data2_cu_int(i-1) + data2_cu(i)*dxdz1
;    ENDFOR
;    data2_xs = reform(data1(xs,*))
;
;  ; z-component
;    ys = ys-hei
;    qname = ['jcpara_dote', 'jcperp_dote', 'jdiagm_dote']
;    fname = '../data/' + qname + '00.gda'
;    readfield, fname, it, nd, data1, nx1, nz1, x1, z1
;    data1 = smooth(data1, [9,9])
;    ratio = nx1/(ze-zs)/2.0
;    im1 = IMAGE(data1(*,zs:ze), x1, z1(zs:ze), /current, $
;        font_size=16, xtitle ='x ($d_i$)', ytitle='z ($d_i$)', $
;        position=[0.10,ys,0.82,ys+hei], $
;        xshowtext = 0, $
;        ;ytickvalues=[200, 400, 600, 800, 1000], $
;        ;max_value=5.0, min_value=0.5, $
;        ;position=[0.07,0.53,0.95,0.96],$
;        rgb_table=rgbtable, axis_style=2, interpolate=1)
;    maxdata = max(data1)
;    mindata = min(data1)
;    print, 'Maximum', maxdata
;    print, 'Minimum', mindata
;    im1.max_value = lims3(1)
;    im1.min_value = lims3(0)
;    pos1 = im1.position
;    ;print, pos1
;    CB1 = COLORBAR(TARGET=im1,ORIENTATION=1,$
;        position=[0.83,pos1(1),0.85,pos1(3)])
;    CB1.TEXTPOS = 1
;    CB1.TICKDIR = 1
;    CB1.FONT_SIZE = 16
;    CB1.TITLE='$\bf j\rm_\perp\cdot\bf E$'
;
;    cn1 = contour(data(*,zs1:ze1), x, z(zs1:ze1), /overplot, $
;        c_label_show=0, color='black')
;    data3_cu = TOTAL(data1,2)
;    data3_cu_int = fltarr(nx1)
;    data3_cu_int(0) = data3_cu(0)
;    FOR i = 1, nx1-1 DO BEGIN
;        data3_cu_int(i) = data3_cu_int(i-1) + data3_cu(i)*dxdz1
;    ENDFOR
;    data3_xs = reform(data1(xs,*))
;    yrmin = pos1(1)
;
;  ; Accumulation of the data over y
;
;    p1 = plot(x1, data1_cu_int, 'r2', font_size=16, /current, $
;        xtitle ='x ($d_i$)', ytitle='Accumulation', axis_style=2,$
;        position=[0.10,ys-hei,0.82,ys], name='Compressional')
;    p2 = plot(x1, data2_cu_int, 'g2', /overplot, name='$\bf j\rm\prime_\perp\cdot\bf E$')
;    p3 = plot(x1, data3_cu_int, 'b2', /overplot, name='$\bf j\rm_\perp\cdot\bf E$')
;
;    leg1 = legend(target=[p1,p2,p3], /auto_text_color, $
;        font_size=16, positio=[0.35, 0.32], transparency=100)
;
;    sz = size(xslice)
;    FOR i = 0, sz(1)-1 DO BEGIN
;        p1 = PLOT([xslice(i),xslice(i)],[1.0,2.0],'k--',/current,$
;            xrange=[x1(0),x1(nx1-1)], yrange=[1.0,2.0],$
;            position=[pos1(0),yrmin,pos1(2),yrmax])
;        ax = p1.axes
;        ax[0].HIDE=1
;        ax[1].HIDE=1
;        ax[2].HIDE=1
;        ax[3].HIDE=1
;    ENDFOR
;
;    fname = 'jdote2d' + STRING(it, FORMAT='(I2.2)') + '.jpg'
;    im1.save, fname, resolution=300
;
;    ys = 0.69
;    yint = 0.26
;    yticks = fltarr(3,sz(1))
;    yticks(*,0) = [0.0,0.0005,0.001]
;    yticks(*,1) = [-0.0005,0.0,0.0005]
;    yticks(*,2) = [-0.0005,0.0,0.0005]
;    w1=window(dimension=[800,400])
;    FOR i = 0, sz(1)-1 DO BEGIN
;        p1 = plot(z1(zs:ze), data1_xs(i,zs:ze), 'r2', font_size=20, $
;            position=[0.2,ys,0.95,ys+yint], /current, $
;            xrange = [z1(zs), z1(ze)], $
;            ytickvalues=yticks(*,i), $
;            name = 'Compressional', xshowtext=0)
;        IF (i EQ (sz(1)-1)) THEN BEGIN
;            p1.xtitle='z ($d_i$)'
;            p1.xshowtext = 1
;        ENDIF
;        IF (i EQ 1) THEN BEGIN
;            p1.ytitle='Energy Conversion'
;        ENDIF
;        p2 = plot(z1(zs:ze), data2_xs(i,zs:ze), /overplot, 'g2', name='$\bf j\rm\prime_\perp\cdot\bf E$')
;        p3 = plot(z1(zs:ze), data3_xs(i,zs:ze), /overplot, 'b2', name='$\bf j\rm_\perp\cdot\bf E$')
;        ys = ys - yint
;    ENDFOR
;    leg1 = legend(target=[p1], /auto_text_color, $
;        font_size=16, position=[0.4,0.4], transparency=100)
;    leg1 = legend(target=[p2], /auto_text_color, $
;        font_size=16, position=[0.8,0.4], transparency=100)
;    leg1 = legend(target=[p3], /auto_text_color, $
;        font_size=16, position=[0.8,0.3], transparency=100)
;
;    fname = 'jdote' + string(it, format='(I2.2)') + '_xs.eps'
;    p1.save, fname
;END
;
;; +
;; Change rate of pressure anisotopy
;; -
;PRO shear, it, icolor, lims, im1
;    ReadDomain, nx, nz, dx, dz, x, z
;    nt = 82
;    readt, nt, dtf, tf, dte
;    fname = '../data/aniso_e_real00.gda'
;    OPENR, lun1, fname, /get_lun
;    data1 = FLTARR(nx, nz)
;    data2 = FLTARR(nx, nz)
;    aniso = FLTARR(nx, nz)
;    aniso1 = aniso
;    data1(*,*) = 0.0
;    data2(*,*) = 0.0
;
;    field = assoc(lun1,data1)
;    data1 = field(it-1)
;    data2 = field(it+1)
;    aniso1 = field(it)
;
;    aniso = (data2-data1)/(dtf*2.0)
;    aniso = smooth(aniso, 9)
;
;    data3 = PTR_NEW(aniso, /no_copy)
;    plot2d, data3, it, lims, 1, 0, im1
;    FREE_LUN, lun1
;
;    fname = '../data/shear00.gda'
;    OPENR, lun1, fname, /get_lun
;    data1 = FLTARR(nx, nz)
;    data = FLTARR(nx, nz)
;    shear = FLTARR(nx, nz)
;    data1(*,*) = 0.0
;
;    field = assoc(lun1,data1)
;    data1 = field(it)
;
;    shear = data1
;
;    data = -3.0*aniso1*shear
;    ;data = shear
;    data = smooth(data,6)
;
;    data3 = PTR_NEW(data, /no_copy)
;    plot2d, data3, it, lims, 1, 0, im1
;    FREE_LUN, lun1
;
;    ;fname = 'img/shear.jpg'
;    ;im1.save, fname
;    ;im1.close
;END
;
;; +
;; Description: test the evolution of parallel and perpendicular pressure
;; -
;PRO dpdt, it, icolor, lims, im1
;    ReadDomain, nx, nz, dx, dz, x, z
;    nt = 82
;    readt, nt, dtf, tf, dte
;    fname = '../data/pepara_real00.gda'
;    ;fname = '../data/peperp_real00.gda'
;    OPENR, lun1, fname, /get_lun
;    data1 = FLTARR(nx, nz)
;    data2 = FLTARR(nx, nz)
;    dpdt1 = FLTARR(nx, nz)
;    data1(*,*) = 0.0
;    data2(*,*) = 0.0
;
;    field = assoc(lun1,data1)
;    data1 = field(it-1)
;    data2 = field(it+1)
;    pe1 = field(it)
;
;    print, dtf
;    dpdt1 = (data2-data1)/(dtf*2.0)
;    dpdt1 = smooth(dpdt1, 9)
;
;    data3 = PTR_NEW(dpdt1, /no_copy)
;    plot2d, data3, it, lims, 1, 0, im1
;    FREE_LUN, lun1
;
;    fname = '../data/divpeparav00.gda'
;    ;fname = '../data/divpeperpv00.gda'
;    OPENR, lun1, fname, /get_lun
;    data1 = FLTARR(nx, nz)
;    divpv = FLTARR(nx, nz)
;    data1(*,*) = 0.0
;
;    field = assoc(lun1,data1)
;    data1 = field(it)
;    divpv = data1
;
;    data3 = PTR_NEW(dpdt1, /no_copy)
;    FREE_LUN, lun1
;
;    fname1 = '../data/shear00.gda'
;    fname2 = '../data/divv00.gda'
;    OPENR, lun1, fname1, /get_lun
;    OPENR, lun2, fname2, /get_lun
;    data1 = FLTARR(nx, nz)
;    data2 = FLTARR(nx, nz)
;    data = FLTARR(nx, nz)
;    data1(*,*) = 0.0
;    data2(*,*) = 0.0
;
;    field1 = assoc(lun1,data1)
;    data1 = field1(it)
;    shear = data1
;    field2 = assoc(lun2,data2)
;    data2 = field2(it)
;    divv = data2
;
;    data = -2.0*pe1*divv/3.0 - 2.0*pe1*shear - divpv
;    ;data = -2.0*pe1*divv/3.0 + pe1*shear - divpv
;    data = smooth(data,9)
;
;    data3 = PTR_NEW(data, /no_copy)
;    plot2d, data3, it, lims*2, 1, 0, im1
;    FREE_LUN, lun1
;    FREE_LUN, lun2
;END
;
;PRO firehose, it, icolor, lims, im1
;    ReadDomain, nx, nz, dx, dz, x, z
;    fname = '../data/firehose00.gda'
;    OPENR, lun1, fname, /get_lun
;    data2 = FLTARR(nx, nz)
;    data = FLTARR(nx, nz)
;    data(*,*) = 0.0
;
;    field = assoc(lun1,data)
;    data = field(it)
;
;;    FOR i = 0, 68 DO BEGIN
;;        data2 = field(i)
;;        data2[where(data2 LT 0, /NULL)] = 0.0
;;        data = data + data2
;;    ENDFOR
;
;    data = smooth(data, 3)
;
;    data1 = PTR_NEW(data, /no_copy)
;
;    plot2d, data1, it, lims, 0, 0, im1
;    fname = 'img/firehose_tot.jpg'
;    im1.save, fname
;    ;im1.close
;    FREE_LUN, lun1
;END
;
;; j dot E with j from the anisotropy term (P_\paralle-P_\perp)*(B cross (B dot \nabla)B).
;PRO jdote1, it, icolor, lims, im1
;    ReadDomain, nx, nz, dx, dz, x, z
;    OPENR, lun1, '../data/jcpara_dote02.gda', /get_lun
;    OPENR, lun2, '../data/jcperp_dote02.gda', /get_lun
;    data1 = FLTARR(nx, nz)
;    data2 = FLTARR(nx, nz)
;    data = FLTARR(nx, nz)
;
;    field = assoc(lun1,data1)
;    data1 = field(it)
;    field = assoc(lun2,data2)
;    data2 = field(it)
;
;    data = SMOOTH(data1+data2, 9)
;    data3 = PTR_NEW(data, /no_copy)
;
;    plot2d, data3, it, lims, 1, 0, im1
;
;    fname = 'img/' + 'jc_dote' + STRING(it, FORMAT='(I2.2)') + '.jpg'
;    im1.save, fname
;    ;im1.close
;    FREE_LUN, lun1
;    FREE_LUN, lun2
;END
;
;; +
;; plotting the curvature magnitude
;; -
;PRO ang_curv, it, lims, im1
;    ReadDomain, nx, nz, dx, dz, x, z
;    OPENR, lun1, '../data/jcparaz.gda', /get_lun
;    OPENR, lun2, '../data/pepara.gda', /get_lun
;    data1 = FLTARR(nx, nz)
;    data2 = FLTARR(nx, nz)
;    data = FLTARR(nx, nz)
;
;    field = assoc(lun1,data1)
;    data1 = field(it)
;    field = assoc(lun2,data2)
;    data2 = field(it)
;
;    data = SMOOTH(data1/data2, 9)
;    data3 = PTR_NEW(data, /no_copy)
;
;    plot2d, data3, it, lims, 0, 0, im1
;
;    ;fname = 'img/' + 'jc_dote' + STRING(it, FORMAT='(I2.2)') + '.jpg'
;    ;im1.save, fname
;    ;im1.close
;    FREE_LUN, lun1
;    FREE_LUN, lun2
;END
;
;*******************************************************************************
; Plot 2D image for dataset.
; Input:
;   data: pointer to the data array.
;   it: the time point ID.
;   lims: minimum and maximum data to plot.
;   icumulate: flag for whether to acculate the data for line plots.
;   icolor: 0 for blue-white-red color, others for the default color.
;   isLimits: whether to use the input limits.
;   ifieldline: flag for whether to overplot magnetic field lines.
;   zLength: the length of the range of z in grids number.
;   isBuffer: wether to buffer the image rather than display it.
; Output:
;   im1: image object
;*******************************************************************************
PRO plot2d, data, it, lims, icumulation, icolor, isLimits, $
    ifieldline, isLog, isBuffer, im1, zLength
    common picinfo
    nInterpolate = 2
    nx1 = FLOOR(nx/nInterpolate)+1
    nz1 = FLOOR(nz/nInterpolate)+1

    data1 = fltarr(nx1,nz1)
    idx1 = findgen(nx1)*nInterpolate
    idz1 = findgen(nz1)*nInterpolate
    x1 = idx1 * dx
    z1 = idz1 * dz
    data1 = interpolate(*data, idx1, idz1, cubic=-0.5, /grid)
    ;data1 = smooth(data1, [3,3])

    ; Center at the middle of the box in z direction.
    zNgrids = zLength / dz ; the number of grids with the length zLength
    zs = nz1/2 - zNgrids/nInterpolate/2
    ze = nz1/2 + zNgrids/nInterpolate/2
    ;zs = nz1/4
    ;ze = nz1*3/4
    ;zs = FLOOR(nz/6.0)
    ;ze = FLOOR(nz*5/6.0)
    ;zs = nz1/4
    ;ze = nz1 * 0.4

    IF (icolor NE 0) THEN BEGIN
        rgbtable = icolor
    ENDIF ELSE BEGIN
        BlueWhiteRed, rgbtable, lims
    ENDELSE
    ; Change the figure canvas size according to zLength.
    cRatio = zNgrids / (nz/2)
    cRatio = 1
    iheight = 720 * cRatio
    ys = 0.15 / cRatio
    IF (icumulation EQ 1) THEN BEGIN
        iheight = 480 * cRatio
        ys = 0.35
    ENDIF
  
    xmax = max(nx1)
    xmin = min(nx1)
    ratio = (xmax-xmin)/(z1(ze)-z1(zs))
    if (isLog eq 1) then begin
        data1 = alog10(data1)
    endif
    if (isBuffer eq 1) then begin
        im1 = IMAGE(data1(*,zs:ze), x1, z1(zs:ze), $
            dimensions=[1440, iheight], /buffer, axis_style=2)
    endif else begin
        im1 = IMAGE(data1(*,zs:ze), x1, z1(zs:ze), $
            dimensions=[1440, iheight], axis_style=2)
    endelse
    im1.aspect_ratio=ratio
    im1.font_size=16
    im1.xtitle='!8x !3($!8d_i$!3)'
    im1.ytitle='!8z !3($!8d_i$!3)'
    im1.position=[0.12,ys,0.85,0.95]
    im1.rgb_table=rgbtable
    im1.interpolate=1
    maxdata = max(data1)
    mindata = min(data1)
    print, 'Maximum', maxdata
    print, 'Minimum', mindata
    if (isLimits EQ 1) then begin
        im1.max_value = lims(1)
        im1.min_value = lims(0)
    endif
    pos1 = im1.position
    CB1 = COLORBAR(TARGET=im1,ORIENTATION=1,$
        position=[0.86,pos1(1),0.88,pos1(3)])
    CB1.TEXTPOS = 1
    CB1.TICKDIR = 1
    CB1.TITLE = '!8$n_e$'
    CB1.FONT_SIZE = 24

    zs1 = zs
    ze1 = ze

    nd = 4
    if (ifieldline EQ 1) then begin
        fname = '../data/Ay.gda'
        readfield, [fname], it, nd, dataAy, nx1, nz1, x1, z1
        ;OPENR, lun1, fname, /get_lun
        ;data = FLTARR(nx, nz)

        ;field = assoc(lun1,data)
        ;data = field(it)
        cn1 = contour(dataAy(*,zs:ze), x1, z1(zs:ze), /overplot, $
            yrange = [z1(zs1), z1(ze1)], $
            c_label_show=0, color='black')
    endif

  ; Plotting cumulated data or not
    IF (icumulation EQ 1) THEN BEGIN
        ;im1.position=[0.05,0.35,0.90,0.95]
        ;im1.image_dimensions=[960, 600]
        im1.xshowtext = 0
        data2 = fltarr(nx)
        data3 = fltarr(nx)
        data2 = total(data1,2)
        data3(0) = data2(0)
        FOR i = 1, nx1-1 DO BEGIN
            data3(i) = data3(i-1) + data2(i)
        ENDFOR
        p1 = PLOT(x1, data3, 'k2', /current, $
            position=[pos1(0),0.12,pos1(2), pos1(1)], $
            font_size=16, xtitle='x ($d_i$)')
    ENDIF
END
;
;; Pressure anisotropy for electrons
;PRO PreAnisotropy, it, x0, lim, im1
;    ReadDomain, nx, nz, dx, dz, x, z
;    zs = FLOOR(nz/6)
;    ze = FLOOR(nz*5/6)
;
;    fname1 = '../data/aniso_e_real00.gda'
;    OPENR, lun1, fname1, /get_lun
;    data1 = FLTARR(nx, nz)
;    data = FLTARR(nx, nz)
;
;    field = assoc(lun1,data1)
;    data1 = field(it)
;
;  ; P_\parallel > P_\perp red.
;  ; P_\parallel < P_\perp blue.
;    data = SMOOTH(data1, 9) - 1.0
;
;    llim = 0.2
;    nd = 4 ; Decrease the size of the data in demensions.
;
;    nx1 = FLOOR(nx/nd)+1
;    nz1 = FLOOR(nz/nd)+1
;
;    data1 = fltarr(nx1,nz1)
;    idx1 = findgen(nx1)*nd
;    idz1 = findgen(nz1)*nd
;    x1 = idx1 * dx
;    z1 = idz1 * dz
;    data1 = interpolate(data, idx1, idz1, cubic=-0.5, /grid)
;
;    zs = nz1/4
;    ze = nz1*3/4
;
;    xs = FLOOR(x0/(dx*nd))
;
;    BlueWhiteRed, rgbtable, [llim-1.0,lim-1.0]
;    ratio = nx1/(ze-zs)/2.0
;    im1 = IMAGE(data1(*,zs:ze), x1, z1(zs:ze), $
;        aspect_ratio=ratio,$
;        font_size=16, xtitle='x ($d_i$)', ytitle='z ($d_i$)', $
;        position=[0.1,0.3,0.75,0.95], $
;        dimensions=[768, 360],$
;        xtickvalues=[0,50,100,150], $
;        ;dimensions=[960, 450],$
;        ;xshowtext = 0, $
;        ;ytickvalues=[200, 400, 600, 800, 1000], $
;        ;max_value=5.0, min_value=0.5, $
;        ;position=[0.07,0.53,0.95,0.96],$
;        rgb_table=rgbtable, axis_style=2, interpolate=1)
;    pos1 = im1.position
;    maxdata = max(data)
;    mindata = min(data)
;    print, 'Maximum', maxdata+1.0
;    print, 'Minimum', mindata+1.0
;    im1.max_value = lim-1.0
;    im1.min_value = llim - 1.0
;    CB1 = COLORBAR(TARGET=im1,ORIENTATION=0,$
;        position=[pos1(0),0.1,pos1(2),0.15])
;    ticken = cb1.tickname
;    CB1.tickvalues = FLOAT(ticken)
;    CB1.tickname = STRING(FLOAT(ticken)+1.0, format='(F4.1)')
;    CB1.TEXTPOS = 0
;    CB1.TICKDIR = 1
;    CB1.FONT_SIZE = 16
;
;    zs1 = nz/4
;    ze1 = nz*3/4
;    fname = '../data1/Ay.gda'
;    OPENR, lun1, fname, /get_lun
;    data = FLTARR(nx, nz)
;
;    field = assoc(lun1,data)
;    data = field(it)
;    cn1 = contour(data(*,zs1:ze1), x, z(zs1:ze1), /overplot, $
;        c_label_show=0, color='black')
;
;    p1 = plot([x1(xs),x1(xs)], [z1(zs),z1(ze)], /overplot, '--')
;
;    maxd = max(data1(xs,zs:ze)+1.0)
;    mind = min(data1(xs,zs:ze)+1.0)
;
;    mind = ceil(mind)
;    maxd = floor(maxd)
;    xtick = fltarr(4)
;    xtick(0) = mind
;    dx = ceil((maxd-mind)/4.0)
;    xtick(1) = xtick(0) + dx
;    xtick(2) = xtick(1) + dx
;    xtick(3) = xtick(2) + dx
;
;    xtick1 = xtick(where(xtick LE maxd, /NULL))
;
;    p2 = PLOT(data1(xs,zs:ze)+1.0,z1(zs:ze), 'k2', /current, $
;        yrange=[z1(zs), z1(ze)],$
;        yshowtext = 0, $
;        xtickvalues=xtick1, $
;        position=[pos1(2),pos1(1),0.95,pos1(3)], $
;        font_size=16)
;
;    t1 = text((pos1(2)+0.95)/2.0,0.10,'$P_{||}/P_\perp$', font_size=20)
;    FREE_LUN, lun1
;
;    fname = 'img/anisotropy_e' + STRING(it, FORMAT='(I2.2)') + '.jpg'
;    im1.save, fname, resolution=300
;    ;im1.close
;END
;
;; +
;; Description: Testing 2D plot for jpolarization dote E
;; -
;PRO jpolar_dote, it, icolor, lim, im1
;    CLOSE, /ALL
;    dims = UINTARR(6)
;    sizes = fltarr(3)
;  ; Get the simulation domain information
;    OPENR, lun, 'data/info', /get_lun, /F77_UNFORMATTED 
;    READU, lun, dims
;    READU, lun, sizes
;    FREE_LUN, lun
;
;    nx = dims(0)
;    nz = dims(4)
;    xmax = sizes(0)
;    zmax = sizes(2)
;    mime = 16.0
;    dx = xmax/nx/sqrt(mime)
;    dz = zmax/nz/sqrt(mime)
;
;    OPENR, lun1, '../data/jpolarx.gda', /get_lun
;    OPENR, lun2, '../data/jpolary.gda', /get_lun
;    OPENR, lun3, '../data/jpolarz.gda', /get_lun
;    OPENR, lun4, '../data1/ex.gda', /get_lun
;    OPENR, lun5, '../data1/ey.gda', /get_lun
;    OPENR, lun6, '../data1/ez.gda', /get_lun
;    data = FLTARR(nx, nz)
;    data1 = FLTARR(nx, nz)
;    data2 = FLTARR(nx, nz)
;    data3 = FLTARR(nx, nz)
;    data4 = FLTARR(nx, nz)
;    data5 = FLTARR(nx, nz)
;    data6 = FLTARR(nx, nz)
;    x = FINDGEN(nx)*dx
;    z = FINDGEN(nz)*dz
;
;    zs = nz/4
;    ze = nz*3/4
;
;    nt = it
;    ;rgbtable = icolor
;    rgbtable = FLTARR(3,256)
;    p = FLTARR(3,5)
;    cindex = FINDGEN(256)*4.0/255.0
;    p = [[0.0,0.0,0.5],[0.0,0.5,1.0],[1.0,1.0,1.0],[1.0,0.0,0.0],[0.5,0.0,0.0]]
;    rgbtable(0,*) = INTERPOLATE(p(0,*),cindex)
;    rgbtable(1,*) = INTERPOLATE(p(1,*),cindex)
;    rgbtable(2,*) = INTERPOLATE(p(2,*),cindex)
;    rgbtable = rgbtable*255
;    FOR i = 0, nt-1 DO BEGIN
;        READU, lun1, data1
;        READU, lun2, data2
;        READU, lun3, data3
;        READU, lun4, data4
;        READU, lun5, data5
;        READU, lun6, data6
;        IF (i EQ it-1) THEN BEGIN
;            ;data = SMOOTH(data1*data4+data2*data5+data3*data6, 2)
;            data = data1*data4+data2*data5+data3*data6
;            print, TOTAL(data)
;            ;FOR k = 2, nz-2 DO BEGIN
;            ;    FOR j = 2, nx-2 DO BEGIN
;            ;        data1(j,k) = TOTAL(data(j-1:j+1,k-1:k+1))/9.0
;            ;    ENDFOR
;            ;ENDFOR
;            im1 = IMAGE(data(*,zs:ze), x, z(zs:ze) ,$
;                aspect_ratio=2,$
;                /current, font_size=16, xtitle='x ($d_i$)', ytitle='z ($d_i$)', $
;                position=[0.05,0.2,0.90,0.95], $
;                dimensions=[960, 450],$
;                ;xshowtext = 0, $
;                ;ytickvalues=[200, 400, 600, 800, 1000], $
;                ;max_value=5.0, min_value=0.5, $
;                ;position=[0.07,0.53,0.95,0.96],$
;                rgb_table=rgbtable, axis_style=2, interpolate=1)
;            CB1 = COLORBAR(TARGET=im1,ORIENTATION=1,$
;                position=[0.85,0.2,0.88,0.95])
;            CB1.TEXTPOS = 1
;            CB1.TICKDIR = 1
;            CB1.FONT_SIZE = 16
;            maxdata = max(data)
;            mindata = min(data)
;            print, maxdata, mindata
;            im1.max_value = lim
;            im1.min_value = -lim
;            ;fname = 'img/' + qname + STRING(i, FORMAT='(I2.2)') + '.jpg'
;            ;fname = 'img/pre' + STRING(i, FORMAT='(I2.2)') + '.png'
;            ;im1.save, fname
;            ;im1.close
;
;            ;p1 = PLOT(z, total(data,1))
;            ;print, max(data), min(data), min(abs(data))
;        ENDIF
;    ENDFOR
;    FREE_LUN, lun1
;END
;
;;+
;; Description: Plotting time evolution of magnetic moment.
;;-
;PRO plotmu, nt
;    readt, nt, dtf, tf, dte
;    data = fltarr(nt)
;    openr, lun, 'data/bmu.dat', /get_lun
;    readf, lun, data
;    free_lun, lun
;    tmax = tf(nt-1)
;    tmin = tf(0)
;    p1 = plot(tf, data, 'r2', $
;        xrange=[tmin, tmax], font_size=16,$
;        xtitle='$t\omega_{ce}$', ytitle='$\mu$')
;END
;
; +
; Read analyzed data, which change with time.
; -
PRO readdata_t, fname, nt, nk, data
    OPENR, lun, fname, /get_lun
    nt1 = 81
    ts1 = FINDGEN(nt)
    data1 = FLTARR(nk)
    data = FLTARR(nk, nt)
    FOR it = 0, nt-1 DO BEGIN
        field = assoc(lun,data1)
        data1 = field(it)
        data(*,it) = data1
    ENDFOR
    FREE_LUN, lun
END

;*******************************************************************************
; Procedure to plot the compressional heating term.
;*******************************************************************************
PRO compression, species
    common picinfo
    common jdote_data
    common consts

    ReadJdotE, ntf, dtf, 0, species, dtwpe, dtwci, 0

    nk1 = 4
    fname = 'data_omp/divv00_' + species + '.gda'
    readdata_t, fname, ntf, nk1, data1
    pdivv = reform(data1(1,*))
    vdot_divp = reform(data1(2,*))

    nk2 = 3
    fname = 'data_omp/shear00_' + species + '.gda'
    readdata_t, fname, ntf, nk2, data2
    pshear = -reform(data2(1,*))

    IntegrateDataInTime, pdivv, ntf, dtf, pdivv_int, ene_norm
    IntegrateDataInTime, vdot_divp, ntf, dtf, vdot_divp_int, ene_norm
    IntegrateDataInTime, pshear, ntf, dtf, pshear_int, ene_norm
    pdivv = pdivv * dtwpe / dtwci
    vdot_divp = vdot_divp * dtwpe / dtwci
    pshear = pshear * dtwpe / dtwci
    pdivv_int = pdivv_int * dtwpe / dtwci
    vdot_divp_int = vdot_divp_int * dtwpe / dtwci
    pshear_int = pshear_int * dtwpe / dtwci

    kenorm = 1.0E4 ; Normalized energy
    pdivv_int = pdivv_int / kenorm
    vdot_divp_int = vdot_divp_int / kenorm
    pshear_int = pshear_int / kenorm
    jqnupara_dote_int = jqnupara_dote_int / kenorm
    if (species eq 'e') then begin
        dke = dkee
        ke = kee / kenorm
    endif else begin
        dke = dkei
        ke = kei / kenorm
    endelse

    tmax = tf(ntf-1)
    tmin = tf(0)
    ymin1 = fltarr(6)
    ymax1 = fltarr(6)
    ymin1(0) = min(pdivv)
    ymin1(1) = min(vdot_divp)
    ymin1(2) = min(pshear)
    ymin1(3) = min(jqnupara_dote)
    ymin1(4) = min(dke)
    ymin1(5) = min(pdivv+pshear)
    ymax1(0) = max(pdivv)
    ymax1(1) = max(vdot_divp)
    ymax1(2) = max(pshear)
    ymax1(3) = max(jqnupara_dote)
    ymax1(4) = max(dke)
    ymax1(5) = max(pdivv+pshear)
    ymax = max(ymax1)
    ymin = min(ymin1)
    yrange_values, ymax, ymin


    ys = 0.56
    yint = 0.36
    pos1 = [0.18,ys,0.95,ys+yint]
    p1 = plot(tf, pdivv, 'r2', $
        xrange=[tmin, tmax], yrange=[ymin, ymax], font_size=16,$
        xtitle='$!8t\omega_{ci}$', ytitle='$!8dE_c!3/!8dt$',$
        position=pos1, dimension=[500, 350],$
        xshowtext=0, $
        name='$-!8P\nabla\cdot\bf!16U$')
    pos1 = p1.position
    p2 = plot(tf, pshear, 'g2', /overplot, $
        name='$-!8(P_{||}-P_\perp)b_ib_j\sigma_{ij}$')
    p3 = plot(tf, vdot_divp, 'b2', /overplot, $
        name='$!16U!8\cdot(\nabla\cdot!15P!8)$')
;    p4 = plot(tf, jqnupara_dote, 'r2--', /overplot, $
;        name='$\bf j\rm_{||}\cdot\bf E$')
    p5 = plot(te, dke, 'k2', /overplot, name='$!8K_e$')
;    p6 = plot(tf, pdivv+pshear, 'k2--', /overplot, $
;        name='$-P\nabla\cdot\bf U\rm-(P_{||}-P_\perp)b_ib_j\sigma_{ij}$')

    p11 = plot([tmin, tmax], [0,0], '--', /overplot)

    leg1 = legend(target=[p1,p2,p3,p5], /auto_text_color, $
        font_size=16,position=[0.50, 0.57], transparency=100)
;    leg1 = legend(target=[p6], /auto_text_color, $
;        font_size=16,position=[0.65, 0.57], transparency=100)

;    fname = 'ene_comp.eps'
;    p1.save, fname
;    p1.save, 'ene_comp.jpg', resolution=300

    ymin1(0) = min(pdivv_int)
    ymin1(1) = min(vdot_divp_int)
    ymin1(2) = min(pshear_int)
    ymin1(3) = min(jqnupara_dote_int)
    ymin1(4) = min(ke-ke(0))
    ymin1(5) = min(pdivv_int+pshear_int)
    ymax1(0) = max(pdivv_int)
    ymax1(1) = max(vdot_divp_int)
    ymax1(2) = max(pshear_int)
    ymax1(3) = max(jqnupara_dote_int)
    ymax1(4) = max(ke-ke(0))
    ymax1(5) = max(pdivv_int+pshear_int)
    ymax = max(ymax1)
    ymin = min(ymin1)
    yrange_values, ymax, ymin

    p1 = plot(tf, pdivv_int, 'r2', $
        xrange=[tmin, tmax], yrange=[ymin, ymax], /current, font_size=16,$
        xtitle='$!8t\omega_{ci}$', ytitle='$!8E_c$',$
        position=[pos1(0),0.17,pos1(2),pos1(1)],$
        name='$\int -P\nabla\cdot\bf U\rm dt$')
    p2 = plot(tf, pshear_int, 'g2', /overplot, $
        name='$\int -(P_{||}-P_\perp)b_ib_j\sigma_{ij} dt$')
    p3 = plot(tf, vdot_divp_int, 'b2', /overplot, $
        name='$\int\bf U\rm\cdot(\nabla\cdot\bf P)\rm dt$')
;    p4 = plot(tf, jqnupara_dote_int, 'r2--', /overplot, $
;        name='$\int\bf j\rm_{||}\cdot\bf E\rm dt$')
    p5 = plot(te, ke-ke(0), 'k2', /overplot, name='$\int dK_e$')
;    p6 = plot(tf, pdivv_int+pshear_int, 'k2--', /overplot, $
;        name='$\int(-P\nabla\cdot\bf U\rm-(P_{||}-P_\perp)b_ib_j\sigma_{ij}) dt$')

    p11 = plot([tmin, tmax], [0,0], '--', /overplot)

;    leg1 = legend(target=[p6,p1,p2,p3,p4,p5], /data, /auto_text_color, $
;        font_size=16,position=[tmax*0.6, ymax*0.95], transparency=100)
    t1 = TEXT(0.01, 0.9, '(I)', font_size=24)

    fname = 'ene_comp_int_' + species + '.eps'
    p1.save, fname
    fname = 'ene_comp_int_' + species + '.jpg'
    p1.save, fname, resolution=300
END

;*******************************************************************************
; Plot time evolution of compression related terms.
;*******************************************************************************
PRO plotcomp, species
    common picinfo
    common jdote_data

    ReadJdotE, ntf, dtf, 0, species, dtwpe, dtwci, 0
    data = fltarr(4,ntf)
    data1 = fltarr(4)
    fname = 'data/divv00_' + species + '.gda'
    openr, lun, fname, /get_lun
    FOR it = 0, ntf-1 DO BEGIN
        field = assoc(lun,data1)
        data1 = field(it)
        data(*,it-1) = data1
    ENDFOR
    data = data * dtwpe / dtwci
    free_lun, lun

    if (species eq 'e') then begin
        ke = kee
        dke = dkee
    endif else begin
        ke = kei
        dke = dkei
    endelse

    tmax = tf(ntf-1)
    tmin = tf(0)
    ymin1 = fltarr(2)
    ymax1 = fltarr(2)
    ymin1(0) = min(data(1:3,*))
    ymax1(0) = max(data(1:3,*))
    ymin1(1) = min(dke)
    ymax1(1) = max(dke)
    ymax = max(ymax1)
    ymin = min(ymin1)
    yrange_values, ymax, ymin
    p1 = plot(tf, data(1,*), 'r2', $
        xrange=[tmin, tmax], yrange=[ymin, ymax], font_size=20,$
        xtitle='$t\omega_{ci}$', ytitle='Energy conversion',$
        position=[0.1,0.15,0.95,0.95], dimension=[1000, 512],$
        name='$P\nabla\cdot\bf V$')
    p2 = plot(tf, data(2,*), 'g2', $
        /overplot, name='$\bf V\rm\cdot(\nabla\cdot\bf P)$')
    p3 = plot(tf, data(3,*), 'b2', /overplot, name='$\bf V\rm\cdot\nabla P$')
    p4 = plot(te, dke, 'k2', /overplot, name='$dK_e\slash dt$')

    pdv = reform(data(1,*))
    ;pdv = pdv * dtwpe / dtwci
    ;p5 = plot(tf, data(0,*), font_size=16)

    data = fltarr(3,ntf)
    data1 = fltarr(3)
    fname = 'data/shear00_' + species + '.gda'
    openr, lun, fname, /get_lun
    FOR it = 0, ntf-1 DO BEGIN
        field = assoc(lun,data1)
        data1 = field(it)
        data(*,it-1) = data1
    ENDFOR
    free_lun, lun
    shear = -reform(data(1,*))
    shear = shear * dtwpe / dtwci

    p5 = plot(tf, shear, 'r2--', /overplot, name='Shear term')
    p6 = plot(tf, shear+pdv, 'k2--', /overplot, $
        name='Shear term + $P\nabla\cdot\bf V$')

    data = fltarr(1,ntf)
    data1 = fltarr(1)
    fname = 'data/jdivpdote00_' + species + '.gda'
    openr, lun, fname, /get_lun
    FOR it = 0, ntf-1 DO BEGIN
        field = assoc(lun,data1)
        data1 = field(it)
        data(*,it-1) = data1
    ENDFOR
    free_lun, lun
    jdivpdote = -reform(data)
    p7 = plot(tf, jdivpdote+jqnupara_dote+jpolar_dote, 'b2--', $
        /overplot, name='$(\nabla\cdot\bf P\rm)\times\bf B\rm\slash B^2$')

    leg1 = legend(target=[p1,p2,p3,p4,p5,p6,p7], /data, /auto_text_color, $
        font_size=16,position=[tmax*0.9, ymax*0.95])
END
;
;*******************************************************************************
; Description: procedure to decide the yrange values
;*******************************************************************************
PRO yrange_values, ymax, ymin
    if (ymax GT 0) then begin
        ymax = ymax * 1.1
    endif else begin
        ymax = ymax * 0.9
    endelse
    if (ymin GT 0) then begin
        ymin = ymin * 0.9
    endif else begin
        ymin = ymin * 1.1
    endelse
END

;*******************************************************************************
; Description: Blue-White-Red color table.
;   Originally from matlab file exchange: 
;   http://www.mathworks.com/matlabcentral/fileexchange/4058-bluewhitered
;*******************************************************************************
PRO BlueWhiteRed, rgbtable, lims
    p = FLTARR(3,5)
    p = [[0.0,0.0,0.5],[0.0,0.5,1.0],[1.0,1.0,1.0],[1.0,0.0,0.0],[0.5,0.0,0.0]]
    ;p = [[0.0,0.0,0.5],[0.0,0.5,1.0],[0.5,0.5,0.5],[1.0,0.0,0.0],[0.5,0.0,0.0]]
    m = 256
    rgbtable = FLTARR(3,m)
    lims = float(lims)
    IF ((lims(0) LT 0) AND (lims(1) GT 0)) THEN BEGIN
      ; It has both negative and positive values.
      ; Find portion of negative values.
        ratio = abs(lims(0)) / (abs(lims(0))+abs(lims(1)))
        neglen = round(m*ratio)
        poslen = m - neglen

      ; Colorbar for negative values
        cindex = FINDGEN(neglen)*2.0/neglen
        rgbtable(0,0:neglen-1) = INTERPOLATE(p(0,0:2),cindex)
        rgbtable(1,0:neglen-1) = INTERPOLATE(p(1,0:2),cindex)
        rgbtable(2,0:neglen-1) = INTERPOLATE(p(2,0:2),cindex)
        rgbtable(*,0:neglen-1) = rgbtable(*,0:neglen-1)*255

      ; Colorbar for positive values
        cindex = FINDGEN(poslen)*2.0/poslen
        rgbtable(0,neglen:m-1) = INTERPOLATE(p(0,2:4),cindex)
        rgbtable(1,neglen:m-1) = INTERPOLATE(p(1,2:4),cindex)
        rgbtable(2,neglen:m-1) = INTERPOLATE(p(2,2:4),cindex)
        rgbtable(*,neglen:m-1) = rgbtable(*,neglen:m-1)*255
    ENDIF ELSE IF (lims(0) GT 0) THEN BEGIN
      ; Just positive values
        cindex = FINDGEN(m)*2.0/m
        rgbtable(0,*) = INTERPOLATE(p(0,2:4),cindex)
        rgbtable(1,*) = INTERPOLATE(p(1,2:4),cindex)
        rgbtable(2,*) = INTERPOLATE(p(2,2:4),cindex)
        rgbtable = rgbtable*255
    ENDIF ELSE IF (lims(1) LT 0) THEN BEGIN
      ; Just negative vaules
        cindex = FINDGEN(m)*2.0/m
        rgbtable(0,*) = INTERPOLATE(p(0,2:0:-1),cindex)
        rgbtable(1,*) = INTERPOLATE(p(1,2:0:-1),cindex)
        rgbtable(2,*) = INTERPOLATE(p(2,2:0:-1),cindex)
        rgbtable = rgbtable*255
    ENDIF
END

;*******************************************************************************
; Routine to create a video
;*******************************************************************************
PRO Video_current, qname, nt, icumulation
;    width = 768
;    height = 360
;    IF (icumulation EQ 1) THEN BEGIN
;        height = 480
;    ENDIF
    height = 720
    ;width = 800
    width = 1440
    frames = nt
    fps = 10

    ; Create object and initialize video/audio streams
    fname_swf = qname + '.swf'
    fname_mp4 = qname + '.mp4'
    fname_avi = qname + '.avi'
    oVid_swf = IDLffVideoWrite(fname_swf)
    oVid_mp4 = IDLffVideoWrite(fname_mp4)
    oVid_avi = IDLffVideoWrite(fname_avi)
    vidStream_swf = oVid_swf.AddVideoStream(width, height, fps, $
        BIT_RATE=8e6)
    vidStream_mp4 = oVid_mp4.AddVideoStream(width, height, fps, $
        BIT_RATE=8e6)
    vidStream_avi = oVid_mp4.AddVideoStream(width, height, fps, $
        BIT_RATE=8e6)
   
    ; Generate video frames
    FOR i = 0, 20 DO BEGIN
        print, i
;        lims = [-1.0,1.0]
;        firehose, i, 70, lims, im
        lims = [-0.0005,0.0005]
        ;testplot, qname, i, icolor, lims, im
;        FieldsImage, 'data', 'jx', i, $
;            0, 0, 0, 0, [-0.1, 0.1], im, 0
;        FieldsImage, 'data', 'ne', i, $
;            0, 0, 0, 0, [0, 5], im, 0
        FieldsImage, 'data', 'ne', i, $
            0, 33, 1, 0, 1, [-1, 1], im, 0, 100, 1, 0
;        FieldsImage, 'data3', 'ene_curv00', i, $
;            0, 39, 1, 1, 1, [-2, 1], im, 0, 50, 1, 1
;        jDriftsDote2d, i, 0, 1, 1, 0, [], 'e', im
;        lims = 5.0
;        preanisotropy, i, 70, lims, im
;        fname = 'img/jy_all00' + STRING(i+1,FORMAT='(I2.2)') + '.png'
;        im = IMAGE(/buffer,fname,dimensions=[width,height])
        time = oVid_swf.Put(vidStream_swf, im.CopyWindow())
        time = oVid_mp4.Put(vidStream_mp4, im.CopyWindow())
        ;time = oVid_avi.Put(vidStream_avi, im.CopyWindow())
        im.Close
    ENDFOR
    ;im.Close
    
    ; Close the file
    oVid_swf.Cleanup
    oVid_mp4.Cleanup
    oVid_avi.Cleanup
END

;*******************************************************************************
; 2D plot for j dote E using calculated current density.
; Input:
;   it: time point ID.
;   ifieldline: flag for whether to plot B field lines.
;   xslice: x positions to plot vertical cut dashed lines.
;   species: 'e' for electron, 'i' for ion.
; Return:
;   im1: image object for handler.
;*******************************************************************************
PRO jdote2d, it, ifieldline, xslice, species, im1
    common picinfo
;    lims1 = [-0.2,0.2]
;    lims2 = [-0.3,0.3]
;    lims3 = [-0.4,0.4]
    lims1 = [-0.001,0.001]
    lims2 = [-0.001,0.001]
    lims3 = [-0.001,0.001]

    nd = 2.0 ; Interpolate grid size
    iheight = 350
    ys = 0.70
    hei = 0.27
    xs = 0.15
    xe = 0.73
    jname = '$!16j_{!8'

    ; first variable
    fname = '../data1/jdiagm_dote00_' + species + '.gda'
    readfield, [fname], it, nd, data1, nx1, nz1, x1, z1
    data1 = smooth(data1, 9)
    ratio = nx1*hei/(ze-zs)/(xe-xs)/1.5
    BlueWhiteRed, rgbtable, lims1
    im1 = IMAGE(data1(*,zs:ze), x1, z1(zs:ze),  $
        aspect_ratio=ratio,$
        font_size=16, xtitle='x ($d_i$)', ytitle='!8z ($d_i$)', $
        position=[xs,ys,xe,ys+hei], $
        dimensions=[500, iheight], $
        xshowtext = 0, $
        ;ytickvalues=[200, 400, 600, 800, 1000], $
        ;max_value=5.0, min_value=0.5, $
        ;position=[0.07,0.53,0.95,0.96],$
        rgb_table=rgbtable, axis_style=2, interpolate=1)
    maxdata = max(data1)
    mindata = min(data1)
    print, 'Maximum', maxdata
    print, 'Minimum', mindata
    im1.max_value = lims1(1)
    im1.min_value = lims1(0)
    pos1 = im1.position
;    ;print, pos1
;    CB1 = COLORBAR(TARGET=im1,ORIENTATION=1,$
;        position=[xe+0.01,pos1(1),xe+0.03,pos1(3)])
;    CB1.TEXTPOS = 1
;    CB1.TICKDIR = 1
;    CB1.FONT_SIZE = 16
;    CB1.TITLE=jname + 'x}$'
;
    if (ifieldline eq 1) then begin
        zs1 = nz/4
        ze1 = nz*3/4
        fname = '../data/Ay.gda'
        OPENR, lun1, fname, /get_lun
        data = FLTARR(nx, nz)

        field = assoc(lun1,data)
        data = field(it)
        cn1 = contour(data(*,zs1:ze1), x, z(zs1:ze1), /overplot, $
            c_label_show=0, color='black')
    endif

    yrmax = pos1(3)

  ; y-component
    ys = ys-hei
    fname1 = '../data1/jcpara_dote00_' + species + '.gda'
    fname2 = '../data1/jcperp_dote00_' + species + '.gda'
    readfield, [fname1, fname2], it, nd, data1, nx1, nz1, x1, z1
    data1 = smooth(data1, 9)
    ;ratio = nx1/(ze-zs)/2.0
    BlueWhiteRed, rgbtable, lims2
    im1 = IMAGE(data1(*,zs:ze), x1, z1(zs:ze), /current, $
        aspect_ratio=ratio,$
        font_size=16, xtitle='x ($d_i$)', ytitle='!8z ($d_i$)', $
        position=[xs,ys,xe,ys+hei], $
        dimensions=[768, iheight], $
        xshowtext = 0, $
        ;ytickvalues=[200, 400, 600, 800, 1000], $
        ;max_value=5.0, min_value=0.5, $
        ;position=[0.07,0.53,0.95,0.96],$
        rgb_table=rgbtable, axis_style=2, interpolate=1)
    maxdata = max(data1)
    mindata = min(data1)
    print, 'Maximum', maxdata
    print, 'Minimum', mindata
    im1.max_value = lims2(1)
    im1.min_value = lims2(0)
    pos2 = im1.position
;    ;print, pos1
;    CB1 = COLORBAR(TARGET=im1,ORIENTATION=1,$
;        position=[xe+0.01,pos1(1),xe+0.03,pos1(3)])
;    CB1.TEXTPOS = 1
;    CB1.TICKDIR = 1
;    CB1.FONT_SIZE = 16
;    CB1.TITLE=jname + 'y}$'
;    
    if (ifieldline eq 1) then begin
        cn1 = contour(data(*,zs1:ze1), x, z(zs1:ze1), /overplot, $
            c_label_show=0, color='black')
    endif

  ; z-component
    ys = ys-hei
    fname = '../data1/jqnupara_dote00_' + species + '.gda'
    readfield, [fname], it, nd, data1, nx1, nz1, x1, z1
    data1 = smooth(data1, 9)
    ;ratio = nx1/(ze-zs)/2.0
    BlueWhiteRed, rgbtable, lims3
    im1 = IMAGE(data1(*,zs:ze), x1, z1(zs:ze), /current, $
        aspect_ratio=ratio,$
        font_size=16, xtitle ='!8x ($d_i$)', ytitle='!8z ($d_i$)', $
        position=[xs,ys,xe,ys+hei], $
        ;xshowtext = 0, $
        ;ytickvalues=[200, 400, 600, 800, 1000], $
        ;max_value=5.0, min_value=0.5, $
        ;position=[0.07,0.53,0.95,0.96],$
        rgb_table=rgbtable, axis_style=2, interpolate=1)
    maxdata = max(data1)
    mindata = min(data1)
    print, 'Maximum', maxdata
    print, 'Minimum', mindata
    im1.max_value = lims3(1)
    im1.min_value = lims3(0)
    pos3 = im1.position
    ;print, pos1
    CB1 = COLORBAR(TARGET=im1,ORIENTATION=1,$
        position=[xe+0.01,pos3(1),xe+0.03,pos1(3)])
    CB1.TEXTPOS = 1
    CB1.TICKDIR = 1
    CB1.FONT_SIZE = 16
    ;CB1.TITLE=jname + 'z}$'
    CB1.TITLE='$!16j_{!8\parallel}\cdot!16E, !16j_{!8ap}\cdot!16E, !16j_{!8d}\cdot!16E$'

    if (ifieldline eq 1) then begin
        cn1 = contour(data(*,zs1:ze1), x, z(zs1:ze1), /overplot, $
            c_label_show=0, color='black')
    endif

    yrmin = pos1(1)

    sz = size(xslice)
    FOR i = 0, sz(1)-1 DO BEGIN
        p1 = PLOT([xslice(i),xslice(i)],[1.0,2.0],'k--',/current,$
            xrange=[x1(0),x1(nx1-1)], yrange=[1.0,2.0],$
            position=[pos1(0),yrmin,pos1(2),yrmax])
        ax = p1.axes
        ax[0].HIDE=1
        ax[1].HIDE=1
        ax[2].HIDE=1
        ax[3].HIDE=1
    ENDFOR

    fname = 'jdote2d' + STRING(it, FORMAT='(I2.2)') + $
        '_' + species + '.jpg'
    im1.save, fname, resolution=300
END

;*******************************************************************************
; 2D plot for j dote E using calculated current density due to particle drifts.
; Input:
;   it: time point ID.
;   ifieldline: flag for whether to plot B field lines.
;   xslice: x positions to plot vertical cut dashed lines.
;   species: 'e' for electron, 'i' for ion.
;   isBuffer: whether to buffer the image rather than display it.
;   isSave: whether to save the image after plotting.
;   isClose: whether to close the image after plotting.
; Return:
;   im1: image object for handler.
;*******************************************************************************
PRO jDriftsDote2d, it, ifieldline, isBuffer, isSave, isClose, xslice, species, im1
    common picinfo
;    lims1 = [-0.2,0.2]
;    lims2 = [-0.3,0.3]
;    lims3 = [-0.4,0.4]
    lims1 = [-0.001,0.001]
    lims2 = [-0.001,0.001]
    lims3 = [-0.001,0.001]

    nd = 2.0 ; Interpolate grid size
    iheight = 720
    ys = 0.75
    hei = 0.21
    xs = 0.12
    xe = 0.80
    jname = '$!16j_{!8'

    ; curvature drift
    fname = '../data1/jcpara_dote00_' + species + '.gda'
    readfield, [fname], it, nd, data1, nx1, nz1, x1, z1
    zs = nz1/4
    ze = zs + nz1/2
    data1 = smooth(data1, [3,3])
    ratio = (ze-zs) / nx1
    BlueWhiteRed, rgbtable, lims1
    accumulatedJdotE1 = total(data1, 2)
    accumulatedJdotE1 = total(accumulatedJdotE1, /CUMULATIVE)
    if (isBuffer eq 1) then begin
        im1 = IMAGE(data1(*,zs:ze), x1, z1(zs:ze), /buffer, $
            font_size=16, dimensions=[800, iheight], $
            axis_style=2)
    endif else begin
        im1 = IMAGE(data1(*,zs:ze), x1, z1(zs:ze), $
            font_size=16, dimensions=[800, iheight], $
            axis_style=2)
    endelse
    im1.aspect_ratio=ratio
    im1.xtitle='x ($d_i$)'
    im1.ytitle='!8z ($d_i$)'
    im1.position=[xs,ys,xe,ys+hei]
    im1.xshowtext = 0
    im1.rgb_table=rgbtable
    im1.interpolate=1
    maxdata = max(data1)
    mindata = min(data1)
    print, 'Maximum', maxdata
    print, 'Minimum', mindata
    im1.max_value = lims1(1)
    im1.min_value = lims1(0)
    pos1 = im1.position
    if (ifieldline eq 1) then begin
        zs1 = nz/4
        ze1 = nz*3/4
        fname = '../data/Ay.gda'
        ;OPENR, lun1, fname, /get_lun
        ;data = FLTARR(nx, nz)
        readfield, [fname], it, nd, dataAy, nx1, nz1, x1, z1

        ;field = assoc(lun1,data)
        ;data = field(it)
        cn1 = contour(dataAy(*,zs:ze), x1, z1(zs:ze), /overplot, $
            c_label_show=0, color='black')
    endif

    yrmax = pos1(3)

  ; gradient drift
    ys = ys-hei
    fname1 = '../data1/jgrad_dote00_' + species + '.gda'
    readfield, [fname1], it, nd, data1, nx1, nz1, x1, z1
    data1 = smooth(data1, [3,3])
    ;ratio = nx1/(ze-zs)/2.0
    BlueWhiteRed, rgbtable, lims2
    accumulatedJdotE2 = total(data1, 2)
    accumulatedJdotE2 = total(accumulatedJdotE2, /CUMULATIVE)
    im1 = IMAGE(data1(*,zs:ze), x1, z1(zs:ze), /current, $
        aspect_ratio=ratio,$
        font_size=16, xtitle='x ($d_i$)', ytitle='!8z ($d_i$)', $
        position=[xs,ys,xe,ys+hei], $
        dimensions=[768, iheight], $
        xshowtext = 0, $
        ;ytickvalues=[200, 400, 600, 800, 1000], $
        ;max_value=5.0, min_value=0.5, $
        ;position=[0.07,0.53,0.95,0.96],$
        rgb_table=rgbtable, axis_style=2, interpolate=1)
    maxdata = max(data1)
    mindata = min(data1)
    print, 'Maximum', maxdata
    print, 'Minimum', mindata
    im1.max_value = lims2(1)
    im1.min_value = lims2(0)
    pos2 = im1.position
;    ;print, pos1
;    CB1 = COLORBAR(TARGET=im1,ORIENTATION=1,$
;        position=[xe+0.01,pos1(1),xe+0.03,pos1(3)])
;    CB1.TEXTPOS = 1
;    CB1.TICKDIR = 1
;    CB1.FONT_SIZE = 16
;    CB1.TITLE=jname + 'y}$'
;    
    if (ifieldline eq 1) then begin
        cn1 = contour(dataAy(*,zs:ze), x1, z1(zs:ze), /overplot, $
            c_label_show=0, color='black')
    endif

  ; magnetization
    ys = ys-hei
    fname = '../data1/jmag_dote00_' + species + '.gda'
    readfield, [fname], it, nd, data1, nx1, nz1, x1, z1
    data1 = smooth(data1, [3,3])
    ;ratio = nx1/(ze-zs)/2.0
    BlueWhiteRed, rgbtable, lims3
    accumulatedJdotE3 = total(data1, 2)
    accumulatedJdotE3 = total(accumulatedJdotE3, /CUMULATIVE)
    im1 = IMAGE(data1(*,zs:ze), x1, z1(zs:ze), /current, $
        aspect_ratio=ratio,$
        font_size=16, xtitle ='!8x ($d_i$)', ytitle='!8z ($d_i$)', $
        position=[xs,ys,xe,ys+hei], $
        xshowtext = 0, $
        ;ytickvalues=[200, 400, 600, 800, 1000], $
        ;max_value=5.0, min_value=0.5, $
        ;position=[0.07,0.53,0.95,0.96],$
        rgb_table=rgbtable, axis_style=2, interpolate=1)
    maxdata = max(data1)
    mindata = min(data1)
    print, 'Maximum', maxdata
    print, 'Minimum', mindata
    im1.max_value = lims3(1)
    im1.min_value = lims3(0)
    pos3 = im1.position
    ;print, pos1
    CB1 = COLORBAR(TARGET=im1,ORIENTATION=1,$
        position=[xe+0.01,pos3(1),xe+0.03,pos1(3)])
    CB1.TEXTPOS = 1
    CB1.TICKDIR = 1
    CB1.FONT_SIZE = 16
    ;CB1.TITLE=jname + 'z}$'
    ;CB1.TITLE='$!16j_{!8m}\cdot!16E$, $!16j_{!8g}\cdot!16E$, $!16j_{!8c}\cdot!16E$'

    t1 = text(xe+0.17, ys+hei/2, '$!16j_{!8m}\cdot!16E$', $
        font_size=16, color='b', orientation=90)
    t1 = text(xe+0.17, ys+hei*3/2, '$!16j_{!8g}\cdot!16E$', $
        font_size=16, color='g', orientation=90)
    t1 = text(xe+0.17, ys+hei*5/2, '$!16j_{!8c}\cdot!16E$', $
        font_size=16, color='r', orientation=90)

    if (ifieldline eq 1) then begin
        cn1 = contour(dataAy(*,zs:ze), x1, z1(zs:ze), /overplot, $
            c_label_show=0, color='black')
    endif

  ; Accumulation of the data over y

    p1 = plot(x1, accumulatedJdotE1, 'r2', font_size=16, /current, $
        xtitle ='!8x ($d_i$)', ytitle='Accumulation', axis_style=2,$
        position=[xs,ys-hei,xe,ys], name='$!16j_{!8c}\cdot!16E$')
    p2 = plot(x1, accumulatedJdotE2, 'g2', /overplot, name='$!16j_{!8g}\cdot!16E$')
    p3 = plot(x1, accumulatedJdotE3, 'b2', /overplot, name='$!16j_{!8m}\cdot!16E$')

    leg1 = legend(target=[p1,p2,p3], /auto_text_color, $
        font_size=16, position=[0.94, 0.30], transparency=100)

;    yrmin = pos1(1)

;    sz = size(xslice)
;    FOR i = 0, sz(1)-1 DO BEGIN
;        p1 = PLOT([xslice(i),xslice(i)],[1.0,2.0],'k--',/current,$
;            xrange=[x1(0),x1(nx1-1)], yrange=[1.0,2.0],$
;            position=[pos1(0),yrmin,pos1(2),yrmax])
;        ax = p1.axes
;        ax[0].HIDE=1
;        ax[1].HIDE=1
;        ax[2].HIDE=1
;        ax[3].HIDE=1
;    ENDFOR

    if (isSave eq 1) then begin
        fname = 'jDriftsDote2d' + STRING(it, FORMAT='(I3.3)') + $
            '_' + species + '.jpg'
        im1.save, fname, resolution=300
    endif
    if (isClose eq 1) then begin
        im1.close
    endif
END

;*******************************************************************************
; 2D plot for j dote E using calculated parallel and perpendicular current
; density.
; Input:
;   it: time point ID.
;   ifieldline: flag for whether to plot B field lines.
;   xslice: x positions to plot vertical cut dashed lines.
;   species: 'e' for electron, 'i' for ion.
; Return:
;   im1: image object for handler.
;*******************************************************************************
PRO jParaPerpDote2d, it, ifieldline, xslice, species, im1
    common picinfo
;    lims1 = [-0.2,0.2]
;    lims2 = [-0.3,0.3]
;    lims3 = [-0.4,0.4]
    lims1 = [-0.001,0.001]
    lims2 = [-0.001,0.001]
    lims3 = [-0.001,0.001]

    nd = 2.0 ; Interpolate grid size
    iheight = 720
    ys = 0.75
    hei = 0.21
    xs = 0.15
    xe = 0.80
    jname = '$!16j_{!8'

    ; parallel current 
    fname = '../data1/jqnupara_dote00_' + species + '.gda'
    readfield, [fname], it, nd, data1, nx1, nz1, x1, z1
    data1 = smooth(data1, [5,5])
    ratio = (ze-zs) / nx1
    BlueWhiteRed, rgbtable, lims1
    accumulatedJdotE1 = total(data1, 2)
    accumulatedJdotE1 = total(accumulatedJdotE1, /CUMULATIVE)

    ; Total current. The perpendicular current will be inclued latter.
    jqnu_tot = data1 

    im1 = IMAGE(data1(*,zs:ze), x1, z1(zs:ze),  $
        aspect_ratio=ratio,$
        font_size=16, xtitle='x ($d_i$)', ytitle='!8z ($d_i$)', $
        position=[xs,ys,xe,ys+hei], $
        dimensions=[800, iheight], $
        xshowtext = 0, $
        ;ytickvalues=[200, 400, 600, 800, 1000], $
        ;max_value=5.0, min_value=0.5, $
        ;position=[0.07,0.53,0.95,0.96],$
        rgb_table=rgbtable, axis_style=2, interpolate=1)
    maxdata = max(data1)
    mindata = min(data1)
    print, 'Maximum', maxdata
    print, 'Minimum', mindata
    im1.max_value = lims1(1)
    im1.min_value = lims1(0)
    pos1 = im1.position
    if (ifieldline eq 1) then begin
        zs1 = nz/4
        ze1 = nz*3/4
        fname = '../data/Ay.gda'
        ;OPENR, lun1, fname, /get_lun
        ;data = FLTARR(nx, nz)
        readfield, [fname], it, nd, dataAy, nx1, nz1, x1, z1

        ;field = assoc(lun1,data)
        ;data = field(it)
        cn1 = contour(dataAy(*,zs:ze), x1, z1(zs:ze), /overplot, $
            c_label_show=0, color='black')
    endif

    yrmax = pos1(3)

  ; perpendicular current
    ys = ys-hei
    fname1 = '../data1/jqnuperp_dote00_' + species + '.gda'
    readfield, [fname1], it, nd, data1, nx1, nz1, x1, z1
    data1 = smooth(data1, [5,5])
    jqnu_tot = jqnu_tot + data1 ; Total current

    ;ratio = nx1/(ze-zs)/2.0
    BlueWhiteRed, rgbtable, lims2
    accumulatedJdotE2 = total(data1, 2)
    accumulatedJdotE2 = total(accumulatedJdotE2, /CUMULATIVE)
    im1 = IMAGE(data1(*,zs:ze), x1, z1(zs:ze), /current, $
        aspect_ratio=ratio,$
        font_size=16, xtitle='x ($d_i$)', ytitle='!8z ($d_i$)', $
        position=[xs,ys,xe,ys+hei], $
        dimensions=[768, iheight], $
        xshowtext = 0, $
        ;ytickvalues=[200, 400, 600, 800, 1000], $
        ;max_value=5.0, min_value=0.5, $
        ;position=[0.07,0.53,0.95,0.96],$
        rgb_table=rgbtable, axis_style=2, interpolate=1)
    maxdata = max(data1)
    mindata = min(data1)
    print, 'Maximum', maxdata
    print, 'Minimum', mindata
    im1.max_value = lims2(1)
    im1.min_value = lims2(0)
    pos2 = im1.position
;    ;print, pos1
;    CB1 = COLORBAR(TARGET=im1,ORIENTATION=1,$
;        position=[xe+0.01,pos1(1),xe+0.03,pos1(3)])
;    CB1.TEXTPOS = 1
;    CB1.TICKDIR = 1
;    CB1.FONT_SIZE = 16
;    CB1.TITLE=jname + 'y}$'
;    
    if (ifieldline eq 1) then begin
        cn1 = contour(dataAy(*,zs:ze), x1, z1(zs:ze), /overplot, $
            c_label_show=0, color='black')
    endif

  ; total current
    ys = ys-hei
    BlueWhiteRed, rgbtable, lims3
    accumulatedJdotE3 = total(jqnu_tot, 2)
    accumulatedJdotE3 = total(accumulatedJdotE3, /CUMULATIVE)
    im1 = IMAGE(jqnu_tot(*,zs:ze), x1, z1(zs:ze), /current, $
        aspect_ratio=ratio,$
        font_size=16, xtitle ='!8x ($d_i$)', ytitle='!8z ($d_i$)', $
        position=[xs,ys,xe,ys+hei], $
        xshowtext = 0, $
        ;ytickvalues=[200, 400, 600, 800, 1000], $
        ;max_value=5.0, min_value=0.5, $
        ;position=[0.07,0.53,0.95,0.96],$
        rgb_table=rgbtable, axis_style=2, interpolate=1)
    maxdata = max(data1)
    mindata = min(data1)
    print, 'Maximum', maxdata
    print, 'Minimum', mindata
    im1.max_value = lims3(1)
    im1.min_value = lims3(0)
    pos3 = im1.position
    ;print, pos1
    CB1 = COLORBAR(TARGET=im1,ORIENTATION=1,$
        position=[xe+0.01,pos3(1),xe+0.03,pos1(3)])
    CB1.TEXTPOS = 1
    CB1.TICKDIR = 1
    CB1.FONT_SIZE = 16
    ;CB1.TITLE=jname + 'z}$'
    ;CB1.TITLE='$!16j_{!8\parallel}\cdot!16E, !16j_{!8\perp}\cdot!16E, !16j_{!8 tot}\cdot!16E$'

    t1 = text(xe+0.17, ys+hei/2, '$!16j_{!8tot}\cdot!16E$', $
        font_size=16, color='b', orientation=90)
    t1 = text(xe+0.17, ys+hei*3/2, '$!16j_{!8\perp}\cdot!16E$', $
        font_size=16, color='g', orientation=90)
    t1 = text(xe+0.17, ys+hei*5/2, '$!16j_{!8\parallel}\cdot!16E$', $
        font_size=16, color='r', orientation=90)

    if (ifieldline eq 1) then begin
        cn1 = contour(dataAy(*,zs:ze), x1, z1(zs:ze), /overplot, $
            c_label_show=0, color='black')
    endif

  ; Accumulation of the data over y

    p1 = plot(x1, accumulatedJdotE1, 'r2', font_size=16, /current, $
        xtitle ='!8x ($d_i$)', ytitle='Accumulation', axis_style=2,$
        position=[xs,ys-hei,xe,ys], name='$!16j_{!8\parallel}\cdot!16E$')
    p2 = plot(x1, accumulatedJdotE2, 'g2', /overplot, name='$!16j_{!8\perp}\cdot!16E$')
    p3 = plot(x1, accumulatedJdotE3, 'b2', /overplot, name='$!16j_{!8 tot}\cdot!16E$')

    leg1 = legend(target=[p1,p2,p3], /auto_text_color, $
        font_size=16, position=[0.94, 0.30], transparency=100)

;    yrmin = pos1(1)

;    sz = size(xslice)
;    FOR i = 0, sz(1)-1 DO BEGIN
;        p1 = PLOT([xslice(i),xslice(i)],[1.0,2.0],'k--',/current,$
;            xrange=[x1(0),x1(nx1-1)], yrange=[1.0,2.0],$
;            position=[pos1(0),yrmin,pos1(2),yrmax])
;        ax = p1.axes
;        ax[0].HIDE=1
;        ax[1].HIDE=1
;        ax[2].HIDE=1
;        ax[3].HIDE=1
;    ENDFOR

    fname = 'jParaPerpDote2d' + STRING(it, FORMAT='(I3.3)') + $
        '_' + species + '.jpg'
    im1.save, fname, resolution=300
END

;*******************************************************************************
; 1D plot for j dote E using calculated current density due to particle drifts.
; The data is a cut either along x or along y.
; Input:
;   it: time point ID.
;   xslice: x positions of the cuts along z directions.
;   zslice: z positions of the cuts along x directions.
;   species: 'e' for electron, 'i' for ion.
; Return:
;   im1: image object for handler.
;*******************************************************************************
PRO jDriftsDote1d, it, xslice, zslice, species, im1
    common picinfo

    nd = 1 ; Interpolate grid size
    sz = size(xslice)
    szx = sz(1)
    sz = size(zslice)
    szz = sz(1)
    x = findgen(nx) * dx
    z = findgen(nz) * dz
    xslice = floor(xslice/dx)
    zslice = floor(zslice/dz)
    ; curvature drift
    fname = '../data1/jcpara_dote00_' + species + '.gda'
    readOriginalField, [fname], it, data
    datax1 = fltarr(nx,szz)
    dataz1 = fltarr(nz,szx)
    for ix = 0, szx - 1 do begin
        dataz1(*,ix) = data(xslice[ix],*)
    endfor
    for iz = 0, szz - 1 do begin
        datax1(*,iz) = data(*,zslice[iz])
    endfor

    ; gradient drift
    fname = '../data1/jgrad_dote00_' + species + '.gda'
    readOriginalField, [fname], it, data
    datax2 = fltarr(nx,szz)
    dataz2 = fltarr(nz,szx)
    for ix = 0, szx - 1 do begin
        dataz2(*,ix) = data(xslice[ix],*)
    endfor
    for iz = 0, szz - 1 do begin
        datax2(*,iz) = data(*,zslice[iz])
    endfor

    ; magnetization
    fname = '../data1/jdiagm_dote00_' + species + '.gda'
    readOriginalField, [fname], it, data
    datax3 = fltarr(nx,szz)
    dataz3 = fltarr(nz,szx)
    for ix = 0, szx - 1 do begin
        dataz3(*,ix) = data(xslice[ix],*)
    endfor
    for iz = 0, szz - 1 do begin
        datax3(*,iz) = data(*,zslice[iz])
    endfor

;    datax1 = smooth(datax1, [5, 1])
;    datax2 = smooth(datax2, [5, 1])
;    datax3 = smooth(datax3, [5, 1])
    dataz1 = smooth(dataz1, [3, 1])
    dataz2 = smooth(dataz2, [3, 1])
    dataz3 = smooth(dataz3, [3, 1])

    zlims = [45, 55]
    xlims = [0, 200]
    xs = 0.18
    xe = 0.95
    ys = 0.16
    ye = 0.92
    ymin1 = fltarr(3)
    ymax1 = fltarr(3)
    ; Cut along z
    for ix = 0, szx - 1 do begin
        p1 = plot(z, dataz1(*,ix), 'r2', font_size=16, $
            xtitle='$!8 z(d_i)$', ytitle='Energy conversions', $
            position=[xs, ys, xe, ye], $
            dimension=[500, 400], $
            ytickunits='numeric', $
            xrange=zlims, name='$!16j_{!8c}\cdot!16E$')
        p2 = plot(z, dataz2(*,ix), 'g2', /overplot, $
            name='$!16j_{!8g}\cdot!16E$')
        p3 = plot(z, dataz3(*,ix), 'b2', /overplot, $
            name='$!16j_{!8m}\cdot!16E$')
        ymin1(0) = min(dataz1(*,ix))
        ymin1(1) = min(dataz2(*,ix))
        ymin1(2) = min(dataz3(*,ix))
        ymax1(0) = max(dataz1(*,ix))
        ymax1(1) = max(dataz2(*,ix))
        ymax1(2) = max(dataz3(*,ix))
        ymin = min(ymin1)
        ymax = max(ymax1)
        yrange_values, ymax, ymin
        p1.yrange = [ymin, ymax]
        leg1 = legend(target=[p1,p2,p3], font_size=16, $
            position=[xe-0.05, ye-0.02], /auto_text_color, $
            transparency=100)
        tickvalues = p1.ytickvalues
        sz = size(tickvalues)
        ticknames = strarr(sz(1)-1)
        ticknames = string(tickvalues*1E3, format='(F4.1)')
        p1.ytickname = ticknames
        t1 = text(xs, ye, '$\times 10^{-3}$', font_size=16)
    endfor
    
;    ; Cut along x
;    for iz = 0, szz - 1 do begin
;        p1 = plot(x, datax1(*,iz), 'r2', font_size=16, $
;            xtitle='$!8 x(d_i)$', ytitle='Energy conversions', $
;            position=[xs, ys, xe, ye], $
;            dimension=[500, 400], $
;            ytickunits='numeric', $
;            xrange=xlims, name='$!16j_{!8c}\cdot!16E$')
;        p2 = plot(x, datax2(*,iz), 'g2', /overplot, $
;            name='$!16j_{!8g}\cdot!16E$')
;        p3 = plot(x, datax3(*,iz), 'b2', /overplot, $
;            name='$!16j_{!8m}\cdot!16E$')
;        ymin1(0) = min(datax1(*,iz))
;        ymin1(1) = min(datax2(*,iz))
;        ymin1(2) = min(datax3(*,iz))
;        ymax1(0) = max(datax1(*,iz))
;        ymax1(1) = max(datax2(*,iz))
;        ymax1(2) = max(datax3(*,iz))
;
;        ymin = min(ymin1)
;        ymax = max(ymax1)
;        yrange_values, ymax, ymin
;        p1.yrange = [ymin, ymax]
;        leg1 = legend(target=[p1,p2,p3], font_size=16, $
;            position=[xe-0.05, ye-0.02], /auto_text_color, $
;            transparency=100)
;        tickvalues = p1.ytickvalues
;        sz = size(tickvalues)
;        ticknames = strarr(sz(1)-1)
;        ticknames = string(tickvalues*1E2, format='(F4.1)')
;        p1.ytickname = ticknames
;        t1 = text(xs, ye, '$\times 10^{-2}$', font_size=16)
;    endfor
END

;*******************************************************************************
; 2D plot for resonance energy with field curvature, out-of-plan electric field
; and particle number density.
; Input:
;   it: time point ID.
;   ifieldline: flag for whether to plot B field lines.
;   xslice: x positions to plot vertical cut dashed lines.
;   species: 'e' for electron, 'i' for ion.
; Return:
;   im1: image object for handler.
;*******************************************************************************
PRO EnergyEfieldRho, it, ifieldline, xslice, species, im1
    common picinfo
;    lims1 = [-0.2,0.2]
;    lims2 = [-0.3,0.3]
;    lims3 = [-0.4,0.4]
    lims1 = [-2, 1]
    lims2 = [-0.1,0.1]
    lims3 = [0.25,6]

    nd = 2.0 ; Interpolate grid size
    iheight = 540
    ys = 0.68
    hei = 0.28
    xs = 0.12
    xe = 0.85
    shift1 = 0.02

    ; parallel current 
    fname = '../data3/ene_curv00.gda'
    readfield, [fname], it, nd, data1, nx1, nz1, x1, z1
    data1 = smooth(data1, [5,5])
    ratio = (ze-zs) / nx1
    ;BlueWhiteRed, rgbtable, lims1

    im1 = IMAGE(alog10(data1(*,zs:ze)), x1, z1(zs:ze),  $
        aspect_ratio=ratio,$
        font_size=16, xtitle='x ($d_i$)', ytitle='!8z ($d_i$)', $
        position=[xs,ys,xe,ys+hei], $
        dimensions=[800, iheight], $
        xshowtext = 0, $
        ;ytickvalues=[200, 400, 600, 800, 1000], $
        ;max_value=5.0, min_value=0.5, $
        ;position=[0.07,0.53,0.95,0.96],$
        rgb_table=39, axis_style=2, interpolate=1)
    maxdata = max(data1)
    mindata = min(data1)
    print, 'Maximum', maxdata
    print, 'Minimum', mindata
    im1.max_value = lims1(1)
    im1.min_value = lims1(0)
    pos1 = im1.position
    if (ifieldline eq 1) then begin
        zs1 = nz/4
        ze1 = nz*3/4
        fname = '../data/Ay.gda'
        ;OPENR, lun1, fname, /get_lun
        ;data = FLTARR(nx, nz)
        readfield, [fname], it, nd, dataAy, nx1, nz1, x1, z1

        ;field = assoc(lun1,data)
        ;data = field(it)
        cn1 = contour(dataAy(*,zs:ze), x1, z1(zs:ze), /overplot, $
            c_label_show=0, color='black')
    endif
    ;print, pos1
    CB1 = COLORBAR(TARGET=im1,ORIENTATION=1,$
        position=[xe+0.01,pos1(1)+shift1,xe+0.03,pos1(3)-shift1])
    CB1.TEXTPOS = 1
    CB1.TICKDIR = 1
    CB1.FONT_SIZE = 16
    cb1.tickvalues = [-2, -1, 0, 1]
    cb1.tickname = ['0.01', '0.10', '1.00', '10.0']

    yrmax = pos1(3)

  ; out of plan electric field
    ys = ys-hei
    fname1 = '../data/ey.gda'
    readfield, [fname1], it, nd, data1, nx1, nz1, x1, z1
    data1 = smooth(data1, [5,5])

    ;ratio = nx1/(ze-zs)/2.0
    BlueWhiteRed, rgbtable, lims2
    im1 = IMAGE(data1(*,zs:ze), x1, z1(zs:ze), /current, $
        aspect_ratio=ratio,$
        font_size=16, xtitle='x ($d_i$)', ytitle='!8z ($d_i$)', $
        position=[xs,ys,xe,ys+hei], $
        dimensions=[768, iheight], $
        xshowtext = 0, $
        ;ytickvalues=[200, 400, 600, 800, 1000], $
        ;max_value=5.0, min_value=0.5, $
        ;position=[0.07,0.53,0.95,0.96],$
        rgb_table=rgbtable, axis_style=2, interpolate=1)
    maxdata = max(data1)
    mindata = min(data1)
    print, 'Maximum', maxdata
    print, 'Minimum', mindata
    im1.max_value = lims2(1)
    im1.min_value = lims2(0)
    pos2 = im1.position
    ;print, pos1
    CB1 = COLORBAR(TARGET=im1,ORIENTATION=1,$
        position=[xe+0.01,pos2(1)+shift1,xe+0.03,pos2(3)-shift1])
    CB1.TEXTPOS = 1
    CB1.TICKDIR = 1
    CB1.FONT_SIZE = 16
    
    if (ifieldline eq 1) then begin
        cn1 = contour(dataAy(*,zs:ze), x1, z1(zs:ze), /overplot, $
            c_label_show=0, color='black')
    endif

  ; number density
    ys = ys-hei
    fname1 = '../data/ne.gda'
    readfield, [fname1], it, nd, data1, nx1, nz1, x1, z1
    data1 = smooth(data1, [5,5])
    ;BlueWhiteRed, rgbtable, lims3
    im1 = IMAGE(data1(*,zs:ze), x1, z1(zs:ze), /current, $
        aspect_ratio=ratio,$
        font_size=16, xtitle ='!8x ($d_i$)', ytitle='!8z ($d_i$)', $
        position=[xs,ys,xe,ys+hei], $
        ;xshowtext = 1, $
        ;ytickvalues=[200, 400, 600, 800, 1000], $
        ;max_value=5.0, min_value=0.5, $
        ;position=[0.07,0.53,0.95,0.96],$
        rgb_table=5, axis_style=2, interpolate=1)
    maxdata = max(data1)
    mindata = min(data1)
    print, 'Maximum', maxdata
    print, 'Minimum', mindata
    im1.max_value = lims3(1)
    im1.min_value = lims3(0)
    pos3 = im1.position
    ;print, pos1
    CB1 = COLORBAR(TARGET=im1,ORIENTATION=1,$
        position=[xe+0.01,pos3(1)+shift1,xe+0.03,pos3(3)-shift1])
    CB1.TEXTPOS = 1
    CB1.TICKDIR = 1
    CB1.FONT_SIZE = 16
    ;CB1.TITLE=jname + 'z}$'
    ;CB1.TITLE='$!16j_{!8\parallel}\cdot!16E, !16j_{!8\perp}\cdot!16E, !16j_{!8 tot}\cdot!16E$'

    shift2 = 0.13
    t1 = text(xe+shift2, ys+hei/2, '$!8n_e$', $
        font_size=16, color='k', orientation=90)
    t1 = text(xe+shift2, ys+hei*3/2, '$!8E_y$', $
        font_size=16, color='k', orientation=90)
    t1 = text(xe+shift2, ys+hei*5/2, '$!8E_r$', $
        font_size=16, color='k', orientation=90)

    if (ifieldline eq 1) then begin
        cn1 = contour(dataAy(*,zs:ze), x1, z1(zs:ze), /overplot, $
            c_label_show=0, color='black')
    endif

;    sz = size(xslice)
;    FOR i = 0, sz(1)-1 DO BEGIN
;        p1 = PLOT([xslice(i),xslice(i)],[1.0,2.0],'k--',/current,$
;            xrange=[x1(0),x1(nx1-1)], yrange=[1.0,2.0],$
;            position=[pos1(0),yrmin,pos1(2),yrmax])
;        ax = p1.axes
;        ax[0].HIDE=1
;        ax[1].HIDE=1
;        ax[2].HIDE=1
;        ax[3].HIDE=1
;    ENDFOR

    fname = 'EnergyEfieldRho2d' + STRING(it, FORMAT='(I3.3)') + $
        '_' + species + '.jpg'
    im1.save, fname, resolution=300
END

;*******************************************************************************
; 2D plot for total particle number density and higher energy percentage.
; Input:
;   it: time point ID.
;   ifieldline: flag for whether to plot B field lines.
;   xslice: x positions to plot vertical cut dashed lines.
;   species: 'e' for electron, 'i' for ion.
; Return:
;   im1: image object for handler.
;*******************************************************************************
PRO ParticleDensity2dPlot, it, ifieldline, xslice, species, im1
    common picinfo
;    lims1 = [-0.2,0.2]
;    lims2 = [-0.3,0.3]
;    lims3 = [-0.4,0.4]
;    lims1 = [0.15, 2.5]
;    lims1 = [-0.5, 0.5]
;    lims1 = [0.01, 2.0]
    lims1 = [-2, 1]
    lims2 = [0, 0.8]

    nd = 2.0 ; Interpolate grid size
    iwidth = 512
    iheight = 256
    ys = 0.58
    hei = 0.38
    xs = 0.12
    xe = 0.85
    shift1 = 0.02

    fontsize=16

    ; total particle number density
    fname = '../data/pe-xx.gda'
    readfield, [fname], it, nd, data1, nx1, nz1, x1, z1
    data1 = smooth(data1, [5,5])
    data = data1
    fname = '../data/pe-yy.gda'
    readfield, [fname], it, nd, data1, nx1, nz1, x1, z1
    data1 = smooth(data1, [5,5])
    data = data + data1
    fname = '../data/pe-zz.gda'
    readfield, [fname], it, nd, data1, nx1, nz1, x1, z1
    data1 = smooth(data1, [5,5])
    data = data + data1

    fname = '../data/absB.gda'
    readfield, [fname], it, nd, btot, nx1, nz1, x1, z1
    btot = smooth(btot, [5,5])

    plasma_beta = (data*2.0) / (3.0*btot^2)


    zs = nz1*2/5
    ze = nz1*3/5
    ratio = (ze-zs) / nx1
    BlueWhiteRed, rgbtable, lims1

    fname = 'beta_e_' + string(it, '(I4.4)') + '.gda'
    Save2DfieldData, nx1, ze-zs+1, x1, z1(zs:ze), plasma_beta(*,zs:ze), fname

    data1 = alog10(plasma_beta)
    im1 = IMAGE(data1(*,zs:ze), x1, z1(zs:ze),  $
        aspect_ratio=ratio,$
        font_size=fontsize, xtitle='x ($d_i$)', ytitle='!8z !7(!8$d_i$!7)', $
        position=[xs,ys,xe,ys+hei], $
        dimensions=[iwidth, iheight], $
        xshowtext = 0, $
        ;ytickvalues=[200, 400, 600, 800, 1000], $
        ;max_value=5.0, min_value=0.5, $
        ;position=[0.07,0.53,0.95,0.96],$
        rgb_table=33, axis_style=2, interpolate=1)
    ;rgbtable = im1.rgb_table
    ;rgbtable1 = rgbtable(*, 255:0:-1)
    ;im1.rgb_table = rgbtable1
    maxdata = max(data1)
    mindata = min(data1)
    print, 'Maximum', maxdata
    print, 'Minimum', mindata
    im1.max_value = lims1(1)
    im1.min_value = lims1(0)
    pos1 = im1.position
    if (ifieldline eq 1) then begin
        zs1 = nz/4
        ze1 = nz*3/4
        fname = '../data/Ay.gda'
        ;OPENR, lun1, fname, /get_lun
        ;data = FLTARR(nx, nz)
        readfield, [fname], it, nd, dataAy, nx1, nz1, x1, z1

        ;field = assoc(lun1,data)
        ;data = field(it)
        cn1 = contour(dataAy(*,zs:ze), x1, z1(zs:ze), /overplot, $
            c_label_show=0, color='white', c_thick=[0.5])

        fname = 'Ay_' + string(it, '(I4.4)') + '.gda'
        Save2DfieldData, nx1, ze-zs+1, x1, z1(zs:ze), dataAy(*,zs:ze), fname
    endif

    ;print, pos1
    CB1 = COLORBAR(TARGET=im1,ORIENTATION=1,$
        position=[xe+0.01,pos1(1)+shift1,xe+0.03,pos1(3)-shift1])
    CB1.TEXTPOS = 1
    CB1.TICKDIR = 1
    CB1.FONT_SIZE = fontsize
    values = cb1.tickvalues
    cb1.tickvalues = [-2, -1, 0, 1]
    ;cb1.tickname = ['$10^{-2}$', '$10^{-1}$', '$1$', '$10^{1}$' ]
    cb1.tickname = ['0.01', '0.1', '1', '10' ]

    xmid = (min(x1) + max(x1)) * 0.5
    zmid = (min(z1) + max(z1)) * 0.5
    plot_rectangle, [xmid, zmid], 160, 2, 'k', 1
    arrow1 = ARROW([[xmid-4, zmid-4],[xmid,zmid-1]], $
        TARGET=im1, /DATA, COLOR='white', THICK=1, head_size=0.2)
    ; Draw the text
    landtext1 = TEXT(xmid-6, zmid-7, TARGET=im1, $
           '4', /DATA, COLOR='white', FONT_SIZE=fontsize)

    xmid = 46.92
    plot_rectangle, [xmid, zmid], 1, 0.5, 'k', 1
    arrow2 = ARROW([[xmid-6, zmid-4],[xmid,zmid-0.5]], $
        TARGET=im1, /DATA, COLOR='white', THICK=1, head_size=0.2)
    ; Draw the text
    landtext2 = TEXT(xmid-7, zmid-7, TARGET=im1, $
           '3', /DATA, COLOR='white', FONT_SIZE=fontsize)

    xmid = 50.83
    plot_rectangle, [xmid, zmid], 1, 0.5, 'k', 1
    arrow3 = ARROW([[xmid, zmid-4],[xmid,zmid-0.5]], $
        TARGET=im1, /DATA, COLOR='white', THICK=1, head_size=0.2)
    ; Draw the text
    landtext3 = TEXT(xmid-1, zmid-7, TARGET=im1, $
           '2', /DATA, COLOR='white', FONT_SIZE=fontsize)

    xmid = 53.76
    plot_rectangle, [xmid, zmid], 1, 0.5, 'k', 1
    arrow4 = ARROW([[xmid+6, zmid-4],[xmid,zmid-0.5]], $
        TARGET=im1, /DATA, COLOR='white', THICK=1, head_size=0.2)
    ; Draw the text
    landtext4 = TEXT(xmid+5, zmid-7, TARGET=im1, $
           '1', /DATA, COLOR='white', FONT_SIZE=fontsize)

    yrmax = pos1(3)

  ; high energy particle percentage
    ys = ys-hei-0.02
    fname1 = '../data/eEB05.gda'
    readfield, [fname1], it, nd, data1, nx1, nz1, x1, z1
    data1 = smooth(data1, [5,5])

    fname = 'ne_' + string(it, '(I4.4)') + '.gda'
    Save2DfieldData, nx1, ze-zs+1, x1, z1(zs:ze), data1(*,zs:ze), fname

    ;ratio = nx1/(ze-zs)/2.0
    BlueWhiteRed, rgbtable, lims2
    im1 = IMAGE(data1(*,zs:ze), x1, z1(zs:ze), /current, $
        aspect_ratio=ratio,$
        font_size=fontsize, xtitle='!8x !7(!8$d_i$!7)', ytitle='!8z !7(!8$d_i$!7)', $
        position=[xs,ys,xe,ys+hei], $
        dimensions=[768, iheight], $
        ;xshowtext = 0, $
        ;ytickvalues=[200, 400, 600, 800, 1000], $
        ;max_value=5.0, min_value=0.5, $
        ;position=[0.07,0.53,0.95,0.96],$
        rgb_table=33, axis_style=2, interpolate=1)
    maxdata = max(data1)
    mindata = min(data1)
    print, 'Maximum', maxdata
    print, 'Minimum', mindata
    im1.max_value = lims2(1)
    im1.min_value = lims2(0)
    pos2 = im1.position
    ;print, pos1
    CB1 = COLORBAR(TARGET=im1,ORIENTATION=1,$
        position=[xe+0.01,pos2(1)+shift1,xe+0.03,pos2(3)-shift1])
    CB1.TEXTPOS = 1
    CB1.TICKDIR = 1
    CB1.FONT_SIZE = fontsize
    
    if (ifieldline eq 1) then begin
        cn1 = contour(dataAy(*,zs:ze), x1, z1(zs:ze), /overplot, $
            c_label_show=0, color='white', c_thick=[0.5])
    endif


    shift2 = 0.12
    t1 = text(xe+shift2, ys+hei/4, '$!8n_{>2.7E_{th}}/n_e$', $
        font_size=fontsize, color='black', orientation=90)
    t2 = text(xe+shift2, ys+hei*3/2, '$!8\beta_e$', $
        font_size=fontsize, color='black', orientation=90)
    ;t2 = text(xe+shift2, ys+hei*3/2, '$!8j_y$', $
    ;    font_size=16, color='k', orientation=90)
    ;t2 = text(xe+shift2, ys+hei*3/2, '$!8n_e$', $
    ;    font_size=16, color='k', orientation=90)
    ;t2 = text(xe+shift2, ys+hei*3/2, '$|!16B!6|$', $
    ;    font_size=16, color='k', orientation=90)

;    if (ifieldline eq 1) then begin
;        cn1 = contour(dataAy(*,zs:ze), x1, z1(zs:ze), /overplot, $
;            c_label_show=0, color='white')
;    endif
;
;    sz = size(xslice)
;    FOR i = 0, sz(1)-1 DO BEGIN
;        p1 = PLOT([xslice(i),xslice(i)],[1.0,2.0],'k--',/current,$
;            xrange=[x1(0),x1(nx1-1)], yrange=[1.0,2.0],$
;            position=[pos1(0),yrmin,pos1(2),yrmax])
;        ax = p1.axes
;        ax[0].HIDE=1
;        ax[1].HIDE=1
;        ax[2].HIDE=1
;        ax[3].HIDE=1
;    ENDFOR

    fname = 'ParticleRho' + STRING(it, FORMAT='(I3.3)') + $
        '_' + species + '.jpg'
    im1.save, fname, resolution=300
    fname = 'ParticleRho' + STRING(it, FORMAT='(I3.3)') + $
        '_' + species + '.eps'
    im1.save, fname 
END

;*******************************************************************************
; Save 2D field data.
; Args:
;   nx, nz: the dimension of the data.
;   x, z: the coodinates of the data.
;   dset: the data set
;   fname: the filename to save the data.
;*******************************************************************************
pro Save2DfieldData, nx, nz, x, z, dset, fname
    openw, lun, fname, /get_lun
    nx = float(nx)
    nz = float(nz)
    writeu, lun, nx, nz, x, z, dset
    print, "The sizes of the data: ", nx, nz
    free_lun, lun
end

;*******************************************************************************
; Overplot one rectangle over a 2D plot.
; Args:
;   center: the center of the rectangle.
;   width, height: the sizes of the rectangle.
;   color: the color of the lines.
;   linewidth: the width of the lines.
;*******************************************************************************
pro plot_rectangle, center, width, height, linecolor, linewidth
    ; only need 5 points
    x = fltarr(5)
    y = fltarr(5)
    ; center coordinates
    xc = center(0)
    yc = center(1)
    hwidth = width * 0.5
    hheight = height * 0.5

    x(0) = xc - hwidth
    y(0) = yc + hheight
    x(1) = xc - hwidth
    y(1) = yc - hheight
    x(2) = xc + hwidth
    y(2) = yc - hheight
    x(3) = xc + hwidth
    y(3) = yc + hheight
    x(4) = x(0)
    y(4) = y(0)
    p1 = plot(x, y, /overplot, thick=linewidth, color=linecolor)
end

;*******************************************************************************
; Plot the time evolution of the total density in different energy bands.
; Args:
;   species: 'e' for electron. 'i' for ion.
;*******************************************************************************
pro ParticleNumberBands, species
    common picinfo
    fname =  'data/rhoBands_' + species + '.dat'
    data = fltarr(5, ntf)
    openr, lun, fname, /get_lun
    readf, lun, data
    free_lun, lun
    ;data = smooth(data, [1, 5])
    p1 = plot(tf, data(0, *), 'k2')
    p2 = plot(tf, data(1, *), /overplot, 'r2')
    p3 = plot(tf, data(2, *), /overplot, 'g2')
    p4 = plot(tf, data(3, *), /overplot, 'b2')
    p5 = plot(tf, data(4, *), /overplot, 'c2')
    
    p6 = plot(tf, total(data, 1), 'k2')
end

;*******************************************************************************
; Plot the time evolution of the bulk flow energy of each component.
; Args:
;   species: 'e' for electron. 'i' for ion.
;*******************************************************************************
pro bulkFlowEnergy, species
    common picinfo
    fname =  'data/bulkEnergy_' + species + '.dat'
    data = fltarr(3, ntf)
    openr, lun, fname, /get_lun
    readf, lun, data
    free_lun, lun
    ;data = smooth(data, [1, 5])
    p1 = plot(tf, data(0, *), 'k2')
    p2 = plot(tf, data(1, *), /overplot, 'r2')
    p3 = plot(tf, data(2, *), /overplot, 'g2')
    
    p6 = plot(tf, total(data, 1), 'k2')
end

;*******************************************************************************
; 2D plot for j dote E using calculated current density due to curvature
; drift and gradient B drift.
; Input:
;   it: time point ID.
;   ifieldline: flag for whether to plot B field lines.
;   xslice: x positions to plot vertical cut dashed lines.
;   species: 'e' for electron, 'i' for ion.
;   isBuffer: whether to buffer the image rather than display it.
;   isSave: whether to save the image after plotting.
;   isClose: whether to close the image after plotting.
; Return:
;   im1: image object for handler.
;*******************************************************************************
PRO jcmDote2d, it, ifieldline, isBuffer, isSave, isClose, xslice, species, im1
    common picinfo
;    lims1 = [-0.2,0.2]
;    lims2 = [-0.3,0.3]
;    lims3 = [-0.4,0.4]
    lims1 = [-0.002,0.002]
    lims2 = [-0.002,0.002]

    nd = 2.0 ; Interpolate grid size
    iheight = 256
    ys = 0.59
    hei = 0.36
    xs = 0.12
    xe = 0.85
    jname = '$!16j_{!8'

    ; curvature drift
    fname = '../data1/jcpara_dote00_' + species + '.gda'
    readfield, [fname], it, nd, data1, nx1, nz1, x1, z1
    zs = nz1*2/5
    ze = nz1*3/5
    data1 = smooth(data1, [3,3])
    ratio = (ze-zs) / nx1
    BlueWhiteRed, rgbtable, lims1
    accumulatedJdotE1 = total(data1, 2)
    accumulatedJdotE1 = total(accumulatedJdotE1, /CUMULATIVE)
    fontsize = 16

    fname = 'jcpara_' + string(it, '(I4.4)') + '.gda'
    Save2DfieldData, nx1, ze-zs+1, x1, z1(zs:ze), data1(*,zs:ze), fname

    if (isBuffer eq 1) then begin
        im1 = IMAGE(data1(*,zs:ze), x1, z1(zs:ze), /buffer, $
            font_size=fontsize, dimensions=[512, iheight], $
            axis_style=2)
    endif else begin
        im1 = IMAGE(data1(*,zs:ze), x1, z1(zs:ze), $
            font_size=fontsize, $
            dimensions=[512, iheight], $
            axis_style=2)
    endelse
    im1.aspect_ratio=ratio
    im1.xtitle='x ($d_i$)'
    im1.ytitle='!8z !7(!8$d_i$!7)'
    im1.position=[xs,ys,xe,ys+hei]
    im1.xshowtext = 0
    im1.rgb_table=rgbtable
    im1.interpolate=1
    maxdata = max(data1)
    mindata = min(data1)
    print, 'Maximum', maxdata
    print, 'Minimum', mindata
    im1.max_value = lims1(1)
    im1.min_value = lims1(0)
    pos1 = im1.position
    if (ifieldline eq 1) then begin
        zs1 = nz/4
        ze1 = nz*3/4
        fname = '../data/Ay.gda'
        ;OPENR, lun1, fname, /get_lun
        ;data = FLTARR(nx, nz)
        readfield, [fname], it, nd, dataAy, nx1, nz1, x1, z1

        ;field = assoc(lun1,data)
        ;data = field(it)
        cn1 = contour(dataAy(*,zs:ze), x1, z1(zs:ze), /overplot, $
            c_label_show=0, color='black', c_thick=[0.5])
    endif

    yrmax = pos1(3)

  ; gradient drift
    ys = ys-hei-0.04
    fname1 = '../data1/jgrad_dote00_' + species + '.gda'
    readfield, [fname1], it, nd, data1, nx1, nz1, x1, z1
    data1 = smooth(data1, [3,3])
    ;ratio = nx1/(ze-zs)/2.0
    BlueWhiteRed, rgbtable, lims2
    accumulatedJdotE2 = total(data1, 2)
    accumulatedJdotE2 = total(accumulatedJdotE2, /CUMULATIVE)

    fname = 'jgrad_' + string(it, '(I4.4)') + '.gda'
    Save2DfieldData, nx1, ze-zs+1, x1, z1(zs:ze), data1(*,zs:ze), fname

    im1 = IMAGE(data1(*,zs:ze), x1, z1(zs:ze), /current, $
        aspect_ratio=ratio,$
        font_size=fontsize, xtitle='!8x !7(!8$d_i$!7)', $
        ytitle='!8z !7(!8$d_i$!7)', $
        position=[xs,ys,xe,ys+hei], $
        dimensions=[768, iheight], $
        ;xshowtext = 0, $
        ;ytickvalues=[200, 400, 600, 800, 1000], $
        ;max_value=5.0, min_value=0.5, $
        ;position=[0.07,0.53,0.95,0.96],$
        rgb_table=rgbtable, axis_style=2, interpolate=1)
    maxdata = max(data1)
    mindata = min(data1)
    print, 'Maximum', maxdata
    print, 'Minimum', mindata
    im1.max_value = lims2(1)
    im1.min_value = lims2(0)
    pos2 = im1.position
    ;print, pos1
    CB1 = COLORBAR(TARGET=im1,ORIENTATION=1,$
        position=[xe+0.01,pos2(1),xe+0.03,pos1(3)])
    CB1.TEXTPOS = 1
    CB1.TICKDIR = 1
    CB1.FONT_SIZE = fontsize
    cb1.tickvalues = [-1.5, -0.5, 0.5, 1.5, 2] * 1E-3
    cb1.tickname = ['-1.5', '-0.5', '0.5', '1.5', '$\times 10^{-3}$']
    ;t11 = text(xe+0.07, pos1(3)-0.02, '$\times 10^{-3}$', $
    ;    font_size = fontsize)
    
    if (ifieldline eq 1) then begin
        cn1 = contour(dataAy(*,zs:ze), x1, z1(zs:ze), /overplot, $
            c_label_show=0, color='black', c_thick=[0.5])
    endif

    t1 = text(xe+0.13, ys+hei/3, '$!16j_{!8g}\cdot!16E$', $
        font_size=fontsize, color='green', orientation=90)
    t2 = text(xe+0.13, ys+hei*3/2, '$!16j_{!8c}\cdot!16E$', $
        font_size=fontsize, color='blue', orientation=90)

;    if (ifieldline eq 1) then begin
;        cn1 = contour(dataAy(*,zs:ze), x1, z1(zs:ze), /overplot, $
;            c_label_show=0, color='black')
;    endif

;  ; Accumulation of the data over y
;
;    p1 = plot(x1, accumulatedJdotE1, 'b2', font_size=16, /current, $
;        xtitle ='!8x ($d_i$)', ytitle='Accumulation', axis_style=2,$
;        position=[xs,ys-hei,xe,ys], name='$!16j_{!8c}\cdot!16E$')
;    p2 = plot(x1, accumulatedJdotE2, 'g2', /overplot, name='$!16j_{!8g}\cdot!16E$')
;
;    leg1 = legend(target=[p1,p2], /auto_text_color, $
;        font_size=16, position=[0.94, 0.30], transparency=100)
;
;    yrmin = pos1(1)

;    sz = size(xslice)
;    FOR i = 0, sz(1)-1 DO BEGIN
;        p1 = PLOT([xslice(i),xslice(i)],[1.0,2.0],'k--',/current,$
;            xrange=[x1(0),x1(nx1-1)], yrange=[1.0,2.0],$
;            position=[pos1(0),yrmin,pos1(2),yrmax])
;        ax = p1.axes
;        ax[0].HIDE=1
;        ax[1].HIDE=1
;        ax[2].HIDE=1
;        ax[3].HIDE=1
;    ENDFOR

    if (isSave eq 1) then begin
        fname = 'jcmDote2d' + STRING(it, FORMAT='(I3.3)') + $
            '_' + species + '.jpg'
        ;im1.save, fname, resolution=300
        im1.save, fname
        fname = 'jcmDote2d' + STRING(it, FORMAT='(I3.3)') + $
            '_' + species + '.eps'
        im1.save, fname
    endif
    if (isClose eq 1) then begin
        im1.close
    endif
END

;*******************************************************************************
; 2D plot for agyrotropy.
; Input:
;   it: time point ID.
;   ifieldline: flag for whether to plot B field lines.
;   xslice: x positions to plot vertical cut dashed lines.
;   species: 'e' for electron, 'i' for ion.
;   isBuffer: whether to buffer the image rather than display it.
;   isSave: whether to save the image after plotting.
;   isClose: whether to close the image after plotting.
; Return:
;   im1: image object for handler.
;*******************************************************************************
pro plotAgyrotropy, it, ifieldline, isBuffer, isSave, isClose, xslice, species, im1
    common picinfo
    lims1 = [0,1.5]

    nd = 2.0 ; Interpolate grid size
    iheight = 256
    ys = 0.59
    hei = 0.36
    xs = 0.12
    xe = 0.85
    jname = '$!16j_{!8'

    ; absB
    fname = '../data/absB.gda'
    readfield, [fname], it, nd, data1, nx1, nz1, x1, z1
    ;zs = nz1*2/5
    ;ze = nz1*3/5
    zs = nz1*49/100
    ze = nz1*51/100
    data1 = smooth(data1, [3,3])
    ratio = (ze-zs) / nx1
    BlueWhiteRed, rgbtable, lims1
    accumulatedJdotE1 = total(data1, 2)
    accumulatedJdotE1 = total(accumulatedJdotE1, /CUMULATIVE)
    fontsize = 16

    fname = 'absB_' + string(it, '(I4.4)') + '.gda'
    Save2DfieldData, nx1, ze-zs+1, x1, z1(zs:ze), data1(*,zs:ze), fname

    ; agyrotropy
    fname = '../data1/agyrotropy00_' + species + '.gda'
    readfield, [fname], it, nd, data1, nx1, nz1, x1, z1
    zs = nz1*2/5
    ze = nz1*3/5
    ;zs = nz1*49/100
    ;ze = nz1*51/100
    print, 'nz1, ze-zs: ', nz1, ze - zs
    data1 = smooth(data1, [3,3])
    ratio = (ze-zs) / nx1
    BlueWhiteRed, rgbtable, lims1
    accumulatedJdotE1 = total(data1, 2)
    accumulatedJdotE1 = total(accumulatedJdotE1, /CUMULATIVE)
    fontsize = 16

    fname = 'agyrotropy1_' + string(it, '(I4.4)') + '.gda'
    Save2DfieldData, nx1, ze-zs+1, x1, z1(zs:ze), data1(*,zs:ze), fname

    if (isBuffer eq 1) then begin
        im1 = IMAGE(data1(*,zs:ze), x1, z1(zs:ze), /buffer, $
            font_size=fontsize, dimensions=[512, iheight], $
            axis_style=2)
    endif else begin
        im1 = IMAGE(data1(*,zs:ze), x1, z1(zs:ze), $
            font_size=fontsize, $
            dimensions=[512, iheight], $
            axis_style=2)
    endelse
    im1.aspect_ratio=ratio
    im1.xtitle='x ($d_i$)'
    im1.ytitle='!8z !7(!8$d_i$!7)'
    im1.position=[xs,ys,xe,ys+hei]
    im1.xshowtext = 0
    im1.rgb_table=33
    im1.interpolate=1
    maxdata = max(data1)
    mindata = min(data1)
    print, 'Maximum', maxdata
    print, 'Minimum', mindata
    im1.max_value = lims1(1)
    im1.min_value = lims1(0)
    pos1 = im1.position
    if (ifieldline eq 1) then begin
        zs1 = nz/4
        ze1 = nz*3/4
        fname = '../data/Ay.gda'
        ;OPENR, lun1, fname, /get_lun
        ;data = FLTARR(nx, nz)
        readfield, [fname], it, nd, dataAy, nx1, nz1, x1, z1

        ;field = assoc(lun1,data)
        ;data = field(it)
        cn1 = contour(dataAy(*,zs:ze), x1, z1(zs:ze), /overplot, $
            c_label_show=0, color='black', c_thick=[0.5])
    endif

    ; jcpara
    fname = '../data1/jcpara_dote00_' + species + '.gda'
    readfield, [fname], it, nd, data1, nx1, nz1, x1, z1
    zs = nz1*2/5
    ze = nz1*3/5
    ;zs = nz1*49/100
    ;ze = nz1*51/100
    data1 = smooth(data1, [3,3])
    ratio = (ze-zs) / nx1
    fontsize = 16

    fname = 'jcpara_' + string(it, '(I4.4)') + '.gda'
    Save2DfieldData, nx1, ze-zs+1, x1, z1(zs:ze), data1(*,zs:ze), fname

    ; jgrad
    fname = '../data1/jgrad_dote00_' + species + '.gda'
    readfield, [fname], it, nd, data1, nx1, nz1, x1, z1
    zs = nz1*2/5
    ze = nz1*3/5
    ;zs = nz1*49/100
    ;ze = nz1*51/100
    data1 = smooth(data1, [3,3])
    ratio = (ze-zs) / nx1
    fontsize = 16

    fname = 'jgrad_' + string(it, '(I4.4)') + '.gda'
    Save2DfieldData, nx1, ze-zs+1, x1, z1(zs:ze), data1(*,zs:ze), fname

    ; Ay
    fname = '../data/Ay.gda'
    readfield, [fname], it, nd, data1, nx1, nz1, x1, z1
    zs = nz1*2/5
    ze = nz1*3/5
    ;zs = nz1*49/100
    ;ze = nz1*51/100
    data1 = smooth(data1, [3,3])
    ratio = (ze-zs) / nx1
    fontsize = 16

    fname = 'Ay_' + string(it, '(I4.4)') + '.gda'
    Save2DfieldData, nx1, ze-zs+1, x1, z1(zs:ze), data1(*,zs:ze), fname

    ; Radius of curvature
    fname = '../data1/curvRadius00_' + species + '.gda'
    readfield, [fname], it, nd, data1, nx1, nz1, x1, z1
    ;zs = nz1*2/5
    ;ze = nz1*3/5
    zs = nz1*49/100
    ze = nz1*51/100
    data1 = smooth(data1, [3,3])
    ratio = (ze-zs) / nx1
    fontsize = 16

    fname = 'curvRadius_' + string(it, '(I4.4)') + '.gda'
    Save2DfieldData, nx1, ze-zs+1, x1, z1(zs:ze), data1(*,zs:ze), fname

    ; length scale of gradient B
    fname = '../data1/lengthGradB00_' + species + '.gda'
    readfield, [fname], it, nd, data1, nx1, nz1, x1, z1
    ;zs = nz1*2/5
    ;ze = nz1*3/5
    zs = nz1*49/100
    ze = nz1*51/100
    data1 = smooth(data1, [3,3])
    ratio = (ze-zs) / nx1
    fontsize = 16

    fname = 'lengthGradB_' + string(it, '(I4.4)') + '.gda'
    Save2DfieldData, nx1, ze-zs+1, x1, z1(zs:ze), data1(*,zs:ze), fname
end

;*******************************************************************************
; Read data and save part of it.
; Args:
;   fpath: file path.
;   varname: variable name.
;   it: time frame.
;   zheight: the height of the box in z direction.
;*******************************************************************************
pro ReadDataSavePart, fpath, varname, it, zheight
    nd = 2
    fname = '../' + fpath + '/' + varname + '.gda'
    readfield, [fname], it, nd, data1, nx1, nz1, x1, z1
    dz1 = z1(1) - z1(0)
    dnz = zheight / dz1
    zmid = nz1 / 2
    ;zs = nz1*9/20
    ;ze = nz1*11/20
    zs = zmid - floor(dnz/2)
    ze = zmid + floor(dnz/2)
    fname = varname + '_sbox_' + string(it, '(I4.4)') + '.gda'
    Save2DfieldData, nx1, ze-zs+1, x1, z1(zs:ze), data1(*,zs:ze), fname

    fname = '../data/Ay.gda'
    readfield, [fname], it, nd, data1, nx1, nz1, x1, z1
    ;zs = nz1*9/20
    ;ze = nz1*11/20
    data1 = smooth(data1, [3,3])
    fname = 'Ay_sbox_' + string(it, '(I4.4)') + '.gda'
    Save2DfieldData, nx1, ze-zs+1, x1, z1(zs:ze), data1(*,zs:ze), fname
end

;*******************************************************************************
; Get the averaged value in a box.
; Args:
;   fpath: file path.
;   varname: variable name.
;   it: time frame.
;   xc, zc: the coordinates of the center of the box in di.
;   xwidth, zheight: the width and height of the box in di.
; Returns:
;*******************************************************************************
