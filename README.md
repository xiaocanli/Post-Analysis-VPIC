# Post-analysis procedures for VPIC simulation.

## Install
Depends on the compiler to use `source module_gcc.sh` or `source module_intel.sh`
Run the following command to build the main program
```
mkdir build
cd build
cmake ..
make
make install
```
On a cray system, the `cmake ..` should be
```
CC=cc CXX=CC FC=ftn cmake ..
```

## Get started

The executables are all saved in this code's root directory.
The configuration files are in **./config_files**.

* _translate_: processing the VPIC raw field data, which saved as an individual
file for each MPI process in PIC simulations. The processed data will be
saved in a single file or multiple files depending on the flags (see below).
When doing _translate_,
**conf.dat** and **analysis_config.dat** may need to be modified
    - conf.dat
        - httx, htty, httz: their produce should be the same as current total
        CPU numbers. Each of them should \(\le\) the MPI topology of the PIC
        simulation
        - tindex_start: the default is 0, so the programe will translate the
        beginning time frame. Change it when doing translate for a restart run,
        so you don't have to start from the beginning
        - tindex_stop: set a large number, so the programe will translate all
        the time frame. When testing the programe, set it small
        output_format: when setting 1, each time frame saves one file.
        When setting 2, all time frames save in a single file.
        - append_to_files: default is 1, so processed data will be appended to
        the existing files
    - analysis_config: is_rel should be 1 when all hydro data are dumped
        - tp1, tp2: starting and ending time frame. They are output time frame,
        not the actual time steps of the PIC simulations.
        - inductive: using whole electric field when it is set 0; using motional
        electric field \(-\mathbf{v}\times\mathbf{B}\) when it is set 1.
        - is_rel: whether relativistic fields are used
        - eratio: the ratio of the energy interval to the initial thermal energy
        You need to calculate this using emax and nex, defined in your
        configuration for this PIC simulation (**.cxx).

* _parspec_: get the particle energy spectrum in a box. The box can be the whole
simulation domain or a local small region. When doing _parspec_,
**spectrum_config.dat** may need to be modified
    - xc, yc, zc are the center of the box (in \(d_e\)). In the PIC simulations,
    \(x\in[0, L_x]\), \(y\in[-L_y/2, L_y/2]\), \(z\in[-L_z/2, L_z/2]\)
    - xsize, ysize and zsize: sizes of each dimension in cell numbers
    - the rest is for velocity distribution. It can be ignored when
    doing _parspec_

* _dissipation_: energy conversion analysis using \(\mathbf{j}\cdot\mathbf{E}\).
**analysis_config.dat** (see above) and **saving_flags.dat** may be modified.
  - saving_flags.dat
      - When a flag is 1, the 2D/3D field data will be saved
