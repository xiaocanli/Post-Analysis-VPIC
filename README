mkdir build
source module_intel.sh: change to module_gcc on mustang
cd build
cmake ..
make
make install
cd ..: the executables are in this directory

Some common ones: translate, parspec
dissipation: calculate jdote

The configuration files are in config_files
* When doing translate, conf.dat and analysis_config.dat may be modified
  -- conf.dat: same as previous one
  -- analysis_config: is_rel should be 1 when all hydro data are dumped

* When doing parspec, spectrum_config.dat should be modified
  -- xc, yc, zc are the center of the box (in de)
  -- xsize, ysize and zsize: sizes of each dimension in cell numbers
  -- the rest is for velocity distribution. It can be ignored when doing parspec

* When doing dissipation, analysis_config.dat and saving_flags.dat may be modified
  
  analysis_config.dat
  -- tp1 and tp2: the starting and ending time frame.
     tp2 can be changed to 1 to test. The screen output will give the total time frames of the simulation
  -- inductive: whether to just use -v*B electric field
  -- is_rel: same as for translate

  saving_flags.dat
  When a flag is 1, the 2D/3D data will be saved
