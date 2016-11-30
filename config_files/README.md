# Description of the configuration files

## conf.dat
It is the configuration file for current analysis.
- **httx**, **htty**, **httz**: MPI topology. Their product should be the same as the total number of CPUs.
Each of them should less the MPI topology of the PIC simulation.
- **tindex_start**: the starting time index. The default is 0.
- **tindex_stop**: the stopping time index. It is set to be a large number, so the programes will run through all time frames.
- **output_format**: the format of output. When it is set as 1, it creats one file for each time frame. When it is set as 2,
data from all time frames are saved in the same file.
- **append_to_file**: whether to append the data to the existing files. The default is 1, indicating that data will be appended to
        the existing files
