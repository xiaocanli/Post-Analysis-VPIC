# Edit the following variables as needed
#HDF_INSTALL = $(HOME)/hdf5
#
CC = mpicc
# define any compile-time flags
CFLAGS = -fopenmp -O3 -Wall -g -std=gnu99

INCLUDES = -I$(HDF5_INCL)
LFLAGS = 
HDF5LIB = -L$(HDF5_ROOT)/lib -lhdf5
LIBS = $(HDF5LIB) -ldl -lm

# define the C source files
SRCS = h5group-sorter.c configuration.c mpi_io.c vpic_data.c package_data.c \
	   qsort-parallel.c get_data.c meta_data.c

SRCS_TRAJ = particle_trajectory.c time_frame_info.c particle_tags.c \
			vpic_data.c get_data.c package_data.c mpi_io.c

# define the C object files 
#
# This uses Suffix Replacement within a macro:
#   $(name:string1=string2)
#         For each word in 'name' replace 'string1' with 'string2'
# Below we are replacing the suffix .c of all words in the macro SRCS
# with the .o suffix
#
OBJS = $(SRCS:.c=.o)

OBJS_TRAJ = $(SRCS_TRAJ:.c=.o)

# define the executable file 
MAIN = h5group-sorter
TRAJ = h5trajectory

#
.PHONY: depend clean

all:	$(MAIN) $(TRAJ) lib/libh5sort.a lib/libtraj.a
	@echo  Programs are successfully compiled!

main:	$(MAIN)
	@echo  $(MAIN) are successfully compiled!

traj:	$(TRAJ)
	@echo  $(TRAJ) is successfully compiled!

$(MAIN): $(OBJS) 
	$(CC) $(CFLAGS) $(INCLUDES) -o $(MAIN) $(OBJS) $(LFLAGS) $(LIBS)

$(TRAJ): $(OBJS_TRAJ) 
	$(CC) $(CFLAGS) $(INCLUDES) -o $(TRAJ) $(OBJS_TRAJ) $(LFLAGS) $(LIBS)

lib/libh5sort.a: $(OBJS)
	ar rc $@ $^ && ranlib $@

lib/libtraj.a: $(OBJS_TRAJ)
	ar rc $@ $^ && ranlib $@

.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $<  -o $@

clean:
	$(RM) *.o *~ $(MAIN) $(TRAJ)

depend: $(SRCS)
	makedepend $(INCLUDES) $^

# DO NOT DELETE THIS LINE -- make depend needs it
