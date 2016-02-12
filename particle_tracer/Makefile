# Edit the following variables as needed
#HDF_INSTALL = $(HOME)/hdf5
VPIC_SORTER_PATH = ../vpic-sorter
#
CC = mpicc
# define any compile-time flags
CFLAGS = -fopenmp -O3 -Wall -g -std=gnu99

INCLUDES = -I$(HDF5_INCL) -I$(VPIC_SORTER_PATH)
LFLAGS = 
HDF5LIB = -L$(HDF5_ROOT)/lib -lhdf5
H5SORTLIB = -L$(VPIC_SORTER_PATH)/lib -lh5sort -ltraj
LIBS = $(HDF5LIB) $(H5SORTLIB) -ldl -lm

# define the C source files
SRCS = binary_to_hdf5.c

# define the C object files 
#
# This uses Suffix Replacement within a macro:
#   $(name:string1=string2)
#         For each word in 'name' replace 'string1' with 'string2'
# Below we are replacing the suffix .c of all words in the macro SRCS
# with the .o suffix
#
OBJS = $(SRCS:.c=.o)

# define the executable file 
MAIN = binary_to_hdf5

#
.PHONY: depend clean

all:	$(MAIN)
	@echo  Programs are successfully compiled!

main:	$(MAIN)
	@echo  $(MAIN) are successfully compiled!

$(MAIN): $(OBJS) 
	$(CC) $(CFLAGS) $(INCLUDES) -o $(MAIN) $(OBJS) $(LFLAGS) $(LIBS)

.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $<  -o $@

clean:
	$(RM) *.o *~ $(MAIN)

depend: $(SRCS)
	makedepend $(INCLUDES) $^

# DO NOT DELETE THIS LINE -- make depend needs it
