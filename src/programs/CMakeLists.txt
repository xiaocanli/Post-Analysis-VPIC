#include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# # For a large number of source files you can create it in a simpler way
# # using file() function:
# file(GLOB analysis_SOURCES *.f90)

# Create an executable files.
add_executable(dissipation dissipation.f90)
target_link_libraries(dissipation ${MPI_Fortran_LIBRARIES} get_info
    analysis_config fields jdote statistics)

add_executable(calc_agyrotropy calc_agyrotropy.f90)
target_link_libraries(calc_agyrotropy ${MPI_Fortran_LIBRARIES} get_info
    analysis_config fields)

add_executable(parallel_potential parallel_potential.f90)
target_link_libraries(parallel_potential ${MPI_Fortran_LIBRARIES}
    fieldline_tracing analysis_config)

add_executable(translate translate.f90)
target_link_libraries(translate ${MPI_Fortran_LIBRARIES} translate_module)

add_executable(parspec parspec.f90)
target_link_libraries(parspec ${MPI_Fortran_LIBRARIES} particle_spectrum
    get_info)

add_executable(spectrum_along_fieldline spectrum_along_fieldline.f90)
target_link_libraries(spectrum_along_fieldline ${MPI_Fortran_LIBRARIES}
    analysis_config particle_spectrum fieldline_tracing)

#add_executable(vdistribution vdistribution.f90)

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})
install (TARGETS dissipation DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS calc_agyrotropy DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS parallel_potential DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS translate DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS parspec DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS spectrum_along_fieldline DESTINATION ${CMAKE_INSTALL_PREFIX})
#install (TARGETS vdistribution DESTINATION ${CMAKE_INSTALL_PREFIX})
