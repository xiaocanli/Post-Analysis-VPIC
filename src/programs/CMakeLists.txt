#include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# # For a large number of source files you can create it in a simpler way
# # using file() function:
# file(GLOB analysis_SOURCES *.f90)

# Create an executable files.
add_executable(dissipation dissipation.f90)
target_link_libraries(dissipation ${MPI_Fortran_LIBRARIES} get_info
    analysis_config fields jdote statistics saving_flags compression_shear)

add_executable(calc_agyrotropy calc_agyrotropy.f90)
target_link_libraries(calc_agyrotropy ${MPI_Fortran_LIBRARIES} get_info
    analysis_config fields)

add_executable(parallel_potential parallel_potential.f90)
target_link_libraries(parallel_potential ${MPI_Fortran_LIBRARIES}
    fieldline_tracing electric_field analysis_config)

add_executable(translate translate.f90)
target_link_libraries(translate ${MPI_Fortran_LIBRARIES} translate_module)

add_executable(parspec parspec.f90)
target_link_libraries(parspec ${MPI_Fortran_LIBRARIES} particle_spectrum
    get_info)

add_executable(spectrum_along_fieldline spectrum_along_fieldline.f90)
target_link_libraries(spectrum_along_fieldline ${MPI_Fortran_LIBRARIES}
    analysis_config particle_spectrum fieldline_tracing particle_fieldline)

add_executable(vdist_2d_along_fieldline vdist_2d_along_fieldline.f90)
target_link_libraries(vdist_2d_along_fieldline ${MPI_Fortran_LIBRARIES}
    analysis_config vdist fieldline_tracing particle_fieldline)

add_executable(vdist_1d_along_fieldline vdist_1d_along_fieldline.f90)
target_link_libraries(vdist_1d_along_fieldline ${MPI_Fortran_LIBRARIES}
    analysis_config vdist fieldline_tracing particle_fieldline)

add_executable(vdistribution vdistribution.f90)
target_link_libraries(vdistribution ${MPI_Fortran_LIBRARIES} vdist)

add_executable(nrho_eband particle_number_eband.f90)
target_link_libraries(nrho_eband ${MPI_Fortran_LIBRARIES} analysis_config
    fields)

add_executable(bulk_internal_energy bulk_internal_energy.f90)
target_link_libraries(bulk_internal_energy ${MPI_Fortran_LIBRARIES}
    analysis_config fields)

add_executable(compression compression.f90)
target_link_libraries(compression ${MPI_Fortran_LIBRARIES} analysis_config
    fields compression_shear)

add_executable(jdote_eband jdote_eband.f90)
target_link_libraries(jdote_eband ${MPI_Fortran_LIBRARIES} jdote_energy_band)

add_executable(particle_based_jdote particle_based_jdote.f90)
target_link_libraries(particle_based_jdote ${MPI_Fortran_LIBRARIES}
    particle_drift interp_emf get_info translate_module)

add_executable(particle_spectrum_box particle_spectrum_box.f90)
target_link_libraries(particle_spectrum_box ${MPI_Fortran_LIBRARIES}
    particle_spectrum get_info)

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})
install (TARGETS dissipation DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS calc_agyrotropy DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS parallel_potential DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS translate DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS parspec DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS spectrum_along_fieldline DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS vdistribution DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS vdist_2d_along_fieldline DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS vdist_1d_along_fieldline DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS nrho_eband DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS bulk_internal_energy DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS compression DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS jdote_eband DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS particle_based_jdote DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS particle_spectrum_box DESTINATION ${CMAKE_INSTALL_PREFIX})
