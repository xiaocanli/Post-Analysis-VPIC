#include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# # For a large number of source files you can create it in a simpler way
# # using file() function:
# file(GLOB analysis_SOURCES *.f90)

# Create an executable files.
add_executable(dissipation.exec dissipation.f90)
target_link_libraries(dissipation.exec ${MPI_Fortran_LIBRARIES} get_info
    analysis_config fields jdote statistics saving_flags compression_shear)

add_executable(calc_agyrotropy.exec calc_agyrotropy.f90)
target_link_libraries(calc_agyrotropy.exec ${MPI_Fortran_LIBRARIES} get_info
    analysis_config fields)

add_executable(parallel_potential.exec parallel_potential.f90)
target_link_libraries(parallel_potential.exec ${MPI_Fortran_LIBRARIES}
    fieldline_tracing electric_field analysis_config)

add_executable(translate.exec translate.f90)
target_link_libraries(translate.exec ${MPI_Fortran_LIBRARIES} translate_module)

add_executable(parspec.exec parspec.f90)
target_link_libraries(parspec.exec ${MPI_Fortran_LIBRARIES} particle_spectrum
    get_info commandline_arguments maximum_energy)

add_executable(spectrum_along_fieldline.exec spectrum_along_fieldline.f90)
target_link_libraries(spectrum_along_fieldline.exec ${MPI_Fortran_LIBRARIES}
    analysis_config particle_spectrum fieldline_tracing particle_fieldline)

add_executable(vdist_2d_along_fieldline.exec vdist_2d_along_fieldline.f90)
target_link_libraries(vdist_2d_along_fieldline.exec ${MPI_Fortran_LIBRARIES}
    analysis_config vdist fieldline_tracing particle_fieldline)

add_executable(vdist_1d_along_fieldline.exec vdist_1d_along_fieldline.f90)
target_link_libraries(vdist_1d_along_fieldline.exec ${MPI_Fortran_LIBRARIES}
    analysis_config vdist fieldline_tracing particle_fieldline)

add_executable(vdistribution.exec vdistribution.f90)
target_link_libraries(vdistribution.exec ${MPI_Fortran_LIBRARIES} vdist)

add_executable(nrho_eband.exec particle_number_eband.f90)
target_link_libraries(nrho_eband.exec ${MPI_Fortran_LIBRARIES} analysis_config
    fields)

add_executable(bulk_internal_energy.exec bulk_internal_energy.f90)
target_link_libraries(bulk_internal_energy.exec ${MPI_Fortran_LIBRARIES}
    analysis_config fields)

add_executable(compression.exec compression.f90)
target_link_libraries(compression.exec ${MPI_Fortran_LIBRARIES} analysis_config
    fields compression_shear)

add_executable(jdote_eband.exec jdote_eband.f90)
target_link_libraries(jdote_eband.exec ${MPI_Fortran_LIBRARIES} jdote_energy_band)

add_executable(particle_based_jdote.exec particle_based_jdote.f90)
target_link_libraries(particle_based_jdote.exec ${MPI_Fortran_LIBRARIES}
    particle_drift interp_emf get_info translate_module)

add_executable(particle_spectrum_box.exec particle_spectrum_box.f90)
target_link_libraries(particle_spectrum_box.exec ${MPI_Fortran_LIBRARIES}
    particle_spectrum get_info)

add_executable(particle_spectrum_vdist_box.exec particle_spectrum_vdist_box.f90)
target_link_libraries(particle_spectrum_vdist_box.exec ${MPI_Fortran_LIBRARIES}
    particle_spectrum_vdist)

add_executable(particle_spectrum_vdist_fieldlines.exec particle_spectrum_vdist_fieldlines.f90)
target_link_libraries(particle_spectrum_vdist_fieldlines.exec ${MPI_Fortran_LIBRARIES}
    particle_spectrum vdist get_info flap)

add_executable(cla_test.exec cla_test.f90)
target_link_libraries(cla_test.exec ${MPI_Fortran_LIBRARIES} cla)

add_executable(parspec_cpu_based.exec parspec_cpu_based.f90)
target_link_libraries(parspec_cpu_based.exec ${MPI_Fortran_LIBRARIES}
    get_info commandline_arguments particle_spectrum)

add_executable(parallel_hdf5.exec parallel_hdf5.f90)
#target_include_directories(parallel_hdf5)
target_link_libraries(parallel_hdf5.exec ${MPI_Fortran_LIBRARIES}
    ${HDF5_Fortran_LIBRARIES} particle_drift interp_emf get_info
    translate_module interp_pfields fields flap)

add_executable(ptl_tracer_hdf5.exec ptl_tracer_hdf5.f90)
target_link_libraries(ptl_tracer_hdf5.exec ${MPI_Fortran_LIBRARIES}
    ${HDF5_Fortran_LIBRARIES} get_info translate_module commandline_arguments)

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})
install (TARGETS dissipation.exec DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS calc_agyrotropy.exec DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS parallel_potential.exec DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS translate.exec DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS parspec.exec DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS spectrum_along_fieldline.exec DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS vdistribution.exec DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS vdist_2d_along_fieldline.exec DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS vdist_1d_along_fieldline.exec DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS nrho_eband.exec DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS bulk_internal_energy.exec DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS compression.exec DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS jdote_eband.exec DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS particle_based_jdote.exec DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS particle_spectrum_box.exec DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS particle_spectrum_vdist_box.exec DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS particle_spectrum_vdist_fieldlines.exec DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS parspec_cpu_based.exec DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS parallel_hdf5.exec DESTINATION ${CMAKE_INSTALL_PREFIX})
install (TARGETS ptl_tracer_hdf5.exec DESTINATION ${CMAKE_INSTALL_PREFIX})
