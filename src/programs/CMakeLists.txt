#include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# # For a large number of source files you can create it in a simpler way
# # using file() function:
# file(GLOB analysis_SOURCES *.f90)

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})

# Create an executable files.

add_executable(translate.exec translate.f90)
target_link_libraries(translate.exec ${MPI_Fortran_LIBRARIES} translate_module flap)
install (TARGETS translate.exec DESTINATION ${CMAKE_INSTALL_PREFIX})

add_executable(dissipation.exec dissipation.f90)
target_link_libraries(dissipation.exec ${MPI_Fortran_LIBRARIES} get_info
    analysis_config fields jdote statistics saving_flags compression_shear flap)
install (TARGETS dissipation.exec DESTINATION ${CMAKE_INSTALL_PREFIX})

add_executable(compression.exec compression.f90)
target_link_libraries(compression.exec ${MPI_Fortran_LIBRARIES} analysis_config
    fields compression_shear flap)
install (TARGETS compression.exec DESTINATION ${CMAKE_INSTALL_PREFIX})

add_executable(parspec.exec parspec.f90)
target_link_libraries(parspec.exec ${MPI_Fortran_LIBRARIES} particle_spectrum
    get_info maximum_energy flap)
install (TARGETS parspec.exec DESTINATION ${CMAKE_INSTALL_PREFIX})

add_executable(parspec_cpu_based.exec parspec_cpu_based.f90)
target_link_libraries(parspec_cpu_based.exec ${MPI_Fortran_LIBRARIES}
    get_info commandline_arguments particle_spectrum)
install (TARGETS parspec_cpu_based.exec DESTINATION ${CMAKE_INSTALL_PREFIX})

add_executable(calc_agyrotropy.exec calc_agyrotropy.f90)
target_link_libraries(calc_agyrotropy.exec ${MPI_Fortran_LIBRARIES} get_info
    analysis_config fields)
install (TARGETS calc_agyrotropy.exec DESTINATION ${CMAKE_INSTALL_PREFIX})

add_executable(particle_spectrum_vdist_fieldlines.exec particle_spectrum_vdist_fieldlines.f90)
target_link_libraries(particle_spectrum_vdist_fieldlines.exec ${MPI_Fortran_LIBRARIES}
    particle_spectrum vdist get_info flap)
install (TARGETS particle_spectrum_vdist_fieldlines.exec DESTINATION ${CMAKE_INSTALL_PREFIX})

add_executable(vdistribution.exec vdistribution.f90)
target_link_libraries(vdistribution.exec ${MPI_Fortran_LIBRARIES} vdist flap)
install (TARGETS vdistribution.exec DESTINATION ${CMAKE_INSTALL_PREFIX})

add_executable(particle_energization.exec particle_energization.f90)
target_link_libraries(particle_energization.exec ${MPI_Fortran_LIBRARIES} flap
    get_info translate_module particle_module interp_emf particle_file
    interp_vel_mom interp_comp_shear)
install (TARGETS particle_energization.exec DESTINATION ${CMAKE_INSTALL_PREFIX})

# # -----------------------------------------------------------------------------
# # Particle tracer procedures
# # -----------------------------------------------------------------------------
# add_executable(parallel_hdf5.exec parallel_hdf5.f90)
# target_link_libraries(parallel_hdf5.exec ${MPI_Fortran_LIBRARIES}
#     ${HDF5_Fortran_LIBRARIES} particle_drift interp_emf get_info
#     translate_module interp_pfields fields flap)
# install (TARGETS parallel_hdf5.exec DESTINATION ${CMAKE_INSTALL_PREFIX})
# 
# add_executable(ptl_tracer_hdf5.exec ptl_tracer_hdf5.f90)
# target_link_libraries(ptl_tracer_hdf5.exec ${MPI_Fortran_LIBRARIES}
#     ${HDF5_Fortran_LIBRARIES} get_info translate_module commandline_arguments)
# install (TARGETS ptl_tracer_hdf5.exec DESTINATION ${CMAKE_INSTALL_PREFIX})

# # -----------------------------------------------------------------------------
# # Procedures that not commonly used
# # -----------------------------------------------------------------------------
# add_executable(parallel_potential.exec parallel_potential.f90)
# target_link_libraries(parallel_potential.exec ${MPI_Fortran_LIBRARIES}
#     fieldline_tracing electric_field analysis_config)
# install (TARGETS parallel_potential.exec DESTINATION ${CMAKE_INSTALL_PREFIX})
# 
# add_executable(spectrum_along_fieldline.exec spectrum_along_fieldline.f90)
# target_link_libraries(spectrum_along_fieldline.exec ${MPI_Fortran_LIBRARIES}
#     analysis_config particle_spectrum fieldline_tracing particle_fieldline)
# install (TARGETS spectrum_along_fieldline.exec DESTINATION ${CMAKE_INSTALL_PREFIX})
# 
# add_executable(vdist_2d_along_fieldline.exec vdist_2d_along_fieldline.f90)
# target_link_libraries(vdist_2d_along_fieldline.exec ${MPI_Fortran_LIBRARIES}
#     analysis_config vdist fieldline_tracing particle_fieldline)
# install (TARGETS vdist_2d_along_fieldline.exec DESTINATION ${CMAKE_INSTALL_PREFIX})
# 
# add_executable(vdist_1d_along_fieldline.exec vdist_1d_along_fieldline.f90)
# target_link_libraries(vdist_1d_along_fieldline.exec ${MPI_Fortran_LIBRARIES}
#     analysis_config vdist fieldline_tracing particle_fieldline)
# install (TARGETS vdist_1d_along_fieldline.exec DESTINATION ${CMAKE_INSTALL_PREFIX})
# 
# add_executable(nrho_eband.exec particle_number_eband.f90)
# target_link_libraries(nrho_eband.exec ${MPI_Fortran_LIBRARIES} analysis_config
#     fields)
# install (TARGETS nrho_eband.exec DESTINATION ${CMAKE_INSTALL_PREFIX})
# 
# add_executable(bulk_internal_energy.exec bulk_internal_energy.f90)
# target_link_libraries(bulk_internal_energy.exec ${MPI_Fortran_LIBRARIES}
#     analysis_config fields)
# install (TARGETS bulk_internal_energy.exec DESTINATION ${CMAKE_INSTALL_PREFIX})
# 
# add_executable(jdote_eband.exec jdote_eband.f90)
# target_link_libraries(jdote_eband.exec ${MPI_Fortran_LIBRARIES} jdote_energy_band)
# install (TARGETS jdote_eband.exec DESTINATION ${CMAKE_INSTALL_PREFIX})
# 
# add_executable(particle_based_jdote.exec particle_based_jdote.f90)
# target_link_libraries(particle_based_jdote.exec ${MPI_Fortran_LIBRARIES}
#     particle_drift interp_emf get_info translate_module)
# install (TARGETS particle_based_jdote.exec DESTINATION ${CMAKE_INSTALL_PREFIX})
# 
# add_executable(particle_spectrum_box.exec particle_spectrum_box.f90)
# target_link_libraries(particle_spectrum_box.exec ${MPI_Fortran_LIBRARIES}
#     particle_spectrum get_info)
# install (TARGETS particle_spectrum_box.exec DESTINATION ${CMAKE_INSTALL_PREFIX})
# 
# add_executable(particle_spectrum_vdist_box.exec particle_spectrum_vdist_box.f90)
# target_link_libraries(particle_spectrum_vdist_box.exec ${MPI_Fortran_LIBRARIES}
#     particle_spectrum_vdist)
# install (TARGETS particle_spectrum_vdist_box.exec DESTINATION ${CMAKE_INSTALL_PREFIX})
