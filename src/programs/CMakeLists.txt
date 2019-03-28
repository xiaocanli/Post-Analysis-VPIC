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

#add_executable(translate_pre.exec translate_pre.f90)
#target_link_libraries(translate_pre.exec ${MPI_Fortran_LIBRARIES} translate_module flap)
#install (TARGETS translate_pre.exec DESTINATION ${CMAKE_INSTALL_PREFIX})
#
#add_executable(translate_post.exec translate_post.f90)
#target_link_libraries(translate_post.exec ${MPI_Fortran_LIBRARIES} translate_module flap)
#install (TARGETS translate_post.exec DESTINATION ${CMAKE_INSTALL_PREFIX})

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
    interp_vel_mom interp_comp_shear pre_post_emf interp_pre_post_b
    interp_pre_post_e interp_vexb ${HDF5_Fortran_LIBRARIES})
install (TARGETS particle_energization.exec DESTINATION ${CMAKE_INSTALL_PREFIX})

add_executable(particle_energization_io.exec particle_energization_io.f90)
target_link_libraries(particle_energization_io.exec ${MPI_Fortran_LIBRARIES} flap
    get_info translate_module particle_module interp_emf particle_file
    interp_vel_mom interp_comp_shear pre_post_emf pre_post_vexb interp_pre_post_b
    interp_pre_post_e interp_pre_post_u interp_vexb interp_gradu
    interp_j interp_grad_vperp interp_pre_post_v interp_pre_post_vexb
    ${HDF5_Fortran_LIBRARIES})
install (TARGETS particle_energization_io.exec DESTINATION ${CMAKE_INSTALL_PREFIX})

add_executable(combine_spectrum_hdf5.exec combine_spectrum_hdf5.f90)
target_link_libraries(combine_spectrum_hdf5.exec ${MPI_Fortran_LIBRARIES}
    flap parameters ${HDF5_Fortran_LIBRARIES})
install (TARGETS combine_spectrum_hdf5.exec DESTINATION ${CMAKE_INSTALL_PREFIX})

add_executable(combine_spectrum_binary.exec combine_spectrum_binary.f90)
target_link_libraries(combine_spectrum_binary.exec ${MPI_Fortran_LIBRARIES}
    flap parameters)
install (TARGETS combine_spectrum_binary.exec DESTINATION ${CMAKE_INSTALL_PREFIX})

add_executable(reduce_spectrum_hdf5.exec reduce_spectrum_hdf5.f90)
target_link_libraries(reduce_spectrum_hdf5.exec ${MPI_Fortran_LIBRARIES}
    flap parameters ${HDF5_Fortran_LIBRARIES})
install (TARGETS reduce_spectrum_hdf5.exec DESTINATION ${CMAKE_INSTALL_PREFIX})

add_executable(reorganize_spectrum_hdf5.exec reorganize_spectrum_hdf5.f90)
target_link_libraries(reorganize_spectrum_hdf5.exec ${MPI_Fortran_LIBRARIES}
    flap parameters ${HDF5_Fortran_LIBRARIES})
install (TARGETS reorganize_spectrum_hdf5.exec DESTINATION ${CMAKE_INSTALL_PREFIX})

add_executable(reduce_organized_spectrum_hdf5.exec reduce_organized_spectrum_hdf5.f90)
target_link_libraries(reduce_organized_spectrum_hdf5.exec ${MPI_Fortran_LIBRARIES}
    flap parameters ${HDF5_Fortran_LIBRARIES})
install (TARGETS reduce_organized_spectrum_hdf5.exec DESTINATION ${CMAKE_INSTALL_PREFIX})

add_executable(spectrum_reconnection_layer.exec spectrum_reconnection_layer.f90)
target_link_libraries(spectrum_reconnection_layer.exec ${MPI_Fortran_LIBRARIES}
    get_info flap parameters ${HDF5_Fortran_LIBRARIES})
install (TARGETS spectrum_reconnection_layer.exec DESTINATION ${CMAKE_INSTALL_PREFIX})

add_executable(fluid_energization.exec fluid_energization.f90)
target_link_libraries(fluid_energization.exec ${MPI_Fortran_LIBRARIES}
    get_info translate_module particle_info mpi_setup fluid_energization_module
    mpi_io_fields fields flap)
install (TARGETS fluid_energization.exec DESTINATION ${CMAKE_INSTALL_PREFIX})

add_executable(temperature_anisotropy.exec temperature_anisotropy.f90)
target_link_libraries(temperature_anisotropy.exec ${MPI_Fortran_LIBRARIES}
    get_info translate_module particle_info mpi_setup mpi_io_fields fields flap)
install (TARGETS temperature_anisotropy.exec DESTINATION ${CMAKE_INSTALL_PREFIX})

add_executable(calc_vexb.exec calc_vexb.f90)
target_link_libraries(calc_vexb.exec ${MPI_Fortran_LIBRARIES}
    get_info translate_module mpi_setup mpi_io_fields fields flap)
install (TARGETS calc_vexb.exec DESTINATION ${CMAKE_INSTALL_PREFIX})

add_executable(bulk_internal_energy.exec bulk_internal_energy.f90)
target_link_libraries(bulk_internal_energy.exec ${MPI_Fortran_LIBRARIES}
    get_info translate_module particle_info mpi_setup mpi_io_fields fields flap)
install (TARGETS bulk_internal_energy.exec DESTINATION ${CMAKE_INSTALL_PREFIX})

add_executable(vdot_kappa.exec vdot_kappa.f90)
target_link_libraries(vdot_kappa.exec ${MPI_Fortran_LIBRARIES}
    get_info translate_module mpi_setup mpi_io_fields fields flap)
install (TARGETS vdot_kappa.exec DESTINATION ${CMAKE_INSTALL_PREFIX})

add_executable(energetic_particle_density.exec energetic_particle_density.f90)
target_link_libraries(energetic_particle_density.exec ${MPI_Fortran_LIBRARIES}
    flap get_info translate_module particle_module particle_file
    ${HDF5_Fortran_LIBRARIES})
install (TARGETS energetic_particle_density.exec DESTINATION ${CMAKE_INSTALL_PREFIX})

add_executable(kappa_dist.exec kappa_dist.f90)
target_link_libraries(kappa_dist.exec ${MPI_Fortran_LIBRARIES}
    get_info translate_module mpi_setup mpi_io_fields fields flap)
install (TARGETS kappa_dist.exec DESTINATION ${CMAKE_INSTALL_PREFIX})

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
