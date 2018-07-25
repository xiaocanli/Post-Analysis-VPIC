#include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Add libraries.
add_library(read_config read_config.f90 global.f90)

add_library(path_info path_info.f90 global.f90)

add_library(get_info STATIC get_info.f90 get_info_translate.f90
    translate_config.f90 global.f90)
target_link_libraries(get_info read_config path_info)

add_library(statistics STATIC statistics.f90)

add_library(mpi_io mpi_io.f90 global.f90)

add_library(saving_flags saving_flags.f90)
target_link_libraries(saving_flags read_config)

add_library(parameters parameters.f90 global.f90)
target_link_libraries(parameters read_config)

add_library(mpi_setup mpi_setup.f90 global.f90 translate_config.f90)
target_link_libraries(mpi_setup get_info)

add_library(particle_info particle_info.f90 global.f90)
target_link_libraries(particle_info get_info)

add_library(mpi_io_fields mpi_io_fields.f90 global.f90)
target_link_libraries(mpi_io_fields parameters get_info particle_info
    mpi_io mpi_setup)

add_library(analysis_config STATIC analysis_config.f90 translate_config.f90
    get_info_translate.f90)
target_link_libraries(analysis_config get_info mpi_io parameters
    commandline_arguments mpi_setup mpi_io_fields)

add_library(magnetic_field STATIC magnetic_field.f90 global.f90)
target_link_libraries(magnetic_field get_info parameters)

add_library(electric_field STATIC electric_field.f90 global.f90)
target_link_libraries(electric_field get_info parameters)

add_library(fieldline_tracing STATIC fieldline_tracing.f90 global.f90)
target_link_libraries(fieldline_tracing magnetic_field)

add_library(pic_fields pic_fields.f90 global.f90)
target_link_libraries(pic_fields get_info saving_flags mpi_io mpi_setup
    get_info mpi_io_fields)

add_library(usingle usingle.f90 global.f90)
target_link_libraries(usingle get_info pic_fields)

add_library(fields STATIC inductive_efield.f90 para_perp_pressure.f90
    emf_derivatives.f90 hydro_derivatives.f90 vperp_derivatives.f90
    neighbors.f90)
target_link_libraries(fields statistics get_info saving_flags pic_fields
    usingle)

add_library(pre_post_hydro STATIC pre_post_hydro.f90)
target_link_libraries(pre_post_hydro pic_fields)

add_library(pre_post_emf STATIC pre_post_emf.f90)
target_link_libraries(pre_post_emf pic_fields)

add_library(exb_drift exb_drift.f90 neighbors.f90 global.f90)
target_link_libraries(exb_drift get_info fields file_header)

add_library(pre_post_vexb STATIC pre_post_vexb.f90)
target_link_libraries(pre_post_vexb exb_drift)

add_library(jdote STATIC current_densities.f90 jdote_module.f90
    neighbors.f90 global.f90)
target_link_libraries(jdote compression_shear pre_post_hydro)

add_library(fluid_energization_module STATIC fluid_energization_module.f90 neighbors.f90)
target_link_libraries(fluid_energization_module pre_post_hydro)

add_library(file_header STATIC file_header.f90)

add_library(translate_module translate_config.f90 emfields.f90
    particle_fields.f90 mpi_io_translate.f90 global.f90 get_info_translate.f90
    topology_translate.f90)
target_link_libraries(translate_module file_header get_info mpi_io parameters)

add_library(spectrum_config spectrum_config.f90)
target_link_libraries(spectrum_config read_config get_info)

add_library(particle_module particle_module.f90)
target_link_libraries(particle_module get_info file_header interp_emf
    magnetic_field particle_info)

add_library(particle_file particle_file.f90)
target_link_libraries(particle_file file_header)

add_library(particle_spectrum particle_frames.f90 particle_spectrum.f90)
target_link_libraries(particle_spectrum file_header magnetic_field
    spectrum_config particle_module particle_file maximum_energy
    commandline_arguments)

add_library(vdist vdist.f90 particle_frames.f90)
target_link_libraries(vdist file_header get_info magnetic_field
    spectrum_config particle_module particle_file)

add_library(particle_fieldline particle_fieldline.f90 particle_frames.f90)
target_link_libraries(particle_fieldline get_info fieldline_tracing
    particle_file mpi_setup)

add_library(pressure_tensor pressure_tensor.f90 neighbors.f90 global.f90)
target_link_libraries(pressure_tensor get_info fields)

add_library(compression_shear compression_shear.f90 global.f90
    para_perp_pressure.f90)
target_link_libraries(compression_shear pressure_tensor get_info pic_fields)

add_library(jdote_energy_band jdote_energy_band.f90 global.f90)
target_link_libraries(jdote_energy_band read_config analysis_config)

add_library(interp_funs interpolation_funs.f90 global.f90)

add_library(interp_emf interpolation_emf.f90 neighbors.f90)
target_link_libraries(interp_emf interp_funs get_info fields file_header)

add_library(interp_gradu interpolation_gradu.f90 neighbors.f90)
target_link_libraries(interp_gradu interp_funs get_info fields file_header)

add_library(interp_grad_vperp interpolation_vperp_derivatives.f90 neighbors.f90)
target_link_libraries(interp_grad_vperp interp_funs get_info fields file_header)

add_library(interp_pfields interpolation_particle_fields.f90 neighbors.f90)
target_link_libraries(interp_pfields interp_funs get_info usingle)

add_library(interp_comp_shear interpolation_comp_shear.f90 neighbors.f90)
target_link_libraries(interp_comp_shear interp_funs get_info fields file_header)

add_library(interp_vexb interpolation_vexb.f90 neighbors.f90)
target_link_libraries(interp_vexb interp_funs get_info fields
    file_header exb_drift)

add_library(interp_vel_mom interpolation_vel_mom.f90 neighbors.f90)
target_link_libraries(interp_vel_mom interp_funs get_info fields file_header)

add_library(interp_j interpolation_current_density.f90 neighbors.f90)
target_link_libraries(interp_j interp_funs get_info fields file_header)

add_library(interp_pre_post_b interpolation_pre_post_bfield.f90 neighbors.f90)
target_link_libraries(interp_pre_post_b interp_funs get_info pre_post_emf
    file_header)

add_library(interp_pre_post_e interpolation_pre_post_efield.f90 neighbors.f90)
target_link_libraries(interp_pre_post_e interp_funs get_info pre_post_emf
    file_header)

add_library(interp_pre_post_u interpolation_pre_post_ufield.f90 neighbors.f90)
target_link_libraries(interp_pre_post_u interp_funs get_info pre_post_hydro
    file_header)

add_library(interp_pre_post_v interpolation_pre_post_vfield.f90 neighbors.f90)
target_link_libraries(interp_pre_post_v interp_funs get_info pre_post_hydro
    file_header)

add_library(interp_pre_post_vexb interpolation_pre_post_vexb.f90 neighbors.f90)
target_link_libraries(interp_pre_post_vexb interp_funs get_info pre_post_vexb
    file_header)

add_library(particle_drift particle_drift.f90 particle_frames.f90)
target_link_libraries(particle_drift file_header magnetic_field interp_emf
    particle_module translate_module particle_file)

add_library(particle_spectrum_vdist particle_spectrum_vdist.f90 global.f90)
target_link_libraries(particle_spectrum_vdist particle_spectrum vdist
    get_info commandline_arguments flap)

add_library(cla cla.f90 kinds.f90)

add_library(commandline_arguments commandline_arguments.f90 global.f90)
target_link_libraries(commandline_arguments cla spectrum_config parameters
    particle_info)

add_library(maximum_energy maximum_energy.f90 global.f90)
target_link_libraries(maximum_energy get_info spectrum_config mpi_io
    path_info mpi_setup)
