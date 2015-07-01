#include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Add libraries.
add_library(read_config read_config.f90 global.f90)

add_library(get_info STATIC get_info.f90 global.f90)
target_link_libraries(get_info read_config)

add_library(statistics STATIC statistics.f90)

add_library(mpi_io mpi_io.f90 global.f90)

add_library(saving_flags saving_flags.f90)
target_link_libraries(saving_flags read_config)

add_library(parameters parameters.f90 global.f90)
target_link_libraries(parameters read_config)

add_library(analysis_config STATIC analysis_config.f90 mpi_setup.f90
    mpi_io_fields.f90 particle_info.f90)
target_link_libraries(analysis_config get_info mpi_io parameters)

add_library(magnetic_field STATIC magnetic_field.f90 global.f90)
target_link_libraries(magnetic_field get_info parameters)

add_library(electric_field STATIC electric_field.f90 global.f90)
target_link_libraries(electric_field get_info parameters)

add_library(fieldline_tracing STATIC fieldline_tracing.f90 global.f90)
target_link_libraries(fieldline_tracing magnetic_field)

add_library(pic_fields pic_fields.f90 global.f90)
target_link_libraries(pic_fields get_info saving_flags)

add_library(usingle usingle.f90 global.f90)
target_link_libraries(usingle get_info pic_fields)

add_library(fields STATIC inductive_efield.f90 para_perp_pressure.f90)
target_link_libraries(fields statistics get_info saving_flags pic_fields
    usingle)
    
add_library(jdote STATIC current_densities.f90 jdote_module.f90
    neighbors.f90 global.f90)

add_library(file_header STATIC file_header.f90)

add_library(translate_module translate_config.f90 emfields.f90
    particle_fields.f90 mpi_io_translate.f90 global.f90 get_info_translate.f90)
target_link_libraries(translate_module file_header get_info mpi_io)

add_library(spectrum_config spectrum_config.f90)
target_link_libraries(spectrum_config read_config)

add_library(particle_spectrum particle_module.f90 particle_frames.f90
    particle_spectrum.f90 particle_file.f90)
target_link_libraries(particle_spectrum file_header magnetic_field
    spectrum_config)

add_library(vdist vdist.f90 particle_module.f90 particle_frames.f90
    particle_file.f90)
target_link_libraries(vdist file_header get_info magnetic_field
    spectrum_config)

add_library(particle_fieldline particle_fieldline.f90 particle_file.f90
    particle_frames.f90 mpi_setup.f90)
target_link_libraries(particle_fieldline get_info fieldline_tracing)

add_library(pressure_tensor pressure_tensor.f90 neighbors.f90 global.f90)
target_link_libraries(pressure_tensor get_info)

add_library(compression compression.f90 global.f90)
