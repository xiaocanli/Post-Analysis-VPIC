f2py --f90exec=mpif90 -m particle_spectrum_vdist -c ../src/modules/particle_spectrum_vdist_python.f90 -I/net/scratch2/guofan/sigma1-mime25-beta001-track-3/pic_analysis/lib/ -L/net/scratch2/guofan/sigma1-mime25-beta001-track-3/pic_analysis/lib/ -lparticle_spectrum_vdist -lmpi -lspectrum_config -lparticle_file -lparticle_module -linterp_emf -lmagnetic_field -lparameters -lget_info -lfile_header -lparticle_spectrum -lpath_info -lvdist -lread_config
