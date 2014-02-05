add_executable (kpsf solver.cc)
target_link_libraries (kpsf ${CERES_LIBRARIES})
target_link_libraries (kpsf ${CFITSIO_LIBRARIES})

install (TARGETS kpsf DESTINATION bin)
