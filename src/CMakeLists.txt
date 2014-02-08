add_executable (kpsf solver.cc)
target_link_libraries (kpsf ${CERES_LIBRARIES})
target_link_libraries (kpsf ${CFITSIO_LIBRARIES})
install (TARGETS kpsf DESTINATION bin)


add_executable (mog mog.cc)
target_link_libraries (mog ${CERES_LIBRARIES})
target_link_libraries (mog ${CFITSIO_LIBRARIES})
install (TARGETS mog DESTINATION bin)
