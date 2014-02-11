set (KPSF_LIBRARY_DEPS
     ${CERES_LIBRARIES}
     ${CFITSIO_LIBRARIES})

# Build and install the library.
add_library (kpsf STATIC kpsf.cc)
target_link_libraries (kpsf ${KPSF_LIBRARY_DEPS})
install (TARGETS kpsf
         RUNTIME DESTINATION bin
         LIBRARY DESTINATION lib${LIB_SUFFIX}
         ARCHIVE DESTINATION lib${LIB_SUFFIX})

# Build the demo code.
add_executable (kpsf-demo demo.cc)
target_link_libraries (kpsf-demo kpsf)
install (TARGETS kpsf-demo DESTINATION bin)

# # Build the mixture-of-Gaussians code.
# add_executable (mog mog.cc)
# target_link_libraries (mog ${CERES_LIBRARIES})
# target_link_libraries (mog ${CFITSIO_LIBRARIES})
# install (TARGETS mog DESTINATION bin)
