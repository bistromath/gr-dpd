# Copyright 2011 Free Software Foundation, Inc.
#
# This file was generated by gr_modtool, a tool from the GNU Radio framework
# This file is a part of gr-dpd
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

########################################################################
# Check if there is C++ code at all
########################################################################
if(NOT dpd_sources)
    MESSAGE(STATUS "No C++ sources... skipping swig/")
    return()
endif(NOT dpd_sources)

########################################################################
# Include swig generation macros
########################################################################
find_package(SWIG)
find_package(PythonLibs)
if(NOT SWIG_FOUND OR NOT PYTHONLIBS_FOUND)
    return()
endif()
include(GrSwig)
include(GrPython)

########################################################################
# Setup swig generation
########################################################################
set(GR_SWIG_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../include $<TARGET_PROPERTY:gnuradio::runtime_swig,INTERFACE_INCLUDE_DIRECTORIES>)
set(GR_SWIG_TARGET_DEPS gnuradio::runtime_swig)

set(GR_SWIG_LIBRARIES gnuradio-dpd)

set(GR_SWIG_DOC_FILE ${CMAKE_CURRENT_BINARY_DIR}/dpd_swig_doc.i)
set(GR_SWIG_DOC_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../include)

GR_SWIG_MAKE(dpd_swig dpd_swig.i)

########################################################################
# Install the build swig module
########################################################################
GR_SWIG_INSTALL(TARGETS dpd_swig DESTINATION ${GR_PYTHON_DIR}/dpd)

########################################################################
# Install swig .i files for development
########################################################################
install(
    FILES
    dpd_swig.i
    ${CMAKE_CURRENT_BINARY_DIR}/dpd_swig_doc.i
    DESTINATION ${GR_INCLUDE_DIR}/dpd/swig
)
