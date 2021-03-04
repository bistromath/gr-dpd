/* -*- c++ -*- */

#define DPD_API

%include "gnuradio.i"           // the common stuff

//for GMP.h python bindings for std::set
%include <typemaps.i>
%include "std_set.i"
%template(iset) std::set<int>;

//load generated python docstrings
%include "dpd_swig_doc.i"

%{
#include "dpd/MP_model_PA.h"
#include "dpd/GMP_model_PA.h"
#include "dpd/predistorter_training.h"
#include "dpd/RLS_postdistorter.h"
#include "dpd/stream_to_gmp_vector.h"
#include "dpd/gain_phase_calibrate.h"
#include "dpd/stream_to_message.h"
#include "dpd/ILA_LMS_estimator.h"
#include "dpd/GMP_model.h"
#include "dpd/GMP.h"
%}

%include <armadillo>
%include "dpd/GMP.h"

%include "dpd/MP_model_PA.h"
GR_SWIG_BLOCK_MAGIC2(dpd, MP_model_PA);
%include "dpd/GMP_model_PA.h"
GR_SWIG_BLOCK_MAGIC2(dpd, GMP_model_PA);
%include "dpd/stream_to_message.h"
GR_SWIG_BLOCK_MAGIC2(dpd, stream_to_message);
%include "dpd/predistorter_training.h"
GR_SWIG_BLOCK_MAGIC2(dpd, predistorter_training);
%include "dpd/RLS_postdistorter.h"
GR_SWIG_BLOCK_MAGIC2(dpd, RLS_postdistorter);
%include "dpd/stream_to_gmp_vector.h"
GR_SWIG_BLOCK_MAGIC2(dpd, stream_to_gmp_vector);
%include "dpd/gain_phase_calibrate.h"
GR_SWIG_BLOCK_MAGIC2(dpd, gain_phase_calibrate);
%include "dpd/ILA_LMS_estimator.h"
GR_SWIG_BLOCK_MAGIC2(dpd, ILA_LMS_estimator);
%include "dpd/GMP_model.h"
GR_SWIG_BLOCK_MAGIC2(dpd, GMP_model);
