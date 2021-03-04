/* -*- c++ -*- */
/*
 * Copyright 2021 Nick Foster.
 *
 * This is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this software; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#ifndef INCLUDED_DPD_GMP_MODEL_IMPL_H
#define INCLUDED_DPD_GMP_MODEL_IMPL_H

#include <dpd/GMP.h>
#include <dpd/GMP_model.h>
#include <armadillo>
#include <mutex>

using namespace arma;

namespace gr {
  namespace dpd {

    class GMP_model_impl : public GMP_model
    {
     private:
        GMP gmp;
        Col<gr_complex> _coeffs; //Coefficients of the model in a complicated and unreadable order
        std::mutex _lock;

     public:
      GMP_model_impl(GMP gmp, const std::vector<gr_complex> &coeffs);
      ~GMP_model_impl();

      void set_coeffs(const std::vector<gr_complex> &coeffs);
      void handle_msg(pmt::pmt_t coeffpmt);

      int work(
              int noutput_items,
              gr_vector_const_void_star &input_items,
              gr_vector_void_star &output_items
      );
    };

  } // namespace dpd
} // namespace gr

#endif /* INCLUDED_DPD_GMP_MODEL_IMPL_H */

