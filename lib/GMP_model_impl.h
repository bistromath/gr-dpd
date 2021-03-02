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

#include <dpd/GMP_model.h>
#include <armadillo>
#include <mutex>

using namespace arma;

namespace gr {
  namespace dpd {

    class GMP_model_impl : public GMP_model
    {
     private:
        std::vector<gr_complex> coeffs;
        size_t K_a, L_a;      //Number of time aligned envelope and signal coefficients -- order and lag
        size_t K_b, L_b, M_b; //Number of lagging envelope and signal coefficients -- order, lag, and memory
        size_t K_c, L_c, M_c; //Number of leading envelope and signal coefficients -- order, lag, and memory

        size_t get_history(); //Return the required number of history samples based on above
        size_t get_future();  //Return the required number of samples after the end of the working buffer based on the above

        size_t get_num_coeffs(); //Return the coefficient vector length required based on the above

        Col<gr_complex> _coeffs; //Coefficients of the model in a complicated and unreadable order

        std::mutex _lock;

     public:
      GMP_model_impl(size_t K_a, size_t L_a, size_t K_b, size_t L_b, size_t M_b, size_t K_c, size_t L_c, size_t M_c, std::vector<gr_complex> coeffs);
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

