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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gnuradio/io_signature.h>
#include "GMP_model_impl.h"
#include "gen_GMP_basis_matrix.h"

using namespace arma;

namespace gr {
  namespace dpd {

    GMP_model::sptr
    GMP_model::make(size_t K_a, size_t L_a, size_t K_b, size_t L_b, size_t M_b, size_t K_c, size_t L_c, size_t M_c, std::vector<gr_complex> coeffs)
    {
      return gnuradio::get_initial_sptr
        (new GMP_model_impl(K_a, L_a, K_b, L_b, M_b, K_c, L_c, M_c, coeffs));
    }


    /*
     * The private constructor
     */

    //TODO FIXME make this have a coeffs message input which can take a vector of coeffs as a PMT
    GMP_model_impl::GMP_model_impl(size_t K_a, size_t L_a,
                                   size_t K_b, size_t L_b, size_t M_b,
                                   size_t K_c, size_t L_c, size_t M_c,
                                   std::vector<gr_complex> coeffs = {})
      : gr::sync_block("GMP_model",
              gr::io_signature::make(1, 1, sizeof(gr_complex)),
              gr::io_signature::make(1, 1, sizeof(gr_complex))),
        K_a(K_a),
        L_a(L_a),
        K_b(K_b),
        L_b(L_b),
        M_b(M_b),
        K_c(K_c),
        L_c(L_c),
        M_c(M_c),
        _coeffs(coeffs)
    {
        _coeffs.resize(get_num_coeffs());
        set_history(get_history());
    }

    /*
     * Our virtual destructor.
     */
    GMP_model_impl::~GMP_model_impl()
    {
    }

    size_t GMP_model_impl::get_history()
    {
        return std::max(L_a, L_b*M_b);
    }

    size_t GMP_model_impl::get_future()
    {
        return M_c;
    }

    size_t GMP_model_impl::get_num_coeffs()
    {
        return K_a*L_a + K_b*L_b*M_b + K_c*L_c*M_c;
    }

    int
    GMP_model_impl::work(int noutput_items,
        gr_vector_const_void_star &input_items,
        gr_vector_void_star &output_items)
    {
      const gr_complex *in = (const gr_complex *) input_items[0];
      gr_complex *out = (gr_complex *) output_items[0];

      size_t nsamps = noutput_items - get_future();
      auto X = gen_GMP_basis_matrix<float,float>(in, nsamps, K_a, L_a, K_b, L_b, M_b, K_c, L_c, M_c);

      //we note that Armadillo column vectors are dense in memory, and can be used as such
      Col<gr_complex> outVec(out, nsamps, false, true);
      outVec = X * _coeffs;

      // Tell runtime system how many output items we produced.
      return nsamps;
    }

  } /* namespace dpd */
} /* namespace gr */

