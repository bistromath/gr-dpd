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
#include <dpd/GMP.h>

using namespace arma;

namespace gr {
  namespace dpd {

    GMP_model::sptr
    GMP_model::make(GMP gmp, const std::vector<gr_complex> &coeffs)
    {
      return gnuradio::get_initial_sptr
        (new GMP_model_impl(gmp, coeffs));
    }


    /*
     * The private constructor
     */

    GMP_model_impl::GMP_model_impl(GMP gmp, const std::vector<gr_complex> &coeffs = {})
      : gr::sync_block("GMP_model",
              gr::io_signature::make(1, 1, sizeof(gr_complex)),
              gr::io_signature::make(1, 1, sizeof(gr_complex))),
        gmp(gmp)
    {
        set_coeffs(coeffs);
        set_history(gmp.history()+1);
        message_port_register_in(pmt::mp("taps"));
        set_msg_handler(pmt::mp("taps"), boost::bind(&GMP_model_impl::handle_msg, this, _1));
    }

    /*
     * Our virtual destructor.
     */
    GMP_model_impl::~GMP_model_impl()
    {
    }

    void GMP_model_impl::set_coeffs(const std::vector<gr_complex> &coeffs) {
      std::lock_guard<std::mutex> lock(_lock);
      _coeffs = conv_to<Col<gr_complex>>::from(coeffs);
      _coeffs.resize(gmp.num_coeffs());
    }

    void GMP_model_impl::handle_msg(pmt::pmt_t coeffpmt) {
        size_t len;
        const gr_complex *ref = pmt::c32vector_elements(coeffpmt, len);
        const std::vector<gr_complex> vecref(ref, ref+len);
        set_coeffs(vecref);
    }

    int
    GMP_model_impl::work(int noutput_items,
        gr_vector_const_void_star &input_items,
        gr_vector_void_star &output_items)
    {
      const gr_complex *in = (const gr_complex *) input_items[0]; //in[0] is time=(0-history)
      gr_complex *out = (gr_complex *) output_items[0];

      size_t nsamps = noutput_items - gmp.future();
      if(nsamps <= 0) return 0;

      auto X = gmp.gen_basis_matrix<float,float>(in, nsamps);

      //we note that Armadillo column vectors are dense in memory, and can be used as such
      Col<gr_complex> outVec(out, nsamps, false, true);
      std::lock_guard<std::mutex> lock(_lock); //don't change coeffs out from under us during calls
      outVec = X * _coeffs;

      // Tell runtime system how many output items we produced.
      return nsamps;
    }

  } /* namespace dpd */
} /* namespace gr */

