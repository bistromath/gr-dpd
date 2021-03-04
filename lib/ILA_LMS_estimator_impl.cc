/* -*- c++ -*- */
/*
 * Copyright 2020 gr-dpd author.
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
#define ARMA_DONT_PRINT_ERRORS
#include "ILA_LMS_estimator_impl.h"
#include <gnuradio/io_signature.h>
#include <armadillo>

using namespace arma;

namespace gr {
namespace dpd {

    ILA_LMS_estimator::sptr ILA_LMS_estimator::make(GMP gmp,
                                        size_t iter_limit, float learning_rate,
                                        size_t block_size, float lambda,
                                        std::vector<gr_complex> initial_taps)
    {
        return gnuradio::get_initial_sptr(
            new ILA_LMS_estimator_impl(gmp, iter_limit, learning_rate, block_size, lambda, initial_taps));
    }


    /*
    * The private constructor
    */
    ILA_LMS_estimator_impl::ILA_LMS_estimator_impl(GMP gmp,
                                                size_t iter_limit, float learning_rate,
                                                size_t block_size, float lambda,
                                                std::vector<gr_complex> initial_taps = {1})
        : gr::sync_block("ILA_LMS_estimator",
                        gr::io_signature::make(2, 2, sizeof(gr_complex)),
                        gr::io_signature::make(0, 0, sizeof(float))),
        d_gmp(gmp),
        d_iter_limit(iter_limit),
        d_iter(0),
        d_learning_rate(learning_rate),
        d_block_size(block_size),
        d_lambda(lambda)
    {
        // set up message ports
        message_port_register_out(pmt::mp("taps"));
        message_port_register_in(pmt::mp("trigger"));
        set_msg_handler(pmt::mp("trigger"), boost::bind(&ILA_LMS_estimator_impl::handle_trigger_msg, this, _1));
        // initialize coefficients
        for (auto tap : initial_taps) {
            d_taps.push_back(tap);
        }
        d_taps.resize(d_gmp.num_coeffs());

        set_history(d_gmp.history()+1);
        set_output_multiple(d_block_size);
    }

    ILA_LMS_estimator_impl::~ILA_LMS_estimator_impl() {}

    void ILA_LMS_estimator_impl::handle_trigger_msg(pmt::pmt_t trigger)
    {
        d_iter = 0;
        d_iter_limit = 1;
    }

    Col<gr_complexd> ILA_LMS_estimator_impl::ls_estimation(Mat<gr_complexd> A, Col<gr_complexd> y)
    {
        Mat<gr_complexd> regularizer(d_gmp.num_coeffs(), d_gmp.num_coeffs(), fill::eye);
        regularizer *= d_lambda;

        auto ls_result = solve( A.t()*A + regularizer, A.t()*y ).eval();
        return ls_result;
    }

    int ILA_LMS_estimator_impl::work(int noutput_items,
                                    gr_vector_const_void_star& input_items,
                                    gr_vector_void_star& output_items)
    {
        // we operate on fixed block sizes so as to retain
        // control over the condition of the solver
        if(noutput_items < d_block_size+d_gmp.future()) {
            return 0;
        }

        // TODO FIXME synchronize the inputs by waiting to hear from both the
        // pre and postdistorters -- they will output a tag that says "new_coeffs"
        // with a UUID after they update. this will ensure that you're listening
        // to new data. obviously, in the real world this won't work, because
        // tags won't propagate through a power amplifier.

        if(d_iter >= d_iter_limit) return noutput_items;

        const gr_complex* error    = (const gr_complex*) input_items[0]; // Error input (predistorted - postdistorted)
        const gr_complex* paoutput = (const gr_complex*) input_items[1]; // PA output (before postdistorter)

        size_t nsamps = d_block_size;

        // recast error input as Armadillo column vector and convert to double for the solver
        const Col<gr_complex> error_col(const_cast<gr_complex*>(error), nsamps, false, true);
        auto errord = conv_to<Col<gr_complexd>>::from(error_col);

        // generate basis matrix from PA output
        auto paout_basis = d_gmp.gen_basis_matrix<float, double>(paoutput, nsamps);

        //run LS estimation on the PA basis and error vector
        auto ls_result = ls_estimation(paout_basis, errord);

        // recast taps vector as a column vector (could be done once, but this is lightweight)
        // and then do the update to taps.
        Col<gr_complex> taps_col(&d_taps[0], d_gmp.num_coeffs(), false, true);
        taps_col = taps_col + conv_to<Col<gr_complex>>::from(ls_result)*d_learning_rate;

        //publish a message!
        pmt::pmt_t pmt_taps = pmt::init_c32vector(d_taps.size(), &d_taps[0]);
        message_port_pub(pmt::mp("taps"), pmt_taps);

        d_iter++;

        return nsamps;
    }
} /* namespace dpd */
} /* namespace gr */
