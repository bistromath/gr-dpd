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
#include "gen_GMP_basis_matrix.h"

using namespace arma;

namespace gr {
namespace dpd {

    ILA_LMS_estimator::sptr ILA_LMS_estimator::make(size_t K_a, size_t L_a,
                                        size_t K_b, size_t L_b, size_t M_b,
                                        size_t K_c, size_t L_c, size_t M_c,
                                        size_t iter_limit, float learning_rate,
                                        size_t block_size, float lambda,
                                        std::vector<gr_complex> initial_taps)
    {
        return gnuradio::get_initial_sptr(
            new ILA_LMS_estimator_impl(K_a, L_a, K_b, L_b, M_b, K_c, L_c, M_c, iter_limit, learning_rate, block_size, lambda, initial_taps));
    }


    /*
    * The private constructor
    */
    ILA_LMS_estimator_impl::ILA_LMS_estimator_impl(size_t K_a, size_t L_a,
                                                size_t K_b, size_t L_b, size_t M_b,
                                                size_t K_c, size_t L_c, size_t M_c,
                                                size_t iter_limit, float learning_rate,
                                                size_t block_size, float lambda,
                                                std::vector<gr_complex> initial_taps = {1})
        : gr::sync_block("ILA_LMS_estimator",
                        gr::io_signature::make(2, 2, sizeof(gr_complex)),
                        gr::io_signature::make(0, 0, sizeof(float))),
        K_a(K_a),
        L_a(L_a),
        K_b(K_b),
        L_b(L_b),
        M_b(M_b),
        K_c(K_c),
        L_c(L_c),
        M_c(M_c),
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
        d_taps.resize(get_num_coeffs());

        set_history(get_history());
        set_output_multiple(d_block_size);
    }

    ILA_LMS_estimator_impl::~ILA_LMS_estimator_impl() {}

    void ILA_LMS_estimator_impl::handle_trigger_msg(pmt::pmt_t trigger)
    {
        d_iter = 0;
        d_iter_limit = 1;
    }

    /* the next 3 helper functions are duplicated with GMP_model_impl.
     * should probably find a way to factor it out.
     * TODO FIXME to redo these blocks with masks:
     * make each input K_a, L_a... a std::vector just listing which
     * orders and which lags you care about. you'll have to recalculate
     * history, num_coeffs, etc based on this. that way you can have
     * separate masks for the fwd model, the postdistorter, etc.
     */
    size_t ILA_LMS_estimator_impl::get_history()
    {
        return std::max({static_cast<int>(L_a)-1,
                         static_cast<int>(L_b)-1+static_cast<int>(M_b),
                         static_cast<int>(L_c)-1})+1;
    }

    size_t ILA_LMS_estimator_impl::get_future()
    {
        return M_c;
    }

    size_t ILA_LMS_estimator_impl::get_num_coeffs()
    {
        return K_a*L_a + K_b*L_b*M_b + K_c*L_c*M_c;
    }

    //TODO FIXME ok let's talk about amplitude and normalization.
    //The output will be related to the input by both a nonlinear transformation (the amplifier's nonlinearity)
    //and a linear scaling factor K, due to both the amplifier gain and the feedback path gain. we need to
    //compensate for that in order to accurately detect nonlinearities, right? yeah. how do we do this?
    //well, we have the amplifier input as well as the amplifier output, as well as the original input.
    //do we want the output scaled to the input, or to the predistorted output? do we want the scaling
    //applied after the postdistorter? no, the goal is the postdistorter should be operating on samples
    //of the same magnitude as the predistorter. that seems to imply that the samples should be normalized
    //to the input of the predistorter -- i.e., the original samples -- and applied to the input of the
    //postdistorter.
    //
    //Obviously, there is distortion in the gain estimate, since nonlinearities imply that the output
    //is not a simple linear transformation of the input. In addition, time-delayed nonlinearities will
    //further complicate that. How do we make an educated guess? Peak amplitude? RMS power?
    //
    //The output and the input, when linearized, will be copies of each other, at least in theory.
    //As such, they will be related by a linear factor K. We should scale the amp output going into
    //the predistorter by that factor K. Oh. Doing that completely removes the AM-AM nonlinearities
    //from the PA output signal. Using RMS works great.
    //
    //So, we've normalized the PA output in, and the postdistorter input. This means that the output
    //of the ILA model will always have the same linear magnitude as the input, and so the first
    //coefficient will always have magnitude 1. Well, no -- it will have the same RMS power as the
    //input signal. That's different. In practice, though, it will be near unity. Is that what we want?
    //Or do we want to normalize peak power? I think RMS is a good target. It lets you stretch the top
    //end of the amplifier.

    Col<gr_complexd> ILA_LMS_estimator_impl::ls_estimation(Mat<gr_complexd> A, Col<gr_complexd> y)
    {
        Mat<gr_complexd> regularizer(get_num_coeffs(), get_num_coeffs(), fill::eye);
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
        if(noutput_items < d_block_size+get_future()) {
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
        auto paout_basis = gen_GMP_basis_matrix<float, double>(paoutput, nsamps, K_a, L_a, K_b, L_b, M_b, K_c, L_c, M_c);

        //run LS estimation on the PA basis and error vector
        auto ls_result = ls_estimation(paout_basis, errord);

        // recast taps vector as a column vector (could be done once, but this is lightweight)
        // and then do the update to taps.
        Col<gr_complex> taps_col(&d_taps[0], get_num_coeffs(), false, true);
        taps_col = taps_col + conv_to<Col<gr_complex>>::from(ls_result)*d_learning_rate;

        //publish a message!
        pmt::pmt_t pmt_taps = pmt::init_c32vector(d_taps.size(), &d_taps[0]);
        message_port_pub(pmt::mp("taps"), pmt_taps);

        d_iter++;

        return nsamps;
    }
} /* namespace dpd */
} /* namespace gr */
