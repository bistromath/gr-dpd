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
                                        size_t iter_limit, ILA_LMS_estimator::methods method,
                                        float learning_rate, size_t block_size,
                                        std::vector<gr_complexd> initial_taps)
    {
        return gnuradio::get_initial_sptr(
            new ILA_LMS_estimator_impl(K_a, L_a, K_b, L_b, M_b, K_c, L_c, M_c, iter_limit, method, learning_rate, block_size, initial_taps));
    }


    /*
    * The private constructor
    */
    ILA_LMS_estimator_impl::ILA_LMS_estimator_impl(size_t K_a, size_t L_a,
                                                size_t K_b, size_t L_b, size_t M_b,
                                                size_t K_c, size_t L_c, size_t M_c,
                                                size_t iter_limit, ILA_LMS_estimator::methods method,
                                                float learning_rate, size_t block_size,
                                                std::vector<gr_complexd> initial_taps = {1})
        : gr::sync_block("ILA_LMS_estimator",
                        gr::io_signature::make(2, 2, sizeof(gr_complex)),
                        gr::io_signature::make(0, 0, 0)),
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
        d_method(method),
        d_learning_rate(learning_rate),
        d_block_size(block_size),
        d_taps(initial_taps)
    {
        // set up message ports
        message_port_register_out(pmt::mp("taps"));
        message_port_register_in(pmt::mp("trigger"));
        set_msg_handler(pmt::mp("trigger"), boost::bind(&ILA_LMS_estimator_impl::handle_trigger_msg, this, _1));

        // initialize coefficients
        d_taps.resize(get_num_coeffs());

        set_history(get_history());
    }

    ILA_LMS_estimator_impl::~ILA_LMS_estimator_impl() {}

    void ILA_LMS_estimator_impl::handle_trigger_msg(pmt::pmt_t trigger)
    {
        d_iter = 0;
    }

    /* the next 3 helper functions are duplicated with GMP_model_impl.
     * should probably find a way to factor it out.
     */
    size_t ILA_LMS_estimator_impl::get_history()
    {
        return std::max(L_a, L_b*M_b);
    }

    size_t ILA_LMS_estimator_impl::get_future()
    {
        return M_c;
    }

    size_t ILA_LMS_estimator_impl::get_num_coeffs()
    {
        return K_a*L_a + K_b*L_b*M_b + K_c*L_c*M_c;
    }

    // TODO FIXME figure out how to run this all at double precision, if necessary
    Col<gr_complex> ILA_LMS_estimator_impl::ls_estimation(Mat<gr_complex> A, Col<gr_complex> y)
    {
        const float lambda = 0.001;
        Mat<gr_complex> regularizer(get_num_coeffs(), get_num_coeffs(), fill::eye);
        regularizer *= lambda;

        auto ls_result = solve( A.t()*A + regularizer, A.t()*y );
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

        const gr_complex* postdist = (const gr_complex*)input_items[0]; // PA output (after postdistorter)
        const gr_complex* painput  = (const gr_complex*)input_items[1]; // PA input (predistorter output)

        // generate basis matrix from postdistorted data
        size_t nsamps = d_block_size;
        auto postdist_basis = gen_GMP_basis_matrix<float, float>(postdist, nsamps, K_a, L_a, K_b, L_b, M_b, K_c, L_c, M_c);

        // recast inputs as Armadillo column vectors
        const Col<gr_complex> postdist_col(const_cast<gr_complex *>(postdist), nsamps, false, true);
        const Col<gr_complex> painput_col (const_cast<gr_complex *>(painput), nsamps, false, true);
        // recast taps vector as a column vector
        Col<gr_complexd> coltaps(&d_taps[0], get_num_coeffs(), false, true);

        if(d_method == ILA_LMS_estimator::methods::NEWTON) {
            auto error = painput_col - postdist_col;
            auto ls_result = ls_estimation(postdist_basis, error);
            auto correction = d_learning_rate * ls_result;
            coltaps = coltaps + correction;
        } else {
            auto ls_result = ls_estimation(postdist_basis, painput_col);
            coltaps = (1.0-d_learning_rate)*coltaps + d_learning_rate*ls_result;
        }

        return nsamps;
    }
} /* namespace dpd */
} /* namespace gr */
