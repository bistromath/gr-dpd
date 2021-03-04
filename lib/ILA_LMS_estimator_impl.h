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

#ifndef INCLUDED_DPD_ILA_LMS_ESTIMATOR_IMPL_H
#define INCLUDED_DPD_ILA_LMS_ESTIMATOR_IMPL_H

#include <dpd/ILA_LMS_estimator.h>
#include <armadillo>
using namespace arma;

namespace gr {
namespace dpd {

class ILA_LMS_estimator_impl : public ILA_LMS_estimator
{
private:
    size_t K_a, L_a,
           K_b, L_b, M_b,
           K_c, L_c, M_c;

    size_t d_iter_limit;
    size_t d_iter;
    float d_learning_rate;
    size_t d_block_size;
    std::vector<gr_complex> d_taps;
    float d_lambda;

    size_t get_num_coeffs();
    size_t get_future();
    size_t get_history();
    Col<gr_complexd> ls_estimation(Mat<gr_complexd> A, Col<gr_complexd> y);

public:
    ILA_LMS_estimator_impl(size_t K_a, size_t L_a,
                           size_t K_b, size_t L_b, size_t M_b,
                           size_t K_c, size_t L_c, size_t M_c,
                           size_t iter_limit, float learning_rate,
                           size_t block_size, float lambda,
                           std::vector<gr_complex> initial_taps);
    ~ILA_LMS_estimator_impl();

    void handle_trigger_msg(pmt::pmt_t trigger);


    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items);
};

} // namespace dpd
} // namespace gr

#endif /* INCLUDED_DPD_ILA_LMS_ESTIMATOR_IMPL_H */
