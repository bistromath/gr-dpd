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

#ifndef INCLUDED_DPD_ILA_LMS_ESTIMATOR_H
#define INCLUDED_DPD_ILA_LMS_ESTIMATOR_H

#include <gnuradio/sync_block.h>
#include <dpd/api.h>

namespace gr {
namespace dpd {

/*!
 * \brief LMS based Algorithm implemented to estimate the coefficients of
 *  the behaviorial model (GMP) of the Power Amplifier and thus
 *  predistorter taps using an indirect learning architecture.
 * \ingroup dpd
 *
 * \details
 *  The block uses either Newton's method or an exponential moving average
 *  method to estimate the nonlinear coefficients of an amplifier model.
 *  Indirect learning compares the input to the PA with an identical model of
 *  the PA applied to the PA output, a so-called postdistorter. When the two
 *  models are the same, the system has converged and the PA output is maximally
 *  linear.
 *
 *  The two inputs are the input to the PA (after the predistorter) and the
 *  output of the postdistorter (after the PA).
 *
 *  The estimated model coefficients are output as messages through message
 *  output port 'taps'. The messages should be sent both to the predistorter
 *  and postdistorter blocks to complete the update loop.
 *
 *  The loop will run for iter_limit iterations and can be triggered again
 *  by sending a PMT of any type to the "trigger" message input.
 *
 */
class DPD_API ILA_LMS_estimator : virtual public gr::sync_block
{
public:
    typedef boost::shared_ptr<ILA_LMS_estimator> sptr;

    enum methods{ NEWTON, EMA };

    /*!
     * \brief Make ILA_LMS_estimator
     *
     * \param dpd_params The (K_a, L_a, K_b, L_b, M_b) int_vector denoting
     *  the GMP model parameters used for predistorter 'taps' estimation.
     *  Total No. of coefficients = ((K_a * L_a) + (K_b * M_b * L_b))
     * \param iter_limit Iteration limit or Max. number of iterations of training
     *  to be performed for predistorter DPD coefficients estimation.
     * \param method Method of LMS algorithm used for coefficients estimation,
     *  i.e., Newton or EMA based method
     * \param learning_rate Learning rate, float value can lie in range 0 to 1.
     *
     */
    static sptr
    make(size_t K_a, size_t L_a,
         size_t K_b, size_t L_b, size_t M_b,
         size_t K_c, size_t L_c, size_t M_c,
         size_t iter_limit, ILA_LMS_estimator::methods method,
         float learning_rate, size_t block_size,
         std::vector<gr_complexd> initial_taps);
};

} // namespace dpd
} // namespace gr

#endif /* INCLUDED_DPD_ILA_LMS_ESTIMATOR_H */
