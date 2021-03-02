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

#ifndef INCLUDED_DPD_GMP_MODEL_H
#define INCLUDED_DPD_GMP_MODEL_H

#include <dpd/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
  namespace dpd {

    /*!
     * \brief <+description of block+>
     * \ingroup dpd
     *
     */
    class DPD_API GMP_model : virtual public gr::sync_block
    {
     public:
      typedef boost::shared_ptr<GMP_model> sptr;

      /*!
       * \brief Return a shared_ptr to a new instance of dpd::GMP_model.
       *
       * To avoid accidental use of raw pointers, dpd::GMP_model's
       * constructor is in a private implementation
       * class. dpd::GMP_model::make is the public interface for
       * creating new instances.
       */
      static sptr make(size_t K_a, size_t L_a, size_t K_b, size_t L_b, size_t M_b, size_t K_c, size_t L_c, size_t M_c, std::vector<gr_complex> coeffs);
    };

  } // namespace dpd
} // namespace gr

#endif /* INCLUDED_DPD_GMP_MODEL_H */

