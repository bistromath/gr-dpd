#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2021 Nick Foster.
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
#


import numpy
from gnuradio import gr
import numpy as np
import scipy.interpolate
import pmt

class lookup_table(gr.sync_block):
    """
    docstring for block lookup_table
    """
    def __init__(self, table=(np.linspace(0,1,11), np.linspace(0,1,11)), method='linear'):  # only default arguments here
        """arguments to this function show up as parameters in GRC"""
        gr.sync_block.__init__(
            self,
            name='Lookup Table',
            in_sig=[np.float32],
            out_sig=[np.float32]
        )
        self.table = table
        self.method = method
        self.interpfn = scipy.interpolate.interp1d(table[0], table[1], kind=method)
        self.message_port_register_in(pmt.intern("table"))
        self.set_msg_handler(pmt.intern("table"), lambda x: self.set_table(pmt.to_python(x)))

    def set_table(self, table):
        self.table = table

    def work(self, input_items, output_items):
        output_items[0][:] = self.interpfn(input_items[0])
        return len(output_items[0])

