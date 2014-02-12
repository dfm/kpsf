#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import glob
from subprocess import Popen, PIPE

# Launch the processes.
jobs = []
for fn in glob.glob("prf/kplr*prf.fits"):
    path, ext = os.path.splitext(fn)
    tmp, basefn = os.path.split(path)
    outfn = os.path.join("mog", basefn + ".mog.fits")
    jobs.append(Popen(["build/bin/kpsf-mog", fn, outfn],
                      stdout=PIPE, stderr=PIPE))

# Wait for the jobs to finish.
for job in jobs:
    print(job.communicate()[0])
