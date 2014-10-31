#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = []

import kplr
import numpy as np
from simplexy import simplexy
import matplotlib.pyplot as pl

from kpsf import TimeSeries, PSF

client = kplr.API()
tpf = client.k2_star(202137899).get_target_pixel_files()[0]

data = tpf.read()
times = data["TIME"]
flux_images = data["FLUX"]
ferr_images = data["FLUX_ERR"]
quality = data["QUALITY"]

flux_images[flux_images == 0.0] = np.nan

psf = PSF(0.25, 0.25, 0.0)
psf.add_component(-1.5, 1.5, 1.5, -0.5, dx=0.05, dy=0.25)

n = -1000
ts = TimeSeries(times[n:], flux_images[n:], ferr_images[n:], quality[n:],
                saddle=0.1)



# pl.figure(figsize=(10, 10))
# for i, j in enumerate(np.arange(len(ts.frames))[ts.good_times]):
#     frame = ts.frames[j]
#     img = frame.predict(psf, origin=ts.motion[j], offsets=ts.offsets)
#     shape = frame.shape

#     pl.clf()
#     pl.subplot(221)
#     pl.imshow(np.log(img.T), cmap="gray", interpolation="nearest")
#     pl.xlim(-0.5, shape[0]-0.5)
#     pl.ylim(-0.5, shape[1]-0.5)

#     pl.subplot(222)
#     pl.imshow(np.log(frame.img.T), cmap="gray", interpolation="nearest")
#     pl.xlim(-0.5, shape[0]-0.5)
#     pl.ylim(-0.5, shape[1]-0.5)

#     pl.subplot(223)
#     pl.imshow((img - frame.img).T, cmap="gray", interpolation="nearest")
#     pl.xlim(-0.5, shape[0]-0.5)
#     pl.ylim(-0.5, shape[1]-0.5)
#     pl.savefig("frames/{0:05d}.png".format(i))

assert 0


img = flux_images[-1000]
img[img == 0.0] = np.nan
frame = Frame(img)
frame.initialize()
print(frame.coords)

assert 0



obs = flux[-1000]
pixel_mask = np.isfinite(obs)
pixel_mask[pixel_mask] *= (obs[pixel_mask] > 0.0)
tmp = np.array(obs)
tmp[~pixel_mask] = np.median(obs[pixel_mask])
coords = simplexy(tmp)

mu = np.median(tmp)
sig = np.sqrt(np.median((tmp - mu) ** 2))

vmin, vmax = mu - sig, mu + 100 * sig

shape = obs.shape
x, y = np.meshgrid(range(shape[0]), range(shape[1]), indexing="ij")
x = np.array(x[pixel_mask], dtype=np.float64)
y = np.array(y[pixel_mask], dtype=np.float64)

psfpars = np.array([0.25, 0.25, 0.0]
                   + [v for j in range(2)
                      for v in [-2.0 - 100 * j, 0.0, 0.0, 2.0+j, 2.0+j, -0.5]])
norm = compute_psf(psfpars, 0.0, 0.0)
fluxes = np.array(coords["flux"] / norm, dtype=np.float64)
frame_center = np.array([0.0, 0.0])
offsets = np.array([[r["x"], r["y"]] for r in coords], dtype=np.float64)
bkg = np.median(coords["bkg"])
print(fluxes)
response = np.ones_like(x)

img = compute_scene(x, y, fluxes, frame_center, offsets, psfpars, bkg,
                    response)

# # Update the fluxes.
# A = np.vander(img - bkg, 2)
# ATA = np.dot(A.T, A)
# w = np.linalg.solve(ATA, np.dot(A.T, obs[pixel_mask] - bkg))
# print(w)
# fluxes *= w[0]
# bkg += w[1]
# print(fluxes)

# # Re-compute the prediction.
# img = compute_scene(x, y, fluxes, frame_center, offsets, psfpars, bkg,
#                          response)

result = np.nan + np.zeros_like(obs)
result[(x.astype(int), y.astype(int))] = img

pl.figure(figsize=(10, 10))
pl.subplot(221)
pl.imshow(result.T, cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)
pl.plot(coords["x"], coords["y"], "+r")
pl.xlim(-0.5, shape[0]-0.5)
pl.ylim(-0.5, shape[1]-0.5)

pl.subplot(222)
pl.imshow(obs.T, cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)
pl.plot(coords["x"], coords["y"], "+r")
pl.xlim(-0.5, shape[0]-0.5)
pl.ylim(-0.5, shape[1]-0.5)

pl.subplot(223)
dv = 0.5 * (vmax - vmin)
pl.imshow((obs - result).T, cmap="gray", interpolation="nearest",
          vmin=-dv, vmax=dv)
pl.plot(coords["x"], coords["y"], "+r")
pl.xlim(-0.5, shape[0]-0.5)
pl.ylim(-0.5, shape[1]-0.5)
pl.savefig("blah.png")
