#ifndef _KPSF_KPSF_H_
#define _KPSF_KPSF_H_

namespace kpsf {

int photometry_all (

        // INPUTS
        const int nw,         // The number of latent dimensions in the pointing model.
        const int nt,         // The number of time points.
        const int npix,       // The number of pixels.
        const double* w,      // The [nt, nw] list of latent parameters.
        const double* xpix,   // The [npix] list of pixel center coordinates.
        const double* ypix,   // ...
        const double* flux,   // The [nt, npix] list of measured pixel values.
        const double* ferr,   // The uncertainties on these values.

        // OUTPUTS
        double* model,        // The [nt, NUM_STARS] list of fluxes.
        double* x0,           // The [NUM_STARS, 2] set of mean star locations.
        double* a,            // The [nw, 2] "a" matrix.
        double* psfpars,      // The PSF parameters.
        double* ff,           // The [npix] list of pixel responses.
        double* bg,           // The [nt] list of background levels.

        // TUNING PARAMETERS
        const double* max_fracs, const double motion_reg, const double ff_reg
);

};

#endif
