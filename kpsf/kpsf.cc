#include <iostream>
#include <ceres/ceres.h>

#include "kpsf.h"
#include "psf.h"
#include "residual.h"

#include <iostream>

using ceres::Solve;
using ceres::Solver;
using ceres::Problem;
using ceres::CostFunction;
using ceres::AutoDiffCostFunction;

using kpsf::PixelResidual;
using kpsf::GaussianPrior;
using kpsf::SmoothPrior;
using kpsf::PSFPrior;

int kpsf::photometry_all (
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
        const double* max_fracs, const double motion_reg, const double ff_reg)
{
    int i, j, ind;
    Problem problem;
    CostFunction* cost;
    for (i = 0; i < nt; ++i) {
        for (j = 0; j < npix; ++j) {
            ind = i*npix+j;
            PixelResidual* res =
                new PixelResidual(nw, &(w[i*nw]), xpix[j], ypix[j],
                                  flux[ind], 1./ferr[ind],
                                  max_fracs);

            cost = new AutoDiffCostFunction<
                PixelResidual,
                1,                   // Size of the residual (one pixel).
                NUM_STARS,           // The fluxes of the stars at this time.
                2 * NUM_STARS,       // The mean position of each star.
                6 * NUM_STARS,       // The "a" matrix for each star.
                6*NUM_PSF_COMP-3,    // The parameters of the Gaussian PSF.
                1,                   // The sky level at this time.
                1                    // The (constant) response of this pixel.
            > (res);

            // ceres::SoftLOneLoss* loss =
            //     new ceres::SoftLOneLoss(loss_scale);

            problem.AddResidualBlock(cost, NULL,
                                     &(model[i*NUM_STARS]),
                                     x0, a, psfpars, &(bg[i]), &(ff[j]));

        }
    }

    // for (k = 0; k < nt; ++k) {
    //     cost = new AutoDiffCostFunction<SmoothPrior, NUM_INT_TIME - 1,
    //                                     2*NUM_INT_TIME> (
    //         new SmoothPrior(motion_reg));
    //     problem.AddResidualBlock(cost, NULL, &(coords[2*k*NUM_INT_TIME]));
    // }

    // for (j = 0; j < npix; ++j) {
    //     cost = new AutoDiffCostFunction<GaussianPrior, 1, 1> (
    //         new GaussianPrior(1.0, ff_reg));
    //     problem.AddResidualBlock(cost, NULL, &(ff[j]));
    // }

    // cost = new AutoDiffCostFunction<PSFPrior, 2*(NUM_PSF_COMP-1),
    //                                 6*NUM_PSF_COMP-3> (
    //     new PSFPrior(1e-4, 2.0));
    // problem.AddResidualBlock(cost, NULL, psfpars);

    // Set up the solver.
    Solver::Options options;
    options.max_num_iterations = 20;
    // options.linear_solver_type = ceres::DENSE_QR;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.dense_linear_algebra_library_type = ceres::LAPACK;
    options.minimizer_progress_to_stdout = true;

    // Do the solve and report the results.
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;

    return 0;
}
