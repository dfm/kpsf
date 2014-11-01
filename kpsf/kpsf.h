#ifndef _KPSF_KPSF_H_
#define _KPSF_KPSF_H_

#include <iostream>
#include <ceres/ceres.h>

#include "residual.h"
#include "constants.h"

using ceres::Solve;
using ceres::Problem;
using ceres::CostFunction;
using ceres::AutoDiffCostFunction;

using kpsf::PixelResidual;

namespace kpsf {

class Solver {
public:
    Solver (
        const unsigned nt,
        const unsigned nx,
        const unsigned ny,
        double* fluxes,     // The n_stars fluxes.
        double* origin,     // The 2-vector coords of the frame.
        double* offsets,    // The (n_stars,2) offset vectors for each star.
        double* psfpars,    // The PSF parameters.
        double* bkg,        // The background level.
        double* response    // The response in the pixel.
    ) : nt_(nt), nx_(nx), ny_(ny), fluxes_(fluxes), origin_(origin),
        offsets_(offsets), psfpars_(psfpars), bkg_(bkg), response_(response) {};

    void add_data_point (
        const unsigned t,
        const unsigned xi,
        const unsigned yi,
        const double flux,
        const double ferr)
    {
        PixelResidual* res = new PixelResidual(xi, yi, NUM_INT_TIME,
                                               NUM_STARS, NUM_PSF_COMP,
                                               flux, 1./ferr);
        CostFunction* cost = new AutoDiffCostFunction<
            PixelResidual,
            1,                   // Size of the residual (one pixel).
            NUM_STARS,           // The fluxes of the stars at this time.
            2 * NUM_INT_TIME,    // The frame origin.
            2 * NUM_STARS,       // The offset for each star.
            6*NUM_PSF_COMP-3,    // The parameters of the Gaussian PSF.
            1,                   // The sky level at this time.
            1                    // The (constant) response of this pixel.
        > (res);

        problem_.AddResidualBlock(cost, NULL,
                                  &(fluxes_[t*NUM_STARS]),
                                  &(origin_[2*NUM_INT_TIME*t]),
                                  offsets_,
                                  psfpars_,
                                  &(bkg_[t]),
                                  &(response_[nx_*xi + yi]));
    };

    void run () {
        // Set up the solver.
        ceres::Solver::Options options;
        options.max_num_iterations = 50;
        // options.linear_solver_type = ceres::DENSE_QR;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.dense_linear_algebra_library_type = ceres::LAPACK;
        options.minimizer_progress_to_stdout = true;

        // Do the solve and report the results.
        ceres::Solver::Summary summary;
        Solve(options, &problem_, &summary);
        std::cout << summary.BriefReport() << std::endl;
    };

private:
    const unsigned nt_, nx_, ny_;
    double *fluxes_, *origin_, *offsets_, *psfpars_, *bkg_, *response_;
    Problem problem_;

};

};

#endif
