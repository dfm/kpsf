#ifndef _KPSF_KPSF_H_
#define _KPSF_KPSF_H_

#include <iostream>
#include <ceres/ceres.h>

#include "residual.h"

using ceres::Solve;
using ceres::Problem;
using ceres::CostFunction;
using ceres::AutoDiffCostFunction;

using kpsf::PixelResidual;

#define ADD_DATA_POINT(T, S, P)                                                   \
        if (nt == (T) && ns == (S) && np == (P)) {                                \
            PixelResidual* res = new PixelResidual(xi, yi, (T), (S), (P),         \
                                                   flux, 1./ferr);                \
            CostFunction* cost = new AutoDiffCostFunction<                        \
                PixelResidual,                                                    \
                1,                   /* Size of the residual (one pixel).      */ \
                NUM_STARS,           /* The fluxes of the stars at this time.  */ \
                2 * NUM_INT_TIME,    /* The frame origin.                      */ \
                2 * NUM_STARS,       /* The offset for each star.              */ \
                6*NUM_PSF_COMP-3,    /* The parameters of the Gaussian PSF.    */ \
                1,                   /* The sky level at this time.            */ \
                1                    /* The (constant) response of this pixel. */ \
            > (res);                                                              \
                                                                                \
            problem_.AddResidualBlock(cost, NULL,                                 \
                                    &(fluxes_[t*NUM_STARS]),                    \
                                    &(origin_[2*NUM_INT_TIME*t]),               \
                                    offsets_,                                   \
                                    psfpars_,                                   \
                                    &(bkg_[t]),                                 \
                                    &(response_[nx_*xi + yi]));                 \
        } else

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


    int add_data_point (
        const unsigned nt,
        const unsigned ns,
        const unsigned np,
        const unsigned t,
        const unsigned xi,
        const unsigned yi,
        const double flux,
        const double ferr)
    {
        // We need to know some constants at compile time so here we're compiling
        // a different function for every combination we'll ever need (we hope).
        //
        // The arguments are:
        // 1. number of times to use in the exposure time integration,
        // 2. number of stars, and
        // 3. the number of PSF components to use.
        //
        ADD_DATA_POINT (1, 1, 1)
        ADD_DATA_POINT (1, 1, 2)
        ADD_DATA_POINT (1, 1, 3)
        ADD_DATA_POINT (1, 1, 4)
        ADD_DATA_POINT (1, 1, 5)
        ADD_DATA_POINT (1, 2, 1)
        ADD_DATA_POINT (1, 2, 2)
        ADD_DATA_POINT (1, 2, 3)
        ADD_DATA_POINT (1, 2, 4)
        ADD_DATA_POINT (1, 2, 5)
        ADD_DATA_POINT (1, 3, 1)
        ADD_DATA_POINT (1, 3, 2)
        ADD_DATA_POINT (1, 3, 3)
        ADD_DATA_POINT (1, 3, 4)
        ADD_DATA_POINT (1, 3, 5)
        ADD_DATA_POINT (1, 4, 1)
        ADD_DATA_POINT (1, 4, 2)
        ADD_DATA_POINT (1, 4, 3)
        ADD_DATA_POINT (1, 4, 4)
        ADD_DATA_POINT (1, 4, 5)
        ADD_DATA_POINT (1, 5, 1)
        ADD_DATA_POINT (1, 5, 2)
        ADD_DATA_POINT (1, 5, 3)
        ADD_DATA_POINT (1, 5, 4)
        ADD_DATA_POINT (1, 5, 5)
        ADD_DATA_POINT (1, 6, 1)
        ADD_DATA_POINT (1, 6, 2)
        ADD_DATA_POINT (1, 6, 3)
        ADD_DATA_POINT (1, 6, 4)
        ADD_DATA_POINT (1, 6, 5)
        ADD_DATA_POINT (1, 7, 1)
        ADD_DATA_POINT (1, 7, 2)
        ADD_DATA_POINT (1, 7, 3)
        ADD_DATA_POINT (1, 7, 4)
        ADD_DATA_POINT (1, 7, 5)
        ADD_DATA_POINT (1, 8, 1)
        ADD_DATA_POINT (1, 8, 2)
        ADD_DATA_POINT (1, 8, 3)
        ADD_DATA_POINT (1, 8, 4)
        ADD_DATA_POINT (1, 8, 5)
        ADD_DATA_POINT (1, 9, 1)
        ADD_DATA_POINT (1, 9, 2)
        ADD_DATA_POINT (1, 9, 3)
        ADD_DATA_POINT (1, 9, 4)
        ADD_DATA_POINT (1, 9, 5)
        ADD_DATA_POINT (3, 1, 1)
        ADD_DATA_POINT (3, 1, 2)
        ADD_DATA_POINT (3, 1, 3)
        ADD_DATA_POINT (3, 1, 4)
        ADD_DATA_POINT (3, 1, 5)
        ADD_DATA_POINT (3, 2, 1)
        ADD_DATA_POINT (3, 2, 2)
        ADD_DATA_POINT (3, 2, 3)
        ADD_DATA_POINT (3, 2, 4)
        ADD_DATA_POINT (3, 2, 5)
        ADD_DATA_POINT (3, 3, 1)
        ADD_DATA_POINT (3, 3, 2)
        ADD_DATA_POINT (3, 3, 3)
        ADD_DATA_POINT (3, 3, 4)
        ADD_DATA_POINT (3, 3, 5)
        ADD_DATA_POINT (3, 4, 1)
        ADD_DATA_POINT (3, 4, 2)
        ADD_DATA_POINT (3, 4, 3)
        ADD_DATA_POINT (3, 4, 4)
        ADD_DATA_POINT (3, 4, 5)
        ADD_DATA_POINT (3, 5, 1)
        ADD_DATA_POINT (3, 5, 2)
        ADD_DATA_POINT (3, 5, 3)
        ADD_DATA_POINT (3, 5, 4)
        ADD_DATA_POINT (3, 5, 5)
        ADD_DATA_POINT (3, 6, 1)
        ADD_DATA_POINT (3, 6, 2)
        ADD_DATA_POINT (3, 6, 3)
        ADD_DATA_POINT (3, 6, 4)
        ADD_DATA_POINT (3, 6, 5)
        ADD_DATA_POINT (3, 7, 1)
        ADD_DATA_POINT (3, 7, 2)
        ADD_DATA_POINT (3, 7, 3)
        ADD_DATA_POINT (3, 7, 4)
        ADD_DATA_POINT (3, 7, 5)
        ADD_DATA_POINT (3, 8, 1)
        ADD_DATA_POINT (3, 8, 2)
        ADD_DATA_POINT (3, 8, 3)
        ADD_DATA_POINT (3, 8, 4)
        ADD_DATA_POINT (3, 8, 5)
        ADD_DATA_POINT (3, 9, 1)
        ADD_DATA_POINT (3, 9, 2)
        ADD_DATA_POINT (3, 9, 3)
        ADD_DATA_POINT (3, 9, 4)
        ADD_DATA_POINT (3, 9, 5)
        ADD_DATA_POINT (5, 1, 1)
        ADD_DATA_POINT (5, 1, 2)
        ADD_DATA_POINT (5, 1, 3)
        ADD_DATA_POINT (5, 1, 4)
        ADD_DATA_POINT (5, 1, 5)
        ADD_DATA_POINT (5, 2, 1)
        ADD_DATA_POINT (5, 2, 2)
        ADD_DATA_POINT (5, 2, 3)
        ADD_DATA_POINT (5, 2, 4)
        ADD_DATA_POINT (5, 2, 5)
        ADD_DATA_POINT (5, 3, 1)
        ADD_DATA_POINT (5, 3, 2)
        ADD_DATA_POINT (5, 3, 3)
        ADD_DATA_POINT (5, 3, 4)
        ADD_DATA_POINT (5, 3, 5)
        ADD_DATA_POINT (5, 4, 1)
        ADD_DATA_POINT (5, 4, 2)
        ADD_DATA_POINT (5, 4, 3)
        ADD_DATA_POINT (5, 4, 4)
        ADD_DATA_POINT (5, 4, 5)
        ADD_DATA_POINT (5, 5, 1)
        ADD_DATA_POINT (5, 5, 2)
        ADD_DATA_POINT (5, 5, 3)
        ADD_DATA_POINT (5, 5, 4)
        ADD_DATA_POINT (5, 5, 5)
        ADD_DATA_POINT (5, 6, 1)
        ADD_DATA_POINT (5, 6, 2)
        ADD_DATA_POINT (5, 6, 3)
        ADD_DATA_POINT (5, 6, 4)
        ADD_DATA_POINT (5, 6, 5)
        ADD_DATA_POINT (5, 7, 1)
        ADD_DATA_POINT (5, 7, 2)
        ADD_DATA_POINT (5, 7, 3)
        ADD_DATA_POINT (5, 7, 4)
        ADD_DATA_POINT (5, 7, 5)
        ADD_DATA_POINT (5, 8, 1)
        ADD_DATA_POINT (5, 8, 2)
        ADD_DATA_POINT (5, 8, 3)
        ADD_DATA_POINT (5, 8, 4)
        ADD_DATA_POINT (5, 8, 5)
        ADD_DATA_POINT (5, 9, 1)
        ADD_DATA_POINT (5, 9, 2)
        ADD_DATA_POINT (5, 9, 3)
        ADD_DATA_POINT (5, 9, 4)
        ADD_DATA_POINT (5, 9, 5)
        ADD_DATA_POINT (7, 1, 1)
        ADD_DATA_POINT (7, 1, 2)
        ADD_DATA_POINT (7, 1, 3)
        ADD_DATA_POINT (7, 1, 4)
        ADD_DATA_POINT (7, 1, 5)
        ADD_DATA_POINT (7, 2, 1)
        ADD_DATA_POINT (7, 2, 2)
        ADD_DATA_POINT (7, 2, 3)
        ADD_DATA_POINT (7, 2, 4)
        ADD_DATA_POINT (7, 2, 5)
        ADD_DATA_POINT (7, 3, 1)
        ADD_DATA_POINT (7, 3, 2)
        ADD_DATA_POINT (7, 3, 3)
        ADD_DATA_POINT (7, 3, 4)
        ADD_DATA_POINT (7, 3, 5)
        ADD_DATA_POINT (7, 4, 1)
        ADD_DATA_POINT (7, 4, 2)
        ADD_DATA_POINT (7, 4, 3)
        ADD_DATA_POINT (7, 4, 4)
        ADD_DATA_POINT (7, 4, 5)
        ADD_DATA_POINT (7, 5, 1)
        ADD_DATA_POINT (7, 5, 2)
        ADD_DATA_POINT (7, 5, 3)
        ADD_DATA_POINT (7, 5, 4)
        ADD_DATA_POINT (7, 5, 5)
        ADD_DATA_POINT (7, 6, 1)
        ADD_DATA_POINT (7, 6, 2)
        ADD_DATA_POINT (7, 6, 3)
        ADD_DATA_POINT (7, 6, 4)
        ADD_DATA_POINT (7, 6, 5)
        ADD_DATA_POINT (7, 7, 1)
        ADD_DATA_POINT (7, 7, 2)
        ADD_DATA_POINT (7, 7, 3)
        ADD_DATA_POINT (7, 7, 4)
        ADD_DATA_POINT (7, 7, 5)
        ADD_DATA_POINT (7, 8, 1)
        ADD_DATA_POINT (7, 8, 2)
        ADD_DATA_POINT (7, 8, 3)
        ADD_DATA_POINT (7, 8, 4)
        ADD_DATA_POINT (7, 8, 5)
        ADD_DATA_POINT (7, 9, 1)
        ADD_DATA_POINT (7, 9, 2)
        ADD_DATA_POINT (7, 9, 3)
        ADD_DATA_POINT (7, 9, 4)
        ADD_DATA_POINT (7, 9, 5)
        {
            return -1;
        }
        return 0;
    }

    void run () {
        // Set up the solver.
        ceres::Solver::Options options;
        options.max_num_iterations = 100;
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
