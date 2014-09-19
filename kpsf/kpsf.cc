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

int kpsf::photometry_all (const int nt, const int npix,
                          const double maxx, const double maxy,
                          const double* xpix,
                          const double* ypix, const double* flux,
                          const double* ferr, double* model,
                          double* coords, double* coeffs,
                          double* ff, double* bg,
                          const double* max_fracs,
                          const double motion_reg,
                          const double ff_reg)
{
    int i, j, k, ind;
    Problem problem;
    CostFunction* cost;
    for (i = 0; i < nt; ++i) {
        for (j = 0; j < npix; ++j) {
            ind = i*npix+j;
            PixelResidual* res =
                new PixelResidual(maxx, maxy,
                                  xpix[j], ypix[j], flux[ind], 1./ferr[ind],
                                  max_fracs);
            cost = new AutoDiffCostFunction<PixelResidual, 1, 1,
                                            2*NUM_INT_TIME, 6*NUM_PSF_COMP-3,
                                            1, 1> (res);
            // ceres::SoftLOneLoss* loss =
            //     new ceres::SoftLOneLoss(loss_scale);
            problem.AddResidualBlock(cost, NULL, &(model[i]),
                                     &(coords[i*2*NUM_INT_TIME]),
                                     coeffs, &(bg[i]), &(ff[j]));

        }
    }

    for (k = 0; k < nt; ++k) {
        cost = new AutoDiffCostFunction<SmoothPrior, NUM_INT_TIME - 1,
                                        2*NUM_INT_TIME> (
            new SmoothPrior(motion_reg));
        problem.AddResidualBlock(cost, NULL, &(coords[2*k*NUM_INT_TIME]));
    }

    for (j = 0; j < npix; ++j) {
        cost = new AutoDiffCostFunction<GaussianPrior, 1, 1> (
            new GaussianPrior(1.0, ff_reg));
        problem.AddResidualBlock(cost, NULL, &(ff[j]));
    }

    // Set up the solver.
    Solver::Options options;
    options.max_num_iterations = 500;
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
