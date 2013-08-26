#include <iostream>
#include "ceres/ceres.h"
#include "ceres/dynamic_autodiff_cost_function.h"

#include "solver.h"
#include "residual.h"

using ceres::Solve;
using ceres::Solver;
using ceres::Problem;
using ceres::CostFunction;
using ceres::AutoDiffCostFunction;

int kpsf_solve (int ntime, int npixels, double *data, double *dim,
                double *coords, double *flat_field, double *bias,
                double *psfpars, int verbose)
{
    int i, j;
    Problem problem;

    for (i = 0; i < ntime; ++i) {
        for (j = 0; j < npixels; ++j) {
            CostFunction *cost =
                new AutoDiffCostFunction<PixelResidual, 1, 3, NPSFPARS, 1> (
                    new PixelResidual(&(data[(i*npixels+j)*4])));
            problem.AddResidualBlock(cost, NULL, &(coords[3*i]),
                                     psfpars, &(flat_field[j]));
        }
    }

    Solver::Options options;
    options.max_num_iterations = 200;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.dense_linear_algebra_library_type = ceres::LAPACK;
    if (verbose > 0)
        options.minimizer_progress_to_stdout = true;

    Solver::Summary summary;
    Solve(options, &problem, &summary);

    if (verbose > 0)
        std::cout << summary.BriefReport() << std::endl;

    return 0;
}
