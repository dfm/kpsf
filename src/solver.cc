#include <iostream>
#include "ceres/ceres.h"
#include "ceres/dynamic_autodiff_cost_function.h"

#include "solver.h"
#include "residual.h"

using ceres::Solve;
using ceres::Solver;
using ceres::Problem;
using ceres::CostFunction;
using ceres::DynamicAutoDiffCostFunction;

int kpsf_solve (int ntime, int npixels, double *data, double *dim,
                double *coords, double *psfpars, int verbose)
{
    Problem problem;

    DynamicAutoDiffCostFunction<KPSFResidual> *cost =
        new DynamicAutoDiffCostFunction<KPSFResidual> (
            new KPSFResidual(ntime, npixels, data, dim));

    cost->AddParameterBlock(3*ntime);
    cost->AddParameterBlock(3);
    cost->SetNumResiduals(ntime*npixels);
    problem.AddResidualBlock (cost, NULL, coords, psfpars);

    Solver::Options options;
    options.max_num_iterations = 25;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    if (verbose > 0)
        options.minimizer_progress_to_stdout = true;

    Solver::Summary summary;
    Solve(options, &problem, &summary);

    if (verbose > 0)
        std::cout << summary.BriefReport() << std::endl;

    return 0;
}
