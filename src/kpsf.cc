#include <iostream>
#include "fits.h"
#include "solver.h"
#include "ceres/ceres.h"

#define WIDTH 10
#define HEIGHT 16

using Eigen::MatrixXi;
using Eigen::MatrixXd;


int main ()
{
    double coords[3] = {3.1234, 4.49815, 1.0},
           psfpars[3] = {1.0, 0.1, 1.0};

    load_target_pixel_file ("../data.fits");

    // PSF<double> psf(psfpars);
    // MatrixXd data(WIDTH, HEIGHT);

    // int ix, iy;
    // for (ix = 0; ix < WIDTH; ++ix)
    //     for (iy = 0; iy < HEIGHT; ++iy)
    //         data(ix, iy) = coords[2] * psf.evaluate(coords[0] - ix,
    //                                                 coords[1] - iy);

    // coords[0] += 1;
    // coords[1] -= 1;
    // coords[2] += 10;

    // psfpars[0] -= 0.5;
    // psfpars[1] += 0.5;
    // psfpars[2] += 0.5;

    // ceres::Problem problem;
    // ceres::CostFunction *cost_function =
    //     new ceres::AutoDiffCostFunction<ImageResidualBlock, ceres::DYNAMIC, 3, 3>(
    //             new ImageResidualBlock(data), HEIGHT*WIDTH);
    // problem.AddResidualBlock(cost_function, NULL, coords, psfpars);

    // ceres::Solver::Options options;
    // options.max_num_iterations = 25;
    // options.linear_solver_type = ceres::DENSE_QR;
    // options.minimizer_progress_to_stdout = true;

    // ceres::Solver::Summary summary;
    // ceres::Solve(options, &problem, &summary);

    // std::cout << summary.BriefReport() << "\n";
    // std::cout << "Final   x: " << coords[0] << " y: " << coords[1] << " f: " << coords[2] << "\n";
    // std::cout << "PSF: " << psfpars[0] << " " << psfpars[1] << " " << psfpars[2] << "\n";

    return 0;
}
