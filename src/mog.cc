#include <vector>
#include <iostream>

#include <ceres/ceres.h>

#include "prf.h"

using std::vector;

using Eigen::MatrixXd;
using Eigen::VectorXd;

using ceres::Solve;
using ceres::Solver;
using ceres::Problem;
using ceres::CostFunction;
using ceres::AutoDiffCostFunction;

using kpsf::load_prfs;

#define N_GAUSSIANS 4

class MOGResidual {

public:

    MOGResidual (MatrixXd img) : img_(img) {};

    template <typename T>
    bool operator() (const T* params, T* residuals) const {
        // Initialize the residuals.
        for (int i = 0; i < DIM_X; ++i)
            for (int j = 0; j < DIM_Y; ++j)
                residuals[i*DIM_Y+j] = T(img_(i, j));

        // Loop over the Gaussians and compute the model.
        for (int k = 0; k < N_GAUSSIANS; ++k) {
            // Ensure that the centers are in the frame.
            const T* pos = &(params[6*k+1]);
            if (pos[0] < T(0) || pos[0] >= T(DIM_X) ||
                    pos[1] < T(0) || pos[1] >= T(DIM_Y)) return false;

            // Compute the determinant and make sure that it's positive.
            const T* cov = &(params[6*k+3]);
            T det = cov[0] * cov[2] - cov[1] * cov[1];
            if (cov[0] <= T(0) || cov[2] <= T(0) || det <= T(0)) return false;

            // Pre-compute the normalization factor.
            T factor = params[6*k] / T(2*M_PI) / sqrt(det);

            // Loop over pixels and compute the model value.
            for (int i = 0; i < DIM_X; ++i) {
                for (int j = 0; j < DIM_Y; ++j) {
                    T dx = pos[0] - T(i),
                      dy = pos[1] - T(j),
                      x = cov[2] * dx - cov[1] * dy,
                      y = cov[0] * dy - cov[1] * dx,
                      v = (dx * x + dy * y) / det;
                    residuals[i*DIM_Y+j] -= factor * exp(T(-0.5) * v);
                }
            }
        }
        return true;
    };

private:

    MatrixXd img_;

};

int main ()
{
    int status;

    // Load the PRF files.
    vector<MatrixXd> prfs;
    status = load_prfs("../data/kplr07.4_2011265_prf.fits", &prfs);
    if (status) return status;

    // Initialize the parameters.
    VectorXd params(6*N_GAUSSIANS);
    for (int k = 0; k < N_GAUSSIANS; ++k) {
        params(6*k)   = 3000.0 / N_GAUSSIANS;
        params(6*k+1) = CENTER_X;
        params(6*k+2) = CENTER_Y;
        params(6*k+3) = (k + 1) * CENTER_X * CENTER_X;
        params(6*k+4) = 100.0;
        params(6*k+5) = (k + 1) * CENTER_Y * CENTER_Y;
    }

    // Set up the problem.
    Problem problem;
    CostFunction *cost =
        new AutoDiffCostFunction<MOGResidual, DIM_X*DIM_Y, 6*N_GAUSSIANS> (
            new MOGResidual (prfs[0]));
    problem.AddResidualBlock(cost, NULL, &(params(0)));

    // Set up the solver.
    Solver::Options options;
    options.max_num_iterations = 50 * N_GAUSSIANS;
    // options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.dense_linear_algebra_library_type = ceres::LAPACK;
    options.minimizer_progress_to_stdout = true;

    Solver::Summary summary;
    Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;

    std::cout << params << std::endl;

    return 0;
}
