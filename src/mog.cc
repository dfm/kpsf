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

#define N_GAUSSIANS 2

class MOGResidual {

public:

    MOGResidual (MatrixXd img) : img_(img) {
        nx_ = img_.rows();
        ny_ = img_.cols();
    };

    template <typename T>
    bool operator() (const T* amps, const T* coords, const T* covs,
                     T* residuals) const {
        // Initialize the residuals.
        for (int i = 0; i < nx_; ++i)
            for (int j = 0; j < ny_; ++j)
                residuals[i*ny_+j] = T(0);

        // Loop over the Gaussians and compute the model.
        for (int k = 0; k < N_GAUSSIANS; ++k) {
            // Ensure that the centers are in the frame.
            const T* pos = &(coords[2*k]);
            if (pos[0] < T(0) || pos[0] >= T(nx_) ||
                    pos[1] < T(0) || pos[1] >= T(ny_)) return false;

            // Compute the determinant and make sure that it's positive.
            const T* cov = &(covs[3*k]);
            T det = cov[0] * cov[2] - cov[1] * cov[1];
            if (cov[0] <= T(0) || cov[2] <= T(0) || det <= T(0)) return false;

            // Loop over pixels and compute the model value.
            for (int i = 0; i < nx_; ++i) {
                for (int j = 0; j < ny_; ++j) {
                    T dx = pos[0] - T(i),
                      dy = pos[1] - T(j),
                      x = cov[2] * dx - cov[1] * dy,
                      y = cov[0] * dy - cov[1] * dx,
                      v = (dx * x + dy * y) / det;
                    residuals[i*ny_+j] -= amps[k] * exp(T(-0.5) * v) / T(2*M_PI) / sqrt(det);
                }
            }
        }
        return true;
    };

private:

    int nx_, ny_;
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
    vector<double> amps(N_GAUSSIANS);
    vector<VectorXd> coords(N_GAUSSIANS),
                     covs(N_GAUSSIANS);

    return 0;
}
