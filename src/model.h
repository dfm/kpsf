#ifndef _MODEL_H_
#define _MODEL_H_

#include <cmath>
#include <cfloat>
#include <vector>
#include <Eigen/Dense>
#include <ceres/numeric_diff_functor.h>

#include "prf.h"

using std::vector;
using Eigen::MatrixXd;
using ceres::NumericDiffFunctor;

namespace kpsf {

double lanczos_kernel (unsigned int a, double x) {
    double fx, px;
    if (x <= DBL_EPSILON) return 1.0;
    fx = fabs(x);
    if (fx >= a) return 0.0;
    px = M_PI * x;
    return a * sin(px / a) * sin(px) / (px * px);
}

class LanczosInterpolator {

public:

    LanczosInterpolator (int order, vector<MatrixXd> * psf_basis)
        : psf_basis_(psf_basis), order_(order) {};

    bool operator() (const double* coeffs, const double* xi, const double* yi,
                     double* result) const {
        double lx, ly;
        int indx, indy, ixi = round(*xi), iyi = round(*yi);

        *result = 0.0;
        for (indx = -order_; indx <= order_; ++indx) {
            lx = lanczos_kernel (order_, indx - *xi + ixi);
            for (indy = -order_; indy <= order_; ++indy) {
                ly = lanczos_kernel (order_, indy - *yi + iyi);
                *result += lx * ly * evaluate (coeffs, ixi+indx, iyi+indy);
            }
        }

        return true;
    };

private:

    int order_;
    vector<MatrixXd> * psf_basis_;

    double evaluate (const double* coeffs, int i, int j) const {
        int ind;
        double value = 0.0, norm = 1.0;
        for (ind = 0; ind < N_PSF_BASIS-1; ++ind) {
            if (coeffs[ind] < 0.0 || coeffs[ind] > 1.0) return false;
            value += coeffs[ind] * (*psf_basis_)[ind](i, j);
            norm -= coeffs[ind];
        }
        if (norm < 0.0) return false;
        value += norm * (*psf_basis_)[N_PSF_BASIS-1](i, j);
        return value;
    };

};

class KeplerPSFResidual {

public:

    typedef NumericDiffFunctor<LanczosInterpolator, ceres::CENTRAL, 1, N_PSF_BASIS-1, 1, 1>
            LanczosInterpolatorFunctor;

    KeplerPSFResidual (int i, int j, double mean, double std,
                       int order, vector<MatrixXd> * psf_basis)
        : i_(i), j_(j), mean_(mean), istd_(1.0 / fabs(std))
    {
        interpolator_.reset(new LanczosInterpolatorFunctor(new LanczosInterpolator(order, psf_basis)));
    };

    template <typename T>
    bool operator() (const T* coords, const T* coeffs,
                     const T* background, const T* response,
                     T* residuals) const {
        // Compute the pixel position in PRF coordinates.
        T xi = T(OVERSAMPLE) * (T(i_) - coords[1]) + T(CENTER_X),
          yi = T(OVERSAMPLE) * (T(j_) - coords[2]) + T(CENTER_Y),
          value;

        // Interpolate the PSF to the position of the pixel.
        if (! ((*interpolator_)(coeffs, &xi, &yi, &value))) return false;

        // Incorporate the response and background.
        value = (*response) * (coords[0] * value + *background);

        // Compute the residuals.
        residuals[0] = (value - T(mean_)) * T(istd_);
        return true;
    };

private:

    int i_, j_;
    double mean_, istd_;
    ceres::internal::scoped_ptr<LanczosInterpolatorFunctor> interpolator_;

};

};

#endif
