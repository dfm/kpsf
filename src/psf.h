#ifndef _PSF_H_
#define _PSF_H_

#include <cmath>
#include <Eigen/Dense>

namespace KPSF {

    class PSF {

    public:

        PSF () {
            Eigen::VectorXd params(3);
            params(0) = 1.0;
            params(1) = 0.0;
            params(2) = 1.0;
            set_params (params);
        };

        void set_params (Eigen::VectorXd params) {
            params_ = params;
            det_ = params_[0] * params[2] - params[1] * params[1];
        }

        template <typename T>
        T evaluate (const T dx, const T dy) {
            return T(0.5) * exp((T(-0.5)
                                 * (T(params_[2]) * dx * dx
                                 + T(params_[0]) * dy * dy)
                                 + T(params_[1]) * dx * dy) / T(det_))
                   / T(det_ * M_PI);
        }

    private:

        double det_;
        Eigen::VectorXd params_;

    };

}

#endif
// _PSF_H_
