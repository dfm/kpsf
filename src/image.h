#ifndef _IMAGE_H_
#define _IMAGE_H_

#include <Eigen/Dense>

namespace KPSF {

    template <class PSFType>
    class Image {

    public:

        Image (int w, int h, PSFType *psf, Eigen::MatrixXi *mask,
               Eigen::MatrixXd *bias, Eigen::MatrixXd *ff) {
            width_ = w;
            height_ = h;
            psf_ = psf;
            mask_ = mask;
            bias_ = bias;
            flat_field_ = ff;
        };
        ~Image () {};

        template <typename T>
        T evaluate (const T xpos, const T ypos, const T flux,
                    const double xi, const double yi) const {
            return T( (*bias_)(xi, yi) )
                   + T( (*flat_field_)(xi, yi) ) * flux
                   * psf_->evaluate (xpos - T(xi), ypos - T(yi));
        };

        Eigen::MatrixXd generate (const double xpos,
                                  const double ypos,
                                  const double flux) {
            int x, y;
            Eigen::MatrixXd img = Eigen::MatrixXd::Zero(width_, height_);
            for (x = 0; x < width_; ++x)
                for (y = 0; y < height_; ++y)
                    img(x, y) = evaluate (xpos, ypos, flux, x, y);
            return img;
        };

    private:

        PSFType *psf_;
        int width_, height_;
        Eigen::MatrixXi *mask_;
        Eigen::MatrixXd *flat_field_, *bias_;

    };

}

#endif
// _IMAGE_H_
