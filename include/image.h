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

        Eigen::MatrixXd evaluate (double xpos, double ypos, double flux) {
            int x, y;
            Eigen::MatrixXd img = Eigen::MatrixXd::Zero(width_, height_);
            for (x = 0; x < width_; ++x)
                for (y = 0; y < height_; ++y)
                    if ((*mask_)(x, y))
                        img(x, y) = (*bias_)(x, y) + (*flat_field_)(x, y)
                                    * flux
                                    * psf_->evaluate(xpos - x, ypos - y);
            return img;
        }

    private:

        PSFType *psf_;
        int width_, height_;
        Eigen::MatrixXi *mask_;
        Eigen::MatrixXd *flat_field_, *bias_;

    };

}

#endif
// _IMAGE_H_
