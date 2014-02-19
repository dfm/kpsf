#ifndef _GAUSSIAN_H_
#define _GAUSSIAN_H_

namespace kpsf {

class GaussianPSF {

public:

    GaussianPSF () {};

    template <typename T>
    T evaluate (const T* params, T xi, T yi) const {
    };

};

};

#endif
