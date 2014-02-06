#include <cstdio>
#include <vector>
#include <iostream>

#include <fitsio.h>
#include <ceres/ceres.h>

// Ignore string warnings. It's cfitsio's fault!
#pragma GCC diagnostic ignored "-Wwrite-strings"

using std::vector;

using Eigen::MatrixXd;
using Eigen::VectorXd;

using ceres::Solve;
using ceres::Solver;
using ceres::Problem;
using ceres::CostFunction;
using ceres::AutoDiffCostFunction;

//
// Constants for the Kepler PRF model. These will be different for other
// models.
//
#define N_PSF_BASIS 5
#define OVERSAMPLE  50
#define DIM_X       550
#define DIM_Y       550
#define CENTER_X    275
#define CENTER_Y    275

//
// The number of Gaussians in our representation.
//
#define N_GAUSSIANS 3
#define PP_GAUSSIAN 6

//
// Read in the Kepler PRF basis images from a given FITS file. Returns a
// vector of images in Eigen::MatrixXd format. Status will be non-zero on
// failure.
//
int load_prfs (const char *fn, vector<MatrixXd>* prfs)
{
    fitsfile *f;
    int status = 0;

    // Open the file.
    if (fits_open_file(&f, fn, READONLY, &status)) {
        fits_report_error(stderr, status);
        return status;
    }

    // Read in the data.
    int anynull, hdutype, i;
    long fpixel = 1;
    double nullval = -1;
    *prfs = vector<MatrixXd>(N_PSF_BASIS);
    for (i = 0; i < N_PSF_BASIS; ++i) {
        // Move to the correct HDU.
        if (fits_movabs_hdu(f, i+2, &hdutype, &status)) {
            fits_report_error(stderr, status);
            return status;
        }

        // Allocate the matrix and read the image.
        (*prfs)[i] = MatrixXd(DIM_X, DIM_Y);
        if (fits_read_img(f, TDOUBLE, fpixel, DIM_X * DIM_Y, &nullval,
                          &((*prfs)[i](0)), &anynull, &status)) {
            fits_report_error(stderr, status);
            return status;
        }
    }

    // Close the file.
    if (fits_close_file(f, &status))
        fits_report_error(stderr, status);

    return status;
};

//
// Write the MOG representation of a PRF image to a file.
//
int write_mog (const char *fn, VectorXd params, int hdu)
{
    fitsfile *f;
    int status = 0;

    // Open the file.
    if (hdu == 1) {
        remove(fn);
        if (fits_create_file(&f, fn, &status)) {
            fits_report_error(stderr, status);
            return status;
        }
    } else {
        if (fits_open_file(&f, fn, READWRITE, &status)) {
            fits_report_error(stderr, status);
            return status;
        }

        // Move to the correct HDU.
    }

    // Create a new binary table.
    char* ttype[] = {"amp", "xpos", "ypos", "xvar", "covar", "yvar"},
        * tform[] = {"1D",  "1D",   "1D",   "1D",   "1D",    "1D"},
        * tunit[] = {"\0",  "\0",   "\0",   "\0",   "\0",    "\0"};
    if (fits_create_tbl(f, BINARY_TBL, N_GAUSSIANS, PP_GAUSSIAN, ttype, tform,
                        tunit, "MOG", &status)) {
        fits_report_error(stderr, status);
        return status;
    }

    // Coerce the data into the correct form.
    VectorXd amp(N_GAUSSIANS), xpos(N_GAUSSIANS), ypos(N_GAUSSIANS),
             xvar(N_GAUSSIANS), covar(N_GAUSSIANS), yvar(N_GAUSSIANS);
    for (int i = 0; i < N_GAUSSIANS; ++i) {
        amp(i)   = params(PP_GAUSSIAN*i);
        xpos(i)  = params(PP_GAUSSIAN*i + 1);
        ypos(i)  = params(PP_GAUSSIAN*i + 2);
        xvar(i)  = params(PP_GAUSSIAN*i + 3);
        covar(i) = params(PP_GAUSSIAN*i + 4);
        yvar(i)  = params(PP_GAUSSIAN*i + 5);
    }

    // Write the columns.
    fits_write_col(f, TDOUBLE, 1, 1, 1, N_GAUSSIANS, &(amp(0)), &status);
    fits_write_col(f, TDOUBLE, 2, 1, 1, N_GAUSSIANS, &(xpos(0)), &status);
    fits_write_col(f, TDOUBLE, 3, 1, 1, N_GAUSSIANS, &(ypos(0)), &status);
    fits_write_col(f, TDOUBLE, 4, 1, 1, N_GAUSSIANS, &(xvar(0)), &status);
    fits_write_col(f, TDOUBLE, 5, 1, 1, N_GAUSSIANS, &(covar(0)), &status);
    fits_write_col(f, TDOUBLE, 6, 1, 1, N_GAUSSIANS, &(yvar(0)), &status);

    // Clean up and return.
    if (status) fits_report_error(stderr, status);
    if (fits_close_file(f, &status)) fits_report_error(stderr, status);
    return status;
}

//
// The residual object used by Ceres to find the NLLS solution to the MOG
// representation model.
//
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
            // Force the amplitudes to be positive.
            T amp = params[PP_GAUSSIAN*k];
            if (amp <= T(0)) return false;

            // Ensure that the centers are in the frame.
            T xpos = T(OVERSAMPLE) * params[PP_GAUSSIAN*k+1],
              ypos = T(OVERSAMPLE) * params[PP_GAUSSIAN*k+2];
            if (xpos < T(0) || xpos >= T(DIM_X) ||
                ypos < T(0) || ypos >= T(DIM_Y)) return false;

            // Compute the determinant and make sure that it's positive.
            T s2 = T(OVERSAMPLE * OVERSAMPLE),
              cov[] = {s2 * params[PP_GAUSSIAN*k+3],
                       s2 * params[PP_GAUSSIAN*k+4],
                       s2 * params[PP_GAUSSIAN*k+5]},
              det = cov[0] * cov[2] - cov[1] * cov[1];
            if (det <= T(0)) return false;

            // Pre-compute the normalization factor.
            T factor = amp / T(2*M_PI) / sqrt(det);

            // Loop over pixels and compute the model value.
            for (int i = 0; i < DIM_X; ++i) {
                for (int j = 0; j < DIM_Y; ++j) {
                    T dx = xpos - T(i),
                      dy = ypos - T(j),
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

    // Format the filenames.
    const char* infn = "../data/kplr07.4_2011265_prf.fits",
              * outfn = "../data/kplr07.4_2011265_prf.mog.fits";

    // Load the PRF files.
    vector<MatrixXd> prfs;
    status = load_prfs(infn, &prfs);
    if (status) return status;

    // Initialize the parameters.
    VectorXd params(PP_GAUSSIAN*N_GAUSSIANS);
    for (int k = 0; k < N_GAUSSIANS; ++k) {
        params(PP_GAUSSIAN*k)   = 300.0 / (k + 1);
        params(PP_GAUSSIAN*k+1) = CENTER_X / OVERSAMPLE;
        params(PP_GAUSSIAN*k+2) = CENTER_Y / OVERSAMPLE;
        params(PP_GAUSSIAN*k+3) = (k + 1) * 2.0;
        params(PP_GAUSSIAN*k+4) = 0.0;
        params(PP_GAUSSIAN*k+5) = (k + 1) * 2.0;
    }

    // Set up the problem.
    Problem problem;
    CostFunction *cost =
        new AutoDiffCostFunction<MOGResidual, DIM_X*DIM_Y, PP_GAUSSIAN*N_GAUSSIANS> (
            new MOGResidual (prfs[1]));
    problem.AddResidualBlock(cost, NULL, &(params(0)));

    // Set up the solver.
    Solver::Options options;
    options.max_num_iterations = 20 * N_GAUSSIANS;
    options.function_tolerance = 1e-5;
    // options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.dense_linear_algebra_library_type = ceres::LAPACK;
    options.minimizer_progress_to_stdout = true;

    Solver::Summary summary;
    Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;

    std::cout << "Writing output file: " << outfn << std::endl;
    status = write_mog(outfn, params);
    if (status) return status;

    return 0;
}
