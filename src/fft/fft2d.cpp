/*******************************************************************************
* Copyright 2020-2022 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
*
*  Content:
*       This example demonstrates use of oneAPI Math Kernel Library (oneMKL)
*       API oneapi::mkl::dft to perform  2-D Double Precision Complex to Complex
*       Fast-Fourier Transform on a SYCL device (Host, CPU, GPU).
*
*       The supported floating point data types for data are:
*           double
*           std::complex<double>
*
*******************************************************************************/
#define SYCL_DEVICES_cpu

#include <vector>
#include <iostream>
#include <CL/sycl.hpp>
#include "oneapi/mkl/dfti.hpp"

#include <stdexcept>
#include <cfloat>
#include <cstddef>
#include <limits>
#include <type_traits>
#include "mkl.h"

// local includes
#define NO_MATRIX_HELPERS
#include "common_for_examples.hpp"

typedef oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::COMPLEX> descriptor_t;

constexpr int SUCCESS = 0;
constexpr int FAILURE = 1;
constexpr double TWOPI = 6.2831853071795864769;

// Compute (K*L)%M accurately
static double moda(int K, int L, int M)
{
    return (double)(((long long)K * L) % M);
}

static void init(double *data, int N1, int N2, int H1, int H2)
{
    // Generalized strides for row-major addressing of data
    int S1 = 1, S2 = N1;

    for (int n2 = 0; n2 < N2; ++n2) {
        for (int n1 = 0; n1 < N1; ++n1) {
            double phase = TWOPI * (moda(n1, H1, N1) / N1
                                        + moda(n2, H2, N2) / N2);
            int index = 2*(n2*S2 + n1*S1);
            data[index+0] = cos(phase) / (N2*N1);
            data[index+1] = sin(phase) / (N2*N1);
        }
    }
}

static int verify_fwd(double *data,
                      int N1, int N2, int H1, int H2)
{
    // Note: this simple error bound doesn't take into account error of
    //       input data
    double errthr = 5.0 * log((double) N2*N1) / log(2.0) * DBL_EPSILON;
    std::cout << "\t\tVerify the result, errthr = " << errthr << std::endl;

    // Generalized strides for row-major addressing of data
    int S1 = 1, S2 = N1;

    double maxerr = 0.0;
    for (int n2 = 0; n2 < N2; n2++) {
        for (int n1 = 0; n1 < N1; n1++) {
            double re_exp = (
                    ((n1-H1) % N1 == 0) &&
                    ((n2-H2) % N2 == 0)
                ) ? 1.0 : 0.0;
            double im_exp = 0.0;

            int index = 2*(n2*S2 + n1*S1);
            double re_got = data[index+0];  // real component
            double im_got = data[index+1];  // imaginary component
            double err  = fabs(re_got - re_exp) + fabs(im_got - im_exp);
            if (err > maxerr) maxerr = err;
            if (!(err < errthr)) {
                std::cout << "\t\tdata[" << n2 << ", " << n1 << "]: "
                          << "expected (" << re_exp << "," << im_exp << "), "
                          << "got (" << re_got << "," << im_got << "), "
                          << "err " << err << std::endl;
                std::cout << "\t\tVerification FAILED" << std::endl;
                return FAILURE;
            }
        }
    }
    std::cout << "\t\tVerified, maximum error was " << maxerr << std::endl;
    return SUCCESS;
}

static int verify_bwd(double* data, int N1, int N2, int H1, int H2) {
    // Note: this simple error bound doesn't take into account error of
    //       input data
    double errthr = 5.0 * log((double) N2*N1) / log(2.0) * DBL_EPSILON;
    std::cout << "\t\tVerify the result, errthr = " << errthr << std::endl;

    // Generalized strides for row-major addressing of data
    int S1 = 1, S2 = N1;

    double maxerr = 0.0;
    for (int n2 = 0; n2 < N2; n2++) {
        for (int n1 = 0; n1 < N1; n1++) {
            double phase = TWOPI * (moda(n1, H1, N1) / N1
                                        + moda(n2, H2, N2) / N2);
            double re_exp = cos(phase) / (N2*N1);
            double im_exp = sin(phase) / (N2*N1);

            int index = 2*(n2*S2 + n1*S1);
            double re_got = data[index+0];  // real component
            double im_got = data[index+1];  // imaginary component
            double err  = fabs(re_got - re_exp) + fabs(im_got - im_exp);
            if (err > maxerr) maxerr = err;
            if (!(err < errthr)) {
                std::cout << "\t\tdata[" << n2 << ", " << n1 << "]: "
                          << "expected (" << re_exp << "," << im_exp << "), "
                          << "got (" << re_got << "," << im_got << "), "
                          << "err " << err << std::endl;
                std::cout << "\t\tVerification FAILED" << std::endl;
                return FAILURE;
            }
        }
    }
    std::cout << "\t\tVerified, maximum error was " << maxerr << std::endl;
    return SUCCESS;
}

int run_dft_example(cl::sycl::device &dev) {
    //
    // Initialize data for DFT
    //
    int N1 = 6, N2 = 13;
    int H1 = -1, H2 = -2;
    int buffer_result = FAILURE;
    int usm_result = FAILURE;
    int result = FAILURE;

    double* in = (double*) mkl_malloc(N2*N1*2*sizeof(double), 64);
    double* h_in = (double*) mkl_malloc(N2*N1*2*sizeof(double), 64);
    double* h_out = (double*) mkl_malloc(N2*N1*2*sizeof(double), 64);
    init(in, N1, N2, H1, H2);
    init(h_in, N1, N2, H1, H2);
    
    //
    // Execute DFT
    //
    try {
        // Catch asynchronous exceptions
        auto exception_handler = [] (cl::sycl::exception_list exceptions) {
            for (std::exception_ptr const& e : exceptions) {
                try {
                    std::rethrow_exception(e);
                } catch(cl::sycl::exception const& e) {
                    std::cout << "Caught asynchronous SYCL exception:" << std::endl
                              << e.what() << std::endl;
                }
            }
        };

        // create execution queue with asynchronous error handling
        cl::sycl::queue queue(dev, exception_handler);

        // Setting up SYCL buffer and initialization
        cl::sycl::buffer<double, 1> inBuffer(in, cl::sycl::range<1>(N2*N1*2));
        inBuffer.set_write_back(false);

        // Setting up USM and initialization
        double *in_usm = (double*) malloc_device(N2*N1*2*sizeof(double), queue.get_device(), queue.get_context());
	queue.memcpy(in_usm, h_in, N2*N1*2*sizeof(double));
	queue.wait();
        //init(in_usm, N1, N2, H1, H2);
        double *out_usm = (double*) malloc_device(N2*N1*2*sizeof(double), queue.get_device(), queue.get_context());

        descriptor_t desc({N2, N1});
        desc.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE, (1.0/(N1*N2)));

        desc.commit(queue);

        // Using SYCL buffer
        std::cout<<"\tUsing SYCL buffers"<<std::endl;
        oneapi::mkl::dft::compute_forward(desc, inBuffer);
        {
          auto inAcc = inBuffer.get_access<cl::sycl::access::mode::read>();
          buffer_result = verify_fwd(inAcc.get_pointer(), N1, N2, H1, H2);
        }

        if (buffer_result == SUCCESS) {
            oneapi::mkl::dft::compute_backward(desc, inBuffer);
            auto inAcc = inBuffer.get_access<cl::sycl::access::mode::read>();
            buffer_result = verify_bwd(inAcc.get_pointer(), N1, N2, H1, H2);
        }

        // Using USM
        desc.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
        desc.commit(queue);
        std::cout<<"\tUsing USM"<<std::endl;
        cl::sycl::event fwd, bwd;
        fwd = oneapi::mkl::dft::compute_forward(desc, in_usm, out_usm);
        fwd.wait();
	
	queue.memcpy(h_out, out_usm, N2*N1*2*sizeof(double));
	queue.wait();
        usm_result = verify_fwd(h_out, N1, N2, H1, H2);

        if (usm_result == SUCCESS) {
	    bwd = oneapi::mkl::dft::compute_backward(desc, out_usm, in_usm);
            bwd.wait();
	    queue.memcpy(h_out, in_usm, N2*N1*2*sizeof(double));
	    queue.wait();
            usm_result = verify_bwd(h_out, N1, N2, H1, H2);
        }

        if ((buffer_result == SUCCESS) && (usm_result == SUCCESS))
            result = SUCCESS;
	// try with independent 1D BATCH
        std::cout<<"\tUsing 1D batches"<<std::endl;
	// create descriptor for first dimension
        descriptor_t desc1(N1);
        desc1.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
        desc1.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, N2);
	// *2 for complex
        desc1.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, N1);
        desc1.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, N1);
        desc1.commit(queue);
	// start first dimension FFT
        fwd = oneapi::mkl::dft::compute_forward(desc1, in_usm, out_usm);
        fwd.wait();
	// create descriptor for second dimension
        std::cout<<"\tUsing batches second dimension"<<std::endl;
        descriptor_t desc2(N2);
        desc2.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
	// set stride
	int64_t st[2] = {0, N1};
        std::cout<<"\tHERE "<<__LINE__<<std::endl;
	desc2.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, st);
        std::cout<<"\tHERE "<<__LINE__<<std::endl;
	desc2.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, st);
	// set batch
        std::cout<<"\tHERE "<<__LINE__<<std::endl;
        desc2.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, N1);
        std::cout<<"\tHERE "<<__LINE__<<std::endl;
        desc2.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, 1);
        std::cout<<"\tHERE "<<__LINE__<<std::endl;
        desc2.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, 1);
        std::cout<<"\tHERE "<<__LINE__<<std::endl;
        desc2.commit(queue);
        std::cout<<"\tHERE "<<__LINE__<<std::endl;
	// start second dimension FFT
        std::cout<<"\tHERE "<<__LINE__<<std::endl;
        fwd = oneapi::mkl::dft::compute_forward(desc2, out_usm, in_usm);
        std::cout<<"\tHERE "<<__LINE__<<std::endl;
        fwd.wait();
        std::cout<<"\tReturned from compute second dimension"<<std::endl;
	// copy back and verify
	queue.memcpy(h_out, in_usm, N2*N1*2*sizeof(double));
        std::cout<<"\tHERE "<<__LINE__<<std::endl;
	queue.wait();
        std::cout<<"\tHERE "<<__LINE__<<std::endl;
	usm_result = verify_fwd(h_out, N1, N2, H1, H2);
        std::cout<<"\tHERE "<<__LINE__<<std::endl;
	result = usm_result;


	
	
        free(in_usm, queue.get_context());
        free(out_usm, queue.get_context());
    }
    catch(cl::sycl::exception const& e) {
        std::cout << "\t\tSYCL exception during FFT" << std::endl;
        std::cout << "\t\t" << e.what() << std::endl;
        std::cout << "\t\tError code: " << get_error_code(e) << std::endl;
    }
    catch(std::runtime_error const& e) {
        std::cout << "\t\truntime exception during FFT" << std::endl;
        std::cout << "\t\t" << e.what() << std::endl;
    }
    mkl_free(in);

    return result;
}

//
// Description of example setup, apis used and supported floating point type precisions
//
void print_example_banner() {
    std::cout << "" << std::endl;
    std::cout << "########################################################################" << std::endl;
    std::cout << "# 2D FFT Complex-Complex Double-Precision Example: " << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Using apis:" << std::endl;
    std::cout << "#   dft" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Supported floating point type precisions:" << std::endl;
    std::cout << "#   double" << std::endl;
    std::cout << "#   std::complex<double>" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "########################################################################" << std::endl;
    std::cout << std::endl;
}

//
// Main entry point for example.
//
// Dispatches to appropriate device types as set at build time with flag:
// -DSYCL_DEVICES_host -- only runs host implementation
// -DSYCL_DEVICES_cpu -- only runs SYCL CPU implementation
// -DSYCL_DEVICES_gpu -- only runs SYCL GPU implementation
// -DSYCL_DEVICES_all (default) -- runs on all: host, cpu and gpu devices
//
//  For each device selected and each supported data type, Basic_Sp_C2C_2D_FFTExample
//  runs is with all supported data types
//
int main() {
    print_example_banner();

    std::list<my_sycl_device_types> list_of_devices;
    set_list_of_devices(list_of_devices);

    int returnCode = 0;
    for (auto it = list_of_devices.begin(); it != list_of_devices.end(); ++it) {
        cl::sycl::device my_dev;
        bool my_dev_is_found = false;
        get_sycl_device(my_dev, my_dev_is_found, *it);

        if (my_dev_is_found) {
            if (!isDoubleSupported(my_dev)) {
                std::cout << "Double precision not supported on this device " << std::endl;
                std::cout << std::endl;
                continue;
            }
            std::cout << "Running tests on " << sycl_device_names[*it] << ".\n";

            std::cout << "\tRunning with double precision complex-to-complex 2-D FFT:" << std::endl;
            int status = run_dft_example(my_dev);
            if (status != SUCCESS) {
                std::cout << "\tTest Failed" << std::endl << std::endl;
                returnCode = status;
            } else {
                std::cout << "\tTest Passed" << std::endl << std::endl;
            }
        } else {
#ifdef FAIL_ON_MISSING_DEVICES
            std::cout << "No " << sycl_device_names[*it] << " devices found; Fail on missing devices is enabled." << std::endl;
            return 1;
#else
            std::cout << "No " << sycl_device_names[*it] << " devices found; skipping " << sycl_device_names[*it] << " tests." << std::endl << std::endl;
#endif
        }
    }

    return returnCode;
}
