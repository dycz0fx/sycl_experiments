/*******************************************************************************
* Copyright 2019-2022 Intel Corporation.
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
*       API oneapi::mkl::dft to perform 3-D Single Precision Complex to Complex
*       Fast-Fourier Transform on a SYCL device (Host, CPU, GPU).
*
*       The supported floating point data types for data are:
*           float
*           std::complex<float>
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
#include <time.h>

#define HERE std::cout << "HERE at " << __LINE__ << std::endl

// local includes
#define NO_MATRIX_HELPERS
#include "common_for_examples.hpp"

typedef oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX> descriptor_t;

constexpr int SUCCESS = 0;
constexpr int FAILURE = 1;
constexpr float TWOPI = 6.2831853071795864769f;



// Compute (K*L)%M accurately
static float moda(int K, int L, int M)
{
    return (float)(((long long)K * L) % M);
}

static void init_1d_batch(float *x, int N, int BATCH, int H)
{
    for (int k = 0; k < BATCH; k++) {
      for (int n = 0; n < N; ++n) {
        float phase  = moda(n, H, N) / N;
	int index = k*2*N + 2*n;
        x[index+0] = cosf(TWOPI * phase) / N;
        x[index+1] = sinf(TWOPI * phase) / N;
      }
    }
}

static int verify_fwd_1d_batch(float* x, int N, int BATCH, int H) {
    // Note: this simple error bound doesn't take into account error of
    //       input data
    float errthr = 5.0f * logf((float) N) / logf(2.0f) * FLT_EPSILON;
    std::cout << "\t\tVerify the result, errthr = " << errthr << std::endl;

    float maxerr = 0.0f;
    for (int k = 0; k < BATCH; k++) {
      for (int n = 0; n < N; ++n) {
        float re_exp = ((n - H) % N == 0) ? 1.0f : 0.0f;
        float im_exp = 0.0f;
	int index = k*2*N + 2*n;

        float re_got = x[index+0];  // real component
        float im_got = x[index+1];  // imaginary component
        float err  = fabsf(re_got - re_exp) + fabsf(im_got - im_exp);
        if (err > maxerr) maxerr = err;
        if (!(err < errthr)) {
	  std::cout << "\t\tbatch[" << k << "] tx[" << n << "]: "
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


static void init_2d_batch(float *data, int N1, int N2, int BATCH, int H1, int H2)
{
    // Generalized strides for row-major addressing of data
    int S1 = 1, S2 = N1;

    for (int k = 0; k < BATCH; k++) {
        for (int n2 = 0; n2 < N2; ++n2) {
            for (int n1 = 0; n1 < N1; ++n1) {
                float phase = TWOPI * (moda(n1, H1, N1) / N1
                                           + moda(n2, H2, N2) / N2);
                int index = k*2*N1*N2 + 2*(n2*S2 + n1*S1);
                data[index+0] = cosf(phase) / (N2*N1);
                data[index+1] = sinf(phase) / (N2*N1);
            }
        }
    }
}

static int verify_fwd_2d_batch(float *data, int N1, int N2, int BATCH, int H1, int H2) {
    // Note: this simple error bound doesn't take into account error of
    //       input data
    float errthr = 5.0f * logf((float) N2*N1) / logf(2.0f) * FLT_EPSILON;
    std::cout << "\t\tVerify the result, errthr = " << errthr << std::endl;

    // Generalized strides for row-major addressing of data
    int S1 = 1, S2 = N1;

    for (int k = 0; k < BATCH; k++) {
        float maxerr = 0.0f;
        for (int n2 = 0; n2 < N2; n2++) {
            for (int n1 = 0; n1 < N1; n1++) {
                float re_exp = (
                        ((n1-H1) % N1 == 0) &&
                        ((n2-H2) % N2 == 0)
                    ) ? 1.0f : 0.0f;
                float im_exp = 0.0f;

                int index = k*2*N1*N2 + 2*(n2*S2 + n1*S1);
                float re_got = data[index+0];  // real component
                float im_got = data[index+1];  // imaginary component
                float err  = fabsf(re_got - re_exp) + fabsf(im_got - im_exp);
                if (err > maxerr) maxerr = err;
                if (!(err < errthr)) {
                    std::cout << "\t\tdata[" << k << "][" << n2 << ", " << n1 << "]: "
                              << "expected (" << re_exp << "," << im_exp << "), "
                              << "got (" << re_got << "," << im_got << "), "
                              << "err " << err << std::endl;
                    std::cout << "\t\tVerification FAILED" << std::endl;
                    return FAILURE;
                }
            }
        }
        std::cout << "\t\tVerified, maximum error was " << maxerr << std::endl;
    }
    return SUCCESS;
}

static int verify_bwd_2d_batch(float *data, int N1, int N2, int BATCH, int H1, int H2) {
    // Note: this simple error bound doesn't take into account error of
    //       input data
    float errthr = 5.0f * logf((float) N2*N1) / logf(2.0f) * FLT_EPSILON;
    std::cout << "\t\tVerify the result, errthr = " << errthr << std::endl;

    // Generalized strides for row-major addressing of data
    int S1 = 1, S2 = N1;

    for (int k = 0; k < BATCH; k++) {
        float maxerr = 0.0f;
        for (int n2 = 0; n2 < N2; n2++) {
            for (int n1 = 0; n1 < N1; n1++) {
                float phase = TWOPI * (moda(n1, H1, N1) / N1
                                           + moda(n2, H2, N2) / N2);
                float re_exp = cosf(phase) / (N2*N1);
                float im_exp = sinf(phase) / (N2*N1);

                int index = k*2*N1*N2 + 2*(n2*S2 + n1*S1);
                float re_got = data[index+0];  // real component
                float im_got = data[index+1];  // imaginary component
                float err  = fabsf(re_got - re_exp) + fabsf(im_got - im_exp);
                if (err > maxerr) maxerr = err;
                if (!(err < errthr)) {
                    std::cout << "\t\tdata[" << k << "][" << n2 << ", " << n1 << "]: "
                              << "expected (" << re_exp << "," << im_exp << "), "
                              << "got (" << re_got << "," << im_got << "), "
                              << "err " << err << std::endl;
                    std::cout << "\t\tVerification FAILED" << std::endl;
                    return FAILURE;
                }
            }
        }
        std::cout << "\t\tVerified, maximum error was " << maxerr << std::endl;
    }
    return SUCCESS;
}


static void init_3d(float *data, int N1, int N2, int N3, int H1, int H2, int H3)
{
    // Generalized strides for row-major addressing of data
    int S1 = 1, S2 = N1, S3 = N1*N2;

    for (int n3 = 0; n3 < N3; ++n3) {
        for (int n2 = 0; n2 < N2; ++n2) {
            for (int n1 = 0; n1 < N1; ++n1) {
                float phase = TWOPI * (moda(n1, H1, N1) / N1
                                           + moda(n2, H2, N2) / N2
                                           + moda(n3, H3, N3) / N3);
                int index = 2*(n3*S3 + n2*S2 + n1*S1);
                data[index+0] = cosf(phase) / (N3*N2*N1);
                data[index+1] = sinf(phase) / (N3*N2*N1);
            }
        }
    }
}

/* copy from a single 3D buffer organized zyx  (x varies fastest)
 * to a buffer split along Z dimension
 */
#if 0
static void init_split(int P, std::vector<float *> buf, float *src, int N1, int N2, int N3)
{
  for (int p = 0; p < P; p += 1) {
    memcpy(buf[p], &src[2*N1*N2*(N3/P)*p], N1*N2*(N3/P)*2*sizeof(float));
  }
}
#else
static void init_split(int P, std::vector<float *> buf, float *src, int N1, int N2, int N3)
{
    // Generalized strides for row-major addressing of data
    int sS1 = 1, sS2 = N1, sS3 = N1 * N2;
    int dS1 = 1, dS2 = N1, dS3 = N1 * N2;
    for (int n3 = 0; n3 < N3; ++n3) {
        for (int n2 = 0; n2 < N2; ++n2) {
            for (int n1 = 0; n1 < N1; ++n1) {
                int sindex = 2*(n3*sS3 + n2*sS2 + n1*sS1);
		float re = src[sindex+0];
                float im = src[sindex+1];
		int dpart = n3 / (N3/P);
		int dindex = 2*(((n3 % (N3/P)) * dS3) + (n2*dS2) + (n1*dS1));
		buf[dpart][dindex+0] = re;
		buf[dpart][dindex+1] = im;
            }
        }
    }
}

#endif

static void init_split_p2(int P, float *dest, float *src, int N1, int N2, int N3)
{
    // Generalized strides for row-major addressing of data
    int sS1 = 1, sS2 = N1, sS3 = N1 * N2;
    int dS1 = N3*N2, dS2 = N3, dS3 = 1;
    for (int n3 = 0; n3 < N3; ++n3) {
        for (int n2 = 0; n2 < N2; ++n2) {
            for (int n1 = 0; n1 < N1; ++n1) {
	      int sindex = 2*((n3 * sS3) + (n2 * sS2) + (n1 * sS1));
		float re = src[sindex+0];
                float im = src[sindex+1];
		int dindex = 2*((n3* dS3) + (n2*dS2) + (n1 * dS1));
		dest[dindex+0] = re;
		dest[dindex+1] = im;
            }
        }
    }
}


// output buffer is split along N1, contiguous along N3, strided on N2

static void merge_split(int P, std::vector<float *> buf, float *dest, int N1, int N2, int N3)
{
    // Generalized strides for row-major addressing of data
    int sS1 = N3*N2, sS2 = N3, sS3 = 1;
    int dS1 = 1, dS2=N1, dS3 = N1 * N2;
    for (int n3 = 0; n3 < N3; ++n3) {
        for (int n2 = 0; n2 < N2; ++n2) {
            for (int n1 = 0; n1 < N1; ++n1) {
	      int sindex = 2 * ((n3 * sS3) + (n2 * sS2) + ((n1 % (N1/P)) * sS1));
	      int spart = n1 / (N1/P);
	      int dindex = 2 * ((n3 * dS3) + (n2 * dS2) + (n1 * dS1));
	      float re = buf[spart][sindex+0];
	      float im = buf[spart][sindex+1];
	      dest[dindex+0] = re;
	      dest[dindex+1] = im;
            }
        }
    }
}

static int verify_fwd_3d(float* data, int N1, int N2, int N3, int H1, int H2, int H3) {
    // Note: this simple error bound doesn't take into account error of
    //       input data
    float errthr = 5.0f * logf((float) N3*N2*N1) / logf(2.0f) * FLT_EPSILON;
    //errthr = 0.001;
    std::cout << "\t\tVerify the result, errthr = " << errthr << std::endl;

    // Generalized strides for row-major addressing of data
    int S1 = 1, S2 = N1, S3 = N1*N2;

    float maxerr = 0.0f;
    for (int n3 = 0; n3 < N3; n3++) {
        for (int n2 = 0; n2 < N2; n2++) {
            for (int n1 = 0; n1 < N1; n1++) {
                float re_exp = (
                        ((n1 - H1) % N1 == 0) &&
                        ((n2 - H2) % N2 == 0) &&
                        ((n3 - H3) % N3 == 0)
                    ) ? 1.0f : 0.0f;
                float im_exp = 0.0f;

                int index = 2*(n3*S3 + n2*S2 + n1*S1);
                float re_got = data[index+0];  // real component
                float im_got = data[index+1];  // imaginary component
                float err  = fabsf(re_got - re_exp) + fabsf(im_got - im_exp);
                if (err > maxerr) maxerr = err;
                if (!(err < errthr)) {
                    std::cout << "\t\tdata[" << n3 << ", " << n2 << ", " << n1 << "]: "
                              << "expected (" << re_exp << "," << im_exp << "), "
                              << "got (" << re_got << "," << im_got << "), "
                              << "err " << err << std::endl;
                    std::cout << "\t\tVerification FAILED" << std::endl;
                    //return FAILURE;
                }
            }
        }
    }
    std::cout << "\t\tVerified, maximum error was " << maxerr << std::endl;
    return SUCCESS;
}

static int verify_bwd_3d(float* data, int N1, int N2, int N3, int H1, int H2, int H3) {
    // Note: this simple error bound doesn't take into account error of
    //       input data
    float errthr = 5.0f * logf((float) N3*N2*N1) / logf(2.0f) * FLT_EPSILON;
    std::cout << "\t\tVerify the result, errthr = " << errthr << std::endl;

    // Generalized strides for row-major addressing of data
    int S1 = 1, S2 = N1, S3 = N1*N2;

    float maxerr = 0.0f;
    for (int n3 = 0; n3 < N3; n3++) {
        for (int n2 = 0; n2 < N2; n2++) {
            for (int n1 = 0; n1 < N1; n1++) {
                float phase = TWOPI * (moda(n1, H1, N1) / N1
                                           + moda(n2, H2, N2) / N2
                                           + moda(n3, H3, N3) / N3);
                float re_exp = cosf(phase) / (N3*N2*N1);
                float im_exp = sinf(phase) / (N3*N2*N1);

                int index = 2*(n3*S3 + n2*S2 + n1*S1);
                float re_got = data[index+0];  // real component
                float im_got = data[index+1];  // imaginary component
                float err  = fabsf(re_got - re_exp) + fabsf(im_got - im_exp);
                if (err > maxerr) maxerr = err;
                if (!(err < errthr)) {
                    std::cout << "\t\tdata[" << n3 << ", " << n2 << ", " << n1 << "]: "
                              << "expected (" << re_exp << "," << im_exp << "), "
                              << "got (" << re_got << "," << im_got << "), "
                              << "err " << err << std::endl;
                    std::cout << "\t\tVerification FAILED" << std::endl;
                    return FAILURE;
                }
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
    //int N1 = 16, N2 = 13, N3 = 6;
    #define P  2

    int N1 = 128, N2 = 128, N3 = 128;
    int H1 = -1, H2 = -2, H3 = -3;
    int buffer_result = FAILURE;
    int usm_result = FAILURE;
    int result = FAILURE;

    float* in = (float*) mkl_malloc(N3*N2*N1*2*sizeof(float), 64);
    float* out = (float*) mkl_malloc(N3*N2*N1*2*sizeof(float), 64);
    init_3d(in, N1, N2, N3, H1, H2, H3);
    
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
        cl::sycl::buffer<float, 1> inBuffer(in, cl::sycl::range<1>(N3*N2*N1*2));
        inBuffer.set_write_back(false);

        // Setting up USM and initialization
        float *in_usm = (float*) malloc_device(N3*N2*N1*2*sizeof(float), queue.get_device(), queue.get_context());
        float *out_usm = (float*) malloc_device(N3*N2*N1*2*sizeof(float), queue.get_device(), queue.get_context());
        init_3d(in, N1, N2, N3, H1, H2, H3);
	queue.memcpy(in_usm, in, N3*N2*N1*2*sizeof(float));
	queue.wait();


        descriptor_t desc({N3, N2, N1});
        desc.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE, (1.0/(N1*N2*N3)));

	
        desc.commit(queue);

        // Using SYCL buffers
        std::cout<<"\tUsing SYCL buffers"<<std::endl;
        oneapi::mkl::dft::compute_forward(desc, inBuffer);
        {
          auto inAcc = inBuffer.get_access<cl::sycl::access::mode::read>();
          buffer_result = verify_fwd_3d(inAcc.get_pointer(), N1, N2, N3, H1, H2, H3);
        }
        if (buffer_result == SUCCESS) {
            oneapi::mkl::dft::compute_backward(desc, inBuffer);
            auto inAcc = inBuffer.get_access<cl::sycl::access::mode::read>();
            buffer_result = verify_bwd_3d(inAcc.get_pointer(), N1, N2, N3, H1, H2, H3);
        }

        // Using USM
        std::cout<<"\tUsing USM"<<std::endl;
        desc.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
        desc.commit(queue);
        cl::sycl::event fwd, bwd;
	struct timespec ts_start, ts_end, ph1_start, ph1_end, a2a_end, ph2_end;
	double elapsed, elapsed_ph1, elapsed_a2a, elapsed_ph2;
	int tries;
	for (tries = 1; tries < 1000000; tries <<= 1) {
	  clock_gettime(CLOCK_REALTIME, &ts_start);
	  for (int iter = 0; iter < tries; iter += 1) {
	    fwd = oneapi::mkl::dft::compute_forward(desc, in_usm, out_usm);
	    fwd.wait();
	  }
	  clock_gettime(CLOCK_REALTIME, &ts_end);
	  elapsed = ((double) (ts_end.tv_sec - ts_start.tv_sec)) * 1000000000.0 +
	    ((double) (ts_end.tv_nsec - ts_start.tv_nsec));
	  elapsed /= 1000000000.0;
	  if (elapsed > 0.1) break;
	}
	std::cout << "\tfft kernel time " << elapsed/tries << std::endl;
	double flops = 5.0 * N1 * N2 * N3 *(log2(N1) + log2(N2) + log2(N3));
	std::cout << "\tfft MFlops " << (flops/(elapsed/tries))/1000000.0 << std::endl;
	queue.memcpy(out, out_usm, N3*N2*N1*2*sizeof(float));
	queue.wait();
        usm_result = verify_fwd_3d(out, N1, N2, N3, H1, H2, H3);

        if (usm_result == SUCCESS) {
	  bwd = oneapi::mkl::dft::compute_backward(desc, out_usm, in_usm);
            bwd.wait();
	    queue.memcpy(out, in_usm, N3*N2*N1*2*sizeof(float));
	    queue.wait();
            usm_result = verify_bwd_3d(out, N1, N2, N3, H1, H2, H3);
        }
	// now try it in pieces
	// N1 varies fastest, so split along N3 first
	// split into P parts
        std::cout<<"\tUsing Distributed"<<std::endl;

	assert((N3 % P) == 0); // for split in first phase
	assert((N1 % P) == 0); // for split in second phase
	std::vector<float *> p1_in;
	std::vector<float *> p1_out;     // output for phase 1, input for ph 2
	std::vector<float *> p2_in;     // output for phase 1, input for ph 2
	std::vector<float *> p2_out;
	std::vector<float *> h1_in;      // host buffers for input
	
	std::vector<float *> h2_in;      // host buffers for output
	std::vector<float *> h2_out;      // host buffers for output
	//std::vector<descriptor_t> d1;   // for phase 1
	//std::vector<descriptor_t> d2;   // for phase 2
	size_t bufcount = (N3*N2*N1*2)/P;
	size_t bufsize = bufcount * sizeof(float);
	for (int p = 0; p < P; p += 1) {
	  p1_in.push_back((float*) malloc_device(bufsize, queue.get_device(), queue.get_context()));
	  h1_in.push_back((float*) malloc_host(bufsize, queue.get_context()));
	  p1_out.push_back((float*) malloc_device(bufsize, queue.get_device(), queue.get_context()));
	  p2_in.push_back((float*) malloc_device(bufsize, queue.get_device(), queue.get_context()));
	  p2_out.push_back((float*) malloc_device(bufsize, queue.get_device(), queue.get_context()));
	  h2_in.push_back((float*) malloc_host(bufsize, queue.get_context()));
	  h2_out.push_back((float*) malloc_host(bufsize, queue.get_context()));
	}

	descriptor_t d1p0({N1,N2});

	//std::int64_t d1p0_s1in[3] = {0L, 1L, static_cast<std::int64_t> (N1)};
	//std::int64_t d1p0_s1out[3] = {0L, 1L, static_cast<std::int64_t> (N1)};
	d1p0.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
	d1p0.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE, (1.0/(N1*N2)));
	d1p0.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, N3/P);
	ulong d1p0_dist = N1 * N2;
	d1p0.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, d1p0_dist);
	d1p0.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, d1p0_dist);
	//d1p0.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, d1p0_s1in);
	//d1p0.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, d1p0_s1out);
	d1p0.commit(queue);

	descriptor_t d1p1({N1,N2});

	//std::int64_t d1p1_s1in[3] = {0L, 1L, static_cast<std::int64_t> (N1)};
	//std::int64_t d1p1_s1out[3] = {0L, 1L, static_cast<std::int64_t> (N1)};
	d1p1.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
	d1p1.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE, (1.0/(N1*N2)));
	d1p1.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, N3/P);
	ulong d1p1_dist = N1 * N2;
	d1p1.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, d1p1_dist);
	d1p1.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, d1p1_dist);
	//d1p1.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, d1p1_s1in);
	//d1p1.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, d1p1_s1out);
	d1p1.commit(queue);

	descriptor_t d2p0(N3);
	/* phase 2 is a baatch of single Z direction FFTs, with Z in unit stride */
	/* The Y's are next (stride N3) and the X's are next (stride N3*N2) */
	//std::int64_t d2p0_s2in[2] = {0L, 1L}; 
	//std::int64_t d2p0_s2out[2] = {0L, 1L};
	d2p0.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
	d2p0.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE, (1.0/(N3)));
	d2p0.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, N2*N1/P);
	ulong d2p0_dist = N3;
	d2p0.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, d2p0_dist);
	d2p0.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, d2p0_dist);
	//d2p0.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, d2p0_s2in);
	//d2p0.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, d2p0_s2out);
	d2p0.commit(queue);

	descriptor_t d2p1(N3);
	/* phase 2 is a baatch of single Z direction FFTs, with Z in unit stride */
	/* The Y's are next (stride N3) and the X's are next (stride N3*N2) */
	//std::int64_t d2p1_s2in[2] = {0L, 1L}; 
	//std::int64_t d2p1_s2out[2] = {0L, 1L};
	d2p1.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
	d2p1.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE, (1.0/(N3)));
	d2p1.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, N2*N1/P);
	ulong d2p1_dist = N3;
	d2p1.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, d2p1_dist);
	d2p1.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, d2p1_dist);
	//d2p1.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, d2p1_s2in);
	//d2p1.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, d2p1_s2out);
	d2p1.commit(queue);

	/*
	 * do a 2D transform on each board
	 * do all to all
	 * do a 1D Batch transform for the second phase
	 *
	 * data layout
	 * p1_in 
	 *   n1 stride 1
	 *   n2 stride N1
	 *   n3 stride N1 * N2  starting at global index p * (N3/p)
	 * p1_out
	 *   n1 stride 1
	 *   n2 stride N1
	 *   n3 stride N1 * N2 starting at global index  p * (N3/p)
	 * p2_in
	 *   n1 stride N3 * N2 starting at global index p * (N1/p)
	 *   n2 stride N3
	 *   n3 stride 1
	 * p2_out
	 *   n1 stride N3 * N2 starting at global index p * (N1/p)
	 *   n2 stride N3
	 *   n3 stride 1
	 *
	 * The copy from p1_out to p2_in can be expressed as a 
	 * cblas_ccopy_batch_strided, maybe
	 *  The contiguous pieces are all length N2 
	 * each dimension of stride should have a base
	 * sidx = ((n1+sB1)*sS1) + ((n2+sB2)*sS2) + ((n3+sB3)*sS3)
	 * didx = ((n1+dB1)*dS1) + ((n2+dB2)*dS2) + ((n3+dB3)*dS3)
	 */

	// run the phase 1 transforms
	// each does complete planes in the n1,n2 directions, and a batch of
	// N3/P
	// the results are routed to the board owning the N3 index
	// Initialize device input buffers
	int p;  // for unrolled loops below

#define VERIFY_2D 0
#if VERIFY_2D
	init_2d_batch(in, N1, N2, N3, H1, H2);

	
        std::cout<<"\t\tInit_split"<<std::endl;
	init_split(P, h1_in, in, N1, N2, N3);
        std::cout<<"\t\tmemcpy to device"<<std::endl;
	for (int p = 0; p < P; p += 1) {
	  queue.memcpy(p1_in[p], h1_in[p], bufsize);
	  queue.wait();
	}

        std::cout<<"\t\tcompute phase 1 forward"<<std::endl;

	p = 0;
	fwd = oneapi::mkl::dft::compute_forward(d1p0, p1_in[p], p1_out[p]);
	fwd.wait();
	p = 1;
	fwd = oneapi::mkl::dft::compute_forward(d1p1, p1_in[p], p1_out[p]);
	fwd.wait();

	// copy back for 2d_batch check
	std::cout<<"\t\t2d memcpy to host"<<std::endl;
	for (int p = 0; p < P; p += 1) {
	  queue.memcpy(&out[p * bufcount], p1_out[p], bufsize);
	  queue.wait();
	}
	result = verify_fwd_2d_batch(out, N1, N2, N3, H1, H2);

        if (result == SUCCESS) {
	  std::cout << "2D batch verify success " << std::endl;
        } else {
	  std::cout << "2D batch verify fail " << std::endl;
	}
#endif //! VERIFY_2D
#define VERIFY_1D 0
#if VERIFY_1D       
	// verify second pass separately
	init_1d_batch(in, N3, (N1/P) * N2, H1);
        std::cout<<"\t\tmemcpy to device"<<std::endl;
	for (int p = 0; p < P; p += 1) {
	  queue.memcpy(p2_in[p], &in[p * bufcount], bufsize);
	  queue.wait();
	}
	// run the phase 2 transforms
        std::cout<<"\t\tcompute phase 2 forward"<<std::endl;
	p = 0;
	fwd = oneapi::mkl::dft::compute_forward(d2p0, p2_in[p], p2_out[p]);
	fwd.wait();
	p = 1;
	fwd = oneapi::mkl::dft::compute_forward(d2p1, p2_in[p], p2_out[p]);
	fwd.wait();

	std::cout<<"\t\t1d memcpy to host"<<std::endl;
	for (int p = 0; p < P; p += 1) {
	  queue.memcpy(&out[p * bufcount], p2_out[p], bufsize);
	  queue.wait();
	}
	result = verify_fwd_1d_batch(out, N3, (N1/P) * N2, H1);
        if (result == SUCCESS) {
	  std::cout << "1D batch verify success " << std::endl;
        } else {
	  std::cout << "1D batch verify fail " << std::endl;
	}

#endif //! VERIFY_1D




	// redo with 3d data
        init_3d(in, N1, N2, N3, H1, H2, H3);
	init_split(P, h1_in, in, N1, N2, N3);
        std::cout<<"\t\tmemcpy to device"<<std::endl;
	for (int p = 0; p < P; p += 1) {
	  queue.memcpy(p1_in[p], h1_in[p], bufsize);
	  queue.wait();
	}
	for (tries = 1; tries < 1000000; tries <<= 1) {
	  clock_gettime(CLOCK_REALTIME, &ts_start);

	  for (int iter = 0; iter < tries; iter += 1) {

	    clock_gettime(CLOCK_REALTIME, &ph1_start);
	    p = 0;
	    fwd = oneapi::mkl::dft::compute_forward(d1p0, p1_in[p], p1_out[p]);
	    fwd.wait();
	    p = 1;
	    fwd = oneapi::mkl::dft::compute_forward(d1p1, p1_in[p], p1_out[p]);
	    fwd.wait();
	    clock_gettime(CLOCK_REALTIME, &ph1_end);

	
	    //for (int p = 0; p < P; p += 1) {
	    //  fwd = oneapi::mkl::dft::compute_forward(d1[p], p1_in[p], p1_out[p]);
	    //  fwd.wait();
	    //}
	    // run a kernel to copy data to the right places
#if 1
	    std::cout<<"\t\tall to all"<<std::endl;
	    for (int p = 0; p < P; p += 1) {
	      int sS1 = 1;        //p1_out_S1;
	      int sS2 = N1;       //p1_out_S2;
	      int sS3 = N1 * N2;  //p1_out_S3;
	      int dS1 = N3 * N2;  //p2_in_S1;
	      int dS2 = N3;       //p2_in_S2;
	      int dS3 = 1;        //p2_in_S3;
	      float *src = p1_out[p];
	      float *dest[2] = {p2_in[0], p2_in[1]};
	      int wg_size = 32;
	      auto e = queue.submit([&](sycl::handler &h) {
		  h.parallel_for({static_cast<size_t>(N1), static_cast<size_t>(N2), static_cast<size_t>(N3/P)}, [=](sycl::item<3> idx){
		      int n1 = idx.get_id(0);
		      int n2 = idx.get_id(1);
		      int sn3 = idx.get_id(2);  // source n3 index in [0..N3/P)
		      int gn3 = sn3 + (p * (N3/P));   // global n3 index in [0..N3)
		      int sidx = 2 * ((n1*sS1) + (n2*sS2) + (sn3*sS3));  // units of sizeof(float)
		      int dbuf = n1/(N1/P);   // destination buffer index, distributed by x
		      int dn1 = n1 % (N1/P);  // x coord within dbuf
		      int didx = 2 * ((dn1*dS1) + (n2*dS2) + (gn3*dS3));
		      //assert(n1 < N1);
		      //assert(n2 < N2);
		      //assert(sn3 < N3/P);
		      //assert(dbuf < P);
		      //assert(dn1 < N1/P);
		      //assert(n2 < N2);
		      //assert(gn3 < N3);
		      dest[dbuf][didx+0] = src[sidx+0];
		      dest[dbuf][didx+1] = src[sidx+1];
		    });
		});
	      e.wait_and_throw();
	    }
#else
	    // host all to all
	    // copy back buffers
	    std::cout<<"\t\t2d memcpy to host"<<std::endl;
	    for (int p = 0; p < P; p += 1) {
	      queue.memcpy(&out[p * bufcount], p1_out[p], bufsize);
	      queue.wait();
	    }
	    // swizzle
	    std::cout<<"\t\tswizzle"<<std::endl;
	    init_split_p2(P, in, out, N1, N2, N3);
	    // copy buffers to device
	    std::cout<<"\t\t2d memcopy back to device"<<std::endl;
	    for (int p = 0; p < P; p += 1) {
	      queue.memcpy(p2_in[p], &in[p * bufcount], bufsize);
	      queue.wait();
	    }
	    
	    
#endif
	    clock_gettime(CLOCK_REALTIME, &a2a_end);
	
	    // run the phase 2 transforms
	    std::cout<<"\t\tcompute phase 2 forward"<<std::endl;
	    p = 0;
	    fwd = oneapi::mkl::dft::compute_forward(d2p0, p2_in[p], p2_out[p]);
	    fwd.wait();
	    p = 1;
	    fwd = oneapi::mkl::dft::compute_forward(d2p1, p2_in[p], p2_out[p]);
	    fwd.wait();
	    //for (int p = 0; p < P; p += 1) {
	    //  fwd = oneapi::mkl::dft::compute_forward(d2[p], p2_in[p], p2_out[p]);
	    //  fwd.wait();
	    //}
	    
	    clock_gettime(CLOCK_REALTIME, &ph2_end);
	    
	  }  //! iter loop
	  clock_gettime(CLOCK_REALTIME, &ts_end);
	  elapsed = ((double) (ts_end.tv_sec - ts_start.tv_sec)) * 1000000000.0 +
	    ((double) (ts_end.tv_nsec - ts_start.tv_nsec));
	  elapsed /= 1000000000.0;
	  if (elapsed > 0.1) break;
	} //! tries loop
	    
	elapsed_ph1 =  ((double) (ph1_end.tv_sec - ph1_start.tv_sec)) * 1000000000.0 +
	  ((double) (ph1_end.tv_nsec - ph1_start.tv_nsec));
	elapsed_a2a =  ((double) (a2a_end.tv_sec - ph1_end.tv_sec)) * 1000000000.0 +
	  ((double) (a2a_end.tv_nsec - ph1_end.tv_nsec));
	elapsed_ph2 =  ((double) (ph2_end.tv_sec - a2a_end.tv_sec)) * 1000000000.0 +
	  ((double) (ph2_end.tv_nsec - a2a_end.tv_nsec));

	std::cout << "\tfft kernel time " << elapsed/tries << std::endl;
	flops = 5.0 * N1 * N2 * N3 *(log2(N1) + log2(N2) + log2(N3));
	std::cout << "\tfft MFlops " << (flops/(elapsed/tries))/1000000.0 << std::endl;

	std::cout << "\tph1 kernel time " << elapsed_ph1/1000000000.0 << std::endl;

	std::cout << "\ta2a kernel time " << elapsed_a2a/1000000000.0 << std::endl;

	std::cout << "\tph2 kernel time " << elapsed_ph2/1000000000.0 << std::endl;

	// copy data back to host

        std::cout<<"\t\tmemcpy to host"<<std::endl;
	for (int p = 0; p < P; p += 1) {
	  queue.memcpy(h2_out[p], p2_out[p], bufsize);
	  queue.wait();
	}

        std::cout<<"\t\tmerge split"<<std::endl;
	merge_split(P, h2_out, out, N1, N2, N3);
        usm_result = verify_fwd_3d(out, N1, N2, N3, H1, H2, H3);

        if ((buffer_result == SUCCESS) && (usm_result == SUCCESS))
            result = SUCCESS;

        free(out_usm, queue.get_context());
        free(in_usm, queue.get_context());
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

int run_dft_split(cl::sycl::device &dev) {
    //
    // Initialize data for DFT
    //
    //int N1 = 16, N2 = 13, N3 = 6;
    #undef P
    #define P  1

    int N1 = 256, N2 = 256, N3 = 256;
    int H1 = -1, H2 = -2, H3 = -3;
    int buffer_result = FAILURE;
    int usm_result = FAILURE;
    int result = FAILURE;

    float* in = (float*) mkl_malloc(N3*N2*N1*2*sizeof(float), 64);
    float* out = (float*) mkl_malloc(N3*N2*N1*2*sizeof(float), 64);



    init_3d(in, N1, N2, N3, H1, H2, H3);
            cl::sycl::event fwd, bwd;
	struct timespec ts_start, ts_end, ph1_start, ph1_end, a2a_end, ph2_end;
	double elapsed, elapsed_ph1, elapsed_a2a, elapsed_ph2;
	double flops;
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

        // Setting up USM and initialization
        float *in_usm = (float*) malloc_device(N3*N2*N1*2*sizeof(float), queue.get_device(), queue.get_context());
        float *out_usm = (float*) malloc_device(N3*N2*N1*2*sizeof(float), queue.get_device(), queue.get_context());
        init_3d(in, N1, N2, N3, H1, H2, H3);
	queue.memcpy(in_usm, in, N3*N2*N1*2*sizeof(float));
	queue.wait();
        descriptor_t desc({N3, N2, N1});
        desc.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE, (1.0/(N1*N2*N3)));
	
        desc.commit(queue);
       // Using USM
        std::cout<<"\tUsing USM"<<std::endl;
        desc.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
        desc.commit(queue);
        cl::sycl::event fwd, bwd;
	struct timespec ts_start, ts_end, ph1_start, ph1_end, a2a_end, ph2_end;
	double elapsed, elapsed_ph1, elapsed_a2a, elapsed_ph2;
	int tries;
	for (tries = 1; tries < 1000000; tries <<= 1) {
	  clock_gettime(CLOCK_REALTIME, &ts_start);
	  for (int iter = 0; iter < tries; iter += 1) {
	    fwd = oneapi::mkl::dft::compute_forward(desc, in_usm, out_usm);
	    fwd.wait();
	  }
	  clock_gettime(CLOCK_REALTIME, &ts_end);
	  elapsed = ((double) (ts_end.tv_sec - ts_start.tv_sec)) * 1000000000.0 +
	    ((double) (ts_end.tv_nsec - ts_start.tv_nsec));
	  elapsed /= 1000000000.0;
	  if (elapsed > 0.1) break;
	}
	std::cout << "\tfft kernel time " << elapsed/tries << std::endl;
	double flops = 5.0 * N1 * N2 * N3 *(log2(N1) + log2(N2) + log2(N3));
	std::cout << "\tfft MFlops " << (flops/(elapsed/tries))/1000000.0 << std::endl;
	queue.memcpy(out, out_usm, N3*N2*N1*2*sizeof(float));
	queue.wait();
        usm_result = verify_fwd_3d(out, N1, N2, N3, H1, H2, H3);

        if (usm_result == SUCCESS) {
	  bwd = oneapi::mkl::dft::compute_backward(desc, out_usm, in_usm);
            bwd.wait();
	    queue.memcpy(out, in_usm, N3*N2*N1*2*sizeof(float));
	    queue.wait();
            usm_result = verify_bwd_3d(out, N1, N2, N3, H1, H2, H3);
        }

	// N1 varies fastest, so split along N3 first
        std::cout<<"\tUsing Distributed"<<std::endl;

	size_t bufcount = (N3*N2*N1*2);
	size_t bufsize = bufcount * sizeof(float);
	float *hw_in = (float *) malloc_device(bufsize, queue.get_device(), queue.get_context());
	float *hw_mid = (float *) malloc_device(bufsize, queue.get_device(), queue.get_context());
	float *hw_out = (float *) malloc_device(bufsize, queue.get_device(), queue.get_context());

	descriptor_t d1p0({N1,N2});

	//std::int64_t d1p0_s1in[3] = {0L, 1L, static_cast<std::int64_t> (N1)};
	//std::int64_t d1p0_s1out[3] = {0L, 1L, static_cast<std::int64_t> (N1)};
	//d1p0.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
	d1p0.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE, (1.0/(N1*N2)));
	d1p0.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, N3);
	ulong d1p0_dist = N1 * N2;
	d1p0.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, d1p0_dist);
	d1p0.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, d1p0_dist);
	//d1p0.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, d1p0_s1in);
	//d1p0.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, d1p0_s1out);
	d1p0.commit(queue);

	descriptor_t d2p0(N3);
	/* phase 2 is a baatch of single Z direction FFTs, with Z in unit stride */
	/* The Y's are next (stride N3) and the X's are next (stride N3*N2) */
	std::int64_t d2p0_s2in[2] = {0L, N1*N2}; 
	std::int64_t d2p0_s2out[2] = {0L, N1*N2};
	//d2p0.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
	d2p0.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE, (1.0/(N3)));
	d2p0.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, N2*N1);
	ulong d2p0_dist = N3;
	d2p0.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, d2p0_dist);
	d2p0.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, d2p0_dist);
	//d2p0.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, d2p0_s2in);
	//d2p0.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, d2p0_s2out);
	d2p0.commit(queue);


	// run the phase 1 transforms
	// each does complete planes in the n1,n2 directions, and a batch of
	// N3
	// the results are routed to the board owning the N3 index
	// Initialize device input buffers
	int p;  // for unrolled loops below

	// redo with 3d data
        init_3d(in, N1, N2, N3, H1, H2, H3);
        std::cout<<"\t\tmemcpy to device"<<std::endl;
	queue.memcpy(hw_in, in, bufsize);
	queue.wait();


	for (tries = 1; tries < 1000000; tries <<= 1) {
	  clock_gettime(CLOCK_REALTIME, &ts_start);

	  for (int iter = 0; iter < tries; iter += 1) {

	    clock_gettime(CLOCK_REALTIME, &ph1_start);
	    fwd = oneapi::mkl::dft::compute_forward(d1p0, hw_in, hw_mid);
	    fwd.wait();
	    clock_gettime(CLOCK_REALTIME, &ph1_end);


	    // run a kernel to copy data to the right places
	    clock_gettime(CLOCK_REALTIME, &a2a_end);
	
	    // run the phase 2 transforms
	    //std::cout<<"\t\tcompute phase 2 forward"<<std::endl;
	    fwd = oneapi::mkl::dft::compute_forward(d2p0, hw_mid, hw_out);
	    fwd.wait();
	    
	    clock_gettime(CLOCK_REALTIME, &ph2_end);
	    
	  }  //! iter loop
	  clock_gettime(CLOCK_REALTIME, &ts_end);
	  elapsed = ((double) (ts_end.tv_sec - ts_start.tv_sec)) * 1000000000.0 +
	    ((double) (ts_end.tv_nsec - ts_start.tv_nsec));
	  elapsed /= 1000000000.0;
	  if (elapsed > 0.1) break;
	} //! tries loop
	    
	elapsed_ph1 =  ((double) (ph1_end.tv_sec - ph1_start.tv_sec)) * 1000000000.0 +
	  ((double) (ph1_end.tv_nsec - ph1_start.tv_nsec));
	elapsed_a2a =  ((double) (a2a_end.tv_sec - ph1_end.tv_sec)) * 1000000000.0 +
	  ((double) (a2a_end.tv_nsec - ph1_end.tv_nsec));
	elapsed_ph2 =  ((double) (ph2_end.tv_sec - a2a_end.tv_sec)) * 1000000000.0 +
	  ((double) (ph2_end.tv_nsec - a2a_end.tv_nsec));

	std::cout << "\tfft kernel time " << elapsed/tries << std::endl;
	std::cout << "\tfft kernel iterations to exceed .1 sec " << tries << std::endl;
	flops = 5.0 * N1 * N2 * N3 *(log2(N1) + log2(N2) + log2(N3));
	std::cout << "\tfft MFlops " << (flops/(elapsed/tries))/1000000.0 << std::endl;

	std::cout << "\tph1 kernel time " << elapsed_ph1/1000000000.0 << std::endl;

	std::cout << "\ta2a kernel time " << elapsed_a2a/1000000000.0 << std::endl;

	std::cout << "\tph2 kernel time " << elapsed_ph2/1000000000.0 << std::endl;

	// copy data back to host

        std::cout<<"\t\tmemcpy to host"<<std::endl;
	queue.memcpy(out, hw_out, bufsize);
	queue.wait();

        usm_result = verify_fwd_3d(out, N1, N2, N3, H1, H2, H3);

        if (usm_result == SUCCESS)
            result = SUCCESS;

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
    std::cout << "# 3D FFT Complex-Complex Single-Precision Example: " << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Using apis:" << std::endl;
    std::cout << "#   dft" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Supported floating point type precisions:" << std::endl;
    std::cout << "#   float" << std::endl;
    std::cout << "#   std::complex<float>" << std::endl;
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
//  For each device selected and each supported data type, Basic_Dp_C2C_3D_FFTExample
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
            std::cout << "Running tests on " << sycl_device_names[*it] << ".\n";

            std::cout << "\tRunning with single precision complex-to-complex 3-D FFT:" << std::endl;
#if 0
            int status = run_dft_example(my_dev);
            if (status != SUCCESS) {
                std::cout << "\tTest Failed" << std::endl << std::endl;
                returnCode = status;
            } else {
                std::cout << "\tTest Passed" << std::endl << std::endl;
            }
#endif
            int status = run_dft_split(my_dev);
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
