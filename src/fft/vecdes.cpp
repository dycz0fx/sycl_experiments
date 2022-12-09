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

/* copy from a single 3D buffer organized zyx  (x varies fastest)
 * to a buffer split along Z dimension
 */


// Compute (K*L)%M accurately
static float moda(int K, int L, int M)
{
    return (float)(((long long)K * L) % M);
}

static void init(float *data, int N1, int N2, int N3, int H1, int H2, int H3)
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

#if 1
static void init_split(int P, std::vector<float *> buf, float *src, int N1, int N2, int N3)
{
  for (int p = 0; p < P; p += 1) {
    memcpy(buf[p], &src[N1*N2*(N3/P)*p], N1*N2*(N3/P)*2* sizeof(float));
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
// output buffer is split along N1, contiguous along N3, strided on N2

static void merge_split(int P, std::vector<float *> buf, float *dest, int N1, int N2, int N3)
{
    // Generalized strides for row-major addressing of data
    int sS1 = N3*N2, sS2 = N3, sS3 = 1;
    int dS1 = 1, dS2=N1, dS3 = N1 * N2;
    for (int n3 = 0; n3 < N3; ++n3) {
        for (int n2 = 0; n2 < N2; ++n2) {
            for (int n1 = 0; n1 < N1; ++n1) {
	      int sindex = 2*((n3 * sS3) + (n2 * sS2) + ((n1 % (N1/P)) * sS1));
	      int spart = n1 / (N1/P);
	      int dindex = 2*((n3*dS3) + (n2*dS2) + (n1 * dS1));
	      float re = buf[spart][sindex+0];
	      float im = buf[spart][sindex+1];
	      dest[dindex+0] = re;
	      dest[dindex+1] = im;
            }
        }
    }
}


static int verify_fwd(float* data, int N1, int N2, int N3, int H1, int H2, int H3) {
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
                    return FAILURE;
                }
            }
        }
    }
    std::cout << "\t\tVerified, maximum error was " << maxerr << std::endl;
    return SUCCESS;
}

static int verify_bwd(float* data, int N1, int N2, int N3, int H1, int H2, int H3) {
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
    int N1 = 64, N2 = 64, N3 = 64;
    int H1 = -1, H2 = -2, H3 = -3;
    int buffer_result = FAILURE;
    int usm_result = FAILURE;
    int result = FAILURE;

    float* in = (float*) mkl_malloc(N3*N2*N1*2*sizeof(float), 64);
    float* out = (float*) mkl_malloc(N3*N2*N1*2*sizeof(float), 64);
    init(in, N1, N2, N3, H1, H2, H3);
    
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

 
	// now try it in pieces
	// N1 varies fastest, so split along N3 first
	// split into P parts
        std::cout<<"\tUsing Distributed"<<std::endl;
	#define P 2
	assert((N3 % P) == 0); // for split in first phase
	assert((N1 % P) == 0); // for split in second phase
	std::vector<float *> p1_in;
	std::vector<float *> p1_out;     // output for phase 1, input for ph 2
	std::vector<float *> p2_in;     // output for phase 1, input for ph 2
	std::vector<float *> p2_out;
	std::vector<float *> h1_in;      // host buffers for input
	std::vector<float *> h2_out;      // host buffers for output
	for (int p = 0; p < P; p += 1) {
	  p1_in.push_back((float*) malloc_device((N3/P)*N2*N1*2*sizeof(float), queue.get_device(), queue.get_context()));
	  h1_in.push_back((float*) malloc_host((N3/P)*N2*N1*2*sizeof(float), queue.get_context()));
	  p1_out.push_back((float*) malloc_device((N3/P)*N2*N1*2*sizeof(float), queue.get_device(), queue.get_context()));
	  p2_in.push_back((float*) malloc_device((N3/P)*N2*N1*2*sizeof(float), queue.get_device(), queue.get_context()));
	  p2_out.push_back((float*) malloc_device((N3/P)*N2*N1*2*sizeof(float), queue.get_device(), queue.get_context()));
	  h2_out.push_back((float*) malloc_host((N3/P)*N2*N1*2*sizeof(float), queue.get_context()));
	}


	std::vector<descriptor_t *> d1;   // for phase 1
	std::vector<descriptor_t *> d2;   // for phase 2

	for (int p = 0; p < P; p += 1) {
	  descriptor_t *d1d = new descriptor_t({N1,N2});
	  d1.push_back(d1d);
	  descriptor_t *d2d = new descriptor_t(N3);
	  d2.push_back(d2d);
	}
	for (int p = 0; p < P; p += 1) {
	  std::int64_t s1in[3] = {0L, 1L, static_cast<std::int64_t> (N1)};
	  std::int64_t s1out[3] = {0L, 1L, static_cast<std::int64_t> (N1)};
	  d1[p]->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
	  d1[p]->set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE, (1.0/(N1*N2)));
	  int nt = N3/P;
	  d1[p]->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, nt);
	  ulong dist1 = N1 * N2;
	  d1[p]->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, dist1);
	  d1[p]->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, dist1);
	  d1[p]->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, s1in);
	  d1[p]->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, s1out);
	  d1[p]->commit(queue);
	  /* phase 2 is a baatch of single Z direction FFTs, with Z in unit stride */
	  /* The Y's are next (stride N3) and the X's are next (stride N3*N2) */
	  std::int64_t s2in[2] = {0L, 1L}; 
	  std::int64_t s2out[2] = {0L, 1L};
	  d2[p]->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
	  d2[p]->set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE, (1.0/(N3)));
	  d2[p]->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, N2*N1/P);
	  ulong dist2 = N3;
	  d2[p]->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, dist2);
	  d2[p]->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, dist2);
	  d2[p]->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, s2in);
	  d2[p]->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, s2out);
	  d2[p]->commit(queue);
	}


	

	// Initialize device input buffers
        std::cout<<"\t\tInit_split"<<std::endl;
	init_split(P, h1_in, in, N1, N2, N3);
        std::cout<<"\t\tmemcpy to device"<<std::endl;
	for (int p = 0; p < P; p += 1) {
	  queue.memcpy(p1_in[p], h1_in[p], (N3/P)*N2*N1*2*sizeof(float));
	  queue.wait();
	}
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
        std::cout<<"\t\tcompute phase 1 forward"<<std::endl;
        cl::sycl::event fwd, bwd;


	for (int p = 0; p < P; p += 1) {
	  fwd = oneapi::mkl::dft::compute_forward(*d1[p], p1_in[p], p1_out[p]);
	  fwd.wait();
	}
	// run a kernel to copy data to the right places
        std::cout<<"\t\tall to all"<<std::endl;
	for (int p = 0; p < P; p += 1) {
	  ulong iterations = N3/P * N2 * N1;
	  int sS1 = 1;        //p1_out_S1;
	  int sS2 = N1;         //p1_out_S2;
	  int sS3 = N1 * N2;   //p1_out_S3;
	  int dS1 = N3 * N2;   //p2_in_S1;
	  int dS2 = N3;        //p2_in_S2;
	  int dS3 = 1;         //p2_in_S3;
	  float *src = p1_out[p];
	  float *dest[2] = {p2_in[0], p2_in[1]};
	  int wg_size = 32;
	  auto e = queue.submit([&](sycl::handler &h) {
	      h.parallel_for({static_cast<size_t>(N1), static_cast<size_t>(N2), static_cast<size_t>(N3/P)}, [=](sycl::item<3> idx){
		  int n1 = idx.get_id(0);
		  int n2 = idx.get_id(1);
		  int sn3 = idx.get_id(2);
		  int gn3 = sn3 + (p * (N3/P) * N1 * N2);
		  int sidx = 2 * ((n1*sS1) + (n2*sS2) + (sn3*sS3));
		  int dbuf = n1/(N1/P);   // distributed by x
		  int dn1 = n1 % (N1/P);  // x coord within dbuf
		  int didx = 2 * ((dn1*dS1) + (n2*dS2) + (gn3*dS3));
		  dest[dbuf][didx+0] = src[sidx+0];
		  dest[dbuf][didx+1] = src[sidx+1];
		});
	    });
	  e.wait_and_throw();
	}
	// run the phase 2 transforms
        std::cout<<"\t\tcompute phase 2 forward"<<std::endl;



	// Initialize device input buffers
        std::cout<<"\t\tInit_split"<<std::endl;
	init_split(P, h1_in, in, N1, N2, N3);
        std::cout<<"\t\tmemcpy to device"<<std::endl;
	for (int p = 0; p < P; p += 1) {
	  queue.memcpy(p1_in[p], h1_in[p], (N3/P)*N2*N1*2*sizeof(float));
	  queue.wait();
	}
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
        std::cout<<"\t\tcompute phase 1 forward"<<std::endl;

	
	for (int p = 0; p < P; p += 1) {
	  fwd = oneapi::mkl::dft::compute_forward(*d1[p], p1_in[p], p1_out[p]);
	  fwd.wait();
	}
	// run a kernel to copy data to the right places
        std::cout<<"\t\tall to all"<<std::endl;
	for (int p = 0; p < P; p += 1) {
	  ulong iterations = N3/P * N2 * N1;
	  int sS1 = 1;        //p1_out_S1;
	  int sS2 = N1;         //p1_out_S2;
	  int sS3 = N1 * N2;   //p1_out_S3;
	  int dS1 = N3 * N2;   //p2_in_S1;
	  int dS2 = N3;        //p2_in_S2;
	  int dS3 = 1;         //p2_in_S3;
	  float *src = p1_out[p];
	  float *dest[2] = {p2_in[0], p2_in[1]};
	  int wg_size = 32;
	  auto e = queue.submit([&](sycl::handler &h) {
	      h.parallel_for({static_cast<size_t>(N1), static_cast<size_t>(N2), static_cast<size_t>(N3/P)}, [=](sycl::item<3> idx){
		  int n1 = idx.get_id(0);
		  int n2 = idx.get_id(1);
		  int sn3 = idx.get_id(2);
		  int gn3 = sn3 + (p * (N3/P) * N1 * N2);
		  int sidx = 2 * ((n1*sS1) + (n2*sS2) + (sn3*sS3));
		  int dbuf = n1/(N1/P);   // distributed by x
		  int dn1 = n1 % (N1/P);  // x coord within dbuf
		  int didx = 2 * ((dn1*dS1) + (n2*dS2) + (gn3*dS3));
		  dest[dbuf][didx+0] = src[sidx+0];
		  dest[dbuf][didx+1] = src[sidx+1];
		});
	    });
	  e.wait_and_throw();
	}
	
	for (int p = 0; p < P; p += 1) {
	  fwd = oneapi::mkl::dft::compute_forward(*d2[p], p2_in[p], p2_out[p]);
	  fwd.wait();
	}
	// copy data back to host

        std::cout<<"\t\tmemcpy to host"<<std::endl;
	for (int p = 0; p < P; p += 1) {
	  queue.memcpy(h2_out[p], p2_out[p], (N1/P)*N2*N3*2*sizeof(float));
	  queue.wait();
	}

        std::cout<<"\t\tmerge split"<<std::endl;
	merge_split(P, h2_out, out, N1, N2, N3);
        usm_result = verify_fwd(out, N1, N2, N3, H1, H2, H3);

        if ((buffer_result == SUCCESS) && (usm_result == SUCCESS))
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
