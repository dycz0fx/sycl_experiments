#include <CL/sycl.hpp>
#include <iostream>
using namespace sycl;

// Array type and data size for this example.
constexpr size_t array_size = (1 << 16);
typedef std::array<int, array_size> IntArray;

double VectorAdd(queue &q, const IntArray &a, const IntArray &b, IntArray &sum) {
   range<1> num_items{a.size()};

   buffer a_buf(a);
   buffer b_buf(b);
   buffer sum_buf(sum.data(), num_items);

   auto t1 = std::chrono::steady_clock::now();   // Start timing

   q.submit([&](handler &h) {
      // Input accessors
      auto a_acc = a_buf.get_access<access::mode::read>(h);
      auto b_acc = b_buf.get_access<access::mode::read>(h);

      // Output accessor
      auto sum_acc = sum_buf.get_access<access::mode::write>(h);

      h.parallel_for(num_items, [=](id<1> i) { sum_acc[i] = a_acc[i] + b_acc[i]; });
   }).wait();

   auto t2 = std::chrono::steady_clock::now();   // Stop timing

   return(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
}

void InitializeArray(IntArray &a) {
   for (size_t i = 0; i < a.size(); i++) a[i] = i;
}

int main() {
  default_selector d_selector;

  IntArray a, b, sum;

  InitializeArray(a);
  InitializeArray(b);

  queue q(d_selector);

  std::cout << "Running on device: "
            << q.get_device().get_info<info::device::name>() << "\n";
  std::cout << "Vector size: " << a.size() << "\n";

  double t = VectorAdd(q, a, b, sum);

  std::cout << "Vector add successfully completed on device in " << t << " microseconds\n";
  return 0;
}
