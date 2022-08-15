#include "movgetmemcpy.h"
#include <string.h>
#include <stdio.h>
#include <assert.h>

char src[262144];
char dst[262144];



int main(int argc, char *argv[])
{

  size_t src_offset;
  size_t dst_offset;
  size_t count;
  printf("dst %p src %p\n", dst, src);
  for (src_offset = 0; src_offset < 262144; src_offset += 1) {
    src[src_offset] = src_offset & 0xff;
  }
  for (src_offset = 0; src_offset < 127; src_offset += 1) {
    for (dst_offset = 0; dst_offset < 127; dst_offset += 1) {
      for (count = 0; count < 1024; count += 1) {
	memset(dst, 0, count);
	//printf("try dst_offset %lu src_offset %lu count %lu\n", dst_offset, src_offset, count);
	movget_memcpy(&dst[dst_offset], &src[src_offset], count);
	if (memcmp(&dst[dst_offset], &src[src_offset], count)!= 0) {
	  printf("fail dst_offset %lu src_offset %lu count %lu\n",
		 dst_offset, src_offset, count);
	  assert(0);
	  }
      }
    }
  }
  for (src_offset = 0; src_offset < 127; src_offset += 1) {
    for (dst_offset = 0; dst_offset < 127; dst_offset += 1) {
      for (count = 0; count < 1024; count += 1) {
	memset(dst, 0, count);
	//printf("try dst_offset %lu src_offset %lu count %lu\n", dst_offset, src_offset, count);
	movget_memcpy_noavx(&dst[dst_offset], &src[src_offset], count);
	if (memcmp(&dst[dst_offset], &src[src_offset], count)!= 0) {
	  printf("fail dst_offset %lu src_offset %lu count %lu\n",
		 dst_offset, src_offset, count);
	  assert(0);
	  }
      }
    }
  }
  for (src_offset = 0; src_offset < 127; src_offset += 1) {
    for (dst_offset = 0; dst_offset < 127; dst_offset += 1) {
      for (count = 0; count < 1024; count += 1) {
	memset(dst, 0, count);
	//printf("try dst_offset %lu src_offset %lu count %lu\n", dst_offset, src_offset, count);
	avx_memcpy(&dst[dst_offset], &src[src_offset], count);
	if (memcmp(&dst[dst_offset], &src[src_offset], count)!= 0) {
	  printf("fail dst_offset %lu src_offset %lu count %lu\n",
		 dst_offset, src_offset, count);
	  assert(0);
	  }
      }
    }
  }
}
