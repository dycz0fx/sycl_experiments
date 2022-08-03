/*
 * define a macro MARKER(id) that works with either SSC marks
 * or Simics magic marker instructions, or possibly others in the future.
 *
 * C. Beckmann (c) Intel, Nov. 2017
 */
#ifndef __MAGIC_MARKER_H__
#define __MAGIC_MARKER_H__

# ifdef SIMICS

#  include "magic-instruction.h"
#  define MARKER(_x_) MAGIC(_x_)

#  define ROI_START_MARKER 0x4321
#  define ROI_END_MARKER   0x5321


# else

#  ifndef __ICC
#   include "ssc_mark.h"
#  endif
#  define MARKER(_x_) __SSC_MARK(_x_)

#  define ROI_START_MARKER 0x44332211
#  define ROI_END_MARKER   0x55332211

# endif

#endif
