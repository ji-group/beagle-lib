/*
 *  BeagleGPUActionImpl.h
 *  BEAGLE
 *
 * Copyright 2024 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * BEAGLE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * BEAGLE is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with BEAGLE.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @author Xiang Ji
 * @author Benjamin Redelings
 * @author Marc Suchard
 */


#ifndef BEAGLE_BEAGLEGPUACTIONIMPL_H
#define BEAGLE_BEAGLEGPUACTIONIMPL_H

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include "libhmsbeagle/GPU/BeagleGPUImpl.h"

namespace beagle {
namespace gpu {

#ifdef CUDA
	namespace cuda {
#else
	namespace opencl {
#endif

BEAGLE_GPU_TEMPLATE
class BeagleGPUActionImpl : public BeagleGPUImpl<BEAGLE_GPU_GENERIC> {
};

BEAGLE_GPU_TEMPLATE
class BeagleGPUActionImplFactory : public BeagleGPUImplFactory<BEAGLE_GPU_GENERIC> {
};

} // namspace device
}	// namespace gpu
}	// namespace beagle

#endif
