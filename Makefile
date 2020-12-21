#******************************************************************************
#MIT License
#
#Copyright (c) 2016 Antti-Pekka Hynninen
#Copyright (c) 2016 Oak Ridge National Laboratory (UT-Batelle)
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
#*******************************************************************************

#################### User Settings ####################

# C++ compiler
CC = g++ -fPIC

# CUDA compiler
ifeq ($(nvcc_path),)
	CUDAC = /usr/local/cuda/bin/nvcc -Xcompiler -fPIC
else
	CUDAC = $(nvcc_path) -Xcompiler -fPIC
endif

# Enable nvvp profiling of CPU code by using "make ENABLE_NVTOOLS=1"
# If aligned_alloc() is not available, use "make NO_ALIGNED_ALLOC=1"

# SM versions for which code is generated must be sm_30 and above
GENCODE_SM35  := -gencode arch=compute_35,code=sm_35
GENCODE_SM50  := -gencode arch=compute_50,code=sm_50
GENCODE_SM52  := -gencode arch=compute_52,code=sm_52
GENCODE_SM60  := -gencode arch=compute_60,code=sm_60
GENCODE_SM75  := -gencode arch=compute_75,code=sm_75
GENCODE_FLAGS := $(GENCODE_SM35) $(GENCODE_SM52) $(GENCODE_SM60) $(GENCODE_SM75)
GENCODE_FLAGS := $(NVCC_GENCODE)

#######################################################

# Detect OS
ifeq ($(shell uname -a|grep Linux|wc -l|tr -d ' '), 1)
OS = linux
endif

ifeq ($(shell uname -a|grep titan|wc -l|tr -d ' '), 1)
OS = linux
endif

ifeq ($(shell uname -a|grep Darwin|wc -l|tr -d ' '), 1)
OS = osx
endif

# Detect x86_64 vs. Power
CPU = unknown

ifeq ($(shell uname -a|grep x86_64|wc -l|tr -d ' '), 1)
CPU = x86_64
endif

ifeq ($(shell uname -a|grep ppc64|wc -l|tr -d ' '), 1)
CPU = ppc64
endif

# Set optimization level
OPTLEV = -O3

# Defines
DEFS =

ifdef ENABLE_NVTOOLS
DEFS += -DENABLE_NVTOOLS
endif

ifdef NO_ALIGNED_ALLOC
DEFS += -DNO_ALIGNED_ALLOC
endif

OBJSLIB = build/cutt.o build/cuttplan.o build/cuttkernel.o build/cuttGpuModel.o build/CudaUtils.o build/cuttTimer.o build/cuttGpuModelKernel.o
OBJSTEST = build/cutt_test.o build/TensorTester.o build/CudaUtils.o build/cuttTimer.o
OBJSBENCH = build/cutt_bench.o build/TensorTester.o build/CudaUtils.o build/cuttTimer.o build/CudaMemcpy.o
OBJS = $(OBJSLIB) $(OBJSTEST) $(OBJSBENCH)

#CUDAROOT = $(subst /bin/,,$(dir $(shell which nvcc)))
#CUDAROOT = $(subst /bin/,,$(dir $(shell which $(CUDAC))))

ifeq ($(nvcc_path),)
	CUDAROOT = /usr/local/cuda
else
	CUDAROOT = $(subst /bin/nvcc,, $(nvcc_path))
endif

CFLAGS = -I${CUDAROOT}/include -std=c++11 $(DEFS) $(OPTLEV) 
ifeq ($(CPU),x86_64)
CFLAGS += -march=native
endif

CUDA_CFLAGS = -I${CUDAROOT}/include -std=c++11 $(OPTLEV) -Xptxas -dlcm=ca -lineinfo $(GENCODE_FLAGS) --resource-usage -Xcompiler "$(CUDA_CCFLAGS)" $(DEFS) -D_FORCE_INLINES

ifeq ($(OS),osx)
CUDA_LFLAGS = -L$(CUDAROOT)/lib
else
CUDA_LFLAGS = -L$(CUDAROOT)/lib64
endif

CUDA_LFLAGS += -Llib -lcudart -lcutt
ifdef ENABLE_NVTOOLS
CUDA_LFLAGS += -lnvToolsExt
endif

all: create_build lib/libcutt.so bin/cutt_test bin/cutt_bench

create_build:
	mkdir -p build

lib/libcutt.so: $(OBJSLIB)
	mkdir -p lib
	rm -f lib/libcutt.so
	g++ -fPIC --share -o lib/libcutt.so $(OBJSLIB)
	mkdir -p include
	cp -f src/cutt.h include/cutt.h

bin/cutt_test : lib/libcutt.so $(OBJSTEST)
	mkdir -p bin
	$(CC) -o bin/cutt_test $(OBJSTEST) $(CUDA_LFLAGS)

bin/cutt_bench : lib/libcutt.so $(OBJSBENCH)
	mkdir -p bin
	$(CC) -o bin/cutt_bench $(OBJSBENCH) $(CUDA_LFLAGS)

clean: 
	rm -f $(OBJS)
	rm -f build/*.d
	rm -f *~
	rm -f lib/libcutt.so
	rm -f bin/cutt_test
	rm -f bin/cutt_bench

# Pull in dependencies that already exist
-include $(OBJS:.o=.d)

build/%.o : src/%.cu
	$(CUDAC) -c $(CUDA_CFLAGS) -o build/$*.o $<
	echo -e 'build/\c' > build/$*.d
	$(CUDAC) -M $(CUDA_CFLAGS) $< >> build/$*.d

build/%.o : src/%.cpp
	$(CC) -c $(CFLAGS) -o build/$*.o $<
	echo -e 'build/\c' > build/$*.d
	$(CC) -M $(CFLAGS) $< >> build/$*.d
