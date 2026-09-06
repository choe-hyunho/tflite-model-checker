###############################################################################
# model-checker
#
# Simple TensorFlow Lite Micro model checker
###############################################################################

PROJECT := model-checker
TARGET  := $(PROJECT)

###############################################################################
# Target Architecture
###############################################################################

ARCH ?= x86_64

BUILD_TYPE ?= default

ifeq ($(ARCH),x86_64)
TFLM_BUILD := linux_x86_64_$(BUILD_TYPE)_gcc
else ifeq ($(ARCH),arm)
TFLM_BUILD := arm920t_x86_64_$(BUILD_TYPE)_gcc
else
$(error Unsupported ARCH "$(ARCH)")
endif

###############################################################################
# Cross Compiler
###############################################################################

CROSS_COMPILE ?=

CC      := $(CROSS_COMPILE)gcc
CXX     := $(CROSS_COMPILE)g++
AR      := $(CROSS_COMPILE)ar
STRIP   := $(CROSS_COMPILE)strip
FLATC   := ../flatbuffers/build/flatc

###############################################################################
# Directories
###############################################################################

SRCDIR := src
INCDIR := include
METADATA_DIR := metadata
GENERATED_DIR := generated

TFLM_ROOT := ../tflite-micro
TFLM_LIB := $(TFLM_ROOT)/gen/$(TFLM_BUILD)/lib/libtensorflow-microlite.a
FLATBUFFERS_INC := \
    $(TFLM_ROOT)/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include
GEMMLOWP_INC := \
    $(TFLM_ROOT)/tensorflow/lite/micro/tools/make/downloads/gemmlowp

###############################################################################
# Compiler flags
###############################################################################

CC_WARNINGS := \
  -Wsign-compare \
  -Wdouble-promotion \
  -Wunused-variable \
  -Wunused-function \
  -Wswitch \
  -Wvla \
  -Wall \
  -Wextra \
  -Wmissing-field-initializers \
  -Wstrict-aliasing \
  -Wno-unused-parameter

COMMON_FLAGS := \
  -fno-unwind-tables \
  -fno-asynchronous-unwind-tables \
  -ffunction-sections \
  -fdata-sections \
  -fmessage-length=0 \
  -DTF_LITE_STATIC_MEMORY \
  -DTF_LITE_DISABLE_X86_NEON \
  $(CC_WARNINGS)

CFLAGS := \
  -Wimplicit-function-declaration \
  $(COMMON_FLAGS)

CXXFLAGS := \
  -fno-rtti \
  -fno-exceptions \
  -fno-threadsafe-statics \
  -Wnon-virtual-dtor \
  $(COMMON_FLAGS)

CFLAGS += -std=c17
CXXFLAGS += -std=c++17

ifeq ($(BUILD_TYPE),debug)
CFLAGS += -g -O0
CXXFLAGS += -g -O0
else
CFLAGS += -Os
CXXFLAGS += -Os
endif

INCLUDES := \
    -I$(INCDIR) \
    -I$(GENERATED_DIR) \
    -isystem $(TFLM_ROOT) \
    -isystem $(FLATBUFFERS_INC) \
    -isystem $(GEMMLOWP_INC)

LDFLAGS :=

LDLIBS := \
    $(TFLM_LIB)

###############################################################################
# Sources
###############################################################################

SRCS := $(sort $(wildcard $(SRCDIR)/*.cpp))
OBJS := $(SRCS:.cpp=.o)
DEPS := $(OBJS:.o=.d)

TFLITE_METADATA_FBS := $(METADATA_DIR)/metadata_schema.fbs
CONVERSION_METADATA_FBS := $(METADATA_DIR)/conversion_metadata.fbs

TFLITE_METADATA_H := $(GENERATED_DIR)/metadata_schema_generated.h
CONVERSION_METADATA_H := $(GENERATED_DIR)/conversion_metadata_generated.h

###############################################################################
# Rules
###############################################################################

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(TFLITE_METADATA_H): $(TFLITE_METADATA_FBS)
	mkdir -p $(GENERATED_DIR)
	$(FLATC) -c --gen-object-api -o $(GENERATED_DIR) $<

$(CONVERSION_METADATA_H): $(CONVERSION_METADATA_FBS)
	mkdir -p $(GENERATED_DIR)
	$(FLATC) -c --gen-object-api -o $(GENERATED_DIR) $<

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

src/metadata_decoder.o: $(TFLITE_METADATA_H) $(CONVERSION_METADATA_H)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

strip: $(TARGET)
	$(STRIP) $(TARGET)

clean:
	rm -f $(TFLITE_METADATA_H) $(CONVERSION_METADATA_H) $(OBJS) $(TARGET)

.PHONY: all clean strip