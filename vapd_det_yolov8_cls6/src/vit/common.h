#pragma once

#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "NvOnnxParser.h"

#include <cuda_runtime_api.h>

#include <iostream>
#include <assert.h>
#include <numeric>
#include <memory>

//#define IS_DEBUG 1

#define TIMERSTART(tag)  auto tag##_start = std::chrono::steady_clock::now(),tag##_end = tag##_start
#define TIMEREND(tag)  tag##_end =  std::chrono::steady_clock::now()
#define DURATION_s(tag) (std::chrono::duration_cast<std::chrono::seconds>(tag##_end - tag##_start).count())
#define DURATION_ms(tag) (std::chrono::duration_cast<std::chrono::milliseconds>(tag##_end - tag##_start).count() )
#define DURATION_us(tag) (std::chrono::duration_cast<std::chrono::microseconds>(tag##_end - tag##_start).count()))
#define DURATION_ns(tag) (std::chrono::duration_cast<std::chrono::nanoseconds>(tag##_end - tag##_start).count()))

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        delete obj;
    }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, InferDeleter>;

using namespace nvinfer1;

class nv_logger : public nvinfer1::ILogger
{
public:
    nv_logger(Severity severity = Severity::kWARNING)
            : reportableSeverity(severity)
    {
    }

    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress messages with severity enum value greater than the reportable
        if (severity > reportableSeverity)
            return;

        switch (severity)
        {
            case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
            case Severity::kERROR: std::cerr << "ERROR: "; break;
            case Severity::kWARNING: std::cerr << "WARNING: "; break;
            case Severity::kINFO: std::cerr << "INFO: "; break;
            default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << __FILE__ << ":" << __LINE__ << msg << std::endl;
    }
    Severity reportableSeverity;
};

int64_t volume(const nvinfer1::Dims& d);

unsigned int
getElementSize(nvinfer1::DataType t);