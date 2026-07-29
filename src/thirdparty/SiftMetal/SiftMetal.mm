// SiftMetal.mm - Metal-accelerated SIFT feature extraction.
// Objective-C++ port of SIFTMetal Swift library by Luke Van In.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "SiftMetal.h"

// Shared C headers for Metal shader parameter structs.
#include "include/ConvolutionSeries.h"
#include "include/NearestNeighbor.h"
#include "include/SIFTDescriptor.h"
#include "include/SIFTExtrema.h"
#include "include/SIFTInterpolate.h"
#include "include/SIFTOrientation.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

// Path to compiled metallib is set by CMake and embedded here.
#ifndef SIFT_METAL_METALLIB_PATH
#define SIFT_METAL_METALLIB_PATH ""
#endif

namespace sift_metal {

static constexpr int kMaxExtrema = 4096;
static constexpr int kMaxKeypoints = 4096;
static constexpr int kMaxDescriptors = 8192;

// ---------------------------------------------------------------------------
// Helper: compute 1D Gaussian kernel weights.
// ---------------------------------------------------------------------------
static std::vector<float> GaussianWeights(float sigma) {
  int radius = static_cast<int>(std::ceil(4.0f * sigma));
  int size = radius * 2 + 1;
  std::vector<float> weights(size);
  float sum = 0.0f;
  float ss = sigma * sigma;
  for (int k = -radius; k <= radius; ++k) {
    float w = std::exp(-0.5f * (float(k * k) / ss));
    weights[k + radius] = w;
    sum += w;
  }
  for (auto& w : weights) w /= sum;
  return weights;
}

// ---------------------------------------------------------------------------
// Octave: manages textures and pipelines for one octave of the pyramid.
// ---------------------------------------------------------------------------
struct Octave {
  int o;                 // octave index
  float delta;           // sampling distance
  int width, height;     // dimensions at this octave
  int num_scales;        // scales per octave (typically 3)
  std::vector<float> sigmas;  // sigma values for each gaussian

  id<MTLTexture> gaussianTextures;   // 2DArray [num_scales+3]
  id<MTLTexture> differenceTextures; // 2DArray [num_scales+2]
  id<MTLTexture> gradientTextures;   // 2DArray, rg32Float

  // Buffers for extrema detection
  id<MTLBuffer> extremaOutputBuffer;
  id<MTLBuffer> extremaIndexBuffer;

  // Buffers for interpolation
  id<MTLBuffer> interpolateInputBuffer;
  id<MTLBuffer> interpolateOutputBuffer;
  id<MTLBuffer> interpolateParamsBuffer;

  // Buffers for orientation
  id<MTLBuffer> orientationInputBuffer;
  id<MTLBuffer> orientationOutputBuffer;
  id<MTLBuffer> orientationParamsBuffer;

  // Buffers for descriptors
  id<MTLBuffer> descriptorInputBuffer;
  id<MTLBuffer> descriptorOutputBuffer;
  id<MTLBuffer> descriptorParamsBuffer;

  // Convolution kernel weights buffers for Gaussian series blur
  struct ConvPair {
    id<MTLBuffer> paramsX;
    id<MTLBuffer> paramsY;
  };
  std::vector<ConvPair> convPairs;
  id<MTLTexture> convWorkTexture; // private storage 2DArray[1]
};

// ---------------------------------------------------------------------------
// SiftMetalExtractorImpl
// ---------------------------------------------------------------------------
class SiftMetalExtractorImpl {
 public:
  bool Init(const Options& opts, int max_w, int max_h);
  bool Extract(const uint8_t* data, int w, int h, ExtractResult* result);

 private:
  void SetupOctaves(int w, int h);
  void SetupOctave(Octave& oct, int o, float delta, int w, int h,
                   int num_scales, const std::vector<float>& sigmas);

  // Pipeline encoding helpers
  void EncodeGrayscaleUpload(id<MTLCommandBuffer> cb, int w, int h);
  void EncodeSeedTexture(id<MTLCommandBuffer> cb);
  void EncodeOctave(id<MTLCommandBuffer> cb, Octave& oct,
                    id<MTLTexture> inputTexture, bool inputIs2D);
  void EncodeGaussianSeries(id<MTLCommandBuffer> cb, Octave& oct);
  void EncodeDifferences(id<MTLCommandBuffer> cb, Octave& oct);
  void EncodeGradients(id<MTLCommandBuffer> cb, Octave& oct);
  void EncodeExtrema(id<MTLCommandBuffer> cb, Octave& oct);

  // Per-octave extraction
  int ReadExtremaCount(Octave& oct);
  void InterpolateKeypoints(Octave& oct, int extrema_count);
  void ComputeOrientations(Octave& oct,
                           const std::vector<Keypoint>& keypoints,
                           std::vector<std::pair<int, float>>& oriented);
  void ComputeDescriptors(Octave& oct,
                          const std::vector<Keypoint>& keypoints,
                          const std::vector<std::pair<int, float>>& oriented,
                          ExtractResult* result);

  // Metal objects
  id<MTLDevice> device_;
  id<MTLCommandQueue> commandQueue_;
  id<MTLLibrary> library_;

  // Compute pipelines
  id<MTLComputePipelineState> bilinearUpScalePipeline_;
  id<MTLComputePipelineState> nearestNeighborDownScalePipeline_;
  id<MTLComputePipelineState> convolutionXPipeline_;
  id<MTLComputePipelineState> convolutionYPipeline_;
  id<MTLComputePipelineState> convolutionSeriesXPipeline_;
  id<MTLComputePipelineState> convolutionSeriesYPipeline_;
  id<MTLComputePipelineState> subtractPipeline_;
  id<MTLComputePipelineState> siftGradientPipeline_;
  id<MTLComputePipelineState> siftExtremaListPipeline_;
  id<MTLComputePipelineState> siftInterpolatePipeline_;
  id<MTLComputePipelineState> siftOrientationPipeline_;
  id<MTLComputePipelineState> siftDescriptorsPipeline_;

  // Seed textures
  id<MTLTexture> luminosityTexture_;  // R32Float, input size
  id<MTLTexture> scaledTexture_;      // R32Float, seed size (2x)
  id<MTLTexture> seedTexture_;        // R32Float, seed size (2x)
  id<MTLTexture> seedConvWorkTexture_; // R32Float, seed size, private

  // Seed Gaussian blur convolution buffers
  id<MTLBuffer> seedConvWeightsBuffer_;
  id<MTLBuffer> seedConvParamsBuffer_;

  // Octaves
  std::vector<Octave> octaves_;

  // Options
  Options options_;
  float sigma_min_ = 0.8f;
  float delta_min_ = 0.5f;
  float sigma_input_ = 0.5f;
  int input_w_ = 0, input_h_ = 0;
  int seed_w_ = 0, seed_h_ = 0;

  // Upload buffer
  id<MTLBuffer> uploadBuffer_;
};

// ---------------------------------------------------------------------------
// Pipeline creation helper
// ---------------------------------------------------------------------------
static id<MTLComputePipelineState> MakePipeline(id<MTLDevice> device,
                                                 id<MTLLibrary> library,
                                                 const char* name) {
  NSString* nsName = [NSString stringWithUTF8String:name];
  id<MTLFunction> func = [library newFunctionWithName:nsName];
  if (!func) {
    NSLog(@"SiftMetal: Failed to find function '%s'", name);
    return nil;
  }
  NSError* error = nil;
  id<MTLComputePipelineState> ps =
      [device newComputePipelineStateWithFunction:func error:&error];
  if (error) {
    NSLog(@"SiftMetal: Pipeline creation error for '%s': %@", name, error);
  }
  return ps;
}

static id<MTLTexture> MakeTexture2D(id<MTLDevice> device, int w, int h,
                                     MTLPixelFormat fmt,
                                     MTLStorageMode storage) {
  MTLTextureDescriptor* desc =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fmt
                                                        width:w
                                                       height:h
                                                    mipmapped:NO];
  desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  desc.storageMode = storage;
  return [device newTextureWithDescriptor:desc];
}

static id<MTLTexture> MakeTexture2DArray(id<MTLDevice> device, int w, int h,
                                          int arrayLen, MTLPixelFormat fmt,
                                          MTLStorageMode storage) {
  MTLTextureDescriptor* desc = [[MTLTextureDescriptor alloc] init];
  desc.textureType = MTLTextureType2DArray;
  desc.pixelFormat = fmt;
  desc.width = w;
  desc.height = h;
  desc.arrayLength = arrayLen;
  desc.mipmapLevelCount = 1;
  desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  desc.storageMode = storage;
  return [device newTextureWithDescriptor:desc];
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
bool SiftMetalExtractorImpl::Init(const Options& opts, int max_w, int max_h) {
  options_ = opts;

  // Get the default Metal device.
  device_ = MTLCreateSystemDefaultDevice();
  if (!device_) return false;

  commandQueue_ = [device_ newCommandQueue];
  if (!commandQueue_) return false;

  // Load the pre-compiled metal library.
  NSError* error = nil;
  NSString* libPath =
      [NSString stringWithUTF8String:SIFT_METAL_METALLIB_PATH];
  if (libPath.length > 0) {
    NSURL* libURL = [NSURL fileURLWithPath:libPath];
    library_ = [device_ newLibraryWithURL:libURL error:&error];
  }
  if (!library_) {
    // Fallback: try default library.
    library_ = [device_ newDefaultLibrary];
  }
  if (!library_) {
    NSLog(@"SiftMetal: Failed to load Metal library: %@", error);
    return false;
  }

  // Create all compute pipelines.
  bilinearUpScalePipeline_ = MakePipeline(device_, library_, "bilinearUpScale");
  nearestNeighborDownScalePipeline_ =
      MakePipeline(device_, library_, "nearestNeighborDownScale");
  convolutionXPipeline_ = MakePipeline(device_, library_, "convolutionX");
  convolutionYPipeline_ = MakePipeline(device_, library_, "convolutionY");
  convolutionSeriesXPipeline_ =
      MakePipeline(device_, library_, "convolutionSeriesX");
  convolutionSeriesYPipeline_ =
      MakePipeline(device_, library_, "convolutionSeriesY");
  subtractPipeline_ = MakePipeline(device_, library_, "subtract");
  siftGradientPipeline_ = MakePipeline(device_, library_, "siftGradient");
  siftExtremaListPipeline_ =
      MakePipeline(device_, library_, "siftExtremaList");
  siftInterpolatePipeline_ =
      MakePipeline(device_, library_, "siftInterpolate");
  siftOrientationPipeline_ =
      MakePipeline(device_, library_, "siftOrientation");
  siftDescriptorsPipeline_ =
      MakePipeline(device_, library_, "siftDescriptors");

  if (!bilinearUpScalePipeline_ || !subtractPipeline_ ||
      !siftExtremaListPipeline_ || !siftInterpolatePipeline_ ||
      !siftOrientationPipeline_ || !siftDescriptorsPipeline_) {
    return false;
  }

  // Determine seed size based on first_octave.
  if (opts.first_octave == -1) {
    delta_min_ = 0.5f;
  } else {
    delta_min_ = 1.0f;
  }

  input_w_ = max_w;
  input_h_ = max_h;
  seed_w_ = static_cast<int>(float(max_w) / delta_min_);
  seed_h_ = static_cast<int>(float(max_h) / delta_min_);

  // Create textures for the seed stage.
  luminosityTexture_ = MakeTexture2D(device_, max_w, max_h,
                                      MTLPixelFormatR32Float,
                                      MTLStorageModeShared);
  scaledTexture_ = MakeTexture2D(device_, seed_w_, seed_h_,
                                  MTLPixelFormatR32Float,
                                  MTLStorageModeShared);
  seedTexture_ = MakeTexture2D(device_, seed_w_, seed_h_,
                                MTLPixelFormatR32Float,
                                MTLStorageModeShared);
  seedConvWorkTexture_ = MakeTexture2D(device_, seed_w_, seed_h_,
                                        MTLPixelFormatR32Float,
                                        MTLStorageModePrivate);

  // Compute seed Gaussian blur kernel.
  float sigma_seed = std::sqrt(sigma_min_ * sigma_min_ -
                                sigma_input_ * sigma_input_) / delta_min_;
  auto seedWeights = GaussianWeights(sigma_seed);
  seedConvWeightsBuffer_ =
      [device_ newBufferWithBytes:seedWeights.data()
                           length:seedWeights.size() * sizeof(float)
                          options:MTLResourceStorageModeShared];
  uint32_t seedWeightCount = static_cast<uint32_t>(seedWeights.size());
  seedConvParamsBuffer_ =
      [device_ newBufferWithBytes:&seedWeightCount
                           length:sizeof(uint32_t)
                          options:MTLResourceStorageModeShared];

  // Upload buffer large enough for max image.
  size_t maxPixels = (size_t)max_w * max_h;
  uploadBuffer_ =
      [device_ newBufferWithLength:maxPixels * sizeof(float)
                           options:MTLResourceStorageModeShared];

  // Setup octaves.
  SetupOctaves(max_w, max_h);

  return true;
}

// ---------------------------------------------------------------------------
// SetupOctaves
// ---------------------------------------------------------------------------
void SiftMetalExtractorImpl::SetupOctaves(int w, int h) {
  int num_octaves = options_.num_octaves;
  if (num_octaves <= 0) {
    // Match SiftGPU's octave count: floor(log2(min(w,h))) - 3
    // But applied to the seed image dimensions (after upscaling).
    int seed_min = std::min(
        static_cast<int>(float(w) / delta_min_),
        static_cast<int>(float(h) / delta_min_));
    num_octaves = static_cast<int>(
        std::floor(std::log2(float(seed_min)))) - 3;
    num_octaves = std::max(1, num_octaves);
  }

  octaves_.resize(num_octaves);
  for (int o = 0; o < num_octaves; ++o) {
    float delta = delta_min_ * std::pow(2.0f, float(o));
    int ow = static_cast<int>(float(w) / delta);
    int oh = static_cast<int>(float(h) / delta);
    if (ow < 8 || oh < 8) {
      octaves_.resize(o);
      break;
    }

    int ns = options_.scales_per_octave;
    std::vector<float> sigmas;
    for (int s = 0; s < ns + 3; ++s) {
      float ratio = delta / delta_min_;
      float scale = std::pow(2.0f, float(s) / float(ns));
      sigmas.push_back(ratio * sigma_min_ * scale);
    }

    SetupOctave(octaves_[o], o, delta, ow, oh, ns, sigmas);
  }
}

void SiftMetalExtractorImpl::SetupOctave(Octave& oct, int o, float delta,
                                          int w, int h, int num_scales,
                                          const std::vector<float>& sigmas) {
  oct.o = o;
  oct.delta = delta;
  oct.width = w;
  oct.height = h;
  oct.num_scales = num_scales;
  oct.sigmas = sigmas;

  int numGaussians = num_scales + 3;
  int numDifferences = num_scales + 2;

  oct.gaussianTextures = MakeTexture2DArray(
      device_, w, h, numGaussians, MTLPixelFormatR32Float,
      MTLStorageModeShared);
  oct.differenceTextures = MakeTexture2DArray(
      device_, w, h, numDifferences, MTLPixelFormatR32Float,
      MTLStorageModeShared);
  oct.gradientTextures = MakeTexture2DArray(
      device_, w, h, numGaussians, MTLPixelFormatRG32Float,
      MTLStorageModeShared);

  // Convolution work texture (single-slice private).
  oct.convWorkTexture = MakeTexture2DArray(
      device_, w, h, 1, MTLPixelFormatR32Float, MTLStorageModePrivate);

  // Build convolution parameter buffers for Gaussian series.
  oct.convPairs.resize(numGaussians - 1);
  for (int s = 1; s < numGaussians; ++s) {
    float sa = sigmas[s - 1];
    float sb = sigmas[s];
    float rho = std::sqrt(sb * sb - sa * sa) / delta;
    auto weights = GaussianWeights(rho);

    // X pass: read from slice [s-1], write to work slice [0]
    ConvolutionParameters paramsX = {};
    paramsX.inputDepth = static_cast<int32_t>(s - 1);
    paramsX.outputDepth = 0;
    paramsX.count = static_cast<int32_t>(weights.size());
    std::memcpy(paramsX.weights, weights.data(),
                std::min(weights.size(), (size_t)CONVOLUTION_WEIGHTS_LENGTH) *
                    sizeof(float));
    oct.convPairs[s - 1].paramsX =
        [device_ newBufferWithBytes:&paramsX
                             length:sizeof(ConvolutionParameters)
                            options:MTLResourceStorageModeShared];

    // Y pass: read from work slice [0], write to slice [s]
    ConvolutionParameters paramsY = {};
    paramsY.inputDepth = 0;
    paramsY.outputDepth = static_cast<int32_t>(s);
    paramsY.count = static_cast<int32_t>(weights.size());
    std::memcpy(paramsY.weights, weights.data(),
                std::min(weights.size(), (size_t)CONVOLUTION_WEIGHTS_LENGTH) *
                    sizeof(float));
    oct.convPairs[s - 1].paramsY =
        [device_ newBufferWithBytes:&paramsY
                             length:sizeof(ConvolutionParameters)
                            options:MTLResourceStorageModeShared];
  }

  // Extrema buffers
  oct.extremaOutputBuffer =
      [device_ newBufferWithLength:kMaxExtrema * sizeof(SIFTExtremaResult)
                           options:MTLResourceStorageModeShared];
  oct.extremaIndexBuffer =
      [device_ newBufferWithLength:sizeof(int32_t)
                           options:MTLResourceStorageModeShared];

  // Interpolation buffers
  oct.interpolateInputBuffer =
      [device_ newBufferWithLength:kMaxKeypoints *
                                       sizeof(SIFTInterpolateInputKeypoint)
                           options:MTLResourceStorageModeShared];
  oct.interpolateOutputBuffer =
      [device_ newBufferWithLength:kMaxKeypoints *
                                       sizeof(SIFTInterpolateOutputKeypoint)
                           options:MTLResourceStorageModeShared];
  oct.interpolateParamsBuffer =
      [device_ newBufferWithLength:sizeof(SIFTInterpolateParameters)
                           options:MTLResourceStorageModeShared];

  // Orientation buffers
  oct.orientationInputBuffer =
      [device_ newBufferWithLength:kMaxKeypoints *
                                       sizeof(SIFTOrientationKeypoint)
                           options:MTLResourceStorageModeShared];
  oct.orientationOutputBuffer =
      [device_ newBufferWithLength:kMaxKeypoints *
                                       sizeof(SIFTOrientationResult)
                           options:MTLResourceStorageModeShared];
  oct.orientationParamsBuffer =
      [device_ newBufferWithLength:sizeof(SIFTOrientationParameters)
                           options:MTLResourceStorageModeShared];

  // Descriptor buffers
  oct.descriptorInputBuffer =
      [device_ newBufferWithLength:kMaxDescriptors * sizeof(SIFTDescriptorInput)
                           options:MTLResourceStorageModeShared];
  oct.descriptorOutputBuffer =
      [device_ newBufferWithLength:kMaxDescriptors *
                                       sizeof(SIFTDescriptorResult)
                           options:MTLResourceStorageModeShared];
  oct.descriptorParamsBuffer =
      [device_ newBufferWithLength:sizeof(SIFTDescriptorParameters)
                           options:MTLResourceStorageModeShared];
}

// ---------------------------------------------------------------------------
// Extract
// ---------------------------------------------------------------------------
bool SiftMetalExtractorImpl::Extract(const uint8_t* data, int w, int h,
                                      ExtractResult* result) {
  result->keypoints.clear();
  result->descriptors.clear();

  // Recreate textures if image size changed.
  if (w != input_w_ || h != input_h_) {
    input_w_ = w;
    input_h_ = h;
    seed_w_ = static_cast<int>(float(w) / delta_min_);
    seed_h_ = static_cast<int>(float(h) / delta_min_);

    luminosityTexture_ = MakeTexture2D(device_, w, h,
                                        MTLPixelFormatR32Float,
                                        MTLStorageModeShared);
    scaledTexture_ = MakeTexture2D(device_, seed_w_, seed_h_,
                                    MTLPixelFormatR32Float,
                                    MTLStorageModeShared);
    seedTexture_ = MakeTexture2D(device_, seed_w_, seed_h_,
                                  MTLPixelFormatR32Float,
                                  MTLStorageModeShared);
    seedConvWorkTexture_ = MakeTexture2D(device_, seed_w_, seed_h_,
                                          MTLPixelFormatR32Float,
                                          MTLStorageModePrivate);

    size_t maxPixels = (size_t)w * h;
    if (uploadBuffer_.length < maxPixels * sizeof(float)) {
      uploadBuffer_ =
          [device_ newBufferWithLength:maxPixels * sizeof(float)
                               options:MTLResourceStorageModeShared];
    }

    SetupOctaves(w, h);
  }

  // Convert uint8 grayscale to float and upload to luminosity texture.
  float* uploadPtr = static_cast<float*>(uploadBuffer_.contents);
  size_t npixels = (size_t)w * h;
  for (size_t i = 0; i < npixels; ++i) {
    uploadPtr[i] = static_cast<float>(data[i]) / 255.0f;
  }
  MTLRegion region = MTLRegionMake2D(0, 0, w, h);
  [luminosityTexture_ replaceRegion:region
                        mipmapLevel:0
                          withBytes:uploadPtr
                        bytesPerRow:w * sizeof(float)];

  // Phase 1: Build scale-space pyramid (DoG + gradients + extrema).
  {
    id<MTLCommandBuffer> cb = [commandQueue_ commandBuffer];
    EncodeSeedTexture(cb);

    // First octave reads from seed texture (2D).
    EncodeOctave(cb, octaves_[0], seedTexture_, true);

    // Subsequent octaves read from previous octave's Gaussian textures.
    for (size_t i = 1; i < octaves_.size(); ++i) {
      EncodeOctave(cb, octaves_[i],
                   octaves_[i - 1].gaussianTextures, false);
    }

    [cb commit];
    [cb waitUntilCompleted];
  }

  // Phase 2: For each octave, read extrema, interpolate, orientate, describe.
  for (auto& oct : octaves_) {
    int extremaCount = ReadExtremaCount(oct);
    if (extremaCount <= 0) continue;
    extremaCount = std::min(extremaCount, kMaxExtrema);

    InterpolateKeypoints(oct, extremaCount);

    // Read interpolated keypoints.
    auto* interpOut = static_cast<SIFTInterpolateOutputKeypoint*>(
        oct.interpolateOutputBuffer.contents);
    float sigmaRatio = oct.sigmas[1] / oct.sigmas[0];

    std::vector<Keypoint> octKeypoints;
    for (int k = 0; k < extremaCount; ++k) {
      auto& p = interpOut[k];
      if (!p.converged) continue;

      Keypoint kp;
      kp.x = p.absoluteX;
      kp.y = p.absoluteY;
      kp.sigma = oct.sigmas[p.scale] * std::pow(sigmaRatio, p.subScale);
      kp.orientation = 0; // Will be set during orientation pass.
      octKeypoints.push_back(kp);
    }

    if (octKeypoints.empty()) continue;

    // Compute orientations.
    std::vector<std::pair<int, float>> oriented; // (keypoint_index, theta)
    ComputeOrientations(oct, octKeypoints, oriented);

    if (oriented.empty()) continue;

    // Compute descriptors.
    ComputeDescriptors(oct, octKeypoints, oriented, result);
  }

  // Sort by scale (descending) and truncate to max_num_features.
  if (options_.max_num_features > 0 &&
      (int)result->keypoints.size() > options_.max_num_features) {
    // Create index array, sort by sigma descending.
    std::vector<int> indices(result->keypoints.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](int a, int b) {
                return result->keypoints[a].sigma >
                       result->keypoints[b].sigma;
              });
    indices.resize(options_.max_num_features);
    std::sort(indices.begin(), indices.end()); // Restore order.

    std::vector<Keypoint> newKp;
    std::vector<float> newDesc;
    newKp.reserve(options_.max_num_features);
    newDesc.reserve(options_.max_num_features * 128);
    for (int idx : indices) {
      newKp.push_back(result->keypoints[idx]);
      newDesc.insert(newDesc.end(),
                     result->descriptors.begin() + idx * 128,
                     result->descriptors.begin() + (idx + 1) * 128);
    }
    result->keypoints = std::move(newKp);
    result->descriptors = std::move(newDesc);
  }

  return true;
}

// ---------------------------------------------------------------------------
// EncodeSeedTexture: upscale grayscale input + Gaussian blur.
// ---------------------------------------------------------------------------
void SiftMetalExtractorImpl::EncodeSeedTexture(id<MTLCommandBuffer> cb) {
  // Bilinear upscale luminosity → scaled.
  if (seed_w_ != input_w_ || seed_h_ != input_h_) {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:bilinearUpScalePipeline_];
    [enc setTexture:scaledTexture_ atIndex:0];
    [enc setTexture:luminosityTexture_ atIndex:1];
    MTLSize tg = {16, 16, 1};
    MTLSize grid = {
        (NSUInteger)(seed_w_ + 15) / 16,
        (NSUInteger)(seed_h_ + 15) / 16, 1};
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
  } else {
    // Same size: just copy.
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromTexture:luminosityTexture_ toTexture:scaledTexture_];
    [blit endEncoding];
  }

  // Gaussian blur scaled → seed (separable 1D convolution).
  {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:convolutionXPipeline_];
    [enc setTexture:seedConvWorkTexture_ atIndex:0];
    [enc setTexture:scaledTexture_ atIndex:1];
    [enc setBuffer:seedConvWeightsBuffer_ offset:0 atIndex:0];
    [enc setBuffer:seedConvParamsBuffer_ offset:0 atIndex:1];
    MTLSize tg = {16, 16, 1};
    MTLSize grid = {(NSUInteger)(seed_w_ + 15) / 16,
                    (NSUInteger)(seed_h_ + 15) / 16, 1};
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
  }
  {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:convolutionYPipeline_];
    [enc setTexture:seedTexture_ atIndex:0];
    [enc setTexture:seedConvWorkTexture_ atIndex:1];
    [enc setBuffer:seedConvWeightsBuffer_ offset:0 atIndex:0];
    [enc setBuffer:seedConvParamsBuffer_ offset:0 atIndex:1];
    MTLSize tg = {16, 16, 1};
    MTLSize grid = {(NSUInteger)(seed_w_ + 15) / 16,
                    (NSUInteger)(seed_h_ + 15) / 16, 1};
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
  }
}

// ---------------------------------------------------------------------------
// EncodeOctave
// ---------------------------------------------------------------------------
void SiftMetalExtractorImpl::EncodeOctave(id<MTLCommandBuffer> cb,
                                           Octave& oct,
                                           id<MTLTexture> inputTexture,
                                           bool inputIs2D) {
  int w = oct.width;
  int h = oct.height;

  // Copy/scale input into gaussian slice 0.
  if (inputIs2D) {
    // 2D texture → first slice of 2DArray.
    if ((int)inputTexture.width == w && (int)inputTexture.height == h) {
      id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
      [blit copyFromTexture:inputTexture
                sourceSlice:0
                sourceLevel:0
              sourceOrigin:MTLOriginMake(0, 0, 0)
                sourceSize:MTLSizeMake(w, h, 1)
                 toTexture:oct.gaussianTextures
          destinationSlice:0
          destinationLevel:0
         destinationOrigin:MTLOriginMake(0, 0, 0)];
      [blit endEncoding];
    }
  } else {
    // Nearest-neighbor downscale from previous octave's gaussian[num_scales].
    auto* params = static_cast<NearestNeighborScaleParameters*>(
        [device_ newBufferWithLength:sizeof(NearestNeighborScaleParameters)
                             options:MTLResourceStorageModeShared].contents);
    id<MTLBuffer> paramsBuf =
        [device_ newBufferWithLength:sizeof(NearestNeighborScaleParameters)
                             options:MTLResourceStorageModeShared];
    auto* p = static_cast<NearestNeighborScaleParameters*>(paramsBuf.contents);
    p->inputSlice = oct.num_scales;
    p->outputSlice = 0;

    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:nearestNeighborDownScalePipeline_];
    [enc setTexture:oct.gaussianTextures atIndex:0];
    [enc setTexture:inputTexture atIndex:1];
    [enc setBuffer:paramsBuf offset:0 atIndex:0];
    MTLSize tg = {16, 16, 1};
    MTLSize grid = {(NSUInteger)(w + 15) / 16,
                    (NSUInteger)(h + 15) / 16, 1};
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
  }

  // Gaussian series blur.
  EncodeGaussianSeries(cb, oct);
  // Differences.
  EncodeDifferences(cb, oct);
  // Gradients.
  EncodeGradients(cb, oct);
  // Extrema detection.
  EncodeExtrema(cb, oct);
}

void SiftMetalExtractorImpl::EncodeGaussianSeries(id<MTLCommandBuffer> cb,
                                                    Octave& oct) {
  int w = oct.width;
  int h = oct.height;
  MTLSize tg = {16, 16, 1};
  MTLSize grid = {(NSUInteger)(w + 15) / 16,
                  (NSUInteger)(h + 15) / 16, 1};

  for (auto& pair : oct.convPairs) {
    // X pass: gaussian → work
    {
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:convolutionSeriesXPipeline_];
      [enc setTexture:oct.convWorkTexture atIndex:0];
      [enc setTexture:oct.gaussianTextures atIndex:1];
      [enc setBuffer:pair.paramsX offset:0 atIndex:0];
      [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
      [enc endEncoding];
    }
    // Y pass: work → gaussian
    {
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:convolutionSeriesYPipeline_];
      [enc setTexture:oct.gaussianTextures atIndex:0];
      [enc setTexture:oct.convWorkTexture atIndex:1];
      [enc setBuffer:pair.paramsY offset:0 atIndex:0];
      [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
      [enc endEncoding];
    }
  }
}

void SiftMetalExtractorImpl::EncodeDifferences(id<MTLCommandBuffer> cb,
                                                 Octave& oct) {
  int w = oct.width;
  int h = oct.height;
  int numDiff = oct.num_scales + 2;

  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:subtractPipeline_];
  [enc setTexture:oct.differenceTextures atIndex:0];
  [enc setTexture:oct.gaussianTextures atIndex:1];
  MTLSize tg = {8, 8, 8};
  MTLSize grid = {(NSUInteger)(w + 7) / 8,
                  (NSUInteger)(h + 7) / 8,
                  (NSUInteger)(numDiff + 7) / 8};
  [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  [enc endEncoding];
}

void SiftMetalExtractorImpl::EncodeGradients(id<MTLCommandBuffer> cb,
                                               Octave& oct) {
  int w = oct.width;
  int h = oct.height;
  int arrayLen = oct.num_scales + 3;

  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:siftGradientPipeline_];
  [enc setTexture:oct.gradientTextures atIndex:0];
  [enc setTexture:oct.gaussianTextures atIndex:1];
  MTLSize tg = {8, 8, 8};
  MTLSize grid = {(NSUInteger)(w + 7) / 8,
                  (NSUInteger)(h + 7) / 8,
                  (NSUInteger)(arrayLen + 7) / 8};
  [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  [enc endEncoding];
}

void SiftMetalExtractorImpl::EncodeExtrema(id<MTLCommandBuffer> cb,
                                             Octave& oct) {
  int w = oct.width;
  int h = oct.height;
  int numDiff = oct.num_scales + 2;

  // Reset index counter.
  auto* idx = static_cast<int32_t*>(oct.extremaIndexBuffer.contents);
  *idx = 0;

  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:siftExtremaListPipeline_];
  [enc setBuffer:oct.extremaOutputBuffer offset:0 atIndex:0];
  [enc setBuffer:oct.extremaIndexBuffer offset:0 atIndex:1];
  [enc setTexture:oct.differenceTextures atIndex:0];

  NSUInteger maxThreads = siftExtremaListPipeline_.maxTotalThreadsPerThreadgroup;
  NSUInteger dim = (NSUInteger)std::cbrt((double)maxThreads);
  MTLSize tg = {dim, dim, dim};
  MTLSize gridSize = {(NSUInteger)(w - 2),
                      (NSUInteger)(h - 2),
                      (NSUInteger)(numDiff - 2)};
  [enc dispatchThreads:gridSize threadsPerThreadgroup:tg];
  [enc endEncoding];
}

// ---------------------------------------------------------------------------
// ReadExtremaCount
// ---------------------------------------------------------------------------
int SiftMetalExtractorImpl::ReadExtremaCount(Octave& oct) {
  auto* idx = static_cast<int32_t*>(oct.extremaIndexBuffer.contents);
  int count = static_cast<int>(*idx);
  *idx = 0;
  return count;
}

// ---------------------------------------------------------------------------
// InterpolateKeypoints
// ---------------------------------------------------------------------------
void SiftMetalExtractorImpl::InterpolateKeypoints(Octave& oct,
                                                    int extremaCount) {
  int count = std::min(extremaCount, kMaxKeypoints);

  // Copy extrema to interpolation input buffer.
  auto* extrema =
      static_cast<SIFTExtremaResult*>(oct.extremaOutputBuffer.contents);
  auto* interpIn = static_cast<SIFTInterpolateInputKeypoint*>(
      oct.interpolateInputBuffer.contents);
  for (int i = 0; i < count; ++i) {
    interpIn[i].x = extrema[i].x;
    interpIn[i].y = extrema[i].y;
    interpIn[i].scale = extrema[i].scale;
  }

  // Set interpolation parameters.
  auto* params = static_cast<SIFTInterpolateParameters*>(
      oct.interpolateParamsBuffer.contents);
  params->dogThreshold = options_.peak_threshold;
  params->maxIterations = 5;
  params->maxOffset = 0.6f;
  params->width = static_cast<int32_t>(oct.width);
  params->height = static_cast<int32_t>(oct.height);
  params->octaveDelta = oct.delta;
  params->edgeThreshold = options_.edge_threshold;
  params->numberOfScales = static_cast<int32_t>(oct.num_scales);

  id<MTLCommandBuffer> cb = [commandQueue_ commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:siftInterpolatePipeline_];
  [enc setBuffer:oct.interpolateOutputBuffer offset:0 atIndex:0];
  [enc setBuffer:oct.interpolateInputBuffer offset:0 atIndex:1];
  [enc setBuffer:oct.interpolateParamsBuffer offset:0 atIndex:2];
  [enc setTexture:oct.differenceTextures atIndex:0];

  NSUInteger maxThreads =
      siftInterpolatePipeline_.maxTotalThreadsPerThreadgroup;
  MTLSize tg = {maxThreads, 1, 1};
  MTLSize gridSize = {(NSUInteger)count, 1, 1};
  [enc dispatchThreads:gridSize threadsPerThreadgroup:tg];
  [enc endEncoding];

  [cb commit];
  [cb waitUntilCompleted];
}

// ---------------------------------------------------------------------------
// ComputeOrientations
// ---------------------------------------------------------------------------
void SiftMetalExtractorImpl::ComputeOrientations(
    Octave& oct, const std::vector<Keypoint>& keypoints,
    std::vector<std::pair<int, float>>& oriented) {
  oriented.clear();

  float delta = oct.delta;
  float lambda = 1.5f;
  float orientThreshold = 0.8f;
  float minX = 1.0f, minY = 1.0f;
  float maxX = float(oct.width - 2);
  float maxY = float(oct.height - 2);

  auto* params =
      static_cast<SIFTOrientationParameters*>(oct.orientationParamsBuffer.contents);
  params->delta = delta;
  params->lambda = lambda;
  params->orientationThreshold = orientThreshold;

  auto* orientIn = static_cast<SIFTOrientationKeypoint*>(
      oct.orientationInputBuffer.contents);

  int validCount = 0;
  std::vector<int> validIndices;
  for (int k = 0; k < (int)keypoints.size() && validCount < kMaxKeypoints; ++k) {
    const auto& kp = keypoints[k];
    float x = kp.x / delta;
    float y = kp.y / delta;
    float sigma = kp.sigma / delta;
    float r = std::ceil(3.0f * lambda * sigma);

    if (std::floor(x - r) < minX || std::ceil(x + r) > maxX ||
        std::floor(y - r) < minY || std::ceil(y + r) > maxY) {
      continue;
    }

    // Determine the scale index by finding the closest sigma.
    int scaleIdx = 0;
    float bestDiff = 1e30f;
    for (int s = 0; s < (int)oct.sigmas.size(); ++s) {
      float diff = std::abs(oct.sigmas[s] - kp.sigma);
      if (diff < bestDiff) {
        bestDiff = diff;
        scaleIdx = s;
      }
    }

    orientIn[validCount].index = static_cast<int32_t>(k);
    orientIn[validCount].absoluteX = static_cast<int32_t>(kp.x);
    orientIn[validCount].absoluteY = static_cast<int32_t>(kp.y);
    orientIn[validCount].scale = static_cast<int32_t>(scaleIdx);
    orientIn[validCount].sigma = kp.sigma;
    validIndices.push_back(k);
    ++validCount;
  }

  if (validCount == 0) return;

  id<MTLCommandBuffer> cb = [commandQueue_ commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:siftOrientationPipeline_];
  [enc setBuffer:oct.orientationOutputBuffer offset:0 atIndex:0];
  [enc setBuffer:oct.orientationInputBuffer offset:0 atIndex:1];
  [enc setBuffer:oct.orientationParamsBuffer offset:0 atIndex:2];
  [enc setTexture:oct.gradientTextures atIndex:0];

  NSUInteger maxThreads =
      siftOrientationPipeline_.maxTotalThreadsPerThreadgroup;
  MTLSize tg = {maxThreads, 1, 1};
  MTLSize gridSize = {(NSUInteger)validCount, 1, 1};
  [enc dispatchThreads:gridSize threadsPerThreadgroup:tg];
  [enc endEncoding];

  [cb commit];
  [cb waitUntilCompleted];

  // Read orientation results.
  auto* orientOut = static_cast<SIFTOrientationResult*>(
      oct.orientationOutputBuffer.contents);
  for (int k = 0; k < validCount; ++k) {
    auto& res = orientOut[k];
    int kpIdx = static_cast<int>(res.keypoint);
    int count = static_cast<int>(res.count);
    int maxOrient = options_.upright ? 1 : options_.max_num_orientations;
    count = std::min(count, maxOrient);
    float* oris = reinterpret_cast<float*>(&res.orientations);
    for (int i = 0; i < count; ++i) {
      float theta = options_.upright ? 0.0f : oris[i];
      oriented.emplace_back(kpIdx, theta);
    }
  }
}

// ---------------------------------------------------------------------------
// ComputeDescriptors
// ---------------------------------------------------------------------------
void SiftMetalExtractorImpl::ComputeDescriptors(
    Octave& oct, const std::vector<Keypoint>& keypoints,
    const std::vector<std::pair<int, float>>& oriented,
    ExtractResult* result) {
  int count = std::min((int)oriented.size(), kMaxDescriptors);
  if (count == 0) return;

  auto* params =
      static_cast<SIFTDescriptorParameters*>(oct.descriptorParamsBuffer.contents);
  params->delta = oct.delta;
  params->scalesPerOctave = static_cast<int32_t>(oct.num_scales);
  params->width = static_cast<int32_t>(oct.width);
  params->height = static_cast<int32_t>(oct.height);

  auto* descIn =
      static_cast<SIFTDescriptorInput*>(oct.descriptorInputBuffer.contents);
  for (int i = 0; i < count; ++i) {
    int kpIdx = oriented[i].first;
    float theta = oriented[i].second;
    const auto& kp = keypoints[kpIdx];

    // Determine scale index.
    int scaleIdx = 0;
    float bestDiff = 1e30f;
    for (int s = 0; s < (int)oct.sigmas.size(); ++s) {
      float diff = std::abs(oct.sigmas[s] - kp.sigma);
      if (diff < bestDiff) {
        bestDiff = diff;
        scaleIdx = s;
      }
    }
    // Compute subScale.
    float sigmaRatio = oct.sigmas[1] / oct.sigmas[0];
    float subScale = std::log(kp.sigma / oct.sigmas[scaleIdx]) /
                     std::log(sigmaRatio);

    descIn[i].keypoint = static_cast<int32_t>(kpIdx);
    descIn[i].absoluteX = static_cast<int32_t>(kp.x);
    descIn[i].absoluteY = static_cast<int32_t>(kp.y);
    descIn[i].scale = static_cast<int32_t>(scaleIdx);
    descIn[i].subScale = subScale;
    descIn[i].theta = theta;
  }

  id<MTLCommandBuffer> cb = [commandQueue_ commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:siftDescriptorsPipeline_];
  [enc setBuffer:oct.descriptorOutputBuffer offset:0 atIndex:0];
  [enc setBuffer:oct.descriptorInputBuffer offset:0 atIndex:1];
  [enc setBuffer:oct.descriptorParamsBuffer offset:0 atIndex:2];
  [enc setTexture:oct.gradientTextures atIndex:0];

  NSUInteger maxThreads =
      siftDescriptorsPipeline_.maxTotalThreadsPerThreadgroup;
  MTLSize tg = {maxThreads, 1, 1};
  MTLSize gridSize = {(NSUInteger)count, 1, 1};
  [enc dispatchThreads:gridSize threadsPerThreadgroup:tg];
  [enc endEncoding];

  [cb commit];
  [cb waitUntilCompleted];

  // Read descriptors.
  auto* descOut = static_cast<SIFTDescriptorResult*>(
      oct.descriptorOutputBuffer.contents);
  for (int i = 0; i < count; ++i) {
    auto& dr = descOut[i];
    if (!dr.valid) continue;

    int kpIdx = oriented[i].first;
    const auto& kp = keypoints[kpIdx];

    Keypoint finalKp;
    finalKp.x = kp.x;
    finalKp.y = kp.y;
    finalKp.sigma = kp.sigma;
    finalKp.orientation = dr.theta;
    result->keypoints.push_back(finalKp);

    // Convert int32 features (0-255) back to float for COLMAP's
    // normalization pipeline.
    for (int j = 0; j < 128; ++j) {
      result->descriptors.push_back(
          static_cast<float>(dr.features[j]) / 512.0f);
    }
  }
}

// ===========================================================================
// Public API
// ===========================================================================

SiftMetalExtractor::SiftMetalExtractor()
    : impl_(std::make_unique<SiftMetalExtractorImpl>()) {}

SiftMetalExtractor::~SiftMetalExtractor() = default;

bool SiftMetalExtractor::Init(const Options& options, int max_w, int max_h) {
  return impl_->Init(options, max_w, max_h);
}

bool SiftMetalExtractor::Extract(const uint8_t* data, int w, int h,
                                  ExtractResult* result) {
  return impl_->Extract(data, w, h, result);
}

}  // namespace sift_metal
