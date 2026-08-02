#pragma once

// ============================================================================
// YOLO Preprocessing Utilities
// ============================================================================
// Optimized image preprocessing functions for YOLO inference including
// letterbox resizing, coordinate scaling, and blob conversion.
//
// Author: YOLOs-CPP Team, https://github.com/Geekgineer/YOLOs-CPP
// ============================================================================

#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>

#include "yolos/core/types.hpp"
#include "yolos/core/utils.hpp"

namespace yolos {
namespace preprocessing {

// ============================================================================
// Pre-allocated Buffer for Inference
// ============================================================================

/// @brief Pre-allocated inference buffer to avoid per-frame allocations
struct InferenceBuffer {
    std::vector<float> blob;         ///< CHW format blob for ONNX
    cv::Mat resized;                 ///< Letterboxed image
    cv::Mat rgbFloat;                ///< RGB float image
    cv::Size lastInputSize;          ///< Last input size (for reuse check)
    cv::Size lastTargetSize;         ///< Last target size
    
    /// @brief Ensure blob has required capacity
    void ensureCapacity(int height, int width, int channels = 3) {
        size_t required = static_cast<size_t>(height * width * channels);
        if (blob.size() < required) {
            blob.resize(required);
        }
    }
};

// ============================================================================
// LetterBox Resizing
// ============================================================================

/// @brief Resize an image with letterboxing to maintain aspect ratio
/// @param image Input image
/// @param outImage Output resized and padded image
/// @param newShape Desired output size
/// @param color Padding color (default is gray 114,114,114)
/// @param autoSize If true, use minimum rectangle to resize
/// @param scaleFill Whether to scale to fill without keeping aspect ratio
/// @param scaleUp Whether to allow scaling up of the image
/// @param stride Stride size for padding alignment
inline void letterBox(const cv::Mat& image,
                      cv::Mat& outImage,
                      const cv::Size& newShape,
                      const cv::Scalar& color = cv::Scalar(114, 114, 114),
                      bool autoSize = true,
                      bool scaleFill = false,
                      bool scaleUp = true,
                      int stride = 32) {
    
    // Calculate the scaling ratio to fit the image within the new shape
    float ratio = std::min(static_cast<float>(newShape.height) / image.rows,
                          static_cast<float>(newShape.width) / image.cols);

    // Prevent scaling up if not allowed
    if (!scaleUp) {
        ratio = std::min(ratio, 1.0f);
    }

    // Calculate new dimensions after scaling (use round to match Ultralytics)
    int newUnpadW = static_cast<int>(std::nearbyint(image.cols * ratio));
    int newUnpadH = static_cast<int>(std::nearbyint(image.rows * ratio));

    // Calculate padding needed to reach the desired shape
    int dw = newShape.width - newUnpadW;
    int dh = newShape.height - newUnpadH;

    if (autoSize) {
        // Ensure padding is a multiple of stride for model compatibility
        dw = dw % stride;
        dh = dh % stride;
    } else if (scaleFill) {
        // Scale to fill without maintaining aspect ratio
        newUnpadW = newShape.width;
        newUnpadH = newShape.height;
        dw = 0;
        dh = 0;
    }

    // Calculate separate padding for left/right and top/bottom
    int padLeft = dw / 2;
    int padRight = dw - padLeft;
    int padTop = dh / 2;
    int padBottom = dh - padTop;

    // Resize the image if dimensions differ
    if (image.cols != newUnpadW || image.rows != newUnpadH) {
        cv::resize(image, outImage, cv::Size(newUnpadW, newUnpadH), 0, 0, cv::INTER_LINEAR);
    } else {
        outImage = image.clone();
    }

    // Apply padding to reach the desired shape
    cv::copyMakeBorder(outImage, outImage, padTop, padBottom, padLeft, padRight,
                       cv::BORDER_CONSTANT, color);
}

/// @brief Alternative letterbox with center option (matches Ultralytics)
/// @param image Input image
/// @param outImage Output resized and padded image
/// @param newShape Desired output size (default 640x640)
/// @param autoSize If true, use minimum rectangle to resize
/// @param scaleFill Whether to scale to fill without keeping aspect ratio
/// @param scaleUp Whether to allow scaling up of the image
/// @param center If true, center the placed image
/// @param stride Stride of the model
/// @param paddingValue Padding value (default is 114)
/// @param interpolation Interpolation method
inline void letterBoxCentered(const cv::Mat& image,
                              cv::Mat& outImage,
                              const cv::Size& newShape = cv::Size(640, 640),
                              bool autoSize = false,
                              bool scaleFill = false,
                              bool scaleUp = true,
                              bool center = true,
                              int stride = 32,
                              const cv::Scalar& paddingValue = cv::Scalar(114, 114, 114),
                              int interpolation = cv::INTER_LINEAR) {
    
    float ratio = std::min(static_cast<float>(newShape.height) / image.rows,
                          static_cast<float>(newShape.width) / image.cols);

    if (!scaleUp) {
        ratio = std::min(ratio, 1.0f);
    }

    // Use round to match Ultralytics
    int newUnpadW = static_cast<int>(std::round(image.cols * ratio));
    int newUnpadH = static_cast<int>(std::round(image.rows * ratio));

    int dw = newShape.width - newUnpadW;
    int dh = newShape.height - newUnpadH;

    if (autoSize) {
        dw = dw % stride;
        dh = dh % stride;
    } else if (scaleFill) {
        newUnpadW = newShape.width;
        newUnpadH = newShape.height;
        dw = 0;
        dh = 0;
    }

    if (center) {
        dw /= 2;
        dh /= 2;
    }

    if (image.cols != newUnpadW || image.rows != newUnpadH) {
        cv::resize(image, outImage, cv::Size(newUnpadW, newUnpadH), 0, 0, interpolation);
    } else {
        outImage = image.clone();
    }

    int top = center ? static_cast<int>(std::round(dh - 0.1f)) : 0;
    int bottom = static_cast<int>(std::round(dh + 0.1f));
    int left = center ? static_cast<int>(std::round(dw - 0.1f)) : 0;
    int right = static_cast<int>(std::round(dw + 0.1f));

    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right,
                       cv::BORDER_CONSTANT, paddingValue);
}

// ============================================================================
// Coordinate Scaling
// ============================================================================

/// @brief Scale detection coordinates from letterbox space back to original image size
/// @param letterboxShape Shape of the letterboxed image used for inference
/// @param coords Bounding box in letterbox coordinates
/// @param originalShape Original image size before letterboxing
/// @param clip Whether to clip coordinates to image boundaries
/// @return Scaled bounding box in original image coordinates
inline BoundingBox scaleCoords(const cv::Size& letterboxShape,
                               const BoundingBox& coords,
                               const cv::Size& originalShape,
                               bool clip = true) {
    
    float gain = std::min(static_cast<float>(letterboxShape.height) / originalShape.height,
                         static_cast<float>(letterboxShape.width) / originalShape.width);

    int padX = static_cast<int>(std::round((letterboxShape.width - originalShape.width * gain) / 2.0f));
    int padY = static_cast<int>(std::round((letterboxShape.height - originalShape.height * gain) / 2.0f));

    BoundingBox result;
    result.x = static_cast<int>(std::round((coords.x - padX) / gain));
    result.y = static_cast<int>(std::round((coords.y - padY) / gain));
    result.width = static_cast<int>(std::round(coords.width / gain));
    result.height = static_cast<int>(std::round(coords.height / gain));

    if (clip) {
        result.x = utils::clamp(result.x, 0, originalShape.width);
        result.y = utils::clamp(result.y, 0, originalShape.height);
        result.width = utils::clamp(result.width, 0, originalShape.width - result.x);
        result.height = utils::clamp(result.height, 0, originalShape.height - result.y);
    }

    return result;
}

/// @brief Scale keypoint coordinates from letterbox space back to original image size
/// @param letterboxShape Shape of the letterboxed image
/// @param keypoint Keypoint in letterbox coordinates
/// @param originalShape Original image size
/// @param clip Whether to clip coordinates to image boundaries
/// @return Scaled keypoint in original image coordinates
inline KeyPoint scaleKeypoint(const cv::Size& letterboxShape,
                              const KeyPoint& keypoint,
                              const cv::Size& originalShape,
                              bool clip = true) {
    
    float gain = std::min(static_cast<float>(letterboxShape.height) / originalShape.height,
                         static_cast<float>(letterboxShape.width) / originalShape.width);

    float padX = (letterboxShape.width - originalShape.width * gain) / 2.0f;
    float padY = (letterboxShape.height - originalShape.height * gain) / 2.0f;

    KeyPoint result;
    result.x = (keypoint.x - padX) / gain;
    result.y = (keypoint.y - padY) / gain;
    result.confidence = keypoint.confidence;

    if (clip) {
        result.x = utils::clamp(result.x, 0.0f, static_cast<float>(originalShape.width - 1));
        result.y = utils::clamp(result.y, 0.0f, static_cast<float>(originalShape.height - 1));
    }

    return result;
}

/// @brief Get letterbox padding and scale parameters
/// @param originalShape Original image size
/// @param letterboxShape Letterboxed image size
/// @param[out] scale Scale factor applied
/// @param[out] padX Horizontal padding
/// @param[out] padY Vertical padding
inline void getLetterboxParams(const cv::Size& originalShape,
                               const cv::Size& letterboxShape,
                               float& scale,
                               float& padX,
                               float& padY) {
    scale = std::min(static_cast<float>(letterboxShape.height) / originalShape.height,
                    static_cast<float>(letterboxShape.width) / originalShape.width);
    padX = (letterboxShape.width - originalShape.width * scale) / 2.0f;
    padY = (letterboxShape.height - originalShape.height * scale) / 2.0f;
}

// ============================================================================
// Optimized Single-Pass Preprocessing
// ============================================================================

/// @brief Fast letterbox writing directly into a caller-owned CHW float slice
/// @param image Input BGR image
/// @param dst Destination CHW slice; must hold at least
///            targetChannels * targetSize.height * targetSize.width floats
/// @param targetChannels Target channels for inference
/// @param targetSize Target size for inference (used verbatim, no stride alignment)
/// @param padColor Padding color value (0-255, default 114)
/// @note Only the requested slice is written, so this is safe to use for one
///       image of a batched N*C*H*W blob.
inline void letterBoxToBlobPtr(const cv::Mat& image,
                               float* dst,
                               int targetChannels,
                               const cv::Size& targetSize,
                               float padColor = 114.0f) {

    const int srcH = image.rows;
    const int srcW = image.cols;
    const int dstH = targetSize.height;
    const int dstW = targetSize.width;

    // Calculate scale and padding (match Ultralytics exactly)
    const float scale = std::min(static_cast<float>(dstH) / srcH,
                                  static_cast<float>(dstW) / srcW);

    // Ultralytics uses round() for new dimensions
    const int newH = static_cast<int>(std::nearbyint(srcH * scale));
    const int newW = static_cast<int>(std::nearbyint(srcW * scale));

    // Ultralytics uses asymmetric padding with -0.1/+0.1 adjustment
    const float dh = (dstH - newH) / 2.0f;
    const float dw = (dstW - newW) / 2.0f;
    const int padTop = static_cast<int>(std::nearbyint(dh - 0.1f));
    const int padLeft = static_cast<int>(std::nearbyint(dw - 0.1f));

    // Fill this slice with padding color (normalized)
    const size_t sliceSize = static_cast<size_t>(dstH) * dstW * targetChannels;
    const float padNorm = padColor / 255.0f;
    std::fill(dst, dst + sliceSize, padNorm);

    // Resize image
    cv::Mat resized;
    if (newW != srcW || newH != srcH) {
        cv::resize(image, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);
    } else {
        resized = image;
    }

    constexpr float scale255 = 1.0f / 255.0f;
    if (targetChannels == 3) {
        // Convert BGR to RGB and normalize directly into blob (CHW format)
        float* rChannel = dst;
        float* gChannel = dst + dstH * dstW;
        float* bChannel = dst + 2 * dstH * dstW;

        for (int y = 0; y < newH; ++y) {
            const int dstY = y + padTop;
            const uchar* row = resized.ptr<uchar>(y);

            for (int x = 0; x < newW; ++x) {
                const int dstX = x + padLeft;
                const int dstIdx = dstY * dstW + dstX;
                const int srcIdx = x * 3;

                // BGR to RGB conversion + normalization
                bChannel[dstIdx] = row[srcIdx + 0] * scale255;
                gChannel[dstIdx] = row[srcIdx + 1] * scale255;
                rChannel[dstIdx] = row[srcIdx + 2] * scale255;
            }
        }
    } else {
        // normalize directly into blob (single channel)
        float *channel = dst;

        for (int y = 0; y < newH; ++y) {
            const int    dstY = y + padTop;
            const uchar *row  = resized.ptr<uchar>(y);

            for (int x = 0; x < newW; ++x) {
                const int dstX   = x + padLeft;
                const int dstIdx = dstY * dstW + dstX;

                channel[dstIdx] = static_cast<float>(row[x]) * scale255;
            }
        }
    }
}

/// @brief Fast letterbox with direct blob output (avoids intermediate copies)
/// @param image Input BGR image
/// @param blob Output CHW float blob (resized if too small)
/// @param targetChannels Target channels for inference
/// @param targetSize Target size for inference
/// @param[out] actualSize Actual output size after letterboxing
/// @param padColor Padding color value (0-255, default 114)
inline void letterBoxToBlob(const cv::Mat& image,
                            std::vector<float>& blob,
                            int targetChannels,
                            const cv::Size& targetSize,
                            cv::Size& actualSize,
                            float padColor = 114.0f) {

    actualSize = targetSize;

    // Ensure blob capacity
    const size_t totalSize = static_cast<size_t>(targetSize.height) * targetSize.width * targetChannels;
    if (blob.size() < totalSize) {
        blob.resize(totalSize);
    }

    letterBoxToBlobPtr(image, blob.data(), targetChannels, targetSize, padColor);
}

/// @brief Letterbox a batch of images into a single contiguous N*C*H*W blob
/// @param images Input BGR images
/// @param blob Output N*C*H*W float blob (resized if too small)
/// @param targetChannels Target channels for inference
/// @param targetSize Target size shared by every image in the batch
/// @param padColor Padding color value (0-255, default 114)
/// @note All images share one letterbox target, which is what a batched tensor
///       requires. Per-image scale/padding still differ and must be recovered
///       with getScalePad(images[i].size(), targetSize, ...) during postprocessing.
inline void letterBoxToBatchBlob(const std::vector<cv::Mat>& images,
                                 std::vector<float>& blob,
                                 int targetChannels,
                                 const cv::Size& targetSize,
                                 float padColor = 114.0f) {

    const size_t sliceSize = static_cast<size_t>(targetSize.height) * targetSize.width * targetChannels;
    const size_t totalSize = sliceSize * images.size();
    if (blob.size() < totalSize) {
        blob.resize(totalSize);
    }

    for (size_t i = 0; i < images.size(); ++i) {
        letterBoxToBlobPtr(images[i], blob.data() + i * sliceSize, targetChannels, targetSize, padColor);
    }
}

/// @brief Fast letterbox with buffer reuse
/// @param image Input BGR image
/// @param buffer Pre-allocated inference buffer
/// @param targetChannels Target channels for inference
/// @param targetSize Target size for inference
/// @param[out] actualSize Actual output size
/// @param dynamicShape Whether to use dynamic shape
inline void letterBoxToBlob(const cv::Mat& image,
                            InferenceBuffer& buffer,
                            int targetChannels,
                            const cv::Size& targetSize,
                            cv::Size& actualSize,
                            bool dynamicShape = false) {
    
    const int srcH = image.rows;
    const int srcW = image.cols;
    int dstH = targetSize.height;
    int dstW = targetSize.width;
    
    // Calculate scale (match Ultralytics exactly)
    const float scale = std::min(static_cast<float>(dstH) / srcH,
                                  static_cast<float>(dstW) / srcW);
    
    // Ultralytics uses round() for new dimensions
    int newH = static_cast<int>(std::nearbyint(srcH * scale));
    int newW = static_cast<int>(std::nearbyint(srcW * scale));
    
    // For dynamic shape, adjust to stride-aligned minimum size
    if (dynamicShape) {
        constexpr int stride = 32;
        dstH = ((newH + stride - 1) / stride) * stride;
        dstW = ((newW + stride - 1) / stride) * stride;
    }
    
    actualSize = cv::Size(dstW, dstH);
    buffer.ensureCapacity(dstH, dstW, targetChannels);
    
    // Ultralytics uses asymmetric padding with -0.1/+0.1 adjustment
    const float dh = (dstH - newH) / 2.0f;
    const float dw = (dstW - newW) / 2.0f;
    const int padTop = static_cast<int>(std::nearbyint(dh - 0.1f));
    const int padLeft = static_cast<int>(std::nearbyint(dw - 0.1f));
    
    // Fill with padding (normalized 114/255)
    constexpr float padNorm = 114.0f / 255.0f;
    std::fill(buffer.blob.begin(), buffer.blob.begin() + dstH * dstW * targetChannels, padNorm);
    
    // Resize if needed
    if (newW != srcW || newH != srcH) {
        cv::resize(image, buffer.resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);
    } else {
        buffer.resized = image;  // Reference, no copy
    }
    
    constexpr float scale255 = 1.0f / 255.0f;
    if (targetChannels == 3) {
        // Direct BGR->RGB + normalize to CHW blob
        float* rChannel = buffer.blob.data();
        float* gChannel = buffer.blob.data() + dstH * dstW;
        float* bChannel = buffer.blob.data() + 2 * dstH * dstW;
        
        for (int y = 0; y < newH; ++y) {
            const int dstY = y + padTop;
            const uchar* row = buffer.resized.ptr<uchar>(y);
            const int rowOffset = dstY * dstW + padLeft;
            
            for (int x = 0; x < newW; ++x) {
                const int dstIdx = rowOffset + x;
                const int srcIdx = x * 3;
                
                bChannel[dstIdx] = row[srcIdx + 0] * scale255;
                gChannel[dstIdx] = row[srcIdx + 1] * scale255;
                rChannel[dstIdx] = row[srcIdx + 2] * scale255;
            }
        }
    } else {
        // normalize directly into blob (single channel)
        float* blobPtr = buffer.blob.data();
        for (int y = 0; y < newH; ++y) {
            const int dstY = y + padTop;
            const uchar* row = buffer.resized.ptr<uchar>(y);
            const int rowOffset = dstY * dstW + padLeft;
            
            for (int x = 0; x < newW; ++x) {
                blobPtr[rowOffset + x] = static_cast<float>(row[x]) * scale255;
            }
        }
    }

    buffer.lastInputSize = cv::Size(srcW, srcH);
    buffer.lastTargetSize = actualSize;
}

/// @brief Get scale and padding info from letterbox operation
/// @param originalSize Original image size
/// @param letterboxSize Letterboxed image size
/// @param[out] scale Scale factor
/// @param[out] padX X padding
/// @param[out] padY Y padding
inline void getScalePad(const cv::Size& originalSize,
                        const cv::Size& letterboxSize,
                        float& scale,
                        float& padX,
                        float& padY) {
    scale = std::min(static_cast<float>(letterboxSize.height) / originalSize.height,
                     static_cast<float>(letterboxSize.width) / originalSize.width);

    // Python round() breaks ties to even; std::nearbyint does the same under the
    // default rounding mode, std::round does not.
    const int newW = static_cast<int>(std::nearbyint(originalSize.width * scale));
    const int newH = static_cast<int>(std::nearbyint(originalSize.height * scale));

    // Return the padding the letterbox ACTUALLY applied, which is an integer:
    // letterBoxToBlob pads by nearbyint(dw - 0.1). Returning the unrounded dw instead
    // descales against padding that was never there, biasing every box by up to half a
    // letterbox pixel - which the gain then magnifies into original-image pixels.
    // This mirrors ultralytics.utils.ops.scale_boxes():
    //     pad_x = round((img1_w - round(img0_w * gain)) / 2 - 0.1)
    const float dw = (letterboxSize.width - newW) / 2.0f;
    const float dh = (letterboxSize.height - newH) / 2.0f;
    padX = std::nearbyint(dw - 0.1f);
    padY = std::nearbyint(dh - 0.1f);
}

/// @brief Fast coordinate descaling (batch operation)
/// @param coords Array of x,y coordinates to descale
/// @param count Number of coordinate pairs
/// @param scale Letterbox scale
/// @param padX X padding
/// @param padY Y padding
inline void descaleCoordsBatch(float* coords, size_t count,
                               float scale, float padX, float padY) {
    const float invScale = 1.0f / scale;
    for (size_t i = 0; i < count; ++i) {
        coords[i * 2 + 0] = (coords[i * 2 + 0] - padX) * invScale;
        coords[i * 2 + 1] = (coords[i * 2 + 1] - padY) * invScale;
    }
}

// ============================================================================
// Antialiased Bilinear Resize (PIL-compatible)
// ============================================================================
// Ultralytics preprocesses classification images with torchvision's
// T.Resize(size, BILINEAR) applied to a PIL image, and PIL's BILINEAR filter is
// *antialiased*: its support widens with the downscale factor so every source
// pixel contributes. cv::resize(INTER_LINEAR) always samples a 2x2 neighborhood
// and therefore aliases when downscaling, which shifted classification
// confidences noticeably and could even change the top-1 class.
//
// The functions below reproduce Pillow's algorithm (src/libImaging/Resample.c).

namespace detail {

/// @brief Per-output-pixel filter weights for one axis
struct ResampleCoeffs {
    int ksize{0};                  ///< Stride between per-pixel weight runs
    std::vector<float> weights;    ///< outSize * ksize weights
    std::vector<int> starts;       ///< First contributing source index
    std::vector<int> counts;       ///< Number of contributing source pixels
};

/// @brief Round-half-up then clamp to a byte, matching Pillow's clip8()
inline uchar clip8(double value) {
    const int rounded = static_cast<int>(std::floor(value + 0.5));
    return static_cast<uchar>(rounded < 0 ? 0 : (rounded > 255 ? 255 : rounded));
}

/// @brief Compute Pillow's triangle-filter weights for one axis
/// @param inSize Source length along the axis
/// @param outSize Destination length along the axis
/// @note The support scales with the downscale factor, which is what produces
///       the antialiasing. When upscaling, filterscale clamps to 1.0 and this
///       degenerates to ordinary bilinear interpolation.
inline ResampleCoeffs computeBilinearCoeffs(int inSize, int outSize) {
    constexpr double filterSupport = 1.0;  // Pillow's BILINEAR support

    const double scale = static_cast<double>(inSize) / outSize;
    const double filterscale = (scale < 1.0) ? 1.0 : scale;
    const double support = filterSupport * filterscale;
    const double invFilterscale = 1.0 / filterscale;

    ResampleCoeffs coeffs;
    coeffs.ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
    coeffs.weights.assign(static_cast<size_t>(outSize) * coeffs.ksize, 0.0f);
    coeffs.starts.resize(outSize);
    coeffs.counts.resize(outSize);

    for (int i = 0; i < outSize; ++i) {
        const double center = (i + 0.5) * scale;

        int start = static_cast<int>(center - support + 0.5);
        if (start < 0) start = 0;
        int end = static_cast<int>(center + support + 0.5);
        if (end > inSize) end = inSize;
        const int count = end - start;

        float* w = &coeffs.weights[static_cast<size_t>(i) * coeffs.ksize];
        double total = 0.0;
        for (int k = 0; k < count; ++k) {
            const double x = std::abs((k + start - center + 0.5) * invFilterscale);
            const double weight = (x < 1.0) ? (1.0 - x) : 0.0;
            w[k] = static_cast<float>(weight);
            total += weight;
        }
        if (total != 0.0) {
            for (int k = 0; k < count; ++k) {
                w[k] = static_cast<float>(w[k] / total);
            }
        }

        coeffs.starts[i] = start;
        coeffs.counts[i] = count;
    }

    return coeffs;
}

} // namespace detail

/// @brief Antialiased bilinear resize matching PIL/Pillow (8-bit input)
/// @param src Source image, CV_8U with any channel count
/// @param dst Destination image, same type as @p src
/// @param dstSize Target size
/// @note Two separable passes with an 8-bit intermediate, exactly like Pillow:
///       the horizontal result is rounded to bytes before the vertical pass, so
///       carrying float accumulators across both passes would NOT match.
inline void resizeAntialiasBilinear(const cv::Mat& src, cv::Mat& dst, const cv::Size& dstSize) {
    CV_Assert(src.depth() == CV_8U && !src.empty());
    CV_Assert(dstSize.width > 0 && dstSize.height > 0);

    const int channels = src.channels();

    // Horizontal pass
    cv::Mat temp;
    if (src.cols != dstSize.width) {
        temp.create(src.rows, dstSize.width, src.type());
        const detail::ResampleCoeffs cx = detail::computeBilinearCoeffs(src.cols, dstSize.width);

        for (int y = 0; y < src.rows; ++y) {
            const uchar* srcRow = src.ptr<uchar>(y);
            uchar* dstRow = temp.ptr<uchar>(y);

            for (int x = 0; x < dstSize.width; ++x) {
                const float* w = &cx.weights[static_cast<size_t>(x) * cx.ksize];
                const int start = cx.starts[x];
                const int count = cx.counts[x];

                for (int c = 0; c < channels; ++c) {
                    double acc = 0.0;
                    for (int k = 0; k < count; ++k) {
                        acc += static_cast<double>(w[k]) * srcRow[(start + k) * channels + c];
                    }
                    dstRow[x * channels + c] = detail::clip8(acc);
                }
            }
        }
    } else {
        temp = src;
    }

    // Vertical pass
    if (temp.rows != dstSize.height) {
        cv::Mat out(dstSize.height, temp.cols, src.type());
        const detail::ResampleCoeffs cy = detail::computeBilinearCoeffs(temp.rows, dstSize.height);

        for (int y = 0; y < dstSize.height; ++y) {
            const float* w = &cy.weights[static_cast<size_t>(y) * cy.ksize];
            const int start = cy.starts[y];
            const int count = cy.counts[y];
            uchar* dstRow = out.ptr<uchar>(y);

            // Hoist the contributing row pointers out of the per-pixel loop
            std::vector<const uchar*> srcRows(static_cast<size_t>(count));
            for (int k = 0; k < count; ++k) {
                srcRows[static_cast<size_t>(k)] = temp.ptr<uchar>(start + k);
            }

            for (int x = 0; x < temp.cols; ++x) {
                for (int c = 0; c < channels; ++c) {
                    const int offset = x * channels + c;
                    double acc = 0.0;
                    for (int k = 0; k < count; ++k) {
                        acc += static_cast<double>(w[k]) * srcRows[static_cast<size_t>(k)][offset];
                    }
                    dstRow[offset] = detail::clip8(acc);
                }
            }
        }
        dst = out;
    } else {
        dst = temp.clone();  // temp may alias src, so never hand back a view
    }
}

// ============================================================================
// Dense Map Rescaling
// ============================================================================

/// @brief Crop letterbox padding from a dense map and rescale to the original size
/// @param map Dense single-channel map in letterbox space (CV_32FC1)
/// @param out Result, sized to @p originalSize; always a fresh buffer, never a view
/// @param originalSize Target size (the un-letterboxed image)
/// @param interpolation Resize interpolation (INTER_LINEAR matches Ultralytics)
/// @note Port of ultralytics.utils.ops.scale_masks(). Gain and padding are derived
///       from @p map's own dimensions rather than from the letterbox size, so this
///       stays correct for maps emitted at reduced resolution.
/// @note segmentation.hpp inlines equivalent logic for mask prototypes; it is left
///       alone deliberately, but this is the natural helper to fold it onto next
///       time that path is touched.
inline void cropLetterboxAndResize(const cv::Mat& map,
                                   cv::Mat& out,
                                   const cv::Size& originalSize,
                                   int interpolation = cv::INTER_LINEAR) {
    CV_Assert(!map.empty() && map.type() == CV_32FC1);
    CV_Assert(originalSize.width > 0 && originalSize.height > 0);

    const int mapH = map.rows;
    const int mapW = map.cols;
    const int dstH = originalSize.height;
    const int dstW = originalSize.width;

    if (mapH == dstH && mapW == dstW) {
        out = map.clone();
        return;
    }

    const double gain = std::min(static_cast<double>(mapH) / dstH,
                                 static_cast<double>(mapW) / dstW);
    // Python round() is round-half-to-even here too, not just below: for a 320x320 map
    // and a 640x241 target, dstH * gain is exactly 120.5, and half-away-from-zero would
    // shift the crop by one row (121 rows instead of 120).
    const double padW = (mapW - std::nearbyint(dstW * gain)) / 2.0;
    const double padH = (mapH - std::nearbyint(dstH * gain)) / 2.0;

    // Ultralytics uses Python round(), which breaks ties to even: std::nearbyint
    // does the same under the default rounding mode.
    int top    = static_cast<int>(std::nearbyint(padH - 0.1));
    int left   = static_cast<int>(std::nearbyint(padW - 0.1));
    int bottom = mapH - static_cast<int>(std::nearbyint(padH + 0.1));
    int right  = mapW - static_cast<int>(std::nearbyint(padW + 0.1));

    // Clamp so a pathological map size cannot produce an invalid ROI
    top    = std::max(0, std::min(top, mapH - 1));
    left   = std::max(0, std::min(left, mapW - 1));
    bottom = std::max(top + 1, std::min(bottom, mapH));
    right  = std::max(left + 1, std::min(right, mapW));

    const cv::Mat cropped = map(cv::Rect(left, top, right - left, bottom - top));
    cv::resize(cropped, out, originalSize, 0, 0, interpolation);
}

} // namespace preprocessing
} // namespace yolos
