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
    int newUnpadW = static_cast<int>(std::round(image.cols * ratio));
    int newUnpadH = static_cast<int>(std::round(image.rows * ratio));

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

/// @brief Fast letterbox with direct blob output (avoids intermediate copies)
/// @param image Input BGR image
/// @param blob Output CHW float blob (pre-allocated)
/// @param targetSize Target size for inference
/// @param[out] actualSize Actual output size after letterboxing
/// @param padColor Padding color value (0-255, default 114)
inline void letterBoxToBlob(const cv::Mat& image,
                            std::vector<float>& blob,
                            const cv::Size& targetSize,
                            cv::Size& actualSize,
                            float padColor = 114.0f) {
    
    const int srcH = image.rows;
    const int srcW = image.cols;
    const int dstH = targetSize.height;
    const int dstW = targetSize.width;
    
    // Calculate scale and padding (match Ultralytics exactly)
    const float scale = std::min(static_cast<float>(dstH) / srcH,
                                  static_cast<float>(dstW) / srcW);
    
    // Ultralytics uses round() for new dimensions
    const int newH = static_cast<int>(std::round(srcH * scale));
    const int newW = static_cast<int>(std::round(srcW * scale));
    
    // Ultralytics uses asymmetric padding with -0.1/+0.1 adjustment
    const float dh = (dstH - newH) / 2.0f;
    const float dw = (dstW - newW) / 2.0f;
    const int padTop = static_cast<int>(std::round(dh - 0.1f));
    const int padLeft = static_cast<int>(std::round(dw - 0.1f));
    
    actualSize = cv::Size(dstW, dstH);
    
    // Ensure blob capacity
    const size_t totalSize = static_cast<size_t>(dstH * dstW * 3);
    if (blob.size() < totalSize) {
        blob.resize(totalSize);
    }
    
    // Fill with padding color (normalized)
    const float padNorm = padColor / 255.0f;
    std::fill(blob.begin(), blob.end(), padNorm);
    
    // Resize image
    cv::Mat resized;
    if (newW != srcW || newH != srcH) {
        cv::resize(image, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);
    } else {
        resized = image;
    }
    
    // Convert BGR to RGB and normalize directly into blob (CHW format)
    float* rChannel = blob.data();
    float* gChannel = blob.data() + dstH * dstW;
    float* bChannel = blob.data() + 2 * dstH * dstW;
    
    constexpr float scale255 = 1.0f / 255.0f;
    
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
}

/// @brief Fast letterbox with buffer reuse
/// @param image Input BGR image
/// @param buffer Pre-allocated inference buffer
/// @param targetSize Target size for inference
/// @param[out] actualSize Actual output size
/// @param dynamicShape Whether to use dynamic shape
inline void letterBoxToBlob(const cv::Mat& image,
                            InferenceBuffer& buffer,
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
    int newH = static_cast<int>(std::round(srcH * scale));
    int newW = static_cast<int>(std::round(srcW * scale));
    
    // For dynamic shape, adjust to stride-aligned minimum size
    if (dynamicShape) {
        constexpr int stride = 32;
        dstH = ((newH + stride - 1) / stride) * stride;
        dstW = ((newW + stride - 1) / stride) * stride;
    }
    
    actualSize = cv::Size(dstW, dstH);
    buffer.ensureCapacity(dstH, dstW, 3);
    
    // Ultralytics uses asymmetric padding with -0.1/+0.1 adjustment
    const float dh = (dstH - newH) / 2.0f;
    const float dw = (dstW - newW) / 2.0f;
    const int padTop = static_cast<int>(std::round(dh - 0.1f));
    const int padLeft = static_cast<int>(std::round(dw - 0.1f));
    
    // Fill with padding (normalized 114/255)
    constexpr float padNorm = 114.0f / 255.0f;
    std::fill(buffer.blob.begin(), buffer.blob.begin() + dstH * dstW * 3, padNorm);
    
    // Resize if needed
    if (newW != srcW || newH != srcH) {
        cv::resize(image, buffer.resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);
    } else {
        buffer.resized = image;  // Reference, no copy
    }
    
    // Direct BGR->RGB + normalize to CHW blob
    float* rChannel = buffer.blob.data();
    float* gChannel = buffer.blob.data() + dstH * dstW;
    float* bChannel = buffer.blob.data() + 2 * dstH * dstW;
    
    constexpr float scale255 = 1.0f / 255.0f;
    
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
    
    // Use round() for new dimensions (matches Ultralytics)
    int newW = static_cast<int>(std::round(originalSize.width * scale));
    int newH = static_cast<int>(std::round(originalSize.height * scale));
    
    // For descaling, use UNROUNDED padding values (matches Ultralytics behavior)
    padX = (letterboxSize.width - newW) / 2.0f;
    padY = (letterboxSize.height - newH) / 2.0f;
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

} // namespace preprocessing
} // namespace yolos
