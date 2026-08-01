/**
 * @file test_batch_and_memory.cpp
 * @brief API tests for batch inference and in-memory model loading
 *
 * These tests run against the tiny synthetic ONNX models produced by
 * make_synthetic_models.py, so they need neither real weights nor Ultralytics.
 *
 * The invariants under test:
 *   - batch* methods return exactly what a per-image loop returns
 *   - each batch slice is postprocessed against its own image, not image 0
 *   - fixed-batch models transparently fall back to the per-image loop
 *   - loading a model from memory behaves identically to loading it from disk
 */

#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <vector>

#include "yolos/yolos.hpp"

#define STRING(x) #x
#define XSTRING(x) STRING(x)

namespace fs = std::filesystem;

namespace {

const std::string kModelDir = std::string(XSTRING(BASE_PATH_API)) + "models/";

std::string modelPath(const std::string& name) { return kModelDir + name; }

/// Images with distinct sizes and distinct mean intensities: the synthetic
/// models key their output off the image mean, and the differing sizes force
/// per-image letterbox scale/padding during postprocessing.
std::vector<cv::Mat> makeTestImages() {
    return {
        cv::Mat(480, 640, CV_8UC3, cv::Scalar(10, 20, 30)),
        cv::Mat(900, 300, CV_8UC3, cv::Scalar(200, 180, 160)),
        cv::Mat(500, 500, CV_8UC3, cv::Scalar(90, 90, 90)),
        cv::Mat(720, 1280, CV_8UC3, cv::Scalar(250, 5, 5)),
    };
}

void expectSameDetections(const std::vector<yolos::det::Detection>& expected,
                          const std::vector<yolos::det::Detection>& actual,
                          const std::string& context) {
    ASSERT_EQ(expected.size(), actual.size()) << context;
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(expected[i].classId, actual[i].classId) << context << " det " << i;
        EXPECT_NEAR(expected[i].conf, actual[i].conf, 1e-4f) << context << " det " << i;
        EXPECT_EQ(expected[i].box.x, actual[i].box.x) << context << " det " << i;
        EXPECT_EQ(expected[i].box.y, actual[i].box.y) << context << " det " << i;
        EXPECT_EQ(expected[i].box.width, actual[i].box.width) << context << " det " << i;
        EXPECT_EQ(expected[i].box.height, actual[i].box.height) << context << " det " << i;
    }
}

} // namespace

// ============================================================================
// Fixtures
// ============================================================================

class SyntheticModelTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        // Check a concrete model, not just the directory: the .onnx files are
        // generated (and gitignored), so a stale empty directory is the likely
        // failure and should produce this message rather than an ORT error.
        ASSERT_TRUE(fs::exists(modelPath("det_dynamic.onnx")))
            << "Synthetic models missing from " << kModelDir
            << "\nRun: python3 tests/api/make_synthetic_models.py tests/api/models";
    }
};

// ============================================================================
// Detection: batched path
// ============================================================================

TEST_F(SyntheticModelTest, DetectionBatchMatchesSequential) {
    yolos::det::YOLODetector detector(modelPath("det_dynamic.onnx"), "", false);
    ASSERT_TRUE(detector.isDynamicBatchSize()) << "det_dynamic.onnx should have a dynamic batch dim";

    const std::vector<cv::Mat> images = makeTestImages();
    ASSERT_TRUE(detector.supportsBatchSize(images.size()));

    std::vector<std::vector<yolos::det::Detection>> sequential;
    for (const auto& image : images) {
        sequential.push_back(detector.detect(image, 0.25f, 0.45f));
    }

    const auto batched = detector.batchDetect(images, 0.25f, 0.45f);

    ASSERT_EQ(images.size(), batched.size());
    for (size_t i = 0; i < images.size(); ++i) {
        expectSameDetections(sequential[i], batched[i], "image " + std::to_string(i));
    }
}

TEST_F(SyntheticModelTest, DetectionBatchProducesNonEmptyPerImageResults) {
    yolos::det::YOLODetector detector(modelPath("det_dynamic.onnx"), "", false);

    const std::vector<cv::Mat> images = makeTestImages();
    const auto batched = detector.batchDetect(images, 0.25f, 0.45f);

    ASSERT_EQ(images.size(), batched.size());
    for (size_t i = 0; i < batched.size(); ++i) {
        EXPECT_FALSE(batched[i].empty()) << "image " << i << " produced no detections";
    }
}

TEST_F(SyntheticModelTest, DetectionBatchSlicesAreImageSpecific) {
    // Guards against every slice being postprocessed from batch element 0:
    // the images have different means, so their detections must differ.
    yolos::det::YOLODetector detector(modelPath("det_dynamic.onnx"), "", false);

    const std::vector<cv::Mat> images = makeTestImages();
    const auto batched = detector.batchDetect(images, 0.25f, 0.45f);

    ASSERT_GE(batched.size(), 2u);
    ASSERT_FALSE(batched[0].empty());
    ASSERT_FALSE(batched[1].empty());
    EXPECT_NE(batched[0][0].conf, batched[1][0].conf)
        << "different images yielded identical confidences";
}

TEST_F(SyntheticModelTest, DetectionBatchWithEmptyInputReturnsEmpty) {
    yolos::det::YOLODetector detector(modelPath("det_dynamic.onnx"), "", false);
    EXPECT_TRUE(detector.batchDetect({}, 0.25f, 0.45f).empty());
}

TEST_F(SyntheticModelTest, DetectionBatchOfOneMatchesDetect) {
    yolos::det::YOLODetector detector(modelPath("det_dynamic.onnx"), "", false);

    const cv::Mat image = makeTestImages()[2];
    const auto expected = detector.detect(image, 0.25f, 0.45f);
    const auto batched = detector.batchDetect({image}, 0.25f, 0.45f);

    ASSERT_EQ(1u, batched.size());
    expectSameDetections(expected, batched[0], "single-image batch");
}

TEST_F(SyntheticModelTest, TwoOutputDetectionBatchMatchesSequential) {
    // Two outputs route through postprocessNAS, which reads both tensors —
    // so this covers slicing more than one output per batch element.
    yolos::det::YOLODetector detector(modelPath("nas_dynamic.onnx"), "", false);
    ASSERT_EQ(2u, detector.getNumOutputNodes());
    ASSERT_TRUE(detector.isDynamicBatchSize());

    const std::vector<cv::Mat> images = makeTestImages();

    std::vector<std::vector<yolos::det::Detection>> sequential;
    for (const auto& image : images) {
        sequential.push_back(detector.detect(image, 0.25f, 0.45f));
    }

    const auto batched = detector.batchDetect(images, 0.25f, 0.45f);

    ASSERT_EQ(images.size(), batched.size());
    for (size_t i = 0; i < images.size(); ++i) {
        EXPECT_FALSE(batched[i].empty()) << "image " << i << " produced no detections";
        expectSameDetections(sequential[i], batched[i], "two-output image " + std::to_string(i));
    }
}

TEST_F(SyntheticModelTest, DynamicInputShapeModelStillBatches) {
    // Batching needs one shared letterbox target, so for a dynamic-input-shape
    // model the batched path uses the model input shape while detect() picks a
    // per-image stride-aligned shape. Results may differ; the batch must still
    // be correct and per-image.
    yolos::det::YOLODetector detector(modelPath("det_dynshape.onnx"), "", false);
    ASSERT_TRUE(detector.isDynamicInputShape());
    ASSERT_TRUE(detector.isDynamicBatchSize());

    const std::vector<cv::Mat> images = makeTestImages();
    const auto batched = detector.batchDetect(images, 0.25f, 0.45f);

    ASSERT_EQ(images.size(), batched.size());
    for (size_t i = 0; i < batched.size(); ++i) {
        EXPECT_FALSE(batched[i].empty()) << "image " << i << " produced no detections";
    }
    EXPECT_NE(batched[0][0].conf, batched[1][0].conf)
        << "different images yielded identical confidences";

    // Same shared target as an explicit batch of one, so those must agree.
    const auto single = detector.batchDetect({images[1]}, 0.25f, 0.45f);
    ASSERT_EQ(1u, single.size());
    expectSameDetections(single[0], batched[1], "dynamic-shape batch of one vs batch of four");
}

// ============================================================================
// Detection: fixed-batch fallback
// ============================================================================

TEST_F(SyntheticModelTest, FixedBatchModelFallsBackToPerImageLoop) {
    yolos::det::YOLODetector detector(modelPath("det_batch1.onnx"), "", false);

    EXPECT_FALSE(detector.isDynamicBatchSize());
    EXPECT_EQ(1, detector.getModelBatchSize());

    const std::vector<cv::Mat> images = makeTestImages();
    EXPECT_FALSE(detector.supportsBatchSize(images.size()));
    EXPECT_TRUE(detector.supportsBatchSize(1));

    std::vector<std::vector<yolos::det::Detection>> sequential;
    for (const auto& image : images) {
        sequential.push_back(detector.detect(image, 0.25f, 0.45f));
    }

    // Falls back internally, but the caller still gets one result per image.
    const auto batched = detector.batchDetect(images, 0.25f, 0.45f);

    ASSERT_EQ(images.size(), batched.size());
    for (size_t i = 0; i < images.size(); ++i) {
        expectSameDetections(sequential[i], batched[i], "fallback image " + std::to_string(i));
    }
}

TEST_F(SyntheticModelTest, ModelThatClaimsDynamicBatchButCannotRunOneFallsBack) {
    // Real Ultralytics exports bake batch=1 into DFL Reshape / anchor Concat
    // nodes even when the declared batch dim is dynamic, so the batched run
    // throws. det_liar.onnx reproduces that: the caller must still get results.
    yolos::det::YOLODetector detector(modelPath("det_liar.onnx"), "", false);

    const std::vector<cv::Mat> images = makeTestImages();
    ASSERT_TRUE(detector.isDynamicBatchSize());
    ASSERT_TRUE(detector.supportsBatchSize(images.size())) << "the model advertises a batch it cannot run";

    std::vector<std::vector<yolos::det::Detection>> sequential;
    for (const auto& image : images) {
        sequential.push_back(detector.detect(image, 0.25f, 0.45f));
    }

    // Batched run fails inside ONNX Runtime; batchDetect must recover.
    const auto batched = detector.batchDetect(images, 0.25f, 0.45f);

    ASSERT_EQ(images.size(), batched.size());
    for (size_t i = 0; i < images.size(); ++i) {
        EXPECT_FALSE(batched[i].empty()) << "image " << i << " produced no detections";
        expectSameDetections(sequential[i], batched[i], "recovered image " + std::to_string(i));
    }
}

// ============================================================================
// In-memory model loading
// ============================================================================

TEST_F(SyntheticModelTest, InMemoryLoadingMatchesFileLoading) {
    const std::vector<std::string> classNames = {"first", "second"};

    yolos::det::YOLODetector fromFile(modelPath("det_dynamic.onnx"), "", false);

    const std::vector<uint8_t> bytes = yolos::utils::readFileBytes(modelPath("det_dynamic.onnx"));
    ASSERT_FALSE(bytes.empty());
    yolos::det::YOLODetector fromMemory(bytes.data(), bytes.size(), classNames, false);

    EXPECT_EQ(fromFile.getInputShape(), fromMemory.getInputShape());
    EXPECT_EQ(fromFile.isDynamicBatchSize(), fromMemory.isDynamicBatchSize());
    EXPECT_EQ(classNames, fromMemory.getClassNames());

    const std::vector<cv::Mat> images = makeTestImages();
    for (size_t i = 0; i < images.size(); ++i) {
        expectSameDetections(fromFile.detect(images[i], 0.25f, 0.45f),
                             fromMemory.detect(images[i], 0.25f, 0.45f),
                             "memory-vs-file image " + std::to_string(i));
    }
}

TEST_F(SyntheticModelTest, InMemoryLoadingSurvivesBufferRelease) {
    // ONNX Runtime copies the model during session creation, which is what lets
    // callers wipe an decrypted buffer immediately after construction.
    std::vector<uint8_t> bytes = yolos::utils::readFileBytes(modelPath("det_dynamic.onnx"));
    ASSERT_FALSE(bytes.empty());

    yolos::det::YOLODetector detector(bytes.data(), bytes.size(), {"first", "second"}, false);

    std::fill(bytes.begin(), bytes.end(), uint8_t{0});
    bytes.clear();
    bytes.shrink_to_fit();

    EXPECT_FALSE(detector.detect(makeTestImages()[0], 0.25f, 0.45f).empty());
}

TEST_F(SyntheticModelTest, InMemoryLoadingFallsBackToOnnxMetadataNames) {
    // det_dynamic.onnx carries Ultralytics-style names metadata {0: alpha, 1: beta}
    const std::vector<uint8_t> bytes = yolos::utils::readFileBytes(modelPath("det_dynamic.onnx"));
    ASSERT_FALSE(bytes.empty());

    yolos::det::YOLODetector detector(bytes.data(), bytes.size(), {}, false);

    const std::vector<std::string> expected = {"alpha", "beta"};
    EXPECT_EQ(expected, detector.getClassNames());
}

TEST_F(SyntheticModelTest, InMemoryLoadingRejectsEmptyBuffer) {
    EXPECT_THROW(
        yolos::det::YOLODetector(nullptr, 0, {"a"}, false),
        std::invalid_argument);

    const std::vector<uint8_t> bytes = {1, 2, 3};
    EXPECT_THROW(
        yolos::det::YOLODetector(bytes.data(), 0, {"a"}, false),
        std::invalid_argument);
}

TEST_F(SyntheticModelTest, InMemoryDetectorSupportsBatchInference) {
    const std::vector<uint8_t> bytes = yolos::utils::readFileBytes(modelPath("det_dynamic.onnx"));
    ASSERT_FALSE(bytes.empty());

    yolos::det::YOLODetector detector(bytes.data(), bytes.size(), {"first", "second"}, false);

    const std::vector<cv::Mat> images = makeTestImages();
    const auto batched = detector.batchDetect(images, 0.25f, 0.45f);
    ASSERT_EQ(images.size(), batched.size());

    for (size_t i = 0; i < images.size(); ++i) {
        expectSameDetections(detector.detect(images[i], 0.25f, 0.45f), batched[i],
                             "in-memory batch image " + std::to_string(i));
    }
}

TEST_F(SyntheticModelTest, CreateDetectorFromMemoryFactory) {
    const std::vector<uint8_t> bytes = yolos::utils::readFileBytes(modelPath("det_dynamic.onnx"));
    ASSERT_FALSE(bytes.empty());

    auto detector = yolos::det::createDetectorFromMemory(
        bytes.data(), bytes.size(), {"first", "second"}, yolos::YOLOVersion::V11, false);

    ASSERT_NE(nullptr, detector);
    EXPECT_FALSE(detector->detect(makeTestImages()[0], 0.25f, 0.45f).empty());
}

// ============================================================================
// Segmentation
// ============================================================================

TEST_F(SyntheticModelTest, SegmentationBatchMatchesSequential) {
    yolos::seg::YOLOSegDetector segmenter(modelPath("seg_dynamic.onnx"), "", false);
    ASSERT_TRUE(segmenter.isDynamicBatchSize());

    const std::vector<cv::Mat> images = makeTestImages();

    std::vector<std::vector<yolos::seg::Segmentation>> sequential;
    for (const auto& image : images) {
        sequential.push_back(segmenter.segment(image, 0.25f, 0.45f));
    }

    const auto batched = segmenter.batchSegment(images, 0.25f, 0.45f);

    ASSERT_EQ(images.size(), batched.size());
    for (size_t i = 0; i < images.size(); ++i) {
        ASSERT_EQ(sequential[i].size(), batched[i].size()) << "image " << i;
        for (size_t d = 0; d < sequential[i].size(); ++d) {
            EXPECT_EQ(sequential[i][d].classId, batched[i][d].classId) << "image " << i;
            EXPECT_NEAR(sequential[i][d].conf, batched[i][d].conf, 1e-4f) << "image " << i;
            EXPECT_EQ(sequential[i][d].box.x, batched[i][d].box.x) << "image " << i;
            EXPECT_EQ(sequential[i][d].box.y, batched[i][d].box.y) << "image " << i;
            EXPECT_EQ(sequential[i][d].box.width, batched[i][d].box.width) << "image " << i;
            EXPECT_EQ(sequential[i][d].box.height, batched[i][d].box.height) << "image " << i;

            // Masks must be identical pixel for pixel, which is what proves the
            // prototype tensor was sliced on the right batch element.
            ASSERT_EQ(sequential[i][d].mask.size(), batched[i][d].mask.size()) << "image " << i;
            EXPECT_EQ(0, cv::countNonZero(sequential[i][d].mask != batched[i][d].mask))
                << "mask mismatch for image " << i << " detection " << d;
        }
    }
}

TEST_F(SyntheticModelTest, SegmentationInMemoryLoading) {
    const std::vector<uint8_t> bytes = yolos::utils::readFileBytes(modelPath("seg_dynamic.onnx"));
    ASSERT_FALSE(bytes.empty());

    yolos::seg::YOLOSegDetector fromMemory(bytes.data(), bytes.size(), {"first", "second"}, false);
    yolos::seg::YOLOSegDetector fromFile(modelPath("seg_dynamic.onnx"), "", false);

    const cv::Mat image = makeTestImages()[1];
    EXPECT_EQ(fromFile.segment(image, 0.25f, 0.45f).size(),
              fromMemory.segment(image, 0.25f, 0.45f).size());
}

// ============================================================================
// Pose
// ============================================================================

TEST_F(SyntheticModelTest, PoseBatchMatchesSequential) {
    yolos::pose::YOLOPoseDetector poser(modelPath("pose_dynamic.onnx"), "", false);
    ASSERT_TRUE(poser.isDynamicBatchSize());

    const std::vector<cv::Mat> images = makeTestImages();

    std::vector<std::vector<yolos::pose::PoseResult>> sequential;
    for (const auto& image : images) {
        sequential.push_back(poser.detect(image, 0.25f, 0.5f));
    }

    const auto batched = poser.batchDetect(images, 0.25f, 0.5f);

    ASSERT_EQ(images.size(), batched.size());
    for (size_t i = 0; i < images.size(); ++i) {
        ASSERT_EQ(sequential[i].size(), batched[i].size()) << "image " << i;
        for (size_t d = 0; d < sequential[i].size(); ++d) {
            EXPECT_NEAR(sequential[i][d].conf, batched[i][d].conf, 1e-4f) << "image " << i;
            EXPECT_EQ(sequential[i][d].box.x, batched[i][d].box.x) << "image " << i;
            EXPECT_EQ(sequential[i][d].box.y, batched[i][d].box.y) << "image " << i;
            ASSERT_EQ(sequential[i][d].keypoints.size(), batched[i][d].keypoints.size());
            for (size_t k = 0; k < sequential[i][d].keypoints.size(); ++k) {
                EXPECT_NEAR(sequential[i][d].keypoints[k].x, batched[i][d].keypoints[k].x, 1e-2f);
                EXPECT_NEAR(sequential[i][d].keypoints[k].y, batched[i][d].keypoints[k].y, 1e-2f);
            }
        }
    }
}

// ============================================================================
// OBB
// ============================================================================

TEST_F(SyntheticModelTest, ObbBatchMatchesSequential) {
    yolos::obb::YOLOOBBDetector obbDetector(modelPath("obb_dynamic.onnx"), "", false);
    ASSERT_TRUE(obbDetector.isDynamicBatchSize());

    const std::vector<cv::Mat> images = makeTestImages();

    std::vector<std::vector<yolos::obb::OBBResult>> sequential;
    for (const auto& image : images) {
        sequential.push_back(obbDetector.detect(image, 0.25f, 0.45f, 300));
    }

    const auto batched = obbDetector.batchDetect(images, 0.25f, 0.45f, 300);

    ASSERT_EQ(images.size(), batched.size());
    for (size_t i = 0; i < images.size(); ++i) {
        ASSERT_EQ(sequential[i].size(), batched[i].size()) << "image " << i;
        for (size_t d = 0; d < sequential[i].size(); ++d) {
            EXPECT_EQ(sequential[i][d].classId, batched[i][d].classId) << "image " << i;
            EXPECT_NEAR(sequential[i][d].conf, batched[i][d].conf, 1e-4f) << "image " << i;
            EXPECT_NEAR(sequential[i][d].box.x, batched[i][d].box.x, 1e-2f) << "image " << i;
            EXPECT_NEAR(sequential[i][d].box.y, batched[i][d].box.y, 1e-2f) << "image " << i;
            EXPECT_NEAR(sequential[i][d].box.angle, batched[i][d].box.angle, 1e-4f) << "image " << i;
        }
    }
}

// ============================================================================
// Classification
// ============================================================================

TEST_F(SyntheticModelTest, ClassificationBatchMatchesSequential) {
    const std::vector<std::string> classNames = {
        "c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"};

    const std::vector<uint8_t> bytes = yolos::utils::readFileBytes(modelPath("cls_dynamic.onnx"));
    ASSERT_FALSE(bytes.empty());
    yolos::cls::YOLOClassifier classifier(bytes.data(), bytes.size(), classNames, false);
    ASSERT_TRUE(classifier.isDynamicBatchSize());

    const std::vector<cv::Mat> images = makeTestImages();

    std::vector<yolos::cls::ClassificationResult> sequential;
    for (const auto& image : images) {
        sequential.push_back(classifier.classify(image));
    }

    const auto batched = classifier.batchClassify(images);

    ASSERT_EQ(images.size(), batched.size());
    for (size_t i = 0; i < images.size(); ++i) {
        EXPECT_EQ(sequential[i].classId, batched[i].classId) << "image " << i;
        EXPECT_EQ(sequential[i].className, batched[i].className) << "image " << i;
        EXPECT_NEAR(sequential[i].confidence, batched[i].confidence, 1e-5f) << "image " << i;
    }
}

TEST_F(SyntheticModelTest, ClassificationBatchHandlesEmptyImages) {
    const std::vector<uint8_t> bytes = yolos::utils::readFileBytes(modelPath("cls_dynamic.onnx"));
    ASSERT_FALSE(bytes.empty());
    yolos::cls::YOLOClassifier classifier(bytes.data(), bytes.size(), {"c0"}, false);

    // An empty Mat forces the per-image path; the empty slot still gets a result.
    std::vector<cv::Mat> images = makeTestImages();
    images.push_back(cv::Mat());

    const auto batched = classifier.batchClassify(images);
    ASSERT_EQ(images.size(), batched.size());
    EXPECT_EQ(-1, batched.back().classId);
}

TEST_F(SyntheticModelTest, ClassificationBatchWithEmptyInputReturnsEmpty) {
    const std::vector<uint8_t> bytes = yolos::utils::readFileBytes(modelPath("cls_dynamic.onnx"));
    ASSERT_FALSE(bytes.empty());
    yolos::cls::YOLOClassifier classifier(bytes.data(), bytes.size(), {"c0"}, false);

    EXPECT_TRUE(classifier.batchClassify({}).empty());
}
