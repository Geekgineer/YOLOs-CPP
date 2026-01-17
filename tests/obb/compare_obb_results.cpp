#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <cmath>
#include <nlohmann/json.hpp>

#define STRING(x) #x
#define XSTRING(x) STRING(x)

using json = nlohmann::json;

constexpr double CONF_ERROR_MARGIN = 0.1;           // ±0.1 difference allowed in confidence scores
constexpr double OBB_CENTER_ERROR_MARGIN = 50.0;   // ±50 pixels difference allowed in center coordinates
constexpr double OBB_SIZE_ERROR_MARGIN = 50.0;     // ±50 pixels difference allowed in width/height
constexpr double OBB_ANGLE_ERROR_MARGIN = 0.2;     // ±0.2 radians difference allowed in rotation angle

constexpr double PI = 3.14159265358979323846;

json read_json(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("File not found: " + path);
    }
    json j;
    f >> j;
    return j;
}

/**
 * @brief Normalize angle to [-PI, PI] range
 */
double normalizeAngle(double angle) {
    while (angle > PI) angle -= 2.0 * PI;
    while (angle < -PI) angle += 2.0 * PI;
    return angle;
}

/**
 * @brief Check if two OBBs match, considering angle ambiguity
 * 
 * Two oriented bounding boxes can represent the same physical box with:
 * 1. Same dimensions and angles, OR
 * 2. Swapped width/height and angle differing by ±π/2
 */
bool obbsMatch(double cx1, double cy1, double w1, double h1, double angle1,
               double cx2, double cy2, double w2, double h2, double angle2) {
    
    // Check center coordinates
    double cx_diff = std::abs(cx1 - cx2);
    double cy_diff = std::abs(cy1 - cy2);
    
    if (cx_diff > OBB_CENTER_ERROR_MARGIN || cy_diff > OBB_CENTER_ERROR_MARGIN) {
        return false;
    }
    
    // Normalize angles to [-PI, PI]
    angle1 = normalizeAngle(angle1);
    angle2 = normalizeAngle(angle2);
    
    // Compute angle difference
    double angle_diff = std::abs(normalizeAngle(angle1 - angle2));
    
    // Case 1: Standard match (same orientation)
    double w_diff = std::abs(w1 - w2);
    double h_diff = std::abs(h1 - h2);
    
    bool standard_match = (w_diff <= OBB_SIZE_ERROR_MARGIN && 
                          h_diff <= OBB_SIZE_ERROR_MARGIN &&
                          angle_diff <= OBB_ANGLE_ERROR_MARGIN);
    
    if (standard_match) {
        return true;
    }
    
    // Case 2: 90-degree rotated match (width/height swapped)
    // When rotated by ±90°, width and height swap
    double w_swap_diff = std::abs(w1 - h2);  // Compare width1 with height2
    double h_swap_diff = std::abs(h1 - w2);  // Compare height1 with width2
    
    // Check if angle differs by approximately ±π/2
    double angle_diff_90 = std::min(
        std::abs(angle_diff - PI/2.0),
        std::abs(angle_diff - 3.0*PI/2.0)
    );
    
    bool rotated_match = (w_swap_diff <= OBB_SIZE_ERROR_MARGIN && 
                         h_swap_diff <= OBB_SIZE_ERROR_MARGIN &&
                         angle_diff_90 <= OBB_ANGLE_ERROR_MARGIN);
    
    return rotated_match;
}

class OBBResultsFixture : public ::testing::Test {
protected:
    json results_ultralytics;
    json results_cpp;
    std::string basePath = XSTRING(BASE_PATH_OBB); // Base path defined in CMakeLists.txt

    void SetUp() override {
        ASSERT_NO_THROW(results_ultralytics = read_json(basePath + "results/results_ultralytics.json"));
        ASSERT_NO_THROW(results_cpp = read_json(basePath + "results/results_cpp.json"));
    }
};

TEST_F(OBBResultsFixture, ResultsNotEmpty) {
    ASSERT_FALSE(results_ultralytics.empty()) << "results_ultralytics is empty";
    ASSERT_FALSE(results_cpp.empty()) << "results_cpp is empty";
}

TEST_F(OBBResultsFixture, CompareModelsNames) {
    std::set<std::string> models_ultra, models_cpp;

    for (auto& el : results_ultralytics.items()) models_ultra.insert(el.key());
    for (auto& el : results_cpp.items()) models_cpp.insert(el.key());

    for (const auto& name : models_ultra) {
        ASSERT_TRUE(models_cpp.count(name)) << "Model " << name << " is missing in results_cpp";
    }
}

TEST_F(OBBResultsFixture, CompareWeightsPaths) {
    for (auto& el : results_ultralytics.items()) {
        const std::string& model_name = el.key();

        std::string weights_ultra = el.value().value("weights_path", "");
        std::string weights_cpp = results_cpp[model_name].value("weights_path", "");

        ASSERT_EQ(weights_ultra, weights_cpp) << "Weights path mismatch for model " << model_name;
    }
}

TEST_F(OBBResultsFixture, CompareImagesCounts) {
    for (auto& el : results_ultralytics.items()) {
        const std::string& model_name = el.key();

        auto& ultra_results = el.value()["results"];
        auto& cpp_results = results_cpp[model_name]["results"];

        ASSERT_EQ(ultra_results.size(), cpp_results.size())
            << "Number of results mismatch for model " << model_name;
    }
}

TEST_F(OBBResultsFixture, CompareImagesPaths) {
    for (auto& el : results_ultralytics.items()) {
        const std::string& model_name = el.key();

        auto& ultra_results = el.value()["results"];
        auto& cpp_results = results_cpp[model_name]["results"];

        for (size_t i = 0; i < ultra_results.size(); ++i) {
            std::string path_ultra = ultra_results[i].value("image_path", "");
            std::string path_cpp = cpp_results[i].value("image_path", "");

            ASSERT_EQ(path_ultra, path_cpp)
                << "Image path mismatch for model " << model_name << ", image " << i;
        }
    }
}

TEST_F(OBBResultsFixture, CompareOBBDetectionsCount) {
    for (auto& el : results_ultralytics.items()) {
        const std::string& model_name = el.key();

        auto& ultra_results = el.value()["results"];
        auto& cpp_results = results_cpp[model_name]["results"];

        for (size_t i = 0; i < ultra_results.size(); ++i) {
            auto detections_ultra = ultra_results[i].value("inference_results", json::array());
            auto detections_cpp = cpp_results[i].value("inference_results", json::array());

            std::string image_path = ultra_results[i].value("image_path", "");

            ASSERT_EQ(detections_ultra.size(), detections_cpp.size())
                << "Number of OBB detections mismatch for model " << model_name << ", image: " << image_path;
        }
    }
}

TEST_F(OBBResultsFixture, CompareOBBDetections) {
    for (auto& el : results_ultralytics.items()) {
        const std::string& model_name = el.key();

        auto& ultra_results = el.value()["results"];
        auto& cpp_results = results_cpp[model_name]["results"];

        for (size_t i = 0; i < ultra_results.size(); ++i) {
            auto detections_ultra = ultra_results[i].value("inference_results", json::array());
            auto detections_cpp = cpp_results[i].value("inference_results", json::array());

            std::string image_path = ultra_results[i].value("image_path", "");

            for (size_t j = 0; j < detections_ultra.size(); ++j) {
                auto& det_ultra = detections_ultra[j];
                int class_id_ultra = det_ultra.value("class_id", -1);
                double conf_ultra = det_ultra.value("confidence", 0.0);
                auto obb_ultra = det_ultra["obb"];

                double cx_ultra = obb_ultra["cx"].get<double>();
                double cy_ultra = obb_ultra["cy"].get<double>();
                double w_ultra = obb_ultra["width"].get<double>();
                double h_ultra = obb_ultra["height"].get<double>();
                double angle_ultra = obb_ultra["angle"].get<double>();
                
                // Find BEST match (smallest center distance) among all valid matches
                int best_match_idx = -1;
                double best_distance = std::numeric_limits<double>::max();
                
                for (size_t k = 0; k < detections_cpp.size(); ++k) {
                    auto& det_cpp = detections_cpp[k];
                    int class_id_cpp = det_cpp.value("class_id", -2);

                    if (class_id_ultra == class_id_cpp) {
                        auto obb_cpp = det_cpp["obb"];

                        double cx_cpp = obb_cpp["cx"].get<double>();
                        double cy_cpp = obb_cpp["cy"].get<double>();
                        double w_cpp = obb_cpp["width"].get<double>();
                        double h_cpp = obb_cpp["height"].get<double>();
                        double angle_cpp = obb_cpp["angle"].get<double>();

                        if (obbsMatch(cx_ultra, cy_ultra, w_ultra, h_ultra, angle_ultra,
                                     cx_cpp, cy_cpp, w_cpp, h_cpp, angle_cpp)) {
                            // Calculate center distance
                            double dist = std::sqrt(std::pow(cx_ultra - cx_cpp, 2) + 
                                                   std::pow(cy_ultra - cy_cpp, 2));
                            if (dist < best_distance) {
                                best_distance = dist;
                                best_match_idx = static_cast<int>(k);
                            }
                        }
                    }
                }
                
                ASSERT_GE(best_match_idx, 0)
                    << "Class ID " << class_id_ultra << " not found in cpp results for model "
                    << model_name << ", image: " << image_path
                    << "\n  Ultralytics OBB: cx=" << obb_ultra["cx"] 
                    << ", cy=" << obb_ultra["cy"]
                    << ", w=" << obb_ultra["width"] 
                    << ", h=" << obb_ultra["height"]
                    << ", angle=" << obb_ultra["angle"];

                // Compare confidence of best match
                auto& det_cpp = detections_cpp[best_match_idx];
                double conf_cpp = det_cpp.value("confidence", 0.0);
                double conf_diff = std::abs(conf_ultra - conf_cpp);

                ASSERT_LE(conf_diff, CONF_ERROR_MARGIN)
                    << "Confidence mismatch for model " << model_name
                    << ", image: " << image_path << ", class_id: " << class_id_ultra
                    << "\n  Ultralytics confidence: " << conf_ultra 
                    << "\n  C++ confidence: " << conf_cpp
                    << "\n  Difference: " << conf_diff;
            }
        }
    }
}