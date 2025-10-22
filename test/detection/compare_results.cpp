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

constexpr double CONF_ERROR_MARGIN = 0.1; // +-0.1 difference allowed in confidence scores
constexpr int BBOX_ERROR_MARGIN = 50;     // +-50 pixels difference allowed in bounding box coordinates

json read_json(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("File not found: " + path);
    }
    json j;
    f >> j;
    return j;
}

class ResultsFixture : public ::testing::Test {
protected:
    json results_ultralytics;
    json results_cpp;
    std::string basePath = XSTRING(BASE_PATH_DETECTION); // Base path defined in CMakeLists.txt


    void SetUp() override {
        ASSERT_NO_THROW(results_ultralytics = read_json(basePath + "results/results_ultralytics.json"));
        ASSERT_NO_THROW(results_cpp = read_json(basePath + "results/results_cpp.json"));
    }
};

TEST_F(ResultsFixture, ResultsNotEmpty) {
    ASSERT_FALSE(results_ultralytics.empty()) << "results_ultralytics is empty";
    ASSERT_FALSE(results_cpp.empty()) << "results_cpp is empty";
}

TEST_F(ResultsFixture, CompareModelsNames) {

    std::set<std::string> models_ultra, models_cpp;

    for (auto& el : results_ultralytics.items()) models_ultra.insert(el.key());
    for (auto& el : results_cpp.items()) models_cpp.insert(el.key());

    for (const auto& name : models_ultra) {
        ASSERT_TRUE(models_cpp.count(name)) << "Model " << name << " is missing in results_cpp";
    }
}

TEST_F(ResultsFixture, CompareWeightsPaths) {

    for (auto& el : results_ultralytics.items()) {

        const std::string& model_name = el.key();

        std::string weights_ultra = el.value().value("weights_path", "");
        std::string weights_cpp = results_cpp[model_name].value("weights_path", "");

        ASSERT_EQ(weights_ultra, weights_cpp) << "Weights path mismatch for model " << model_name;
    }
}

TEST_F(ResultsFixture, CompareImagesCounts) {

    for (auto& el : results_ultralytics.items()) {

        const std::string& model_name = el.key();

        auto& ultra_results = el.value()["results"];
        auto& cpp_results = results_cpp[model_name]["results"];

        ASSERT_EQ(ultra_results.size(), cpp_results.size())
            << "Number of results mismatch for model " << model_name;
    }
}

TEST_F(ResultsFixture, CompareImagesPaths) {

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

TEST_F(ResultsFixture, CompareDetectionsCount) {

    for (auto& el : results_ultralytics.items()) {

        const std::string& model_name = el.key();

        auto& ultra_results = el.value()["results"];
        auto& cpp_results = results_cpp[model_name]["results"];

        for (size_t i = 0; i < ultra_results.size(); ++i) {

            auto detections_ultra = ultra_results[i].value("inference_results", json::array());
            auto detections_cpp = cpp_results[i].value("inference_results", json::array());

            std::string image_path = ultra_results[i].value("image_path", "");

            ASSERT_EQ(detections_ultra.size(), detections_cpp.size())
                << "Number of detections mismatch for model " << model_name << ", image: " << image_path;
        }
    }
}

TEST_F(ResultsFixture, CompareDetections) {

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
                auto bbox_ultra = det_ultra["bbox"];
                bool is_class_found = false;

                for (size_t k = 0; k < detections_cpp.size(); ++k) {

                    auto& det_cpp = detections_cpp[k];

                    int class_id_cpp = det_cpp.value("class_id", -2);

                    if (class_id_ultra == class_id_cpp) {

                        auto bbox_cpp = det_cpp["bbox"];

                        int left_diff = std::abs(bbox_ultra["left"].get<int>() - bbox_cpp["left"].get<int>());
                        int top_diff = std::abs(bbox_ultra["top"].get<int>() - bbox_cpp["top"].get<int>());
                        int width_diff = std::abs(bbox_ultra["width"].get<int>() - bbox_cpp["width"].get<int>());
                        int height_diff = std::abs(bbox_ultra["height"].get<int>() - bbox_cpp["height"].get<int>());

                        if (left_diff <= BBOX_ERROR_MARGIN &&
                            top_diff <= BBOX_ERROR_MARGIN &&
                            width_diff <= BBOX_ERROR_MARGIN &&
                            height_diff <= BBOX_ERROR_MARGIN) {

                            double conf_cpp = det_cpp.value("confidence", 0.0);
                            double conf_diff = std::abs(conf_ultra - conf_cpp);

                            ASSERT_LE(conf_diff, CONF_ERROR_MARGIN)
                                << "Confidence mismatch for model " << model_name
                                << ", image: " << image_path << ", class_id: " << class_id_ultra
                                << ": ultralytics: " << conf_ultra << " != cpp: " << conf_cpp;

                            is_class_found = true;
                            break;
                        }
                    }
                }
                
                ASSERT_TRUE(is_class_found)
                    << "Class ID " << class_id_ultra << " not found in cpp results for model "
                    << model_name << ", image: " << image_path;
            }
        }
    }
}
