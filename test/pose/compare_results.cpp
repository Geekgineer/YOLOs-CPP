#include <gtest/gtest.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <map>
#include <vector>
#include <cmath>
#include <nlohmann/json.hpp>

#define STRING(x) #x
#define XSTRING(x) STRING(x)

using json = nlohmann::json;

constexpr double CONF_ERROR_MARGIN = 0.1;  // ±0.1 difference allowed in confidence scores
constexpr int KEYPOINT_ERROR_MARGIN = 10;  // ±10 pixels difference allowed in keypoint coordinates

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
    std::string basePath = XSTRING(BASE_PATH_POSE);

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

        ASSERT_EQ(weights_ultra, basePath + weights_cpp) << "Weights path mismatch for model " << model_name;
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

        std::set<std::string> ultra_filenames, cpp_filenames;

        for (const auto& result : ultra_results) {
            std::string path = result.value("image_path", "");
            ultra_filenames.insert(path.substr(path.find_last_of("/\\") + 1));
        }

        for (const auto& result : cpp_results) {
            std::string path = result.value("image_path", "");
            cpp_filenames.insert(path.substr(path.find_last_of("/\\") + 1));
        }

        ASSERT_EQ(ultra_filenames, cpp_filenames)
            << "Image filenames mismatch for model " << model_name;
        
    }
}



TEST_F(ResultsFixture, ComparePosesCount) {

    for (auto& el : results_ultralytics.items()) {

        const std::string& model_name = el.key();

        auto& ultra_results = el.value()["results"];
        auto& cpp_results = results_cpp[model_name]["results"];

        for (size_t i = 0; i < ultra_results.size(); ++i) {

            auto poses_ultra = ultra_results[i].value("poses", json::array());
            auto poses_cpp = cpp_results[i].value("poses", json::array());

            std::string image_path = ultra_results[i].value("image_path", "");

            ASSERT_EQ(poses_ultra.size(), poses_cpp.size())
                << "Number of poses mismatch for model " << model_name << ", image: " << image_path;
        }
    }
}

TEST_F(ResultsFixture, ComparePoses) {

    for (auto& el : results_ultralytics.items()) {

        const std::string& model_name = el.key();

        auto& ultra_results = el.value()["results"];
        auto& cpp_results = results_cpp[model_name]["results"];

        for (size_t i = 0; i < ultra_results.size(); ++i) {

            auto poses_ultra = ultra_results[i].value("poses", json::array());
            auto poses_cpp = cpp_results[i].value("poses", json::array());

            std::string image_path = ultra_results[i].value("image_path", "");

            for (size_t j = 0; j < poses_ultra.size(); ++j) {

                auto& det_ultra = poses_ultra[j];
                int class_id_ultra = det_ultra.value("class_id", -1);
                double conf_ultra = det_ultra.value("confidence", 0.0);
                auto keypoints_ultra = det_ultra["keypoints"];
                bool is_class_found = false;

                for (size_t k = 0; k < poses_cpp.size(); ++k) {

                    auto& det_cpp = poses_cpp[k];

                    int class_id_cpp = det_cpp.value("class_id", -2);

                    if (class_id_ultra == class_id_cpp) {

                        auto keypoints_cpp = det_cpp["keypoints"];

                        bool keypoints_match = true;

                        for (size_t l = 0; l < keypoints_ultra.size(); ++l) {

                            float x_ultra = keypoints_ultra[l]["x"].get<float>();
                            float y_ultra = keypoints_ultra[l]["y"].get<float>();
                            float x_cpp = keypoints_cpp[l]["x"].get<float>();
                            float y_cpp = keypoints_cpp[l]["y"].get<float>();

                            if (std::abs(x_ultra - x_cpp) > KEYPOINT_ERROR_MARGIN ||
                                std::abs(y_ultra - y_cpp) > KEYPOINT_ERROR_MARGIN) {

                                keypoints_match = false;
                                break;
                            }
                        }

                        if (keypoints_match) {

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
                    << "class ID " << class_id_ultra << " not found in cpp results for model "
                    << model_name << ", image: " << image_path;
            }
        }
    }
}