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

            auto poses_ultra = ultra_results[i].value("inference_results", json::array());
            std::string image_path = ultra_results[i].value("image_path", "");

            for (size_t j = 0; j < cpp_results.size(); ++j) {

                if (cpp_results[j].value("image_path", "") == image_path) {

                    auto poses_cpp = cpp_results[j].value("inference_results", json::array());
                   
                    ASSERT_EQ(poses_ultra.size(), poses_cpp.size())
                        << "Number of poses mismatch for model " << model_name << ", image: " << image_path;
                
                    break;
                }
            }
        }
    }
}

TEST_F(ResultsFixture, ComparePoses) {

    for (auto& el : results_ultralytics.items()) {

        const std::string& model_name = el.key();

        auto& ultra_results = el.value()["results"];
        auto& cpp_results = results_cpp[model_name]["results"];

        for (size_t i = 0; i < ultra_results.size(); ++i) {

            auto poses_ultra = ultra_results[i].value("inference_results", json::array());

            std::string image_path = ultra_results[i].value("image_path", "");

            for( size_t j = 0; j < cpp_results.size(); ++j) {
    
                if (cpp_results[j].value("image_path", "") == image_path) {
                    
                    auto poses_cpp = cpp_results[j].value("inference_results", json::array());

                    for (size_t j = 0; j < poses_ultra.size(); ++j) {

                        auto& det_ultra = poses_ultra[j];
                        
                        int class_id_ultra = det_ultra.value("class_id", -1);
                        double conf_ultra = det_ultra.value("confidence", 0.0);
                        
                        auto keypoints_ultra = det_ultra["keypoints"];
                        
                        auto bbox_ultra = det_ultra["bbox"];
                        
                        bool is_class_found = false;

                        for (size_t k = 0; k < poses_cpp.size(); ++k) {

                            auto& det_cpp = poses_cpp[k];

                            int class_id_cpp = det_cpp.value("class_id", -2);

                            bool already_matched = poses_cpp[k].value("_matched", false);

                            if (!already_matched && class_id_ultra == class_id_cpp) {

                                auto bbox_cpp = det_cpp["bbox"];

                                bool bbox_match = false;

                                int left_diff = std::abs(bbox_ultra["left"].get<int>() - bbox_cpp["left"].get<int>());
                                int top_diff = std::abs(bbox_ultra["top"].get<int>() - bbox_cpp["top"].get<int>());
                                int width_diff = std::abs(bbox_ultra["width"].get<int>() - bbox_cpp["width"].get<int>());
                                int height_diff = std::abs(bbox_ultra["height"].get<int>() - bbox_cpp["height"].get<int>());

                                if (left_diff <= BBOX_ERROR_MARGIN &&
                                    top_diff <= BBOX_ERROR_MARGIN &&
                                    width_diff <= BBOX_ERROR_MARGIN &&
                                    height_diff <= BBOX_ERROR_MARGIN) {
                                        bbox_match = true;
                                    }

                                ASSERT_EQ(bbox_match, true)
                                    << "Bounding box mismatch for model " << model_name
                                    << ", image: " << image_path << ", class_id: " << class_id_ultra
                                    << ": ultralytics bbox: " << bbox_ultra << " != cpp bbox: " << bbox_cpp;

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

                                ASSERT_EQ(keypoints_match, true)
                                    << "Keypoints mismatch for model " << model_name
                                    << ", image: " << image_path << ", class_id: " << class_id_ultra;

                                if (bbox_match && keypoints_match) {

                                    double conf_cpp = det_cpp.value("confidence", 0.0);
                                    double conf_diff = std::abs(conf_ultra - conf_cpp);

                                    ASSERT_LE(conf_diff, CONF_ERROR_MARGIN)
                                        << "Confidence mismatch for model " << model_name
                                        << ", image: " << image_path << ", class_id: " << class_id_ultra
                                        << ": ultralytics: " << conf_ultra << " != cpp: " << conf_cpp;
                                    
                                    is_class_found = true;

                                    poses_cpp[k]["_matched"] = true;

                                    break;
                                }
                            }
                        }
                        ASSERT_TRUE(is_class_found)
                            << "class ID " << class_id_ultra << " not found in cpp results for model "
                            << model_name << ", image: " << image_path;
                    }
                    break;
                }
            }
        }
    }
}