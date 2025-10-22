#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include <set>
#include <vector>
#include <nlohmann/json.hpp>

#define STRING(x) #x
#define XSTRING(x) STRING(x)

using json = nlohmann::json;

constexpr double CONF_ERROR_MARGIN = 0.1; // +-0.1 difference allowed in confidence scores

json read_json(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("File not found: " + path);
    }
    json j;
    f >> j;
    return j;
}

class ResultsFixtureCls : public ::testing::Test {
protected:
    json results_ultralytics;
    json results_cpp;
    std::string basePath = XSTRING(BASE_PATH_CLASSIFICATION);

    void SetUp() override {
        ASSERT_NO_THROW(results_ultralytics = read_json(basePath + "results/results_ultralytics.json"));
        ASSERT_NO_THROW(results_cpp = read_json(basePath + "results/results_cpp.json"));
    }
};

TEST_F(ResultsFixtureCls, ResultsNotEmpty) {
    ASSERT_FALSE(results_ultralytics.empty()) << "results_ultralytics is empty";
    ASSERT_FALSE(results_cpp.empty()) << "results_cpp is empty";
}

TEST_F(ResultsFixtureCls, CompareModelsNames) {
    std::set<std::string> models_ultra, models_cpp;
    for (auto& el : results_ultralytics.items()) models_ultra.insert(el.key());
    for (auto& el : results_cpp.items()) models_cpp.insert(el.key());
    for (const auto& name : models_ultra) {
        ASSERT_TRUE(models_cpp.count(name)) << "Model " << name << " is missing in results_cpp";
    }
}

TEST_F(ResultsFixtureCls, CompareWeightsPaths) {
    for (auto& el : results_ultralytics.items()) {
        const std::string& model_name = el.key();
        std::string weights_ultra = el.value().value("weights_path", "");
        std::string weights_cpp = results_cpp[model_name].value("weights_path", "");
        ASSERT_EQ(weights_ultra, weights_cpp) << "Weights path mismatch for model " << model_name;
    }
}

TEST_F(ResultsFixtureCls, CompareImagesCounts) {
    for (auto& el : results_ultralytics.items()) {
        const std::string& model_name = el.key();
        auto& ultra_results = el.value()["results"];
        auto& cpp_results = results_cpp[model_name]["results"];
        ASSERT_EQ(ultra_results.size(), cpp_results.size())
            << "Number of results mismatch for model " << model_name;
    }
}

TEST_F(ResultsFixtureCls, CompareImagesPaths) {
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

TEST_F(ResultsFixtureCls, CompareTop1Classification) {
    for (auto& el : results_ultralytics.items()) {
        const std::string& model_name = el.key();
        auto& ultra_results = el.value()["results"];
        auto& cpp_results = results_cpp[model_name]["results"];

        for (size_t i = 0; i < ultra_results.size(); ++i) {
            auto ultra_infs = ultra_results[i].value("inference_results", json::array());
            auto cpp_infs = cpp_results[i].value("inference_results", json::array());

            std::string image_path = ultra_results[i].value("image_path", "");

            ASSERT_FALSE(ultra_infs.empty()) << "Ultralytics inference empty for model " << model_name << ", image: " << image_path;
            ASSERT_FALSE(cpp_infs.empty()) << "CPP inference empty for model " << model_name << ", image: " << image_path;

            // Both are expected to contain a single top-1 entry
            int class_id_ultra = ultra_infs[0].value("class_id", -1);
            double conf_ultra = ultra_infs[0].value("confidence", 0.0);

            int class_id_cpp = cpp_infs[0].value("class_id", -2);
            double conf_cpp = cpp_infs[0].value("confidence", 0.0);

            ASSERT_EQ(class_id_ultra, class_id_cpp)
                << "Top-1 class mismatch for model " << model_name << ", image: " << image_path;

            double conf_diff = std::abs(conf_ultra - conf_cpp);
            ASSERT_LE(conf_diff, CONF_ERROR_MARGIN)
                << "Confidence mismatch for model " << model_name
                << ", image: " << image_path << ": ultralytics: " << conf_ultra << " != cpp: " << conf_cpp;
        }
    }
}


