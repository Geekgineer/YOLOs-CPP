/**
 * @file compare_results.cpp
 * @brief Compare YOLOs-CPP depth maps against the Ultralytics reference
 *
 * Dense maps are compared per pixel, not by summary statistics: a map can have
 * the right range and still be wrong everywhere.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#define STRING(x) #x
#define XSTRING(x) STRING(x)

using json = nlohmann::json;

namespace {

// Thresholds. Tighten after the first CI run records the real residual; loosen
// only with a recorded reason. Issue #137 is the precedent: a tolerance loose
// enough to hide a real bug is worse than no test.
constexpr double ABS_REL_MAX = 0.01;    // mean |d_cpp - d_ref| / d_ref
constexpr double DELTA1_MIN = 0.99;     // fraction of pixels within 1%
constexpr double RANGE_REL_MAX = 0.01;  // min/max agreement

json readJson(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("File not found: " + path);
    json j;
    f >> j;
    return j;
}

std::vector<float> readRaw(const std::string& path, size_t count) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Depth file not found: " + path);
    std::vector<float> data(count);
    f.read(reinterpret_cast<char*>(data.data()),
           static_cast<std::streamsize>(count * sizeof(float)));
    if (!f) throw std::runtime_error("Short read on depth file: " + path);
    return data;
}

} // namespace

class ResultsFixtureDepth : public ::testing::Test {
protected:
    json ultra;
    json cpp;
    std::string basePath = XSTRING(BASE_PATH_DEPTH);

    void SetUp() override {
        // Skip rather than fail when the parity run has not been executed: the
        // self-contained tests in this same binary must stay meaningful for anyone
        // who builds without downloading weights. test_depth.sh always generates
        // both files first, so a real parity run never skips.
        const std::string ultraPath = basePath + "results/results_ultralytics.json";
        const std::string cppPath = basePath + "results/results_cpp.json";
        if (!std::ifstream(ultraPath).good() || !std::ifstream(cppPath).good()) {
            GTEST_SKIP() << "Parity results missing. Run tests/test_depth.sh to generate "
                         << ultraPath << " and " << cppPath;
        }
        ASSERT_NO_THROW(ultra = readJson(ultraPath));
        ASSERT_NO_THROW(cpp = readJson(cppPath));
    }
};

TEST_F(ResultsFixtureDepth, ResultsNotEmpty) {
    ASSERT_FALSE(ultra.empty()) << "results_ultralytics is empty";
    ASSERT_FALSE(cpp.empty()) << "results_cpp is empty";
}

TEST_F(ResultsFixtureDepth, CompareModelsNames) {
    for (auto& el : ultra.items()) {
        ASSERT_TRUE(cpp.contains(el.key())) << "Model " << el.key() << " missing in results_cpp";
    }
}

TEST_F(ResultsFixtureDepth, CompareImagesCounts) {
    for (auto& el : ultra.items()) {
        ASSERT_EQ(el.value()["results"].size(), cpp[el.key()]["results"].size())
            << "Result count mismatch for " << el.key();
    }
}

TEST_F(ResultsFixtureDepth, CompareDepthMapShapes) {
    for (auto& el : ultra.items()) {
        auto& u = el.value()["results"];
        auto& c = cpp[el.key()]["results"];
        for (size_t i = 0; i < u.size(); ++i) {
            ASSERT_EQ(u[i].value("height", -1), c[i].value("height", -2))
                << "Height mismatch for " << el.key() << " image " << i;
            ASSERT_EQ(u[i].value("width", -1), c[i].value("width", -2))
                << "Width mismatch for " << el.key() << " image " << i;
        }
    }
}

TEST_F(ResultsFixtureDepth, CompareDepthRanges) {
    for (auto& el : ultra.items()) {
        auto& u = el.value()["results"];
        auto& c = cpp[el.key()]["results"];
        for (size_t i = 0; i < u.size(); ++i) {
            const double uMin = u[i].value("min", 0.0);
            const double uMax = u[i].value("max", 0.0);
            const double cMin = c[i].value("min", 0.0);
            const double cMax = c[i].value("max", 0.0);

            ASSERT_GT(uMin, 0.0) << "reference min must be positive metric depth";
            EXPECT_LE(std::abs(cMin - uMin) / uMin, RANGE_REL_MAX)
                << el.key() << " image " << i << ": min " << cMin << " vs " << uMin;
            EXPECT_LE(std::abs(cMax - uMax) / uMax, RANGE_REL_MAX)
                << el.key() << " image " << i << ": max " << cMax << " vs " << uMax;
        }
    }
}

TEST_F(ResultsFixtureDepth, ComparePerPixelDepth) {
    for (auto& el : ultra.items()) {
        auto& u = el.value()["results"];
        auto& c = cpp[el.key()]["results"];

        for (size_t i = 0; i < u.size(); ++i) {
            const int h = u[i].value("height", 0);
            const int w = u[i].value("width", 0);
            ASSERT_GT(h * w, 0);
            const size_t count = static_cast<size_t>(h) * static_cast<size_t>(w);

            std::vector<float> ref, got;
            ASSERT_NO_THROW(ref = readRaw(basePath + u[i].value("depth_file", ""), count));
            ASSERT_NO_THROW(got = readRaw(basePath + c[i].value("depth_file", ""), count));

            double sumAbsRel = 0.0;
            size_t valid = 0;
            size_t within = 0;
            for (size_t p = 0; p < count; ++p) {
                if (!(ref[p] > 0.0f) || !(got[p] > 0.0f)) continue;
                ++valid;
                sumAbsRel += std::abs(static_cast<double>(got[p]) - ref[p]) / ref[p];
                const double ratio = std::max(static_cast<double>(got[p]) / ref[p],
                                              static_cast<double>(ref[p]) / got[p]);
                if (ratio < 1.01) ++within;
            }

            ASSERT_GT(valid, count / 2) << "too few valid reference pixels";

            const double absRel = sumAbsRel / static_cast<double>(valid);
            const double delta1 = static_cast<double>(within) / static_cast<double>(valid);

            // Always print, not just on failure: the thresholds above are meant to be
            // pinned from observed values, which requires seeing them on a passing run.
            std::cout << "[metrics] " << el.key() << " " << u[i].value("image_path", "")
                      << " AbsRel=" << absRel << " delta1=" << delta1 << std::endl;

            EXPECT_LE(absRel, ABS_REL_MAX)
                << el.key() << " image " << i << " (" << u[i].value("image_path", "")
                << "): AbsRel " << absRel;
            EXPECT_GE(delta1, DELTA1_MIN)
                << el.key() << " image " << i << " (" << u[i].value("image_path", "")
                << "): delta1 " << delta1;
        }
    }
}
