#pragma once

// Parse Ultralytics ONNX custom metadata "names" (Python dict string).

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cctype>
#include <regex>
#include <string>
#include <utility>
#include <vector>

namespace yolos {
namespace onnxmeta {

inline std::vector<std::string> parseUltralyticsNamesMetadata(const std::string& raw) {
    std::vector<std::pair<int, std::string>> pairs;
    // e.g. {0: 'person', 1: 'car', ...}
    std::regex re(R"((\d+)\s*:\s*['\"]([^'\"]*)['\"])");
    for (std::sregex_iterator it(raw.begin(), raw.end(), re), end; it != end; ++it) {
        const int idx = std::stoi((*it)[1].str());
        pairs.emplace_back(idx, (*it)[2].str());
    }
    if (pairs.empty()) {
        return {};
    }
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<std::string> out;
    out.reserve(pairs.size());
    for (const auto& p : pairs) {
        out.push_back(p.second);
    }
    return out;
}

/// Read Ultralytics `names` custom metadata from an ONNX session, if present.
inline std::vector<std::string> tryGetExportedClassNames(const Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::ModelMetadata meta = session.GetModelMetadata();
    auto namesAlloc = meta.LookupCustomMetadataMapAllocated("names", allocator);
    if (!namesAlloc) {
        return {};
    }
    const char* p = namesAlloc.get();
    if (!p || !*p) {
        return {};
    }
    return parseUltralyticsNamesMetadata(std::string(p));
}

} // namespace onnxmeta
} // namespace yolos
