#pragma once
#include <vector>
#include <string>
#include <map>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "det/YOLO.hpp"   // YOLODetector, Detection, BoundingBox
using namespace std;

// ----------------------- Config / Stats -----------------------
struct EvalConfig {
    int imgSize = 640;
    float confThreshold = 0.001f; // collect almost all preds for PR curve
    float nmsThreshold = 0.7f;
    std::vector<float> iouThresholds;

    EvalConfig() {
        for (float t = 0.50f; t <= 0.95f + 1e-9f; t += 0.05f)
            iouThresholds.push_back(t);
    }
};

struct EvalStats {
    // combined inference time measured per image (ms)
    std::vector<double> perImageTimes;

    double totalInferenceTimeMs() const {
        return std::accumulate(perImageTimes.begin(), perImageTimes.end(), 0.0);
    }

    int imagesProcessed() const { return (int)perImageTimes.size(); }

    double avgTimeMs() const {
        return perImageTimes.empty() ? 0.0 : totalInferenceTimeMs() / perImageTimes.size();
    }

    double stddevMs() const {
        if (perImageTimes.size() <= 1) return 0.0;
        double mean = avgTimeMs();
        double accum = 0.0;
        for (double t : perImageTimes) accum += (t - mean) * (t - mean);
        return std::sqrt(accum / perImageTimes.size());
    }

    double fps() const {
        double a = avgTimeMs();
        return a > 0.0 ? 1000.0 / a : 0.0;
    }
};

// ----------------------- Data structs --------------------------
struct GTBox { int cls; float x, y, w, h; /* top-left px */ };
struct PredBox { int image_id; float x, y, w, h, score; int cls; };

// ----------------------- Utility IoU ---------------------------
static inline float bbox_iou_px(const PredBox &a, const GTBox &b) {
    float ax1 = a.x;
    float ay1 = a.y;
    float ax2 = a.x + a.w;
    float ay2 = a.y + a.h;

    float bx1 = b.x;
    float by1 = b.y;
    float bx2 = b.x + b.w;
    float by2 = b.y + b.h;

    float interX1 = max(ax1, bx1);
    float interY1 = max(ay1, by1);
    float interX2 = min(ax2, bx2);
    float interY2 = min(ay2, by2);

    float interW = max(0.0f, interX2 - interX1);
    float interH = max(0.0f, interY2 - interY1);
    float interArea = interW * interH;
    float areaA = max(0.0f, a.w * a.h);
    float areaB = max(0.0f, b.w * b.h);
    float unionArea = areaA + areaB - interArea;
    if (unionArea <= 0.0f) return 0.0f;
    return interArea / unionArea;
}

// ----------------------- I/O helpers ---------------------------
static inline vector<string> listImages(const string &folder) {
    namespace fs = std::filesystem;
    vector<string> out;
    for (const auto &e : fs::directory_iterator(folder)) {
        if (!e.is_regular_file()) continue;
        auto ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".png" || ext == ".jpeg" || ext == ".bmp" || ext == ".tif" || ext == ".tiff")
            out.push_back(e.path().string());
    }
    sort(out.begin(), out.end());
    return out;
}

// Load GT in YOLO format (cls cx cy w h) normalized --> convert to pixel top-left
static inline vector<GTBox> loadGT_yolo_to_px(const string &labelPath, int img_w, int img_h) {
    vector<GTBox> boxes;
    ifstream f(labelPath);
    if (!f.is_open()) return boxes;
    int cls; float cx, cy, w, h;
    while (f >> cls >> cx >> cy >> w >> h) {
        float bw = w * img_w;
        float bh = h * img_h;
        float cx_px = cx * img_w;
        float cy_px = cy * img_h;
        float x = cx_px - bw / 2.0f;
        float y = cy_px - bh / 2.0f;
        // Clip to image bounds (defensive)
        x = std::max(0.0f, std::min(x, float(img_w - 1)));
        y = std::max(0.0f, std::min(y, float(img_h - 1)));
        bw = std::max(0.0f, std::min(bw, float(img_w - x)));
        bh = std::max(0.0f, std::min(bh, float(img_h - y)));
        boxes.push_back({cls, x, y, bw, bh});
    }
    return boxes;
}

// ----------------------- Evaluator class -----------------------
class Evaluator {
public:
    Evaluator(YOLODetector &detector, EvalConfig cfg) : det(detector), config(cfg) {}

    // Main entry: imageFolder contains images, labelFolder contains YOLO txt labels (same stems)
    void evaluate(const string &imageFolder, const string &labelFolder) {
        auto images = listImages(imageFolder);
        cout << "Found " << images.size() << " images.\n";

        // dataset-wide storage
        map<int, vector<PredBox>> detections_by_class;
        map<int, int> gt_count_by_class;
        map<int, map<int, vector<GTBox>>> gt_by_image_and_class; // image_id -> (class -> [GTBoxes])

        stats.perImageTimes.clear();
        bool first_time = true;
        int image_id = 0;
        for (const auto &imgPath : images) {
            cv::Mat img = cv::imread(imgPath);
            if (img.empty()) { ++image_id; continue; }
            int img_w = img.cols;
            int img_h = img.rows;

            // load and convert GT
            string stem = std::filesystem::path(imgPath).stem().string();
            string labelPath = labelFolder + "/" + stem + ".txt";
            auto gt_boxes = loadGT_yolo_to_px(labelPath, img_w, img_h);

            // store GT
            for (const auto &g : gt_boxes) {
                gt_by_image_and_class[image_id][g.cls].push_back(g);
                gt_count_by_class[g.cls] += 1;
            }
            if(first_time){
                det.detect(img, config.confThreshold, config.nmsThreshold);
                first_time = false;
            }
            // run detector (measure combined inference + postprocess time)
            auto t0 = chrono::high_resolution_clock::now();
            vector<Detection> preds = det.detect(img, config.confThreshold, config.nmsThreshold);
            auto t1 = chrono::high_resolution_clock::now();
            double time_ms = chrono::duration_cast<chrono::microseconds>(t1 - t0).count() / 1000.0; // ms with fractional
            stats.perImageTimes.push_back(time_ms);

            // convert preds to PredBox and add to detections_by_class
            for (const auto &p : preds) {
                PredBox pb;
                pb.image_id = image_id;
                pb.cls = p.classId;
                pb.score = p.conf;
                // ASSUMPTION: p.box are in original image pixel coords top-left x,y,width,height.
                // If your detector returns boxes in 640x640 letterbox coords, convert them back here before adding.
                pb.x = static_cast<float>(p.box.x);
                pb.y = static_cast<float>(p.box.y);
                pb.w = static_cast<float>(p.box.width);
                pb.h = static_cast<float>(p.box.height);
                detections_by_class[pb.cls].push_back(pb);
            }

            image_id++;
        } // end images loop

        // For each IoU threshold, compute per-class AP and then average
        vector<float> ap_per_iou;
        float AP50 = 0.0f;
        for (float iou_thr : config.iouThresholds) {
            // per-class APs
            vector<float> ap_per_class;
            for (const auto &kv : detections_by_class) {
                int cls = kv.first;
                auto &dets = kv.second;
                int n_gt = gt_count_by_class[cls];
                if (n_gt == 0) continue; // COCO ignores classes with 0 GTs

                // Sort detections by score desc
                auto dets_sorted = dets;
                sort(dets_sorted.begin(), dets_sorted.end(), [](const PredBox &a, const PredBox &b){ return a.score > b.score; });

                // prepare matched flags for GTs per image
                map<int, vector<char>> matched_flag_per_image; // image_id -> matched bools (per GT index)
                for (const auto &img_gkv : gt_by_image_and_class) {
                    int imgid = img_gkv.first;
                    auto it = img_gkv.second.find(cls);
                    if (it != img_gkv.second.end()) {
                        matched_flag_per_image[imgid] = vector<char>(it->second.size(), 0);
                    }
                }

                // accumulate tp/fp arrays
                vector<int> tps; tps.reserve(dets_sorted.size());
                vector<int> fps; fps.reserve(dets_sorted.size());

                for (const auto &d : dets_sorted) {
                    // find GTs in same image and class
                    auto it_img = gt_by_image_and_class.find(d.image_id);
                    int best_gt_idx = -1;
                    float best_iou = 0.0f;
                    if (it_img != gt_by_image_and_class.end()) {
                        auto it_cls = it_img->second.find(cls);
                        if (it_cls != it_img->second.end()) {
                            const auto &gts_in_img = it_cls->second;
                            // find best unmatched GT
                            for (size_t gi = 0; gi < gts_in_img.size(); ++gi) {
                                if (matched_flag_per_image[d.image_id].size() <= gi) continue; // safety
                                if (matched_flag_per_image[d.image_id][gi]) continue;
                                float iou = bbox_iou_px(d, gts_in_img[gi]);
                                if (iou > best_iou) { best_iou = iou; best_gt_idx = int(gi); }
                            }
                        }
                    }
                    if (best_iou >= iou_thr && best_gt_idx >= 0) {
                        // true positive
                        tps.push_back(1);
                        fps.push_back(0);
                        matched_flag_per_image[d.image_id][best_gt_idx] = 1;
                    } else {
                        // false positive
                        tps.push_back(0);
                        fps.push_back(1);
                    }
                } // end dets loop

                // if no detections, AP = 0
                if (tps.empty()) {
                    ap_per_class.push_back(0.0f);
                    continue;
                }

                // cumulative sums
                vector<int> tp_cum(tps.size()), fp_cum(fps.size());
                int s = 0; for (size_t i=0;i<tps.size();++i){ s += tps[i]; tp_cum[i]=s; }
                s = 0; for (size_t i=0;i<fps.size();++i){ s += fps[i]; fp_cum[i]=s; }

                vector<float> precision(tp_cum.size()), recall(tp_cum.size());
                for (size_t i=0;i<tp_cum.size();++i) {
                    float tpv = float(tp_cum[i]);
                    float fpv = float(fp_cum[i]);
                    precision[i] = tpv / (tpv + fpv + 1e-12f);
                    recall[i] = tpv / (float)n_gt + 1e-12f;
                }

                // compute AP via 101-point interpolation (COCO style)
                const int NUM_POINTS = 101;
                float ap = 0.0f;
                for (int ri = 0; ri < NUM_POINTS; ++ri) {
                    float r_thr = ri / float(NUM_POINTS - 1); // 0.0 ... 1.0
                    float p_max = 0.0f;
                    // find max precision where recall >= r_thr
                    for (size_t k = 0; k < recall.size(); ++k) {
                        if (recall[k] >= r_thr) {
                            if (precision[k] > p_max) p_max = precision[k];
                        }
                    }
                    ap += p_max;
                }
                ap /= NUM_POINTS;
                ap_per_class.push_back(ap);
            } // end per-class

            // average AP across classes (COCO averages over classes present in GT)
            if (ap_per_class.empty()) {
                ap_per_iou.push_back(0.0f);
            } else {
                float sum = 0.0f;
                for (float v : ap_per_class) sum += v;
                ap_per_iou.push_back(sum / float(ap_per_class.size()));
            }

            if (fabs(iou_thr - 0.50f) < 1e-6f) AP50 = ap_per_iou.back();
        } // end IoU thresholds loop

        // mAP50-95 is mean across IoU thresholds
        float mAP5095 = 0.0f;
        if (!ap_per_iou.empty()) {
            for (float v : ap_per_iou) mAP5095 += v;
            mAP5095 /= float(ap_per_iou.size());
        }

        // print results
        cout << "\n=== Evaluation Results ===\n";
        for (size_t i = 0; i < config.iouThresholds.size(); ++i) {
            cout << "IoU " << config.iouThresholds[i] << "  AP=" << ap_per_iou[i] << "\n";
        }
        cout << "\nAP50 = " << AP50 << "\n";
        cout << "mAP50-95 = " << mAP5095 << "\n";

        cout << "\n=== Speed ===\n";
        cout << "Images processed = " << stats.imagesProcessed() << "\n";
        cout << "Inference time (mean ± stddev): " << stats.avgTimeMs() << " ± " << stats.stddevMs() << " ms\n";
        cout << "FPS (from mean): " << stats.fps() << "\n";

        // store in class members
        this->AP50 = AP50;
        this->mAP5095 = mAP5095;
        this->stats_local = stats; // copy out if caller wants it
    }

    // Results
    float AP50 = 0;
    float mAP5095 = 0;
    EvalStats stats_local;

private:
    YOLODetector &det;
    EvalConfig config;
    EvalStats stats;
}; // end Evaluator
