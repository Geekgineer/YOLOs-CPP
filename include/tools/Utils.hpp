#include <opencv2/opencv.hpp>

/**
 * @brief Struct to represent a bounding box.
 */
struct BoundingBox
{
    int x;
    int y;
    int width;
    int height;

    BoundingBox() : x(0), y(0), width(0), height(0) {}
    BoundingBox(int x_, int y_, int width_, int height_)
        : x(x_), y(y_), width(width_), height(height_) {}
};

/**
 * @brief Struct to represent a detection.
 */
struct Detection
{
    BoundingBox box;
    float conf{};
    int classId{};
};

/**
 * @namespace utils
 * @brief Namespace containing utility functions for the YOLO11Detector.
 */
namespace utils
{

    class MathUtils
    {
    public:
        /**
         * @brief A robust implementation of a clamp function.
         *        Restricts a value to lie within a specified range [low, high].
         *
         * @tparam T The type of the value to clamp. Should be an arithmetic type (int, float, etc.).
         * @param value The value to clamp.
         * @param low The lower bound of the range.
         * @param high The upper bound of the range.
         * @return const T& The clamped value, constrained to the range [low, high].
         *
         * @note If low > high, the function swaps the bounds automatically to ensure valid behavior.
         */
        template <typename T>
        static typename std::enable_if<std::is_arithmetic<T>::value, T>::type inline clamp(const T &value, const T &low, const T &high)
        {
            // Ensure the range [low, high] is valid; swap if necessary
            T validLow = low < high ? low : high;
            T validHigh = low < high ? high : low;

            // Clamp the value to the range [validLow, validHigh]
            if (value < validLow)
                return validLow;
            if (value > validHigh)
                return validHigh;
            return value;
        }

        /**
         * @brief Computes the product of elements in a vector.
         *
         * @param vector Vector of integers.
         * @return size_t Product of all elements.
         */
        static size_t vectorProduct(const std::vector<int64_t> &vector)
        {
            return std::accumulate(vector.begin(), vector.end(), 1ull, std::multiplies<size_t>());
        }
    };

    class ImagePreprocessingUtils
    {
    public:
        /**
         * @brief Resizes an image with letterboxing to maintain aspect ratio.
         *
         * @param image Input image.
         * @param outImage Output resized and padded image.
         * @param newShape Desired output size.
         * @param color Padding color (default is gray).
         * @param auto_ Automatically adjust padding to be multiple of stride.
         * @param scaleFill Whether to scale to fill the new shape without keeping aspect ratio.
         * @param scaleUp Whether to allow scaling up of the image.
         * @param stride Stride size for padding alignment.
         */
        static inline void letterBox(const cv::Mat &image, cv::Mat &outImage,
                                     const cv::Size &newShape,
                                     const cv::Scalar &color = cv::Scalar(114, 114, 114),
                                     bool auto_ = true,
                                     bool scaleFill = false,
                                     bool scaleUp = true,
                                     int stride = 32)
        {
            // Calculate the scaling ratio to fit the image within the new shape
            float ratio = std::min(static_cast<float>(newShape.height) / image.rows,
                                   static_cast<float>(newShape.width) / image.cols);

            // Prevent scaling up if not allowed
            if (!scaleUp)
            {
                ratio = std::min(ratio, 1.0f);
            }

            // Calculate new dimensions after scaling
            int newUnpadW = static_cast<int>(std::round(image.cols * ratio));
            int newUnpadH = static_cast<int>(std::round(image.rows * ratio));

            // Calculate padding needed to reach the desired shape
            int dw = newShape.width - newUnpadW;
            int dh = newShape.height - newUnpadH;

            if (auto_)
            {
                // Ensure padding is a multiple of stride for model compatibility
                dw = (dw % stride) / 2;
                dh = (dh % stride) / 2;
            }
            else if (scaleFill)
            {
                // Scale to fill without maintaining aspect ratio
                newUnpadW = newShape.width;
                newUnpadH = newShape.height;
                ratio = std::min(static_cast<float>(newShape.width) / image.cols,
                                 static_cast<float>(newShape.height) / image.rows);
                dw = 0;
                dh = 0;
            }
            else
            {
                // Evenly distribute padding on both sides
                // Calculate separate padding for left/right and top/bottom to handle odd padding
                int padLeft = dw / 2;
                int padRight = dw - padLeft;
                int padTop = dh / 2;
                int padBottom = dh - padTop;

                // Resize the image if the new dimensions differ
                if (image.cols != newUnpadW || image.rows != newUnpadH)
                {
                    cv::resize(image, outImage, cv::Size(newUnpadW, newUnpadH), 0, 0, cv::INTER_LINEAR);
                }
                else
                {
                    // Avoid unnecessary copying if dimensions are the same
                    outImage = image;
                }

                // Apply padding to reach the desired shape
                cv::copyMakeBorder(outImage, outImage, padTop, padBottom, padLeft, padRight, cv::BORDER_CONSTANT, color);
                return; // Exit early since padding is already applied
            }

            // Resize the image if the new dimensions differ
            if (image.cols != newUnpadW || image.rows != newUnpadH)
            {
                cv::resize(image, outImage, cv::Size(newUnpadW, newUnpadH), 0, 0, cv::INTER_LINEAR);
            }
            else
            {
                // Avoid unnecessary copying if dimensions are the same
                outImage = image;
            }

            // Calculate separate padding for left/right and top/bottom to handle odd padding
            int padLeft = dw / 2;
            int padRight = dw - padLeft;
            int padTop = dh / 2;
            int padBottom = dh - padTop;

            // Apply padding to reach the desired shape
            cv::copyMakeBorder(outImage, outImage, padTop, padBottom, padLeft, padRight, cv::BORDER_CONSTANT, color);
        }

        /**
         * @brief Scales detection coordinates back to the original image size.
         *
         * @param imageShape Shape of the resized image used for inference.
         * @param bbox Detection bounding box to be scaled.
         * @param imageOriginalShape Original image size before resizing.
         * @param p_Clip Whether to clip the coordinates to the image boundaries.
         * @return BoundingBox Scaled bounding box.
         */
        static BoundingBox scaleCoords(const cv::Size &imageShape, BoundingBox coords,
                                       const cv::Size &imageOriginalShape, bool p_Clip)
        {
            BoundingBox result;
            float gain = std::min(static_cast<float>(imageShape.height) / static_cast<float>(imageOriginalShape.height),
                                  static_cast<float>(imageShape.width) / static_cast<float>(imageOriginalShape.width));

            int padX = static_cast<int>(std::round((imageShape.width - imageOriginalShape.width * gain) / 2.0f));
            int padY = static_cast<int>(std::round((imageShape.height - imageOriginalShape.height * gain) / 2.0f));

            result.x = static_cast<int>(std::round((coords.x - padX) / gain));
            result.y = static_cast<int>(std::round((coords.y - padY) / gain));
            result.width = static_cast<int>(std::round(coords.width / gain));
            result.height = static_cast<int>(std::round(coords.height / gain));

            if (p_Clip)
            {
                result.x = utils::MathUtils::clamp(result.x, 0, imageOriginalShape.width);
                result.y = utils::MathUtils::clamp(result.y, 0, imageOriginalShape.height);
                result.width = utils::MathUtils::clamp(result.width, 0, imageOriginalShape.width - result.x);
                result.height = utils::MathUtils::clamp(result.height, 0, imageOriginalShape.height - result.y);
            }
            return result;
        }
    };

    class DrawingUtils
    {
    public:
        /**
         * @brief Generates a vector of colors for each class name.
         *
         * @param classNames Vector of class names.
         * @param seed Seed for random color generation to ensure reproducibility.
         * @return std::vector<cv::Scalar> Vector of colors.
         */
        static inline std::vector<cv::Scalar> generateColors(const std::vector<std::string> &classNames, int seed = 42)
        {
            // Static cache to store colors based on class names to avoid regenerating
            static std::unordered_map<size_t, std::vector<cv::Scalar>> colorCache;

            // Compute a hash key based on class names to identify unique class configurations
            size_t hashKey = 0;
            for (const auto &name : classNames)
            {
                hashKey ^= std::hash<std::string>{}(name) + 0x9e3779b9 + (hashKey << 6) + (hashKey >> 2);
            }

            // Check if colors for this class configuration are already cached
            auto it = colorCache.find(hashKey);
            if (it != colorCache.end())
            {
                return it->second;
            }

            // Generate unique random colors for each class
            std::vector<cv::Scalar> colors;
            colors.reserve(classNames.size());

            std::mt19937 rng(seed);                         // Initialize random number generator with fixed seed
            std::uniform_int_distribution<int> uni(0, 255); // Define distribution for color values

            for (size_t i = 0; i < classNames.size(); ++i)
            {
                colors.emplace_back(cv::Scalar(uni(rng), uni(rng), uni(rng))); // Generate random BGR color
            }

            // Cache the generated colors for future use
            colorCache.emplace(hashKey, colors);

            return colorCache[hashKey];
        }

        /**
         * @brief Draws bounding boxes and labels on the image based on detections.
         *
         * @param image Image on which to draw.
         * @param detections Vector of detections.
         * @param classNames Vector of class names corresponding to object IDs.
         * @param colors Vector of colors for each class.
         */
        static inline void drawBoundingBox(cv::Mat &image, const std::vector<Detection> &detections,
                                           const std::vector<std::string> &classNames, const std::vector<cv::Scalar> &colors, const float confidence_threshold = 0.0)
        {
            // Iterate through each detection to draw bounding boxes and labels
            for (const auto &detection : detections)
            {
                // Skip detections below the confidence threshold
                if (detection.conf <= confidence_threshold)
                    continue;

                // Ensure the object ID is within valid range
                if (detection.classId < 0 || static_cast<size_t>(detection.classId) >= classNames.size())
                    continue;

                // Select color based on object ID for consistent coloring
                const cv::Scalar &color = colors[detection.classId % colors.size()];

                // Draw the bounding box rectangle
                cv::rectangle(image, cv::Point(detection.box.x, detection.box.y),
                              cv::Point(detection.box.x + detection.box.width, detection.box.y + detection.box.height),
                              color, 2, cv::LINE_AA);

                // Prepare label text with class name and confidence percentage
                std::string label = classNames[detection.classId] + ": " + std::to_string(static_cast<int>(detection.conf * 100)) + "%";

                // Define text properties for labels
                int fontFace = cv::FONT_HERSHEY_SIMPLEX;
                double fontScale = std::min(image.rows, image.cols) * 0.0008;
                const int thickness = std::max(1, static_cast<int>(std::min(image.rows, image.cols) * 0.002));
                int baseline = 0;

                // Calculate text size for background rectangles
                cv::Size textSize = cv::getTextSize(label, fontFace, fontScale, thickness, &baseline);

                // Define positions for the label
                int labelY = std::max(detection.box.y, textSize.height + 5);
                cv::Point labelTopLeft(detection.box.x, labelY - textSize.height - 5);
                cv::Point labelBottomRight(detection.box.x + textSize.width + 5, labelY + baseline - 5);

                // Draw background rectangle for label
                cv::rectangle(image, labelTopLeft, labelBottomRight, color, cv::FILLED);

                // Put label text
                cv::putText(image, label, cv::Point(detection.box.x + 2, labelY - 2), fontFace, fontScale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
            }
        }

        /**
         * @brief Draws bounding boxes and semi-transparent masks on the image based on detections.
         *
         * @param image Image on which to draw.
         * @param detections Vector of detections.
         * @param classNames Vector of class names corresponding to object IDs.
         * @param classColors Vector of colors for each class.
         * @param maskAlpha Alpha value for the mask transparency.
         */
        static inline void drawBoundingBoxMask(cv::Mat &image, const std::vector<Detection> &detections,
                                               const std::vector<std::string> &classNames, const std::vector<cv::Scalar> &classColors,
                                               float maskAlpha = 0.4f, const float confidence_threshold = 0.0)
        {
            // Validate input image
            if (image.empty())
            {
                std::cerr << "ERROR: Empty image provided to drawBoundingBoxMask." << std::endl;
                return;
            }

            const int imgHeight = image.rows;
            const int imgWidth = image.cols;

            // Precompute dynamic font size and thickness based on image dimensions
            const double fontSize = std::min(imgHeight, imgWidth) * 0.0006;
            const int textThickness = std::max(1, static_cast<int>(std::min(imgHeight, imgWidth) * 0.001));

            // Create a mask image for blending (initialized to zero)
            cv::Mat maskImage(image.size(), image.type(), cv::Scalar::all(0));

            // Pre-filter detections to include only those above the confidence threshold and with valid class IDs
            std::vector<const Detection *> filteredDetections;
            for (const auto &detection : detections)
            {
                if (detection.conf > confidence_threshold &&
                    detection.classId >= 0 &&
                    static_cast<size_t>(detection.classId) < classNames.size())
                {
                    filteredDetections.emplace_back(&detection);
                }
            }

            // Draw filled rectangles on the mask image for the semi-transparent overlay
            for (const auto *detection : filteredDetections)
            {
                cv::Rect box(detection->box.x, detection->box.y, detection->box.width, detection->box.height);
                const cv::Scalar &color = classColors[detection->classId];
                cv::rectangle(maskImage, box, color, cv::FILLED);
            }

            // Blend the maskImage with the original image to apply the semi-transparent masks
            cv::addWeighted(maskImage, maskAlpha, image, 1.0f, 0, image);

            // Draw bounding boxes and labels on the original image
            for (const auto *detection : filteredDetections)
            {
                cv::Rect box(detection->box.x, detection->box.y, detection->box.width, detection->box.height);
                const cv::Scalar &color = classColors[detection->classId];
                cv::rectangle(image, box, color, 2, cv::LINE_AA);

                std::string label = classNames[detection->classId] + ": " + std::to_string(static_cast<int>(detection->conf * 100)) + "%";
                int baseLine = 0;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, fontSize, textThickness, &baseLine);

                int labelY = std::max(detection->box.y, labelSize.height + 5);
                cv::Point labelTopLeft(detection->box.x, labelY - labelSize.height - 5);
                cv::Point labelBottomRight(detection->box.x + labelSize.width + 5, labelY + baseLine - 5);

                // Draw background rectangle for label
                cv::rectangle(image, labelTopLeft, labelBottomRight, color, cv::FILLED);

                // Put label text
                cv::putText(image, label, cv::Point(detection->box.x + 2, labelY - 2), cv::FONT_HERSHEY_SIMPLEX, fontSize, cv::Scalar(255, 255, 255), textThickness, cv::LINE_AA);
            }

            DEBUG_PRINT("Bounding boxes and masks drawn on image.");
        }
    };

    /**
     * @brief Loads class names from a given file path.
     *
     * @param path Path to the file containing class names.
     * @return std::vector<std::string> Vector of class names.
     */
    std::vector<std::string> getClassNames(const std::string &path)
    {
        std::vector<std::string> classNames;
        std::ifstream infile(path);

        if (infile)
        {
            std::string line;
            while (getline(infile, line))
            {
                // Remove carriage return if present (for Windows compatibility)
                if (!line.empty() && line.back() == '\r')
                    line.pop_back();
                classNames.emplace_back(line);
            }
        }
        else
        {
            std::cerr << "ERROR: Failed to access class name path: " << path << std::endl;
        }

        DEBUG_PRINT("Loaded " << classNames.size() << " class names from " + path);
        return classNames;
    }

    /**
     * @brief Performs Non-Maximum Suppression (NMS) on the bounding boxes.
     *
     * @param boundingBoxes Vector of bounding boxes.
     * @param scores Vector of confidence scores corresponding to each bounding box.
     * @param scoreThreshold Confidence threshold to filter boxes.
     * @param nmsThreshold IoU threshold for NMS.
     * @param indices Output vector of indices that survive NMS.
     */
    // Optimized Non-Maximum Suppression Function
    void NMSBoxes(const std::vector<BoundingBox> &boundingBoxes,
                  const std::vector<float> &scores,
                  float scoreThreshold,
                  float nmsThreshold,
                  std::vector<int> &indices)
    {
        indices.clear();

        const size_t numBoxes = boundingBoxes.size();
        if (numBoxes == 0)
        {
            DEBUG_PRINT("No bounding boxes to process in NMS");
            return;
        }

        // Step 1: Filter out boxes with scores below the threshold
        // and create a list of indices sorted by descending scores
        std::vector<int> sortedIndices;
        sortedIndices.reserve(numBoxes);
        for (size_t i = 0; i < numBoxes; ++i)
        {
            if (scores[i] >= scoreThreshold)
            {
                sortedIndices.push_back(static_cast<int>(i));
            }
        }

        // If no boxes remain after thresholding
        if (sortedIndices.empty())
        {
            DEBUG_PRINT("No bounding boxes above score threshold");
            return;
        }

        // Sort the indices based on scores in descending order
        std::sort(sortedIndices.begin(), sortedIndices.end(),
                  [&scores](int idx1, int idx2)
                  {
                      return scores[idx1] > scores[idx2];
                  });

        // Step 2: Precompute the areas of all boxes
        std::vector<float> areas(numBoxes, 0.0f);
        for (size_t i = 0; i < numBoxes; ++i)
        {
            areas[i] = boundingBoxes[i].width * boundingBoxes[i].height;
        }

        // Step 3: Suppression mask to mark boxes that are suppressed
        std::vector<bool> suppressed(numBoxes, false);

        // Step 4: Iterate through the sorted list and suppress boxes with high IoU
        for (size_t i = 0; i < sortedIndices.size(); ++i)
        {
            int currentIdx = sortedIndices[i];
            if (suppressed[currentIdx])
            {
                continue;
            }

            // Select the current box as a valid detection
            indices.push_back(currentIdx);

            const BoundingBox &currentBox = boundingBoxes[currentIdx];
            const float x1_max = currentBox.x;
            const float y1_max = currentBox.y;
            const float x2_max = currentBox.x + currentBox.width;
            const float y2_max = currentBox.y + currentBox.height;
            const float area_current = areas[currentIdx];

            // Compare IoU of the current box with the rest
            for (size_t j = i + 1; j < sortedIndices.size(); ++j)
            {
                int compareIdx = sortedIndices[j];
                if (suppressed[compareIdx])
                {
                    continue;
                }

                const BoundingBox &compareBox = boundingBoxes[compareIdx];
                const float x1 = std::max(x1_max, static_cast<float>(compareBox.x));
                const float y1 = std::max(y1_max, static_cast<float>(compareBox.y));
                const float x2 = std::min(x2_max, static_cast<float>(compareBox.x + compareBox.width));
                const float y2 = std::min(y2_max, static_cast<float>(compareBox.y + compareBox.height));

                const float interWidth = x2 - x1;
                const float interHeight = y2 - y1;

                if (interWidth <= 0 || interHeight <= 0)
                {
                    continue;
                }

                const float intersection = interWidth * interHeight;
                const float unionArea = area_current + areas[compareIdx] - intersection;
                const float iou = (unionArea > 0.0f) ? (intersection / unionArea) : 0.0f;

                if (iou > nmsThreshold)
                {
                    suppressed[compareIdx] = true;
                }
            }
        }

        DEBUG_PRINT("NMS completed with " + std::to_string(indices.size()) + " indices remaining");
    }

};