# Contributing to YOLOs-CPP

Thank you for your interest in contributing! This guide will help you get started.

## Ways to Contribute

- **Bug reports** — Found an issue? Open a GitHub issue
- **Feature requests** — Have an idea? Let's discuss it
- **Code contributions** — Fix bugs or add features
- **Documentation** — Improve docs, add examples
- **Testing** — Add test cases, improve coverage

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/YOLOs-CPP.git
cd YOLOs-CPP
```

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 3. Build and Test

```bash
# Build
./build.sh 1.20.1 0

# Run tests
cd tests && ./test_all.sh
```

## Code Guidelines

### Style

- **C++17** standard
- **4 spaces** for indentation (no tabs)
- **snake_case** for variables and functions
- **PascalCase** for classes and types
- Max line length: **100 characters**

### Documentation

- Document public APIs with `///` comments
- Include `@brief`, `@param`, `@return` for functions
- Add code examples where helpful

### Example

```cpp
/// @brief Detect objects in an image
/// @param image Input image (BGR format)
/// @param confThreshold Confidence threshold [0, 1]
/// @param iouThreshold IoU threshold for NMS [0, 1]
/// @return Vector of detections
[[nodiscard]] std::vector<Detection> detect(
    const cv::Mat& image,
    float confThreshold = 0.25f,
    float iouThreshold = 0.45f
);
```

## Commit Messages

Use conventional commits:

```
feat: add YOLO-World support
fix: correct NMS threshold handling
docs: update installation guide
test: add segmentation edge cases
refactor: simplify preprocessing pipeline
```

## Pull Request Process

### 1. Before Submitting

- [ ] Code compiles without warnings
- [ ] All tests pass (`./test_all.sh`)
- [ ] New code has tests
- [ ] Documentation is updated

### 2. Submit PR

- Open PR against `main` branch
- Fill out the PR template
- Link related issues

### 3. Review Process

- Maintainers will review within 3-5 days
- Address feedback promptly
- Once approved, we'll merge

## Testing

### Run All Tests

```bash
cd tests
./test_all.sh
```

### Run Specific Tests

```bash
./test_detection.sh
./test_segmentation.sh
./test_pose.sh
./test_obb.sh
./test_classification.sh
```

### Add New Tests

1. Add model to `tests/<task>/models/`
2. Update inference script
3. Add comparison cases

## Reporting Issues

### Bug Reports

Include:
- YOLOs-CPP version
- OS and compiler version
- ONNX Runtime version
- Minimal reproduction steps
- Error messages / logs

### Feature Requests

Include:
- Use case description
- Expected behavior
- Any relevant examples

## Community

- **GitHub Issues** — Bug reports, feature requests
- **GitHub Discussions** — Questions, ideas
- **Pull Requests** — Code contributions

## License

By contributing, you agree that your contributions will be licensed under AGPL-3.0.

---

Thank you for helping make YOLOs-CPP better!
