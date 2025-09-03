# Contributing to YOLOs-CPP

We welcome contributions from the community! Here's how to get started:

## 🪄 Setup Your Environment
1. **Fork** the repository from GitHub
2. **Clone** your fork:
```bash
git clone https://github.com/YOUR_USERNAME/YOLOs-CPP.git
cd YOLOs-CPP
```
3. **Create a new branch**:
```bash
git checkout -b feature/YourFeatureName
```

## 📦 Make Your Changes
- Add headers or model support
- Fix bugs in inference logic
- Optimize performance or reduce latency
- Improve documentation in `docs/`

## ✅ Commit and Push
```bash
git add .
git commit -m "Add: Your feature description"
git push origin feature/YourFeatureName
```

## 🚀 Submit Pull Request
- Go to your GitHub fork
- Click **"New Pull Request"**
- Add a descriptive title and explanation
- Reference related issues if any

## 📏 Code Style Guidelines
- Follow modern C++14 practices
- Prefer descriptive variable/function names
- Use consistent formatting (follow existing code style)

## 📊 Benchmarking and Testing
Want to contribute benchmarks?
- Profile inference time using `cv::TickMeter` or `std::chrono`
- Test across GPU/CPU
- Validate accuracy vs expected outputs

For model-specific development, refer to `docs/DEVELOPMENT.md`.

