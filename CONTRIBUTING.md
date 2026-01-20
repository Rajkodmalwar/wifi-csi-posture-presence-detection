# Contributing Guidelines

Thank you for interest in contributing to the WiFi CSI Detection project!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/wifi-csi-detection.git`
3. Create a feature branch: `git checkout -b feature/your-feature`
4. Make changes and test them

## Testing Your Changes

Before submitting a pull request, test your changes:

```bash
# Test inference pipeline
python test_direct_inference.py

# Test API endpoints
python test_api_quick.py

# Run all tests
python test_endpoints.py
```

## Code Style

- Follow PEP 8 conventions
- Add docstrings to functions and classes
- Use type hints where applicable
- Keep functions focused and modular

## Example: Adding a New Detection Task

1. Add model to `src/model/`
2. Add preprocessing to `src/preprocessing/`
3. Add endpoint to `api/main.py`
4. Add test in `test_endpoints.py`
5. Document in README.md

## Submitting Changes

1. Ensure all tests pass
2. Commit with clear messages: `git commit -m "Add feature: description"`
3. Push to your fork: `git push origin feature/your-feature`
4. Create a Pull Request with:
   - Description of changes
   - Why the change is needed
   - Any new dependencies

## Reporting Issues

When reporting bugs, include:
- Python version and OS
- Error message and traceback
- Steps to reproduce
- Expected vs actual behavior

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and improve

Thank you! 🚀
