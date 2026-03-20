# Contributing to YOLO Vision Analytics

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/ultimate144z/yolo-vision-analytics.git
cd yolo-vision-analytics
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running Tests

```bash
pytest tests/ -v
pytest tests/ --cov=src
```

## Making Changes

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Add or update tests if applicable
4. Run the test suite to make sure everything passes
5. Submit a pull request

## Code Style

- Follow PEP 8
- Use type hints for function signatures
- Keep functions focused and small
- Add docstrings for public methods

## Reporting Bugs

Use the [bug report template](https://github.com/ultimate144z/yolo-vision-analytics/issues/new?template=bug_report.yml) to file issues.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
