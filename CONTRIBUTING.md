# Contributing to JEE-LLM-Fine-tuning

Thank you for your interest in contributing! This guide covers everything you need to get started.

## 🚀 Quick Start

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/JEE-LLM-Fine-tuning.git
cd JEE-LLM-Fine-tuning

# Setup dev environment
conda create -n jee-llm-dev python=3.10 -y
conda activate jee-llm-dev
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install
```

## 📋 Development Workflow

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** with proper tests.

3. **Run tests and linting**:
   ```bash
   make lint   # Black + isort + flake8
   make test   # pytest suite
   ```

4. **Commit** using [Conventional Commits](https://www.conventionalcommits.org/):
   ```
   feat: add numerical accuracy tolerance setting
   fix: correct answer extraction for multiple-correct MCQ
   docs: add training guide for Mistral models
   ```

5. **Open a Pull Request** targeting `main`.

## 📁 Areas for Contribution

| Area | Description | Difficulty |
|------|-------------|------------|
| 🗃️ **Dataset** | Add more JEE problems, improve CoT quality | ⭐ Easy |
| 🧪 **Evaluation** | New benchmarks, better metrics | ⭐⭐ Medium |
| 🎓 **Training** | New PEFT methods, better hyperparameters | ⭐⭐⭐ Hard |
| 🌐 **Inference** | Improve Gradio UI, add features | ⭐⭐ Medium |
| 📚 **Documentation** | Tutorials, guides, notebooks | ⭐ Easy |
| 🔧 **Tooling** | CI/CD, Docker, packaging | ⭐⭐ Medium |

## 🧪 Testing Guidelines

- Write tests in `tests/` using `pytest`
- Test files: `test_<module_name>.py`
- Each new function needs a unit test
- Aim for ≥ 80% coverage on new code

```python
# Example test
def test_extract_boxed_answer():
    text = "...therefore \\boxed{\\pi}"
    assert extract_boxed(text) == "\\pi"
```

## 📐 Code Style

- **Formatter**: `black` (line length 100)
- **Import order**: `isort`
- **Type hints**: Required for all public functions
- **Docstrings**: Google style

```python
def solve(question: str, subject: str = "auto") -> SolveResponse:
    """
    Solve a JEE problem.

    Args:
        question: The problem text (may contain LaTeX).
        subject: Subject area: 'mathematics', 'physics', 'chemistry', or 'auto'.

    Returns:
        SolveResponse with chain-of-thought and answer.

    Raises:
        ValueError: If question is empty.
    """
```

## 📦 Dataset Contributions

To contribute JEE problems:

1. Add entries to `dataset/samples/` in JSONL format
2. Follow the schema in [docs/DATASET_GUIDE.md](docs/DATASET_GUIDE.md)
3. Ensure CoT solutions are correct and clearly structured
4. Include proper source attribution (year, paper, question number)

## 🔒 Code of Conduct

- Be respectful and constructive
- No spam or self-promotion
- Credit sources for JEE problems (publicly available exam papers)

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Questions? Open a Discussion or reach out via GitHub Issues.
