# Changelog

All notable changes to JEE-LLM-Fine-tuning are documented here.

Format: [Semantic Versioning](https://semver.org/) | [Keep a Changelog](https://keepachangelog.com/)

---

## [Unreleased]

### Planned
- Mixture-of-Experts adapter merging
- Tool-augmented reasoning (Python code interpreter)
- RAG over NCERT textbooks and HC Verma
- Multi-modal support for diagram-based problems
- Mobile deployment (GGUF + llama.cpp)

---

## [1.0.0] — 2025-05-01

### Added
- **Stage 1 SFT**: QLoRA fine-tuning pipeline for LLaMA-3-8B on 50K JEE CoT examples
- **Stage 2 DPO**: Direct Preference Optimization for answer quality alignment
- **Stage 2 GRPO**: Group Relative Policy Optimization with mathematical correctness reward
- **Stage 3 SPIN**: Self-Play Fine-Tuning for iterative improvement
- **Gradio UI**: Full-featured daily-use web interface with LaTeX rendering
- **FastAPI Server**: Production REST API with streaming SSE support
- **Evaluation Suite**: JEEBench evaluation with HTML report generation
- **Dataset Builder**: GPT-4o CoT distillation pipeline
- **DeepSpeed ZeRO-3**: Multi-GPU training support
- **WandB Integration**: Full experiment tracking
- **GitHub Actions**: CI/CD pipeline with linting, testing, config validation

### Models
- JEE-LLM v1 (LLaMA-3-8B base): 58.7% JEEBench overall accuracy
  - Mathematics: 61.4%
  - Physics: 56.7%
  - Chemistry: 58.1%

### Datasets
- `jee_advanced_2003_2024`: 8,500 problems with solutions
- `jee_mains_2013_2024`: 32,000 problems
- `jee_cot_gpt4o`: 50,000 Chain-of-Thought examples
- `jee_preference_pairs`: 12,000 DPO preference pairs
- `jeebench_eval`: 515-problem evaluation benchmark

---

## [0.1.0] — 2025-03-15

### Added
- Initial repository structure
- Basic README and project description
- Proof-of-concept training script

[Unreleased]: https://github.com/Aranya2801/JEE-LLM-Fine-tuning/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Aranya2801/JEE-LLM-Fine-tuning/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/Aranya2801/JEE-LLM-Fine-tuning/releases/tag/v0.1.0
