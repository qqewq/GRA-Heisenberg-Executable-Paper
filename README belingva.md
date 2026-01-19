# GRA–Heisenberg Reasoning Architecture

An executable research architecture for stabilizing long-horizon reasoning
under solution degeneracy via orthogonal constraint collapse.

---

## 🇬🇧 English

### Overview

Modern large language model (LLM) systems can generate plausible
hypotheses, but often fail to converge on stable reasoning when
multiple plausible solutions exist (degeneracy).

This repository proposes a two-loop architecture combining:

1. A **Generalized Resonance Algorithm (GRA)** for degeneracy detection
2. A **Heisenberg-style uncertainty bound** for controlled collapse
3. A **meta-control outer loop**
4. A clear separation between hypothesis generation (LLM)
   and reasoning stabilization

This is **not a chatbot**. It is a conceptual architecture
for *stable AI reasoning*.

### Contents

- `ARCHITECTURE.md` — architecture design (EN/RU)  
- `THEORY.md` — theoretical foundations (EN/RU)  
- `EXAMPLES.md` — reasoning examples (EN/RU)  
- `src/` — reference implementation modules  
- `simulator/` — interactive visualization README  
- `LICENSE` — MIT license

### Quick demo

If the simulator is deployed via GitHub Pages, the link will appear here.

---

## 🇷🇺 Русский

### Обзор

Современные системы LLM способны порождать правдоподобные гипотезы,
но часто не способны устойчиво рассуждать, когда существует множество
равнозначных решений (дегенерация).

Этот репозиторий описывает двухконтурную архитектуру, сочетающую:

1. **Обобщённый резонансный алгоритм (GRA)** для обнаружения вырожденности
2. **Ограничение по неопределённости (Heisenberg)**
3. **Метаконтрольный внешний цикл**
4. Чёткое разделение генерации гипотез (LLM)
   и стабилизации рассуждений

Это **не чат-бот**. Это концептуальная архитектура
для *устойчивого ИИ-рассуждения*.

### Содержание

См. разделы выше.

---

## License

MIT License — free for research and experimentation.