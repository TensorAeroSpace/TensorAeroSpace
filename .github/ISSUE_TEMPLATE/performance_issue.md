---
name: 🚀 Performance issue
about: Report a performance degradation or optimization opportunity
title: '[PERF] '
labels: ['performance', 'needs-triage']
assignees: []

---

## 🚀 Description
Describe the performance issue (slow step, memory usage, etc.).

## 📊 Metrics / Evidence
Provide numbers, logs, profiles, flamegraphs, screenshots or reproducible metrics.

## 🔄 Reproduction steps
Steps and code to reproduce the performance problem:
```python
# Minimal code that demonstrates the performance issue
```

## 🖥️ Environment
- OS / Hardware: [CPU/GPU specs]
- Python version:
- TensorAeroSpace version:
- Frameworks (TensorFlow/PyTorch/etc.) and versions:

## 💡 Expected vs. Actual
- Expected: [e.g., training finishes in <5 min]
- Actual: [e.g., training takes 20+ min]

## 🔧 Potential root causes (optional)
- [ ] Algorithmic complexity
- [ ] Inefficient data pipeline
- [ ] Missing vectorization / batching
- [ ] Unnecessary CPU↔GPU transfers
- [ ] Debug logging in hot path
- [ ] Other: ...

## ✅ Checklist
- [ ] Reproduced with a minimal example
- [ ] Attached metrics or profiler output
- [ ] Specified environment details
