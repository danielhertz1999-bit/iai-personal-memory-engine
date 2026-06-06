"""IAI-MCP benchmark harness.

 benchmarks:
- bench.tokens -- (steady <=3000) + (fresh <=8000)
- bench.verbatim -- (verbatim recall >=99% on pinned records)

Both runners are invokable as CLIs (`python -m bench.tokens`, `python -m bench.verbatim`)
and exit non-zero on failure. They fall back to a heuristic token count when
ANTHROPIC_API_KEY is absent so CI (and first-time users) can run the suite offline.
"""
