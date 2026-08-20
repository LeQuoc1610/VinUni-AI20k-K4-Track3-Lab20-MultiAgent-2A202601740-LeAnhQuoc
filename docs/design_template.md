# Design Template

## Problem

Nhận một research query (text tự do, tối thiểu 5 ký tự) và trả về một câu trả lời tổng hợp,
có trích dẫn nguồn, cho một audience xác định (mặc định "technical learners"). Hệ thống phải
so sánh được với một baseline đơn giản trên latency, cost, quality, citation coverage, và
failure rate.

## Why multi-agent?

Single-agent (baseline) chỉ có 1 lần gọi LLM, không có tool access, nên: (1) không thể tra
cứu nguồn mới ngoài kiến thức sẵn có của model, (2) không có bước so sánh/đánh giá độ tin cậy
giữa các nguồn, (3) không có bước kiểm tra citation trước khi trả kết quả. Multi-agent tách
các trách nhiệm này thành các bước có thể quan sát riêng (`state.agent_results`,
`state.trace`), cho phép retry/fallback theo từng stage thay vì toàn bộ pipeline, và cho phép
Critic kiểm tra output của Writer một cách độc lập. Đánh đổi: nhiều lần gọi LLM hơn (latency,
cost cao hơn) — xem `reports/benchmark_report.md` để so sánh cụ thể.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Routing: chọn stage tiếp theo dựa trên field nào trong state còn thiếu; enforce `max_iterations`; retry-then-fallback khi một stage không tiến triển | `state.route_history`, `state.iteration`, các field đã có trong state | `state.route_history[-1]` (route mới) | Không tự fail; nếu 1 stage bị stuck ≥2 lần, tự động bỏ qua và ghi `state.errors` |
| Researcher | Gọi `SearchClient` lấy nguồn, tổng hợp `research_notes` có trích dẫn số | `request.query`, `request.max_sources` | `state.sources`, `state.research_notes` | Search trả về rỗng → ghi lỗi, dừng sớm; LLM synthesis fail → fallback dùng snippet thô |
| Analyst | Trích key claims, so sánh nguồn, đánh giá độ tin cậy (đặc biệt nguồn `is_synthetic`) | `state.research_notes`, `state.sources` | `state.analysis_notes` | LLM fail → fallback dùng `research_notes` nguyên văn, đánh dấu chưa phân tích |
| Writer | Tổng hợp `final_answer` có trích dẫn `[n]` khớp source list | `state.research_notes`, `state.analysis_notes`, `state.sources` | `state.final_answer` | LLM fail → fallback ghép `analysis_notes`/`research_notes` + source list trực tiếp |
| Critic (bonus) | Rule-based check: citation coverage, source index hợp lệ, final_answer không rỗng | `state.final_answer`, `state.sources` | `state.critic_notes` | Không tự fail (không gọi LLM); ghi cảnh báo vào `state.errors` nếu coverage thấp |

## Shared state

`ResearchState` (`src/multi_agent_research_lab/core/state.py`):

- `request`: query gốc — cần cho mọi agent để biết audience/max_sources.
- `iteration`, `route_history`: để Supervisor enforce max_iterations và tính số lần retry mỗi stage (`route_history.count(stage)`).
- `sources`, `research_notes`: output của Researcher — Analyst/Writer/Critic đều cần.
- `analysis_notes`: output của Analyst — Writer cần để viết có chiều sâu hơn snippet thô.
- `final_answer`: output của Writer — Critic kiểm tra, benchmark đo citation coverage/quality trên field này.
- `critic_notes`: output của Critic — Supervisor dùng để biết đã chạy critic hay chưa.
- `agent_results`: lưu latency/token/cost mỗi lần gọi LLM — nguồn dữ liệu chính cho benchmark cost.
- `trace`, `errors`: audit trail cho debug và cho `failure_rate`/notes trong benchmark report.

## Routing policy

```text
supervisor -> (thiếu sources/research_notes?) -> researcher -> supervisor
           -> (thiếu analysis_notes?)          -> analyst    -> supervisor
           -> (thiếu final_answer?)            -> writer     -> supervisor
           -> (critic_notes is None?)          -> critic     -> supervisor
           -> (đủ hết, hoặc max_iterations)    -> done (END)
```

Mỗi stage được thử tối đa 2 lần (đếm qua `route_history`); nếu vẫn không tiến triển,
Supervisor bỏ qua thẳng đến `writer` (dùng dữ liệu partial) rồi `critic`, thay vì lặp vô hạn.

## Guardrails

- Max iterations: `settings.max_iterations` (default 6, từ `.env`/`configs/lab_default.yaml`); Supervisor tự dừng và ghi lỗi nếu chạm giới hạn mà chưa có `final_answer`.
- Timeout: `settings.timeout_seconds` truyền vào `OpenAI` client và vào `urllib` request tới Tavily.
- Retry: `LLMClient.complete` dùng `tenacity` (3 lần, exponential backoff) cho lỗi transient (timeout/rate-limit/API error); Supervisor retry ở cấp stage (2 lần) cho lỗi persistent.
- Fallback: mỗi worker agent có fallback rule-based khi LLM fail (xem bảng Agent roles); `SearchClient` fallback sang offline corpus khi Tavily fail hoặc không có key.
- Validation: `ResearchQuery`/`SourceDocument`/`BenchmarkMetrics` là Pydantic models; LangGraph trả `dict`, được convert lại bằng `ResearchState.model_validate(...)` để re-validate toàn bộ state sau mỗi lần chạy graph.

## Benchmark plan

- Query set: `configs/lab_default.yaml` → `benchmark.queries` (3 câu mặc định).
- Metrics đo tự động (`evaluation/benchmark.py`): latency (wall-clock), estimated cost (tổng `cost_usd` từ `agent_results`, 0 với model free-tier), citation coverage (tỉ lệ câu trong `final_answer` có `[n]`), quality (heuristic 0-10, không thay thế peer review), failure rate (`final_answer` rỗng / tổng số query).
- Chạy: `make benchmark` (hoặc `python -m multi_agent_research_lab.cli benchmark`) → ghi `reports/benchmark_report.md`.
- Expected outcome: multi-agent có citation coverage và quality proxy cao hơn baseline (có source thật + bước phân tích), nhưng latency/cost cao hơn do nhiều lần gọi LLM hơn — xem report thực tế để so sánh số liệu cụ thể.
