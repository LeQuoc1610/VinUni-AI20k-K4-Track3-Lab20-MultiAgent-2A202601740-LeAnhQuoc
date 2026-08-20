# Lab Guide: Multi-Agent Research System

## Scenario

Bạn cần xây dựng một research assistant có thể nhận câu hỏi dài, tìm thông tin, phân tích và viết câu trả lời cuối cùng. Lab yêu cầu so sánh hai cách làm:

1. **Single-agent baseline**: một agent làm toàn bộ.
2. **Multi-agent workflow**: Supervisor điều phối Researcher, Analyst, Writer.

## Quy tắc quan trọng

- Không thêm agent nếu không có lý do rõ ràng.
- Mỗi agent phải có responsibility riêng.
- Shared state phải đủ rõ để debug.
- Phải có trace hoặc log cho từng bước.
- Phải benchmark, không chỉ nhìn output bằng cảm tính.

## Milestone 1: Baseline

File gợi ý:

- `src/multi_agent_research_lab/cli.py`
- `src/multi_agent_research_lab/services/llm_client.py`

TODO(student): thay baseline placeholder bằng một call LLM thật.

## Milestone 2: Supervisor

File gợi ý:

- `src/multi_agent_research_lab/agents/supervisor.py`
- `src/multi_agent_research_lab/graph/workflow.py`

TODO(student): implement routing policy.

Gợi ý câu hỏi thiết kế:

- Khi nào gọi Researcher?
- Khi nào gọi Analyst?
- Khi nào gọi Writer?
- Khi nào stop?
- Nếu agent fail thì retry hay fallback?

## Milestone 3: Worker agents

File gợi ý:

- `src/multi_agent_research_lab/agents/researcher.py`
- `src/multi_agent_research_lab/agents/analyst.py`
- `src/multi_agent_research_lab/agents/writer.py`

TODO(student): implement từng worker.

## Milestone 4: Trace và benchmark

File gợi ý:

- `src/multi_agent_research_lab/observability/tracing.py`
- `src/multi_agent_research_lab/evaluation/benchmark.py`
- `src/multi_agent_research_lab/evaluation/report.py`

Benchmark tối thiểu:

| Metric | Cách đo gợi ý |
|---|---|
| Latency | wall-clock time |
| Cost | token usage hoặc provider usage |
| Quality | rubric 0-10 do peer review |
| Citation coverage | số claims có source / tổng claims chính |
| Failure rate | số query fail / tổng query |

## Troubleshooting

### macOS: lỗi SSL certificate khi gọi API qua HTTPS (Tavily, OpenAI, ...)

Triệu chứng: khi implement `SearchClient` (hoặc bất kỳ HTTPS call nào) trên macOS, bạn có thể gặp lỗi kiểu:

```
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
unable to get local issuer certificate
```

Nguyên nhân: Python cài từ python.org trên macOS **không dùng** certificate store của hệ điều hành, nên không tìm thấy CA bundle hợp lệ. Đây là lỗi môi trường, **không phải** do API key sai.

Cách khắc phục (chọn 1 trong 3):

1. **Chạy script cài certificate đi kèm Python** (nhanh nhất):

   ```bash
   /Applications/Python\ 3.12/Install\ Certificates.command
   ```

   (thay `3.12` bằng version Python của bạn)

2. **Dùng `certifi` trong code** — thêm `certifi` vào dependencies, rồi tạo SSL context khi gọi HTTPS:

   ```python
   import certifi
   import ssl
   from urllib.request import urlopen

   ssl_context = ssl.create_default_context(cafile=certifi.where())
   urlopen(request, timeout=timeout, context=ssl_context)
   ```

3. **Set biến môi trường** trỏ tới CA bundle của certifi (không cần đổi code):

   ```bash
   export SSL_CERT_FILE=$(python -m certifi)
   ```

## Exit ticket

Mỗi nhóm trả lời 2 câu:

1. Case nào nên dùng multi-agent? Vì sao?
2. Case nào không nên dùng multi-agent? Vì sao?

### Trả lời

**1. Nên dùng multi-agent khi task có thể decompose thành các sub-task cần loại thông tin
hoặc loại kiểm chứng khác nhau, và verification độc lập thực sự tạo giá trị.**

Ví dụ trong repo này: câu hỏi "Research GraphRAG state-of-the-art and write a 500-word
summary" cần (a) thu thập nhiều nguồn, (b) so sánh/đánh giá độ tin cậy giữa các nguồn, (c)
viết lại có trích dẫn — ba năng lực khác nhau. Tách Researcher/Analyst/Writer cho phép mỗi
agent có prompt và trách nhiệm hẹp, dễ debug (`state.trace`, `state.agent_results` cho biết
agent nào tốn bao nhiêu token, bao lâu), và Critic có thể kiểm tra citation coverage độc lập
với Writer thay vì tự chấm bài của chính nó. Lý do cốt lõi (theo corpus offline, topic 01,
article A01/A07): multi-agent "defensible" khi task decomposition tạo ra evidence hoặc
verification path *độc lập*, không phải chỉ vì task "nghe có vẻ phức tạp". Các case cụ thể
nên dùng: research report tổng hợp nhiều nguồn mâu thuẫn nhau, task cần fact-check tách biệt
khỏi generation, hoặc task có nhiều sub-câu hỏi độc lập có thể xử lý song song.

**2. Không nên dùng multi-agent khi task ngắn, có một đường thông tin rõ ràng, và output có
thể verify rẻ.**

Ví dụ: một câu hỏi factual đơn giản trả lời được trực tiếp từ 1-2 tài liệu đã có sẵn, hoặc
một yêu cầu tóm tắt ngắn không cần trích dẫn nhiều nguồn. Trong benchmark của repo, baseline
(1 LLM call) thường có latency thấp hơn nhiều so với multi-agent (4 LLM call: researcher →
analyst → writer → critic không gọi LLM nhưng vẫn thêm 1 supervisor hop), trong khi chất
lượng không tăng tương ứng nếu câu hỏi không thực sự cần so sánh nguồn. Corpus offline (case
study CASE-01-B, "Simple task where orchestration loses") mô tả đúng hiện tượng này: pipeline
nhiều agent lặp lại cùng một source extract và tốn phần lớn thời gian cho handoff/synthesis
thay vì tạo thêm giá trị. Coordination có chi phí thật (latency, token, độ phức tạp khi
debug); nếu không có failure mode cụ thể mà multi-agent giải quyết được, baseline đơn giản là
lựa chọn đúng — đây cũng là nguyên tắc "quy tắc quan trọng" đầu bài: *không thêm agent nếu
không có lý do rõ ràng.*
