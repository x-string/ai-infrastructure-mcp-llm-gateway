"""Streams shaped like those of real OpenAI-compatible providers, each with its source.

Every stream is reconstructed from a cited source: a docs page, a source file at a commit, or an
issue. Ids and texts are invented; the shape is what the source shows. The expected outcome is
the gateway's rule for that shape:

* ``success``: the text relayed per channel (before redaction), the finish reason sent to the
  client, and whether usage is available;
* ``reported_error``: the provider said the generation failed;
* ``protocol_error``: the gateway cannot vouch for the stream (an unexpected channel, invalid
  data, or a stream that ended without a finish reason, so possibly truncated).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

Outcome = Literal["success", "reported_error", "protocol_error"]


@dataclass(frozen=True)
class ProviderStream:
    name: str
    provider: str
    source: str
    body: bytes
    outcome: Outcome
    text: str = ""
    refusal: str = ""
    finish: str | None = None
    usage: bool = False


def d(payload: object) -> str:
    """One ``data:`` event."""
    return "data: " + json.dumps(payload, separators=(",", ":")) + "\n\n"


DONE = "data: [DONE]\n\n"
USAGE = {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}


def chunk(choices: list[dict], **top: object) -> dict:
    base: dict = {"id": "chatcmpl-x", "object": "chat.completion.chunk", "created": 1, "model": "m"}
    base.update(top)
    base["choices"] = choices
    return base


def ch(
    delta: dict | None = None, finish: str | None = None, index: int = 0, **extra: object
) -> dict:
    choice: dict = {"index": index, "finish_reason": finish}
    if delta is not None:
        choice["delta"] = delta
    choice.update(extra)
    return choice


STREAMS: list[ProviderStream] = []


def stream(name: str, provider: str, source: str, body: str, outcome: Outcome, **expected: object):
    STREAMS.append(ProviderStream(name, provider, source, body.encode(), outcome, **expected))


# --- OpenAI ---------------------------------------------------------------------------------------

stream(
    "openai_basic_include_usage",
    "OpenAI",
    "openai-python c6fc264 types/chat/chat_completion_chunk.py (obfuscation, usage); "
    "chat_completion_stream_options_param.py (the usage chunk has choices: [])",
    d(chunk([ch({"role": "assistant", "content": "", "refusal": None})], obfuscation="aZ",
            usage=None, service_tier="default", system_fingerprint="fp_1"))
    + d(chunk([ch({"content": "Hello"})], obfuscation="Qx", usage=None))
    + d(chunk([ch({"content": " world"})], obfuscation="k", usage=None))
    + d(chunk([ch({}, "stop")], obfuscation="pp", usage=None))
    + d(chunk([], usage=USAGE, obfuscation="z"))
    + DONE,
    "success", text="Hello world", finish="stop", usage=True,
)  # fmt: skip
stream(
    "openai_refusal_channel",
    "OpenAI",
    "openai-python chat_completion_chunk.py: ChoiceDelta.refusal",
    d(chunk([ch({"role": "assistant", "content": None, "refusal": ""})]))
    + d(chunk([ch({"content": None, "refusal": "I can't help with "})]))
    + d(chunk([ch({"content": None, "refusal": "that."})]))
    + d(chunk([ch({}, "stop")]))
    + DONE,
    "success", refusal="I can't help with that.", finish="stop",
)  # fmt: skip
stream(
    "openai_moderation_chunk_nested_type_error",
    "OpenAI",
    "openai-python chat_completion_chunk.py: Moderation.output may be Error{code, message, "
    "type: 'error'}, nested, so not a stream error",
    d(chunk([ch({"role": "assistant", "content": "Hi"})]))
    + d(chunk([ch({}, "stop")]))
    + d(chunk([], moderation={"input": {"type": "moderation_results", "results": []},
                              "output": {"type": "error", "code": "timeout", "message": "late"}}))
    + DONE,
    "success", text="Hi", finish="stop",
)  # fmt: skip

# --- Azure OpenAI ---------------------------------------------------------------------------------

_FILTERS = {"hate": {"filtered": False, "severity": "safe"}}
stream(
    "azure_prompt_filter_first_chunk",
    "Azure OpenAI",
    "learn.microsoft.com/en-us/azure/foundry/openai/concepts/content-streaming (2026-09-08): first "
    "frame has id '', object '', created 0, model '', prompt_filter_results and choices []",
    d({"id": "", "object": "", "created": 0, "model": "", "choices": [], "usage": None,
       "prompt_filter_results": [{"prompt_index": 0, "content_filter_results": _FILTERS}]})
    + d(chunk([ch({"role": "assistant"})], usage=None))
    + d(chunk([ch({"content": "Color"}, content_filter_results=_FILTERS)], usage=None))
    + d(chunk([ch({}, "stop")], usage=None))
    + DONE,
    "success", text="Color", finish="stop",
)  # fmt: skip
stream(
    "azure_async_annotation_after_stop",
    "Azure OpenAI",
    "content-streaming page, 'Sample response stream (passes filters)': an annotation frame "
    "(no delta, object '') arrives after the finish_reason 'stop' chunk",
    d({"id": "", "object": "", "created": 0, "model": "", "choices": [], "usage": None,
       "prompt_annotations": [{"prompt_index": 0}]})
    + d(chunk([ch({"role": "assistant"})]))
    + d(chunk([ch({"content": "Color is"})]))
    + d({"id": "", "object": "", "created": 0, "model": "", "usage": None,
         "choices": [ch(None, None, content_filter_results=_FILTERS,
                        content_filter_offsets={"check_offset": 44, "start_offset": 44})]})
    + d(chunk([ch({}, "stop")]))
    + d({"id": "", "object": "", "created": 0, "model": "", "usage": None,
         "choices": [ch(None, None, content_filter_results=_FILTERS,
                        content_filter_offsets={"check_offset": 506, "start_offset": 44})]})
    + DONE,
    "success", text="Color is", finish="stop",
)  # fmt: skip
stream(
    "azure_async_blocked_content_filter",
    "Azure OpenAI",
    "content-streaming page, 'Sample response stream (blocked by filters)': finish_reason "
    "'content_filter' arrives in an annotation frame with no delta, then [DONE]",
    d(chunk([ch({"role": "assistant"})]))
    + d(chunk([ch({"content": "Hey Jude,"})]))
    + d({"id": "", "object": "", "created": 0, "model": "", "usage": None,
         "choices": [ch(None, "content_filter",
                        content_filter_results={"protected_material_text": {"filtered": True}})]})
    + DONE,
    "success", text="Hey Jude,", finish="content_filter",
)  # fmt: skip
stream(
    "azure_filter_unavailable_nested_error",
    "Azure OpenAI",
    "learn.microsoft.com/azure/ai-foundry/openai/concepts/content-filter, scenario 6: "
    "content_filter_results.error {code: content_filter_error} is not a stream error",
    d(chunk([ch({"role": "assistant", "content": "ok"},
                content_filter_results={"error": {"code": "content_filter_error",
                                                  "message": "The contents are not filtered"}})]))
    + d(chunk([ch({}, "length")]))
    + DONE,
    "success", text="ok", finish="length",
)  # fmt: skip

# --- vLLM -----------------------------------------------------------------------------------------

_VLLM_ERROR = {"message": "Internal server error", "type": "InternalServerError", "param": None,
               "code": 500}  # fmt: skip
stream(
    "vllm_midstream_error_then_done",
    "vLLM >= 0.10.1",
    "vllm c59d545 chat_completion/serving.py:497-508 (error frame, then an unconditional "
    "'data: [DONE]'); serve/engine/protocol.py:79-87 ErrorResponse{error: ErrorInfo}",
    d(chunk([ch({"role": "assistant", "content": ""})]))
    + d(chunk([ch({"content": "partial john.do"})]))
    + d({"error": _VLLM_ERROR})
    + DONE,
    "reported_error",
)  # fmt: skip
stream(
    "vllm_old_flat_error_then_done",
    "vLLM <= 0.10.0",
    "vllm v0.10.0 entrypoints/openai/protocol.py:80-85 ErrorResponse{object: 'error', message, "
    "type, param, code}",
    d(chunk([ch({"role": "assistant", "content": ""})]))
    + d(chunk([ch({"content": "partial"})]))
    + d({"object": "error", "message": "Internal server error", "type": "InternalServerError",
         "param": None, "code": 500})
    + DONE,
    "reported_error",
)  # fmt: skip
stream(
    "vllm_continuous_usage_reasoning",
    "vLLM",
    "vllm c59d545 serving.py:598-645 (role chunk, continuous usage), 801-823 (content, "
    "finish_reason and stop_reason in one chunk), 875-925 (usage chunk with choices []); "
    "generate/base/protocol.py:356 DeltaMessage.reasoning",
    d(chunk([ch({"role": "assistant", "content": ""})], usage=USAGE))
    + d(chunk([ch({"reasoning": "think about john@example.com"})], usage=USAGE))
    + d(chunk([ch({"content": "Answer"})], usage=USAGE))
    + d(chunk([ch({"content": "."}, "stop", stop_reason=None)], usage=USAGE))
    + d(chunk([], usage=USAGE, system_fingerprint="fp"))
    + DONE,
    "success", text="Answer.", finish="stop", usage=True,
)  # fmt: skip
stream(
    "vllm_013_mirrored_reasoning_fields",
    "vLLM 0.11.1-0.13",
    "vllm v0.13.0 entrypoints/openai/protocol.py:1522-1534 copies reasoning into "
    "reasoning_content",
    d(chunk([ch({"role": "assistant", "content": ""})]))
    + d(chunk([ch({"reasoning": "hmm", "reasoning_content": "hmm"})]))
    + d(chunk([ch({"content": "Hi"})]))
    + d(chunk([ch({}, "stop")]))
    + DONE,
    "success", text="Hi", finish="stop",
)  # fmt: skip
stream(
    "vllm_abort_finish_reason",
    "vLLM",
    "vllm c59d545 v1/engine/__init__.py:37 FINISH_REASON_STRINGS includes 'abort'",
    d(chunk([ch({"role": "assistant", "content": ""})]))
    + d(chunk([ch({"content": "cut john.do"})]))
    + d(chunk([ch({}, "abort")]))
    + DONE,
    "reported_error",
)  # fmt: skip
stream(
    "vllm_repetition_finish_reason",
    "vLLM",
    "vllm c59d545 v1/engine/__init__.py:69-80 'repetition': repetitive token pattern detected",
    d(chunk([ch({"role": "assistant", "content": "la la la"})]))
    + d(chunk([ch({}, "repetition")]))
    + DONE,
    "success", text="la la la", finish="stop",
)  # fmt: skip

# --- llama.cpp ------------------------------------------------------------------------------------

stream(
    "llamacpp_old_error_field_then_done",
    "llama.cpp < b6520 (2025-09)",
    "llama.cpp b6000 tools/server/utils.hpp:520 writes an 'error: {...}' field line, then "
    "server.cpp:4319-4327 writes 'data: [DONE]'; fixed by PR #16109. The unknown field is "
    "ignored, and [DONE] without a finish reason is a truncated stream",
    d(chunk([ch({"role": "assistant", "content": None})]))
    + d(chunk([ch({"content": "Mail john.do"})]))
    + "error: " + json.dumps({"code": 500, "message": "slot crashed", "type": "server_error"})
    + "\n\n"
    + DONE,
    "protocol_error",
)  # fmt: skip
stream(
    "llamacpp_new_midstream_error_no_done",
    "llama.cpp >= PR #16109",
    "llama.cpp e14a779 server-context.cpp:5492-5501 format_oai_sse({error: ...}) then stop; "
    "server-common.cpp:93-135 format_error_response {code, message, type}",
    d(chunk([ch({"role": "assistant", "content": None})]))
    + d(chunk([ch({"content": "Mail john.do"})]))
    + d({"error": {"code": 500, "message": "slot crashed", "type": "server_error"}}),
    "reported_error",
)  # fmt: skip
stream(
    "llamacpp_ping_reasoning_timings",
    "llama.cpp",
    "llama.cpp e14a779 server-context.cpp:5525 ping ':\\n\\n'; server-task.cpp:1122-1160 first "
    "delta {role, content: null} and timings; server-chat.cpp:653-660 reasoning_content",
    d(chunk([ch({"role": "assistant", "content": None})], system_fingerprint="b1"))
    + ":\n\n"
    + d(chunk([ch({"reasoning_content": "Let me think"})], timings={"predicted_n": 1}))
    + d(chunk([ch({"content": "Done"})], timings={"predicted_n": 2}))
    + d(chunk([ch({}, "stop")]))
    + d(chunk([], usage=USAGE, timings={"predicted_n": 3}))
    + DONE,
    "success", text="Done", finish="stop", usage=True,
)  # fmt: skip

# --- Ollama ---------------------------------------------------------------------------------------

stream(
    "ollama_v012_role_every_chunk",
    "Ollama <= 2026-08",
    "ollama v0.12.0 openai/openai.go:284-305 toChunk: role 'assistant' on every chunk, "
    "finish_reason on the chunk with the last content; issue #7626",
    d(chunk([ch({"role": "assistant", "content": "Hel"})], system_fingerprint="fp_ollama"))
    + d(chunk([ch({"role": "assistant", "content": "lo"})], system_fingerprint="fp_ollama"))
    + d(chunk([ch({"role": "assistant", "content": ""}, "stop")], system_fingerprint="fp_ollama"))
    + DONE,
    "success", text="Hello", finish="stop",
)  # fmt: skip
stream(
    "ollama_midstream_error_truncation",
    "Ollama (inferred)",
    "ollama 02480ca server/routes.go:2644-2653 writes an error after the 200; "
    "middleware/openai.go:178-185 turns it into an empty content chunk with no finish and no "
    "[DONE], so the stream looks truncated",
    d(chunk([ch({"role": "assistant", "content": "Hel"})], system_fingerprint="fp_ollama"))
    + d(chunk([ch({"content": "lo john.do"})], system_fingerprint="fp_ollama"))
    + d(chunk([ch({"content": ""})], system_fingerprint="fp_ollama")),
    "protocol_error",
)  # fmt: skip
stream(
    "ollama_current_reasoning_and_usage",
    "Ollama main 2026-11",
    "ollama 02480ca openai/openai.go:43-49 Delta.reasoning, :158-167 timings, :422-449 "
    "FinishChunk; middleware/openai.go:131-160 usage chunk with choices [] then [DONE]",
    d(chunk([ch({"role": "assistant", "reasoning": "thinking"})], system_fingerprint="fp_ollama"))
    + d(chunk([ch({"content": "Yes"})], system_fingerprint="fp_ollama"))
    + d(chunk([ch({}, "stop")], system_fingerprint="fp_ollama"))
    + d(chunk([], system_fingerprint="fp_ollama", usage=USAGE, timings={"prompt_n": 5}))
    + DONE,
    "success", text="Yes", finish="stop", usage=True,
)  # fmt: skip

# --- LiteLLM proxy --------------------------------------------------------------------------------

stream(
    "litellm_ping_error_then_done",
    "LiteLLM proxy",
    "litellm 3f13c1c proxy_server.py:10886-10889 ': ping'; :11037-11061 error frame then the "
    "[DONE] frame; common_request_processing.py:1250-1277 code is str(status)",
    ": ping\n\n"
    + d(chunk([ch({"role": "assistant", "content": "Hi"})]))
    + d({"error": {"message": "upstream reset", "type": "None", "param": None, "code": "500"}})
    + DONE,
    "reported_error",
)  # fmt: skip
stream(
    "litellm_fallback_metadata_chunk",
    "LiteLLM proxy",
    "litellm 3f13c1c proxy_server.py:10250-10268 id 'litellm-fallback-metadata', choices [], "
    "litellm_fallback.errors (not a stream error)",
    d(chunk([ch({"role": "assistant", "content": "Hi", "provider_specific_fields": {"x": 1}})]))
    + d({"id": "litellm-fallback-metadata", "object": "chat.completion.chunk", "created": 1,
         "model": "b", "choices": [],
         "litellm_fallback": {"fallback_model": "b", "errors": [{"message": "a failed"}]}})
    + d(chunk([ch({}, "stop")]))
    + DONE,
    "success", text="Hi", finish="stop",
)  # fmt: skip

# --- OpenRouter -----------------------------------------------------------------------------------

stream(
    "openrouter_comments_duplicate_finish_usage",
    "OpenRouter",
    "openrouter.ai/docs/api/reference/streaming.md lines 15-25 (': OPENROUTER PROCESSING'), "
    "84-94 (the usage chunk repeats finish_reason)",
    ": OPENROUTER PROCESSING\n\n"
    + ": OPENROUTER PROCESSING\n\n"
    + d(chunk([ch({"role": "assistant", "content": "Hi"}, native_finish_reason=None)]))
    + d(chunk([ch({"content": "", "role": "assistant"}, "stop", native_finish_reason="stop")]))
    + d(chunk([ch({"content": "", "role": "assistant"}, "stop", native_finish_reason="stop")],
              usage=USAGE))
    + DONE,
    "success", text="Hi", finish="stop", usage=True,
)  # fmt: skip
stream(
    "openrouter_midstream_error",
    "OpenRouter",
    "openrouter streaming.md lines 181-193: top-level error and choices[0].finish_reason 'error'",
    d(chunk([ch({"role": "assistant", "content": "Hi"})], provider="openai"))
    + d(chunk([ch({"content": ""}, "error")], provider="openai",
              error={"code": "server_error", "message": "Provider disconnected unexpectedly"})),
    "reported_error",
)  # fmt: skip
stream(
    "openrouter_reasoning_details",
    "OpenRouter",
    "openrouter.ai/docs/guides/best-practices/reasoning-tokens.md lines 375-509: delta.reasoning "
    "and delta.reasoning_details; annotations with url_citation",
    d(chunk([ch({"role": "assistant", "content": "", "reasoning": "secret john@example.com",
                 "reasoning_details": [{"type": "reasoning.text", "text": "secret"}]})]))
    + d(chunk([ch({"content": "Answer", "annotations": [
        {"type": "url_citation", "url_citation": {"url": "https://e.x", "title": "t"}}]})]))
    + d(chunk([ch({}, "stop")]))
    + DONE,
    "success", text="Answer", finish="stop",
)  # fmt: skip

# --- Groq -----------------------------------------------------------------------------------------

stream(
    "groq_x_groq_usage",
    "Groq",
    "groq-python 55066d9 types/chat/chat_completion_chunk.py XGroq{id, usage, seed}",
    d(chunk([ch({"role": "assistant", "content": ""})], x_groq={"id": "req_1"}))
    + d(chunk([ch({"content": "Hi"})]))
    + d(chunk([ch({}, "stop")], x_groq={"id": "req_1", "usage": USAGE, "seed": 1}))
    + DONE,
    "success", text="Hi", finish="stop", usage=True,
)  # fmt: skip
stream(
    "groq_x_groq_error",
    "Groq",
    "groq-python chat_completion_chunk.py XGroq.error: 'An error string indicating why a stream "
    "was stopped early.'",
    d(chunk([ch({"role": "assistant", "content": "Hi john.do"})], x_groq={"id": "req_1"}))
    + d(chunk([ch({}, "stop")], x_groq={"id": "req_1", "error": "stream stopped: capacity"}))
    + DONE,
    "reported_error",
)  # fmt: skip

# --- DeepSeek -------------------------------------------------------------------------------------

stream(
    "deepseek_keepalive_reasoning_usage_in_finish",
    "DeepSeek",
    "api-docs.deepseek.com/quick_start/rate_limit (': keep-alive' comments); "
    "api-docs.deepseek.com/api/create-chat-completion (reasoning_content, usage on the last chunk)",
    ": keep-alive\n\n"
    + ": keep-alive\n\n"
    + d(chunk([ch({"role": "assistant", "content": None, "reasoning_content": ""})]))
    + d(chunk([ch({"content": None, "reasoning_content": "The user"})]))
    + d(chunk([ch({"content": "Hello", "reasoning_content": None})]))
    + d(chunk([ch({"content": ""}, "stop", logprobs=None)],
              usage=dict(USAGE, prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=5)))
    + DONE,
    "success", text="Hello", finish="stop", usage=True,
)  # fmt: skip
stream(
    "deepseek_insufficient_system_resource",
    "DeepSeek",
    "api-docs.deepseek.com/api/create-chat-completion finish_reason enum includes "
    "'insufficient_system_resource'",
    d(chunk([ch({"role": "assistant", "content": "Partial john.do"})]))
    + d(chunk([ch({"content": ""}, "insufficient_system_resource")]))
    + DONE,
    "reported_error",
)  # fmt: skip

# --- Mistral --------------------------------------------------------------------------------------

stream(
    "mistral_thinking_list_then_string",
    "Mistral",
    "mistralai/client-python d3da3bd models/deltamessage.py content: str | List[ContentChunk]; "
    "thinkchunk.py; docs.mistral.ai/capabilities/reasoning (thinking phase is a list with a "
    "ThinkChunk, the answer a plain string)",
    d({"id": "x", "model": "m", "choices": [ch({"role": "assistant", "content": [
        {"type": "thinking", "thinking": [{"type": "text", "text": "secret 123-45-6789"}]}]})]})
    + d({"id": "x", "model": "m", "choices": [ch({"content": "Answer"})]})
    + d({"id": "x", "model": "m", "choices": [ch({"content": ""}, "stop")], "usage": USAGE})
    + DONE,
    "success", text="Answer", finish="stop", usage=True,
)  # fmt: skip
stream(
    "mistral_text_chunk_list",
    "Mistral (zai-glm-5-3)",
    "docs.mistral.ai/capabilities/reasoning: for zai-glm-5-3 content 'is always returned as a "
    "list of chunks (thinking + text)'",
    d({"id": "x", "model": "m", "choices": [ch({"role": "assistant", "content": [
        {"type": "thinking", "thinking": [{"type": "text", "text": "hmm"}]}]})]})
    + d({"id": "x", "model": "m",
         "choices": [ch({"content": [{"type": "text", "text": "Mail john.doe"}]})]})
    + d({"id": "x", "model": "m",
         "choices": [ch({"content": [{"type": "text", "text": "@example.com"}]})]})
    + d({"id": "x", "model": "m", "choices": [ch({"content": ""}, "stop")]})
    + DONE,
    "success", text="Mail john.doe@example.com", finish="stop",
)  # fmt: skip
stream(
    "mistral_unknown_content_chunk_type",
    "Mistral",
    "mistralai models/contentchunk.py ContentChunk union includes image_url, document_url, "
    "audio, file and reference",
    d({"id": "x", "model": "m", "choices": [ch({"role": "assistant", "content": [
        {"type": "image_url", "image_url": "data:image/png;base64,AAAA"}]})]})
    + d({"id": "x", "model": "m", "choices": [ch({"content": ""}, "stop")]})
    + DONE,
    "protocol_error",
)  # fmt: skip
stream(
    "mistral_reference_chunk_ignored",
    "Mistral",
    "mistralai models/referencechunk.py ReferenceChunk{reference_ids, type: reference} carries "
    "no text",
    d({"id": "x", "model": "m", "choices": [ch({"role": "assistant", "content": [
        {"type": "text", "text": "See [1]"}, {"type": "reference", "reference_ids": [1]}]})]})
    + d({"id": "x", "model": "m", "choices": [ch({"content": ""}, "stop")]})
    + DONE,
    "success", text="See [1]", finish="stop",
)  # fmt: skip

# --- Together, Fireworks, Gemini, TGI, SGLang -----------------------------------------------------

stream(
    "together_role_every_chunk_eos",
    "Together AI",
    "togethercomputer/together-py f45e98b types/chat/chat_completion_chunk.py: delta.role, "
    "token_id, logprobs as a float, finish_reason includes 'eos'",
    d(chunk([ch({"role": "assistant", "content": "Hi", "token_id": 9}, logprobs=-0.1)]))
    + d(chunk([ch({"role": "assistant", "content": " there", "token_id": 10}, logprobs=-0.2)]))
    + d(chunk([ch({"role": "assistant", "content": "", "token_id": 2}, "eos", logprobs=0.0)],
              usage=USAGE))
    + DONE,
    "success", text="Hi there", finish="stop", usage=True,
)  # fmt: skip
stream(
    "fireworks_usage_chunk_perf_metrics",
    "Fireworks",
    "docs.fireworks.ai/api-reference/post-chatcompletions: usage in a separate final chunk, "
    "perf_metrics on the finish chunk, reasoning_content",
    d(chunk([ch({"role": "assistant", "content": "", "reasoning_content": "hm"})]))
    + d(chunk([ch({"content": "Ok"})]))
    + d(chunk([ch({}, "stop")], perf_metrics={"ttft": 0.1}))
    + d(chunk([], usage=USAGE))
    + DONE,
    "success", text="Ok", finish="stop", usage=True,
)  # fmt: skip
stream(
    "gemini_thought_tags_inline",
    "Gemini OpenAI compatibility (inferred)",
    "raine/consult-llm src/models.rs:140-153 strips '<thought>' tags from Gemini /v1beta/openai/ "
    "content when include_thoughts is set: the thoughts arrive as content, so they are redacted",
    d(chunk([ch({"role": "assistant", "content": "<thought>user is jane@x.org</thought>Hi"})]))
    + d(chunk([ch({}, "stop")]))
    + DONE,
    "success", text="<thought>user is jane@x.org</thought>Hi", finish="stop",
)  # fmt: skip
stream(
    "tgi_error_then_done",
    "Hugging Face TGI",
    "text-generation-inference b4adbf2 router/src/infer/mod.rs:445-466 {error: {message, "
    "http_status_code: 422}}; server.rs:1240-1244 then data: [DONE]",
    d(chunk([ch({"role": "assistant", "content": "Hi"})]))
    + d({"error": {"message": "Request failed during generation", "http_status_code": 422}})
    + DONE,
    "reported_error",
)  # fmt: skip
stream(
    "tgi_legacy_string_error",
    "Hugging Face TGI (generate_stream)",
    "text-generation-inference router/src/lib.rs:1516-1519 ErrorResponse{error: String, "
    "error_type}",
    d(chunk([ch({"role": "assistant", "content": "Hi"})]))
    + d({"error": "Input validation error", "error_type": "validation"}),
    "reported_error",
)  # fmt: skip
stream(
    "sglang_error_then_done",
    "SGLang",
    "sglang e867a9b serving_base.py:223-237 {error: {object: 'error', ..., code: int}}; "
    "serving_chat.py:2880-2886 error frame then data: [DONE]",
    d(chunk([ch({"role": "assistant", "content": "Hi", "reasoning_content": None})]))
    + d({"error": {"object": "error", "message": "boom", "type": "BadRequestError", "param": None,
                   "code": 400}})
    + DONE,
    "reported_error",
)  # fmt: skip

# --- shapes that must fail closed -----------------------------------------------------------------

stream(
    "named_error_event_anthropic_style",
    "generic (Anthropic-style error event)",
    "platform.claude.com/docs/en/api/errors.md:43 (mid-stream error events); SSE 'event: error'",
    d(chunk([ch({"role": "assistant", "content": "Hi"})]))
    + "event: error\ndata: "
    + json.dumps({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})
    + "\n\n",
    "reported_error",
)  # fmt: skip
stream(
    "unknown_named_event",
    "generic",
    "OpenAI-compatible chat streams never name events; a named event means another protocol",
    "event: content_block_delta\n" + d({"type": "content_block_delta", "delta": {"text": "Hi"}}),
    "protocol_error",
)  # fmt: skip
stream(
    "unexpected_tool_calls",
    "generic",
    "the request allowlist rejects tools; arguments are JSON text, so PII can hide in escapes "
    "such as \\u0040",
    d(chunk([ch({"role": "assistant", "content": None, "tool_calls": [
        {"index": 0, "id": "c1", "type": "function",
         "function": {"name": "f", "arguments": "{\"e\":\"a\\u0040b.com\"}"}}]})]))
    + d(chunk([ch({}, "tool_calls")]))
    + DONE,
    "protocol_error",
)  # fmt: skip
stream(
    "unexpected_audio",
    "generic (OpenAI audio output shape)",
    "openai-python chat_completion_chunk.py ChoiceDeltaAudio{id, data, expires_at, transcript}",
    d(chunk([ch({"role": "assistant",
                 "audio": {"id": "a1", "transcript": "my ssn is 123-45-6789", "data": "AAAA"}})]))
    + d(chunk([ch({}, "stop")]))
    + DONE,
    "protocol_error",
)  # fmt: skip
stream(
    "done_without_finish_reason",
    "generic (truncation)",
    "old llama.cpp and some proxies: [DONE] is not proof of success",
    d(chunk([ch({"role": "assistant", "content": "Mail john.do"})])) + DONE,
    "protocol_error",
)  # fmt: skip
stream(
    "eof_after_finish_no_done",
    "generic",
    "plan: an upstream EOF after the choice has finished counts as success",
    d(chunk([ch({"role": "assistant", "content": "Hi"})])) + d(chunk([ch({}, "stop")])),
    "success", text="Hi", finish="stop",
)  # fmt: skip
stream(
    "eof_without_finish",
    "generic (truncation)",
    "the connection closed mid-generation",
    d(chunk([ch({"role": "assistant", "content": "Mail john.do"})])),
    "protocol_error",
)  # fmt: skip
stream(
    "unterminated_final_event",
    "generic (truncation)",
    "WHATWG SSE and openai-python 3.46: an event without its blank line at EOF is discarded",
    d(chunk([ch({"role": "assistant", "content": "Hi"})]))
    + d(chunk([ch({}, "stop")])).removesuffix("\n"),
    "protocol_error",
)  # fmt: skip
stream(
    "content_after_finish",
    "generic",
    "text after a finish reason for the same choice",
    d(chunk([ch({"role": "assistant", "content": "Hi"})]))
    + d(chunk([ch({}, "stop")]))
    + d(chunk([ch({"content": " more"})]))
    + DONE,
    "protocol_error",
)  # fmt: skip
stream(
    "non_string_content",
    "generic",
    "content must be a string, null, or a list of typed parts",
    d(chunk([ch({"role": "assistant", "content": 123})])) + d(chunk([ch({}, "stop")])) + DONE,
    "protocol_error",
)  # fmt: skip
stream(
    "second_choice_index",
    "generic",
    "the request allows n=1 only; a choice with index 1 means that assumption broke",
    d(chunk([ch({"role": "assistant", "content": "A"}),
             ch({"role": "assistant", "content": "B"}, index=1)]))
    + d(chunk([ch({}, "stop"), ch({}, "stop", index=1)]))
    + DONE,
    "protocol_error",
)  # fmt: skip
stream(
    "invalid_json",
    "generic",
    "a truncated or garbage data payload",
    d(chunk([ch({"role": "assistant", "content": "Hi"})])) + 'data: {"id": "x", "choi\n\n' + DONE,
    "protocol_error",
)  # fmt: skip
stream(
    "unknown_finish_reason",
    "generic",
    "an unmapped finish reason fails closed",
    d(chunk([ch({"role": "assistant", "content": "Hi john.do"})]))
    + d(chunk([ch({}, "weird")]))
    + DONE,
    "protocol_error",
)  # fmt: skip
stream(
    "side_channels_ignored",
    "generic",
    "logprobs tokens, x_groq.debug, vLLM token_ids and prompt_text, llama.cpp __verbose: "
    "never relayed, so they can be ignored",
    d(chunk([ch({"role": "assistant", "content": "Hi"},
                logprobs={"content": [{"token": "Hi", "logprob": -0.1, "top_logprobs": []}]},
                token_ids=[1, 2])],
            prompt_text="user: my ssn is 123-45-6789", __verbose={"content": "Hi"},
            x_groq={"debug": {"output_tokens": ["Hi"]}}))
    + d(chunk([ch({}, "stop")]))
    + DONE,
    "success", text="Hi", finish="stop",
)  # fmt: skip
stream(
    "empty_data_keepalive",
    "generic",
    "'data:' with an empty value dispatches an event with data '' (SSE spec): a keep-alive",
    "data:\n\n"
    + d(chunk([ch({"role": "assistant", "content": "Hi"})]))
    + d(chunk([ch({}, "stop")]))
    + DONE,
    "success", text="Hi", finish="stop",
)  # fmt: skip
stream(
    "legacy_text_completion_object",
    "generic (wrong endpoint: /v1/completions)",
    "legacy completions stream: object 'text_completion', text in choices[].text, no delta",
    d({"id": "cmpl-1", "object": "text_completion", "created": 1, "model": "m",
       "choices": [{"index": 0, "text": "Hi", "finish_reason": None}]})
    + d({"id": "cmpl-1", "object": "text_completion", "created": 1, "model": "m",
         "choices": [{"index": 0, "text": "", "finish_reason": "stop"}]})
    + DONE,
    "protocol_error",
)  # fmt: skip
stream(
    "top_level_json_array",
    "generic",
    "a data payload that is not a JSON object",
    d([{"choices": []}]) + DONE,
    "protocol_error",
)  # fmt: skip
