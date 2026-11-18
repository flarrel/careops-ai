# Local agent framework comparison

This benchmark runs the same synthetic claim review in LangGraph, Semantic Kernel, Google ADK, and AutoGen. The approved path reads the clean synthetic claim and then retrieves one cited passage from the pinned INA-CBG PDF. All four runtimes share the typed fixture validation, checklist rules, and attempt budget. The policy fixture uses an exact local lookup so model generation does not affect this comparison.

The eight scenarios cover approval, denial, a missing claim, missing attachments, a transient claim timeout, persistent policy timeouts, the tool budget, and an instruction embedded in the policy query. Each run checks the expected outcome and reason, full synthetic lookup result, source citation, and exact tool-attempt order. Reported timeout errors are injected at the local lookup boundary.

## Runtime design

| Framework | Native execution and trace |
|---|---|
| LangGraph | The maintained workflow pauses on its checkpointer and resumes with the operator decision. Its trace records approval, attempts, and completion. |
| Semantic Kernel | A review plugin invokes the two registered lookup functions through the kernel. A function invocation filter records the executed functions. |
| Google ADK | A sequential agent runs approval, claim, and policy stages. Session state controls eligibility between stages. FunctionTool executes lookups and the runner emits stage and tool events. |
| AutoGen | A review actor sends typed messages to a lookup actor through the local runtime. FunctionTool executes each read, and the receiving actor records dispatch. |

The comparison reuses the same retry budget across frameworks. Its failure results describe the adapters and shared policy exercised in the suite. LangGraph remains the maintained runtime because its existing approval pause, transitions, and traces serve the local review workflow. The other implementations are isolated benchmarks. [Semantic Kernel invocation API](https://learn.microsoft.com/en-us/python/api/semantic-kernel/semantic_kernel.kernel.kernel?view=semantic-kernel-python), [Google ADK custom agents](https://google.github.io/adk-docs/agents/custom-agents/), [AutoGen FunctionTool](https://microsoft.github.io/autogen/0.7.1/reference/python/autogen_core.tools.html)

## Results

| Framework | Correct and trace-matched | Warm approved p50 / max | Peak process RSS |
|---|---:|---:|---:|
| LangGraph 0.2.60 | 8/8 | 4.573 / 5.568 ms | 82.5 MiB |
| Semantic Kernel 1.37.0 | 8/8 | 0.288 / 0.374 ms | 92.1 MiB |
| Google ADK 1.18.0 | 8/8 | 1.760 / 2.201 ms | 350.4 MiB |
| AutoGen Core 0.7.5 | 8/8 | 1.097 / 6.772 ms | 52.5 MiB |

The [machine-readable run](../evals/results/agent-frameworks.json) records the input hashes, dependencies, results, and native event order for each framework. All 32 outputs validated against the agent result schema. A transient lookup recovered within budget, while persistent failures and exhausted budgets escalated in all four runtimes. Denied requests had zero tool calls.

Measurements use Python 3.12 on an Apple M3 Pro. Each framework runs in its own environment and process. Latency measures seven approved runs after the correctness sweep, including orchestration, lookup, and validation. Peak RSS includes imports and fixture loading. Operator wait and model inference are outside the timing. The small fixture and sample size limit what latency differences imply for a deployed application.

## Reproduce

Run from the repository root with Python 3.12. Fetch the checksummed PDFs first using the [corpus guide](corpus.md). Use a separate environment for each adapter.

```sh
lab_dir=$(mktemp -d)
for framework in langgraph semantic_kernel adk autogen; do
  case "$framework" in
    langgraph) extra=agent ;;
    semantic_kernel) extra=comparison-sk ;;
    adk) extra=comparison-adk ;;
    autogen) extra=comparison-autogen ;;
  esac
  python3 -m venv "$lab_dir/$framework"
  runner="$lab_dir/$framework/bin/python"
  "$runner" -m pip install -e ".[comparison,$extra]"
  OTEL_SDK_DISABLED=true "$runner" -m careops.agent_comparison.benchmark "$framework" \
    > "$lab_dir/$framework.json"
done
```

The outputs should agree on correctness, tool attempts, citation IDs, and the shared input hashes. Trace event names and performance are runtime-specific.
