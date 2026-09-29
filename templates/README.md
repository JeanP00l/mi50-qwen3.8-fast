# Chat template

`qwen3.8-agent.jinja` is Qwen3.8's chat template (Apache-2.0, Qwen team) with Unsloth's fixes
(developer role, merged system messages, tool calling) plus two agent tweaks:

- a system message in the middle of the conversation is rendered in place instead of raising
  `System message must be at the beginning` (Claude Code sends such messages);
- default reasoning effort is `medium`, and `high` maps to `medium`: with the stock mapping to `xhigh`
  the model thought for ~10 000 tokens before its first tool call. `xhigh` only when explicitly requested.

`TEMPLATE=model ./run.sh` uses the template embedded in the GGUF instead.
