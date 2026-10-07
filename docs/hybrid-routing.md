# Hybrid routing (`jarvis-auto`)

Pick the virtual model **`jarvis-auto`** in the UI (Ctrl+K) and each message is
sent either to your local model (Ollama) or to Claude.

## How it decides (first match wins)

1. **Directive** at the start of your message: `/local`, `/claude`, `/opus`
   (the directive is removed before the model sees the text).
2. **Privacy words** (`privado`, `confidencial`, `secreto`, `password`...) -> local.
3. **Escalation phrases** (`usa claude`, `piensa a fondo`, `analiza a fondo`...) -> Claude.
4. **Long message** (>= `cloud_min_chars`, default 4000) -> Claude.
5. **Complexity score** >= `cloud_threshold` (default 0.35) -> Claude.
6. Otherwise -> local.

If `ANTHROPIC_API_KEY` is missing, everything stays local.

## See why, without spending anything

```bash
uv run --no-sync jarvis route "analiza paso a paso este problema"
```

or `POST /v1/route/explain` with `{"text": "..."}`.

## Tuning (`~/.openjarvis/config.toml`)

```toml
[hybrid_routing]
enabled = true
cloud_model = "claude-sonnet-5-5"   # automatic escalation
heavy_model = "claude-opus-5-5"     # only with /opus
cloud_threshold = 0.35              # lower -> more goes to Claude
cloud_min_chars = 4000
cloud_keywords = "usa claude,piensa a fondo,analiza a fondo"
local_keywords = "privado,confidencial,secreto,password"
```

The complexity scorer recognises English and Spanish. The UI shows the model you
selected (`jarvis-auto`); the model that actually answered is in the
`model` field of the API response and in the server log.
