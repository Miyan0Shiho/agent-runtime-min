# Q2 — Adversarial review of `arm.tools.schema.normalize_schema`

> Spike · 2026-10-12 · reviewer: subagent (Q2)
> Target: `src/arm/tools/schema.py` (`normalize_schema`, `to_wire`), as it stood at
> the start of this review, plus the two rewrites the Lead landed while it ran.
> Environment: `.venv` — Python 3.11.14, pydantic 2.14.1, jsonschema 4.26.0
> (installed separately, see §0).

---

## TL;DR — lead with the bugs

**The normalizer is now correct for everything pydantic can emit, and wrong for
three hand-constructible inputs.**

Nine defects were found. **All nine were fixed by the Lead during this review**
(including one regression that the first fix introduced). A second adversarial
pass against the *rewritten* code found **three surviving defects**:

| # | Defect | Effect | Reachable from pydantic? |
|---|--------|--------|--------------------------|
| **10** | `properties` merge is one level deep — a same-named sub-property is overwritten | **WIDENS** | **Yes** — `Field(json_schema_extra={"properties": ...})` |
| **11** | `definitions` is deleted and re-emitted as `$defs`, but `#/definitions/…` refs survive verbatim | **dangling pointer** → schema unusable in the provider | No (pydantic emits `$defs`); yes via `tool(parameters={...})` |
| **12** | an `additionalProperties: false` sibling is merged into the target, where it suddenly sees the target's `properties` | **WIDENS** | No; hand-written schema only |

Failing tests for all three are in `test_findings_bugs.py` as
`xfail(strict=True)`: they fail today, and will turn into a **hard failure**
(`XPASS(strict)`) the moment each is fixed, so a fix cannot be silently reverted.

**The single most important finding is not a bug in the code — it is that the
module is optimising the wrong keywords.** OpenAI's own documentation lists
`$ref` and `$defs` as **supported** (its recursive-schema example ships both)
while explicitly **rejecting** `allOf`, `not`, `if`/`then`/`else` and
`dependentSchemas` — and it errors rather than ignoring them. See §5. The
normalizer inlines the former and (now) refuses the latter, which is right; but
the premise in the module docstring ("`$ref` … is the wrong output for a
function-calling API") is not what the vendors document, and a *recursive* schema
is now correctly passed through with `$ref`/`$defs` intact rather than refused.

**A second structural finding:** `schema_of()` — the only caller of
`normalize_schema` — is **dead code**. Nothing in `src/` calls it, so the
normalizer is not on the live path at all. See §6.

---

## 0. Commands

Everything below is reproducible with:

```bash
cd /Users/liuminxuan/Desktop/job-prep/02-projects/agent-runtime-min/spikes
../.venv/bin/python -m pytest q2_schema          # the test suite
../.venv/bin/python q2_schema/differential.py    # raw-vs-normalized differential
../.venv/bin/python q2_schema/hostile.py         # 29 hostile inputs
../.venv/bin/python q2_schema/evidence.py        # every transcript quoted below
```

The `jsonschema` reference library is **not** a dependency of this spike. The
differential test uses an independent subset validator written for the purpose
(`jsonschema_subset.py`) so that one validator is applied to *both* schemas — the
only way to compare them honestly. When the real library happens to be
importable, two extra tests cross-check the mini-validator against it:

```bash
Q2_JSONSCHEMA_PATH=/tmp/q2venv/lib/python3.11/site-packages \
  ../.venv/bin/python -m pytest q2_schema
```

Result of that cross-check: **the mini-validator and `jsonschema` 4.26.0 agree on
every payload of every case** (1,623 catalogue payloads plus every hostile probe,
on both the raw and the normalized schema). Without it, a bug in the
mini-validator could have manufactured or hidden a finding. With it, the
divergences reported below are real.

---

## 1. What pydantic actually emits

`../.venv/bin/python q2_schema/evidence.py 1`

22 models were built covering every construct named in the brief. Raw output is
long; the keyword sets are the point. (`schema_keywords` walks **schema
positions only** — a property *named* `pattern` is data, not the `pattern`
keyword.)

```
flat                  ['description', 'properties', 'required', 'title', 'type']
optional_with_default ['anyOf', 'default', 'description', 'properties', 'title', 'type']
optional_no_default   ['anyOf', 'description', 'properties', 'required', 'title', 'type']
nested_model          ['$defs', '$ref', 'default', 'description', 'properties', 'required', 'title', 'type']
literal               ['const', 'description', 'enum', 'properties', 'required', 'title', 'type']
enum                  ['$defs', '$ref', 'default', 'description', 'enum', 'properties', 'required', 'title', 'type']
list_of_model         ['$defs', '$ref', 'items', 'properties', 'required', 'title', 'type']
dict_of_model         ['$defs', '$ref', 'additionalProperties', 'properties', 'required', 'title', 'type']
union_plain           ['$defs', '$ref', 'anyOf', 'const', 'description', 'properties', 'required', 'title', 'type']
union_discriminated   ['$defs', '$ref', 'const', 'description', 'discriminator', 'oneOf', 'properties', 'required', 'title', 'type']
constrained           ['description', 'items', 'maxItems', 'maxLength', 'maximum', 'minItems',
                       'minLength', 'minimum', 'multipleOf', 'pattern', 'properties', 'required', 'title', 'type']
recursive             ['$defs', '$ref', 'anyOf', 'default', 'description', 'properties', 'required', 'title', 'type']
tuple                 ['description', 'items', 'maxItems', 'minItems', 'prefixItems', 'properties', 'required', 'title', 'type']
scalars               ['anyOf', 'description', 'format', 'properties', 'required', 'title', 'type']
extra_forbid          ['additionalProperties', 'properties', 'required', 'title', 'type']
extra_allow           ['additionalProperties', 'properties', 'required', 'title', 'type']
ref_sibling_validation['$defs', '$ref', 'description', 'properties', 'required', 'title', 'type']
schema_extra_validation['description', 'minProperties', 'properties', 'required', 'title', 'type']
set_field             ['items', 'properties', 'required', 'title', 'type', 'uniqueItems']
deep_chain_13         ['$defs', '$ref', 'description', 'properties', 'required', 'title', 'type']
aliased_dollar_ref    ['$defs', '$ref', 'description', 'properties', 'required', 'title', 'type']
union_enum_none       ['$defs', '$ref', 'anyOf', 'enum', 'properties', 'required', 'title', 'type']

UNION over all 22 models (27 keywords):
['$defs', '$ref', 'additionalProperties', 'anyOf', 'const', 'default', 'description',
 'discriminator', 'enum', 'format', 'items', 'maxItems', 'maxLength', 'maximum',
 'minItems', 'minLength', 'minProperties', 'minimum', 'multipleOf', 'oneOf', 'pattern',
 'prefixItems', 'properties', 'required', 'title', 'type', 'uniqueItems']
```

`test_catalogue_matches_frozen_pydantic_emission` freezes this table, so a
pydantic upgrade that changes emission is noticed rather than assumed away.

### Findings that surprised me

* **`Literal[X]` emits `const`, not `enum`.** `Literal["a", "b"]` emits `enum`;
  `Literal["a"]` (one value) emits `const`. `Literal[1, "x", None]` emits `enum`
  with a `null` member and **no `type`**.
* **`Optional[X]` with a default is *not* `required`; without a default it *is`.**
  `Optional[int] = None` → `anyOf[int, null]`, absent from `required`.
  `Optional[int]` (no default) → `anyOf[int, null]`, **present** in `required`.
* **`Decimal` emits `anyOf[{type: number}, {type: string}]`**, not `type: number`.
* **`datetime`/`date`/`UUID` emit `type: string` + `format`**; `bytes` emits
  `type: string` + `format: binary`. `format` is an annotation: pydantic will
  *reject* `"not-a-date"` at validation time but its JSON Schema *accepts* it.
* **`tuple[int, str]` emits `prefixItems` + `minItems`/`maxItems`**; `tuple[int, ...]`
  emits plain `items`.
* **`extra="forbid"` emits `additionalProperties: false`; `extra="allow"` emits
  `additionalProperties: true`** (not absent).
* **`Field(discriminator=...)` emits `oneOf` + a `discriminator` object.** The
  `discriminator` is stripped by the normalizer and `oneOf` survives — so a
  discriminated union reaches the wire as a bare `oneOf`, which is *weaker* than
  the discriminator it replaced (it loses the "exactly one branch" tag but keeps
  the oneOf semantics, so validation is unchanged; the *model* loses the hint).
* **No construct in this catalogue emits `allOf`.** The normalizer's `allOf`
  handling was therefore untestable from pydantic alone and had to be attacked
  with hand-written schemas (§3).
* **No construct emits a single-branch `anyOf`.** This matters: the
  single-branch-collapse code path, which turned out to be the source of the
  worst bug, is **not reachable from `model_json_schema()`** for these models —
  only from `json_schema_extra` or a hand-written schema.

### pydantic's JSON Schema is not a faithful contract

Worth recording, because it bounds what *any* normalizer can promise. Comparing
`Model.model_validate()` against the schema pydantic itself emits, over the same
1,623 payloads:

```
payloads compared               : 1623
pydantic accepts, schema rejects: 28
schema accepts, pydantic rejects: 46
```

Examples: `{"a": true}` is accepted by `model_validate` for `a: int` but rejected
by the emitted `{"type": "integer"}`; `{"dt": "wrong-type"}` is rejected by
pydantic but accepted by `{"type": "string", "format": "date-time"}` (format is
an annotation). So **the raw schema is already not equivalent to the model's
validator**, and the normalizer cannot be blamed for — nor can it fix — that gap.
The property actually under test is narrower and is the right one: *normalization
must not change the set of payloads the **schema** accepts.*

---

## 2. Does normalization change the accepted payload set?

`../.venv/bin/python q2_schema/differential.py`

```
cases=22 payloads=1623 divergences=0
  widened=0 narrowed=0 refused=0
```

**Zero divergences across all 22 models and 1,623 payloads.** The corpus is
*defined* by the raw schema (a payload is "accepted" iff the raw schema accepts
it), so the reviewer's own judgement is not a source of error. Payloads are
curated boundary cases plus deterministic mutations of every valid payload: drop
a key, null it, swap its type, add an unknown key, empty a container, push a
number out of range.

This is the result *after* the Lead's fixes. Before them the same harness
reported divergences for the `$ref`-sibling cases (§3, BUG 1) and refusals for
`recursive` and `deep_chain_13`.

---

## 3. Hostile inputs

`../.venv/bin/python q2_schema/hostile.py` — 29 attacks.

```
attacks=29
  crashed            : 0 []
  refused            : 7 ['ref_to_boolean_false_def', 'single_branch_anyof_with_sibling',
                          'defs_not_a_mapping', 'ref_inside_unlisted_schema_keyword',
                          'boolean_root_schema', 'not_keyword', 'if_then_else']
  payload divergence : 1 ['ref_sibling_additional_properties']
  dangling $ref      : 1 ['ref_to_missing_def']
  banned kw on wire  : 0 []
```

Per-attack verdicts. "Correct?" judges the **current** code.

| Attack | Verdict | Correct? |
|---|---|---|
| `$ref` to a missing def | passed through verbatim, `$defs` kept | **Yes** — dropping it would widen; but see the note below |
| `$ref` → boolean `false` def | `SchemaNormalizationError` | **Yes** — unsatisfiable, correctly refused |
| `$ref` → boolean `true` def | resolves to the siblings | **Yes** |
| `$ref` chain A→B→C | inlined transitively | **Yes** |
| mutual recursion A→B→A | reference kept, `$defs` kept | **Yes** |
| `~1` / `~0` escaped pointer tokens | decoded and inlined | **Yes** |
| `~01` decode order | decodes to `~1`, not `/` | **Yes** |
| `$ref` + sibling `properties` (new name) | merged | **Yes** |
| `$ref` + sibling `properties` (**colliding name**) | overwritten | **NO — BUG 10** |
| `$ref` + sibling `required` | union | **Yes** |
| `$ref` + sibling `additionalProperties: false` | merged into target | **NO — BUG 12** |
| `$ref` + sibling `description` | sibling wins | **Yes** (annotations do not conflict) |
| single-branch `anyOf` + sibling | refused | **Yes** — fail-closed, was a widening |
| single-branch `oneOf` + sibling `type` | refused | **Yes** — same |
| `allOf`, two object branches | merged | **Yes** |
| `allOf` with a `$ref` branch | inlined | **Yes** |
| unreferenced `$defs` at root | dropped | **Yes** |
| property literally named `$ref` | `$defs` dropped, property kept | **Yes** |
| `$defs` not a mapping | `SchemaNormalizationError` | **Yes** |
| `additionalProperties: false` (no ref) | preserved verbatim | **Yes** |
| properties named `type` / `properties` | treated as opaque names | **Yes** |
| deep `anyOf` in `items` in `properties` | inlined | **Yes** |
| `$ref` inside `dependentSchemas` | refused | **Yes** — cannot be expressed |
| recursion through `allOf` | refused | **Yes** |
| root-level non-recursive `$ref` | inlined | **Yes** |
| boolean root schema | `SchemaNormalizationError` | **Yes** |
| empty root schema `{}` | passed through | **Yes** |
| `not` | refused | **Yes** |
| `if`/`then`/`else` | refused | **Yes** |
| `$defs` referenced only from a stripped keyword | `$defs` dropped | **Yes** |
| draft-07 `definitions` (cyclic) | `definitions` dropped, ref survives | **NO — BUG 11** |

### The three live defects

#### BUG 10 — the `properties` merge is shallow (WIDENS)

`../.venv/bin/python q2_schema/evidence.py 3b`

```
RAW : {"$ref": "#/$defs/T", "properties": {"x": {"type": "string"}}}
NORM: {"type": "object", "properties": {"x": {"type": "string"}}}
  raw=False norm=False   {"m": {"x": 1}}
  raw=False norm=True  <-- WIDENED  {"m": {"x": "s"}}
```

The raw schema requires `m.x` to be *both* an integer and a string —
unsatisfiable, so nothing is accepted. `_conjoin` merges `properties` with
`{**existing, **value}`, which is correct when the names differ (verified:
`{"x": integer, "y": string}`) but silently drops the target's constraint when
they collide.

**Reachable from pydantic**, via a documented feature:

```python
class M(BaseModel):
    inner: Annotated[Inner, Field(json_schema_extra={"properties": {"value": {"type": "string"}}})]
```

**Minimal fix.** The merge must be a genuine conjunction at every depth, or it
must refuse:

```python
if key in _SCHEMA_MAP_KEYWORDS and isinstance(existing, Mapping) and isinstance(value, Mapping):
    merged[key] = _merge_schema_maps(existing, value, ref=ref)
    continue
```

where `_merge_schema_maps` calls `_conjoin` on names present in both maps and
concatenates the rest. For a scalar collision `_conjoin` will raise — which is
the correct outcome, since two different `type` values have no single-schema form
here. **At minimum, a collision must never be resolved by preferring one side.**

#### BUG 11 — `definitions` is dropped while `#/definitions/…` refs survive

`../.venv/bin/python q2_schema/evidence.py 3b`

```
NORM: {"type": "object", "properties": {"a": {"$ref": "#/definitions/Node"}},
       "$defs": {"Node": {"type": "object", "properties": {"child": {"$ref": "#/definitions/Node"}}}}}
  refs in output : ['#/definitions/Node', '#/definitions/Node']
  keys in output : ['$defs', 'properties', 'type']
```

The definition moved from `definitions` to `$defs`; the reference text did not.
Dereferencing fails, verified with the reference library:

```
jsonschema._WrappedReferencingError: PointerToNowhere:
'/definitions/Node' does not exist within {...}
```

So the schema is unusable for any payload containing `a` — and it fails *in the
provider*, not here. The same happens for a non-cyclic `#/definitions/B` sitting
alongside a `$defs`: `definitions` is popped unconditionally while
`b: {"$ref": "#/definitions/B"}` survives.

**Reachability.** pydantic 2.x emits `$defs`, never `definitions` (verified), so
this needs a hand-written schema — and `tool(parameters={...})` accepts a raw
mapping. `#/definitions/` is what draft-07 emitters and several MCP servers
produce, so this is a realistic input for an MCP bridge.

**Minimal fix.** Keep the two spellings consistent. Either (a) write kept
definitions back into the keyword the surviving refs use, or (b) rewrite
`#/definitions/X` → `#/$defs/X` in the same pass that moves the definition.
(b) is smaller and keeps one output spelling, which is what a model-facing subset
wants. The invariant to test: **every `$ref` in the output must resolve within
the output.**

#### BUG 12 — an `additionalProperties: false` sibling changes meaning when merged (WIDENS)

`../.venv/bin/python q2_schema/evidence.py 3b`

```
RAW  sibling: {"$ref": "#/$defs/Inner", "additionalProperties": false}
NORM sibling: {"type": "object", "properties": {"value": {"type": "integer"}},
               "required": ["value"], "additionalProperties": false}
  raw=False norm=True  <-- WIDENED  {"inner": {"value": 1}}
  raw=False norm=False   {"inner": {}}
```

This one is subtle. `additionalProperties` applies to the
`properties`/`patternProperties` of **its own subschema**. The sibling subschema
`{"$ref": ..., "additionalProperties": false}` declares no `properties` of its
own, so it forbids *every* property — the conjunction with the target's
`required: ["value"]` is unsatisfiable. Moving the keyword into the target makes
it see `properties: {value}` and permit `value`. Confirmed against the reference
library:

```python
Draft202012Validator({"$ref": "#/$defs/Inner", "additionalProperties": false,
                      "$defs": {...}}).is_valid({"value": 1})   # False
```

**Minimal fix.** Do not merge a sibling `additionalProperties` into the target.
Either refuse when a sibling carries `additionalProperties` and the target has
`properties`/`patternProperties` (fail-closed, smallest change), or express the
conjunction structurally. Refusing is the honest answer: this is a case the
normalizer cannot express in the subset.

### Fixed during this review

These nine were found, reported, and fixed while the review ran. They are now
`xfail(strict=True)`-free guards in `test_fixed_regressions.py`.

| # | Defect | Effect | Evidence |
|---|--------|--------|----------|
| 1 | `$ref` siblings **overwrote** the target (`merged[key] = value`) | **widened** *and* **narrowed** | `{"inner": {"value": "not-an-integer"}}` accepted; `required` became `["zzz"]`, losing `value` |
| 2 | `_collapse` replaced the `anyOf` **array** with the branch, so `_collapse_combinator` never saw a list | **structurally invalid** `"anyOf": {...}` | `{"anyOf": [{"type": "string"}]}` → `{"anyOf": {"type": "string"}}` |
| 3 | `dict(target)` on a boolean target | `TypeError: 'bool' object is not iterable` | `$ref` → `$defs: {"Nope": false}` |
| 3 | `schema.get("$defs")` on a boolean root | `AttributeError: 'bool' object has no attribute 'get'` | `normalize_schema(True)` |
| 3 | `dict(...)` on a non-mapping `$defs` | `ValueError: dictionary update sequence element #0 has length 1` | `"$defs": ["x"]` |
| 4 | `depth` incremented on *every* recursive call, not just `$ref` inlining | **non-recursive** schema 13 models deep refused as "recursive" | `n=12 OK`, `n=13 SchemaNormalizationError` |
| 5 | `$defs` walked eagerly | an **unreferenced** recursive def refused the whole schema | `$defs: {"Unused": {…"$ref": "#/$defs/Unused"}}` |
| 6 | `_contains_ref` scanned every dict value for the `$ref` key | a property **named** `$ref` pinned `$defs` | pydantic `Field(alias="$ref")` |
| 7 | `_collapse` handled only `anyOf`/`oneOf` | `allOf` reached the wire | single-branch `allOf` survived verbatim |
| 8 | `_resolve_ref` did not decode RFC 6901 escapes | escaped refs never inlined | `#/$defs/A~1B` |
| 9 | **REGRESSION** from the BUG 1 fix: `_conjoin` compared annotation keywords | `Annotated[Doc, Field(description=…)]` **refused** when `Doc` has a docstring | `SchemaNormalizationError: cannot conjoin 'description'` |

BUG 9 is worth dwelling on: the first fix for BUG 1 treated `description` as a
constraint to reconcile. Because pydantic puts a model's docstring on its `$defs`
entry as `description`, and `Annotated[Model, Field(description=...)]` is *the*
idiomatic way to describe a nested tool argument, that fix made a whole class of
ordinary tools unregisterable. The lesson generalises: **when adding a conflict
check, decide first which keywords are assertions.** The fix was to introduce
`ANNOTATION_KEYWORDS` and skip them in `_conjoin` — annotations are collected,
not intersected.

---

## 4. Is the `$defs` removal logic sound?

`../.venv/bin/python q2_schema/evidence.py 4`

**The original logic was unsound; the current logic is sound except for BUG 11.**

The original code dropped `$defs` "if nothing references it" via `_contains_ref`,
which scans **every dict value for the key `$ref`**. That conflates two different
things, and both directions of error were reachable:

* **False positive (kept a def that was not needed).** A property literally named
  `$ref` — pydantic emits this for `Field(alias="$ref")` — kept the whole `$defs`
  block alive after every real reference had been inlined. Result: a dangling
  `$defs` shipped to the model, and a `$ref`-shaped key in `properties` that a
  naive consumer could try to resolve. **Fixed**: liveness is now decided by
  `still_referenced`, populated during the walk, so only genuine references count.

* **False negative (dropped a def that *was* needed) — still live as BUG 11.**
  The reference text is preserved verbatim while the definition is re-emitted
  under a *different keyword*, so a `#/definitions/X` ref survives while
  `definitions` is popped. This is the "drop a `$defs` that was still needed"
  failure mode, in its `definitions` spelling.

The remaining probes are handled correctly:

```
unreferenced defs            -> $defs dropped
property literally named $ref-> $defs dropped, property kept
ref inside dependentSchemas  -> REFUSED (cannot be expressed in the subset)
unresolvable ref             -> $ref kept, $defs dropped  (see caveat)
recursive def, referenced    -> $ref kept, $defs kept
```

**Note on the depth ceiling.** After the BUG 4 fix the ceiling is
`_MAX_NESTING = 400`, which is no longer reachable from pydantic: pydantic's own
recursion limit fires first (measured — `n=40` both fine; `n=55` pydantic raises
`RecursionError` inside `model_json_schema()`, so the normalizer is never
entered). The ceiling now only guards hand-written or hostile documents, which is
what it should do.

**Caveat on the unresolvable ref.** `{"x": {"$ref": "#/$defs/Missing"}}` is
passed through with `$defs` dropped. Passing it through is right — silently
dropping a constraint would widen — but the output now contains a reference that
cannot resolve *anywhere*. Since the raw schema was already broken, this is not a
regression; it is a judgement call between "propagate the breakage" and "refuse
loudly". Refusing would be more in the spirit of the rest of the module. Flagged
as a design question, not a bug.

---

## 5. Real-world risk: what the vendors actually document

Network was available. Primary source fetched directly (not from a search
snippet):

**OpenAI, "Structured Outputs"** —
<https://platform.openai.com/docs/guides/structured-outputs>

Verbatim, from the "Supported schemas" section:

> Structured Outputs supports a subset of the JSON Schema language.
> **Supported types** The following types are supported for Structured Outputs:
> String, Number, Boolean, Integer, Object, Array, Enum, **anyOf**

> **Supported properties** … Supported string properties: `pattern` … `format` …
> Supported number properties: `multipleOf`, `maximum`, `exclusiveMaximum`,
> `minimum`, `exclusiveMinimum` … Supported array properties: `minItems`, `maxItems`

And, decisively:

> **Some type-specific keywords are not yet supported**
> Composition: `allOf`, `not`, `dependentRequired`, `dependentSchemas`, `if`,
> `then`, `else`
> …
> **If you turn on Structured Outputs by supplying `strict: true` and call the
> API with an unsupported JSON Schema, you will receive an error.**

> **Root objects must not be anyOf and must be an object**

> **All fields must be required** To use Structured Outputs, all fields or
> function parameters must be specified as required.

`$ref`/`$defs` are **not** in the unsupported list, and OpenAI's own recursive
example ships them:

> `"children": { "type": "array", "items": { "$ref": "#/$defs/component" } }` …
> `"$defs": { "component": { … } }`

Corroborating third-party evidence that `$ref` is broadly fine but *some*
OpenAI-compatible servers reject it:

* Zed rejects MCP tool schemas containing `$ref` —
  <https://github.com/zed-industries/zed/pull/60165>:
  > `ERROR Schema cannot be made compatible because it contains "$ref"` …
  > This is hit by every provider that uses `JsonSchemaSubset` format (Google
  > Gemini, xAI Grok, OpenAI-compatible proxies, Vercel AI Gateway, …).
  > Providers using `JsonSchema` (Anthropic direct, OpenAI direct) don't hit this
  > check — `$ref` is passed through to the API.
* Ollama silently strips nearly every keyword —
  <https://github.com/ollama/ollama/issues/11444>:
  > Only the following fields are preserved in tool parameters: `type`, `$defs`
  > (mapped from `defs`), `items`, `required`, `properties`. All other JSON
  > schema properties are silently dropped.
* LiteLLM's `unpack_defs()` mis-resolves `$ref` inside `anyOf`/`items` —
  <https://github.com/BerriAI/litellm/issues/11372>.
* pydantic's own issue tracker, on exactly this question —
  <https://github.com/pydantic/pydantic/issues/12232> ("Tell Pydantic to not
  generate JSON schema with `$ref` `$defs`"). Maintainer response: a `ref_mode`
  parameter is under discussion, tracked in pydantic#12023. The practical
  workarounds suggested there are `jsonref`, or avoiding `StrEnum` in favour of
  `Literal`.
* Gemini accepts only a subset of OpenAPI 3.0 —
  <https://ai.google.dev/gemini-api/docs/function-calling>:
  > Only a subset of the OpenAPI schema is supported. For any mode, the API may
  > reject very large or deeply nested schemas.

### What this means for the normalizer

The module docstring says `$ref`/`$defs` are "the wrong output for a
function-calling API". **The vendors do not say that.** OpenAI documents both as
supported; Zed passes them straight through for OpenAI and Anthropic, and rejects
them only for the Gemini/xAI/proxy path. Meanwhile the keywords OpenAI
**explicitly errors on** are `allOf`, `not`, `if`/`then`/`else`,
`dependentSchemas` — none of which pydantic emits for any model in this
catalogue, and all of which the normalizer now correctly refuses.

So the priority order was inverted: the normalizer spent its effort eliminating
the *supported* keywords while the *rejected* ones passed through. That is now
fixed for the rejected set (they refuse), and inlining `$ref` remains defensible
as **portability insurance** for the Gemini/xAI/Ollama-class providers — but it
should be justified that way, not as an OpenAI requirement, and it must not cost
correctness (which is what made the original recursive refusal a bug: inlining is
impossible for a cycle, and refusing made a recursive tool unregisterable).

**Two guarantees the normalizer does *not* provide, and should not be assumed to:**

1. **It does not make a pydantic schema `strict: true`-valid.** OpenAI requires
   *every* property in `required` and `additionalProperties: false`. pydantic
   omits optional fields from `required`, so `OptionalDefault` normalizes to a
   schema that strict mode rejects. No amount of `$ref` inlining fixes this; it
   needs a different transform (optional → nullable, and promote every property
   into `required`).
2. **`format` is annotation-only.** pydantic rejects `"not-a-date"` for
   `datetime`, but the emitted `{"type": "string", "format": "date-time"}` — and
   therefore the normalized schema — accepts it. The schema is a *weaker*
   contract than the validator behind it (§1).

**Residual uncertainty.** Only OpenAI's docs were fetched in full. The Anthropic
tool-use page documents `input_schema` as "A JSON Schema object" without
enumerating a subset (<https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implement-tool-use>);
Gemini's page says "a subset of the OpenAPI schema" without listing it, so the
Gemini list in the table below comes from the third-party
`gemini_schema.py` allow-list, not from Google. No claim here rests on a page I
could not read, and nothing is quoted that I did not fetch.

---

## 6. The normalizer is not on the live path

Not asked for, but it changes what the above is worth.

```bash
$ grep -rn "schema_of" src/ tests/ --include=*.py
src/arm/tools/definition.py:261:def schema_of(model: type[ParameterModel]) -> Mapping[str, Any]:
```

`schema_of` — the only caller of `normalize_schema` — is **defined and never
called**. The two paths that actually put a schema on the wire are:

* `src/arm/tools/builtin.py:303` — `parameters=params.model_json_schema()`, raw;
* `src/arm/prompt/assembler.py:87` — `"parameters": to_wire(tool.parameters)`,
  which thaws whatever was stored, normalized or not.

Observed on the live path (`evidence.py 5`), for the builtin tool set:

```
read     ['default', 'description', 'properties', 'required', 'title', 'type']
write    ['description', 'properties', 'required', 'title', 'type']
...
```

`title` and `default` are still present — i.e. **the builtin tools ship the raw
pydantic schema**. `tool(parameters=<Mapping>)` also passes a mapping through
unnormalized.

So today the normalizer's correctness is a property of a code path nothing
executes. Wiring it in is a separate decision (it would change the model-facing
contract for every existing tool), but it should be made deliberately rather than
discovered later.

---

## 7. Files

| File | What it is |
|---|---|
| `jsonschema_subset.py` | Independent JSON Schema 2020-12 subset validator. `$ref` is conjunctive with siblings; boolean schemas supported; `format` annotation-only. |
| `schema_lint.py` | Structural well-formedness linter (`anyOf` must be an **array**) + schema-position keyword walker. |
| `catalogue.py` | 22 pydantic models + payload corpora + deterministic mutation generator. |
| `differential.py` | Raw-vs-normalized differential over every payload. Runnable. |
| `hostile.py` | 29 hostile schemas with declared expectations. Runnable. |
| `evidence.py` | Transcript generator — every quote above comes from it. Runnable. |
| `test_findings_bugs.py` | **The 3 live defects**, as `xfail(strict=True)`. |
| `test_fixed_regressions.py` | Guards for the 9 fixed defects. |
| `test_schema_normalizer.py` | Catalogue, differential, properties, reference cross-check. |

### Test status

```
$ cd spikes && ../.venv/bin/python -m pytest q2_schema
34 passed, 2 skipped, 5 xfailed

$ Q2_JSONSCHEMA_PATH=/tmp/q2venv/lib/python3.11/site-packages \
    ../.venv/bin/python -m pytest q2_schema
36 passed, 5 xfailed
```

The 5 `xfailed` are BUG 10 (×2), BUG 11 (×2) and BUG 12 (×1): each currently
**fails**, which `strict=True` reports as `xfail`. The suite is green, the
defects are documented, and each will become a hard failure the moment it is
fixed — so a fix cannot be reverted without the suite going red.

The 2 `skipped` are the reference-library cross-checks, skipped when
`jsonschema` is not importable (see §0).

Nothing under `src/`, `tests/`, `docs/`, `README.md` or `REQUIREMENTS.md` was
modified by this spike.
