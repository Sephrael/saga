# Proposed Redesign of Phase 2 Step 5: The Renovation

This document proposes a new approach for the patch-based revision step in SAGA.
The goal is to produce higher quality chapters while keeping the pipeline fast
and deterministic.

## Current Issues

* **Patch Quality Varies:** Generated patches sometimes fail to correct the
  identified problem or introduce new inconsistencies.
* **Overlapping Fixes:** Multiple problems can target the same sentence leading
  to conflicting patches.
* **Unclear Validation:** After applying a patch it is not obvious whether the
  fix resolves the underlying critique.

## Key Changes

1. **Structured Patch Requests**
   * Problems are grouped by their sentence offsets prior to calling the LLM.
   * A single request generates patches for an entire group, ensuring coherent
     changes when multiple issues overlap.
   * The LLM receives a table summarising all issues for that group along with
     a snippet of the original text.

2. **Patch Verification Phase**
   * A lightweight `PatchValidationAgent` reviews each generated patch.
   * It confirms that the replacement addresses all issues in the group and
     preserves continuity with the surrounding context.
   * If validation fails, the patch is discarded and the group is queued for
     a full rewrite fallback.

3. **Semantic Re-mapping**
   * Sentence embeddings are stored when the chapter draft is first produced.
   * When applying a patch, the system uses these embeddings to more reliably
     locate sentences even if offsets shift during earlier revisions.
   * This reduces patch drift when multiple cycles occur.

4. **Adaptive Retry Logic**
   * Each patch group is allowed up to two attempts.
   * On the first failure the system rewrites the prompt with stricter
     instructions. On the second failure it escalates to a full rewrite of the
     affected section.

5. **Metrics and Logging**
   * Patch generation and validation usage statistics are tracked separately.
   * Detailed logs make it easier to trace why a patch was discarded or kept.

## Benefits

* **Improved Accuracy:** Grouping related problems and verifying patches reduces
  the risk of partial fixes.
* **Better Stability:** Embedding-based sentence mapping ensures patches apply to
  the correct text even after earlier revisions.
* **Clearer Failures:** The validation phase explicitly signals when a patch is
  insufficient, triggering a fallback rather than silently accepting bad output.

This redesign keeps Step 5 aligned with the overall chapter-generation loop while
making patch-based revision more reliable and transparent.
