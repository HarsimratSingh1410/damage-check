# Damage-Check Module: Project Plan

## 1. Goal

Automatically detect new damage on borrowed items by comparing **before** photos (taken at handover) with **after** photos (taken at return). Flagged items go to an admin for review. The system never decides blame on its own.

## 2. Scope

**In scope**
- Guided photo capture requirements (spec for the app team)
- Before/after change-detection pipeline
- Structured results for the admin review screen
- Config-driven thresholds and swappable models

**Out of scope (handled by the main app)**
- Users, auth, and item catalog
- Notifications and payments/penalties
- The admin UI itself (the module only supplies data for it)

## 3. Key Constraints

- Items can be anything, so no per-item-type training.
- Each item is its own reference: the "before" photos of that same item.
- Photos are taken by borrowers and lenders on their own phones, with varying backgrounds and lighting.
- The module must be **modular and easy to integrate** while the rest of the team builds other features.
- Missing real damage is worse than an extra review, so tune for high recall and let admins filter.

## 4. Architecture

A self-contained module with a narrow contract. It knows nothing about the app's users or database: images in, structured results out.

```
damage_check/
  core/
    pipeline.py        # orchestrates the stages
    interfaces.py      # Segmenter, Aligner, Comparator, Reviewer, Storage
    models.py          # pydantic input/output schemas
  stages/
    segment_rembg.py   # baseline segmentation
    segment_sam.py     # SAM segmentation
    align_sift.py      # SIFT/ORB + homography
    align_loftr.py     # LoFTR
    compare_ssim.py    # baseline comparison
    compare_dino.py    # DINOv2 patch features
    review_vlm.py      # optional VLM filter
  service/
    api.py             # FastAPI wrapper (jobs, status, results)
  storage/
    local.py
    s3.py
  config.py            # thresholds, model choices, device, VLM on/off
  cli.py               # run on local before/after pairs
  tests/
    fixtures/          # before/after pairs with expected outcomes
```

**Integration modes**
1. Python library (import and call `pipeline.run(job)`)
2. Standalone HTTP service (recommended): keeps ML/GPU dependencies out of the main app and scales independently

## 5. Contract (Schema v1)

**Input**
```json
{
  "schema_version": "1",
  "job_id": "abc123",
  "item_id": "item_42",
  "views": [
    {"view_name": "front", "before_url": "...", "after_url": "..."},
    {"view_name": "left",  "before_url": "...", "after_url": "..."}
  ]
}
```

**Output**
```json
{
  "schema_version": "1",
  "job_id": "abc123",
  "status": "flagged",
  "views": [
    {
      "view_name": "front",
      "quality": {"aligned": true, "alignment_score": 0.91, "issues": []},
      "regions": [
        {
          "bbox": [120, 340, 80, 60],
          "score": 0.78,
          "vlm_label": "new scratch",
          "crop_before_url": "...",
          "crop_after_url": "..."
        }
      ],
      "heatmap_url": "..."
    }
  ]
}
```

**Status values:** `ok`, `flagged`, `needs_retake`, `low_confidence`

**Service endpoints (async)**
- `POST /jobs` submits a job and returns `job_id`
- `GET /jobs/{job_id}` returns status and results
- Optional webhook callback on completion

## 6. Pipeline (per view pair)

1. **Quality gate:** check blur, brightness, glare, and object size. If it fails, return `needs_retake`.
2. **Segment** the object to remove the background (rembg, then SAM).
3. **Align** the after image to the before image (SIFT + homography, later LoFTR), then normalize brightness/color (CLAHE or histogram matching). If alignment fails, return `needs_retake`.
4. **Compare**
   - Baseline: SSIM difference map
   - Target: DINOv2 patch-embedding distance to produce a change heatmap
5. **Filter candidates:** drop edge artifacts from slight misalignment, tiny blobs (dust), and glare regions. Merge nearby regions.
6. **Review (optional):** crop each region from both images and ask a VLM whether it is new damage (scratch, dent, crack, stain, missing part). Attach a short description.
7. **Aggregate** per-view results into an overall job status.

## 7. Capture Requirements (spec for the app team)

- In-app camera only (no gallery uploads), with trusted timestamps.
- Fixed set of views: front, back, left, right, top, bottom, plus close-ups of existing marks.
- **Ghost overlay** of the "before" photo while taking the "after" photo.
- Client-side quality checks (blur, darkness, glare, object too small) with immediate retake prompts.
- Borrower confirms the "before" set at handover.
- Originals are stored unmodified, with SHA-256 hashes and timestamps for dispute evidence.

## 8. Admin Review Data (what the UI needs)

- Before/after side by side per view
- Flagged region boxes, heatmap toggle, confidence score, VLM description
- Actions: confirm damage, dismiss, request retake
- Every decision is logged and fed back as labeled data

## 9. Milestones

| Phase | Deliverable | Exit criteria |
|---|---|---|
| **0. Setup** | Repo skeleton, interfaces, pydantic schemas, config, CLI stub | Schema reviewed and agreed with the app team |
| **1. Baseline** | rembg + SIFT alignment + SSIM diff, CLI, heatmap output | Runs end to end on 10+ real before/after pairs |
| **2. Service** | FastAPI async job API, storage interface (local + S3), Docker image | App team can submit a job and fetch results |
| **3. Better comparison** | DINOv2 comparator, SAM segmenter, LoFTR aligner behind the same interfaces | Measurable drop in false positives vs. baseline |
| **4. VLM filter** | Optional review stage with readable labels | False-positive rate reduced without hurting recall |
| **5. Admin integration** | Admin UI consumes schema v1, decision logging | Admins can confirm/dismiss flags in the app |
| **6. Tuning** | Threshold tuning from admin decisions, low-confidence routing | Targets in section 10 met |

## 10. Evaluation

Build a test set of before/after pairs covering: no change, lighting-only change, background change, minor shift, real scratch/dent/crack/stain, and difficult materials (shiny, transparent, fabric).

**Metrics**
- Recall on real damage (primary target: very high)
- False-positive rate per item (what admins actually feel)
- Alignment failure and retake rate
- Processing time per job

Run regression tests against the fixture set on every change.

## 11. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| False positives from shadows, reflections, fingerprints | Glare filter, deep-feature comparison, VLM second pass, admin review |
| Shiny, transparent, or uniform items | Mark as `low_confidence` and route to manual review |
| Items that legitimately change shape (fabric, cables) | Lower sensitivity and route to manual review |
| Poor photos from borrowers | Quality gate, guided capture, ghost overlay, forced retakes |
| Pre-existing damage blamed on borrower | Compare before vs. after only; borrower confirms "before" set |
| Disputes over evidence | Immutable originals, hashes, timestamps, decision logs |
| ML dependencies complicating the main stack | Separate service with its own container |
| VLM hallucinations | Use as a filter and explanation only, never the sole decision-maker |

## 12. Open Questions (to tailor the plan)

- Backend language/framework (library vs. service decision)
- Mobile/web stack for capture (React Native, Flutter, native)
- Image storage and referencing (S3, Firebase, own server)
- GPU availability in deployment, or CPU only
- Expected volume (returns per day) and required response time
- Whether the admin review UI exists or is being built
- Privacy/retention rules for stored photos

## 13. Next Steps

1. Review and agree on the schema (section 5) with the team.
2. Build the Phase 0 skeleton and Phase 1 baseline.
3. Collect 10-20 real before/after pairs to use as the first fixtures.
4. Answer the open questions above so the service and storage layers can be finalized.
