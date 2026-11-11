# Test Fixtures

Place before/after image pairs here for regression testing.

## Naming Convention

```
fixtures/
  <item_id>/
    <view_name>_before.jpg
    <view_name>_after.jpg
    expected.json          # expected JobResult status and region count
```

## Categories to Cover (from plan.md §10)

| Folder | Description |
|---|---|
| `no_change/` | Identical or near-identical (lighting shifts only) |
| `background_change/` | Different backgrounds, same item |
| `minor_shift/` | Slight misalignment / rotation |
| `scratch/` | Real scratch added |
| `dent/` | Real dent |
| `crack/` | Crack or fracture |
| `stain/` | Liquid stain |
| `missing_part/` | Detached piece |
| `shiny/` | Reflective materials (low confidence expected) |
| `transparent/` | Glass / plastic items |
| `fabric/` | Deformable items (fabric, cables) |

## `expected.json` Schema

```json
{
  "status": "flagged",
  "min_regions": 1,
  "max_regions": 5,
  "notes": "Free-text explanation for human reviewers."
}
```
