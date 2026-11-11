#!/usr/bin/env python3
"""CLI to run the damage-check pipeline on local before/after image pairs."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import typer

app = typer.Typer(help="Damage-Check CLI — compare before/after photos for damage.")


def _build_pipeline(
    output_dir: str,
    segmenter_name: Optional[str] = None,
    aligner_name: Optional[str] = None,
    comparator_name: Optional[str] = None,
):
    from .config import settings as cfg
    from .storage.local import LocalStorage
    from .core.pipeline import Pipeline

    storage = LocalStorage(output_dir=output_dir)

    seg_choice = segmenter_name or cfg.segmenter
    if seg_choice == "rembg":
        from .stages.segment_rembg import RembgSegmenter
        segmenter = RembgSegmenter()
    elif seg_choice == "sam":
        from .stages.segment_sam import SAMSegmenter
        segmenter = SAMSegmenter()
    elif seg_choice in ("none", "passthrough", "dummy"):
        from .core.interfaces import Segmenter
        class PassthroughSegmenter(Segmenter):
            def segment(self, image: np.ndarray) -> np.ndarray:
                return np.ones(image.shape[:2], dtype=np.uint8) * 255
        segmenter = PassthroughSegmenter()
    else:
        raise ValueError(f"Unknown segmenter: {seg_choice!r}")

    align_choice = aligner_name or cfg.aligner
    if align_choice == "sift":
        from .stages.align_sift import SIFTAligner
        aligner = SIFTAligner()
    elif align_choice == "loftr":
        from .stages.align_loftr import LoFTRAligner
        aligner = LoFTRAligner()
    else:
        raise ValueError(f"Unknown aligner: {align_choice!r}")

    comp_choice = comparator_name or cfg.comparator
    if comp_choice == "ssim":
        from .stages.compare_ssim import SSIMComparator
        comparator = SSIMComparator()
    elif comp_choice == "dino":
        from .stages.compare_dino import DINOv2Comparator
        comparator = DINOv2Comparator()
    else:
        raise ValueError(f"Unknown comparator: {comp_choice!r}")

    return Pipeline(
        segmenter=segmenter,
        aligner=aligner,
        comparator=comparator,
        storage=storage,
    )


@app.command()
def run(
    before: List[Path] = typer.Option(..., help="Before image path(s)"),
    after: List[Path] = typer.Option(..., help="After image path(s)"),
    view_names: Optional[List[str]] = typer.Option(
        None, "--view", help="View name(s) (default: front, back, …)"
    ),
    job_id: str = typer.Option("cli_job", help="Job ID"),
    item_id: str = typer.Option("cli_item", help="Item ID"),
    output_dir: str = typer.Option("output", help="Directory for output files"),
    segmenter: Optional[str] = typer.Option(None, help="Segmenter: 'rembg' or 'sam'"),
    aligner: Optional[str] = typer.Option(None, help="Aligner: 'sift' or 'loftr'"),
    comparator: Optional[str] = typer.Option(None, help="Comparator: 'ssim' or 'dino'"),
    json_output: bool = typer.Option(False, "--json", help="Print result as JSON"),
) -> None:
    """Run the damage-check pipeline on before/after image pairs."""
    if len(before) != len(after):
        typer.echo(
            "ERROR: Number of --before and --after paths must match.", err=True
        )
        raise typer.Exit(1)

    from .core.models import JobInput, ViewInput

    default_names = ["front", "back", "left", "right", "top", "bottom"]
    names = view_names or [
        default_names[i] if i < len(default_names) else f"view_{i}"
        for i in range(len(before))
    ]

    views = [
        ViewInput(view_name=n, before_url=str(b), after_url=str(a))
        for n, b, a in zip(names, before, after)
    ]
    job = JobInput(job_id=job_id, item_id=item_id, views=views)

    pipeline = _build_pipeline(
        output_dir,
        segmenter_name=segmenter,
        aligner_name=aligner,
        comparator_name=comparator,
    )
    result = pipeline.run(job)

    if json_output:
        typer.echo(result.model_dump_json(indent=2))
    else:
        typer.echo(f"Job: {result.job_id}  Status: {result.status.value}")
        for vr in result.views:
            typer.echo(
                f"  View: {vr.view_name}  Aligned: {vr.quality.aligned}"
                f"  Regions: {len(vr.regions)}"
            )
            for reg in vr.regions:
                typer.echo(
                    f"    Region bbox={reg.bbox} score={reg.score:.3f}"
                    f" label={reg.vlm_label}"
                )
        typer.echo(f"  Heatmap(s) and crops saved to: {output_dir}/")


@app.command("version")
def version():
    """Show the Damage-Check module version."""
    typer.echo("damage-check version 1.0.0")


if __name__ == "__main__":
    app()
