"""Visualise recorded episodes as a single HTML page.

    poetry run so101-visualize                     # last 10 episodes
    poetry run so101-visualize --last 5            # last 5
    poetry run so101-visualize --episodes 0 1 2    # specific indices
    poetry run so101-visualize --all
    poetry run so101-visualize --open              # xdg-open after generating

Output: `episodes.html` (default; override with --out PATH). Each episode
shows the 3 video streams side by side + per-joint state/action plots
(matplotlib PNG inlined as base64).

This is a read-only inspection tool — it does NOT touch the dataset,
the motors, or any calibration.
"""

from __future__ import annotations

import argparse
import base64
import io
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from so101_ros2.settings import DATASET_REPO_ID, DATASET_ROOT, JOINT_NAMES


def _video_urls(ds: LeRobotDataset, episode_index: int, root: Path) -> dict[str, str]:
    """LeRobotDataset v3 concatenates many episodes into a single mp4 per
    camera; per-episode time ranges live in `meta.episodes`. Build
    `file://...mp4#t=START,END` URLs so the browser plays just this
    episode's segment via HTML5 media fragments."""
    ep_row = ds.meta.episodes[episode_index]
    out: dict[str, str] = {}
    for key, feat in ds.meta.features.items():
        if feat.get("dtype") != "video":
            continue
        chunk = ep_row[f"videos/{key}/chunk_index"]
        fidx  = ep_row[f"videos/{key}/file_index"]
        t0    = ep_row[f"videos/{key}/from_timestamp"]
        t1    = ep_row[f"videos/{key}/to_timestamp"]
        mp4 = root / ds.repo_id / "videos" / key / f"chunk-{chunk:03d}" / f"file-{fidx:03d}.mp4"
        if mp4.exists():
            out[key] = f"file://{quote(str(mp4.resolve()))}#t={t0:.3f},{t1:.3f}"
    return out


def _plot_episode_state_action(ep_rows, episode_index: int) -> str:
    """Return base64-encoded PNG of a 2-row × 6-col plot:
    row 0 = observation.state, row 1 = action, columns = joints."""
    state  = np.array(ep_rows["observation.state"])
    action = np.array(ep_rows["action"])
    t = np.arange(state.shape[0]) / 30.0  # assume 30 Hz; cosmetic

    fig, axes = plt.subplots(2, 6, figsize=(18, 4.5), sharex=True)
    fig.suptitle(f"Episode {episode_index}  ({state.shape[0]} frames, {t[-1]:.1f}s)",
                 fontsize=11)
    for j, name in enumerate(JOINT_NAMES):
        axes[0, j].plot(t, state[:, j],  color="#1f77b4", linewidth=1.0)
        axes[0, j].set_title(name, fontsize=9)
        axes[0, j].tick_params(labelsize=7)
        axes[0, j].grid(alpha=0.3)
        axes[1, j].plot(t, action[:, j], color="#d62728", linewidth=1.0)
        axes[1, j].tick_params(labelsize=7)
        axes[1, j].grid(alpha=0.3)
        axes[1, j].set_xlabel("s", fontsize=8)
    axes[0, 0].set_ylabel("state (deg)", fontsize=9)
    axes[1, 0].set_ylabel("action (deg)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=85, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _html_for_episode(ds: LeRobotDataset, episode_index: int, root: Path) -> str:
    # Pull just this episode's rows
    ep = ds.hf_dataset.filter(lambda ex: ex["episode_index"] == episode_index)
    plot_b64 = _plot_episode_state_action(ep, episode_index)

    videos = _video_urls(ds, episode_index, root)
    vid_html = ""
    for name, url in videos.items():
        vid_html += (
            f'<div class="cam"><div class="cam-label">{name}</div>'
            f'<video controls preload="metadata" muted style="width:100%;max-width:480px">'
            f'<source src="{url}" type="video/mp4"></video></div>'
        )

    return f"""
    <section class="ep">
      <h2>Episode {episode_index}</h2>
      <div class="cams">{vid_html}</div>
      <img class="plot" src="data:image/png;base64,{plot_b64}" alt="state/action plot"/>
    </section>
    """


def _html_skeleton(title: str, body: str) -> str:
    return f"""<!doctype html>
<html><head>
  <meta charset="utf-8"/>
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 16px; background:#fafafa; color:#222 }}
    h1 {{ font-size: 18px; margin: 0 0 12px }}
    .ep {{ background:#fff; border:1px solid #ddd; border-radius:6px; padding:12px; margin:0 0 16px }}
    .ep h2 {{ font-size: 14px; margin: 0 0 8px; color:#555 }}
    .cams {{ display:flex; gap:8px; flex-wrap:wrap; margin-bottom:8px }}
    .cam {{ flex:1; min-width:240px; max-width:480px }}
    .cam-label {{ font-size:11px; color:#777; margin-bottom:2px }}
    .plot {{ width:100%; max-width:1200px; border:1px solid #eee }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  {body}
</body></html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default=DATASET_REPO_ID)
    p.add_argument("--root",    default=str(DATASET_ROOT))
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--episodes", type=int, nargs="+", help="specific indices")
    grp.add_argument("--last",     type=int, default=10, help="last N episodes (default 10)")
    grp.add_argument("--all",      action="store_true")
    p.add_argument("--out",        default="episodes.html")
    p.add_argument("--open",       action="store_true", help="xdg-open after generating")
    args = p.parse_args()

    ds_root = Path(args.root) / args.repo_id
    if not ds_root.exists():
        sys.stderr.write(f"ERROR: dataset not found at {ds_root}\n")
        sys.exit(1)

    print(f"loading dataset: {args.repo_id} at {ds_root}")
    ds = LeRobotDataset(repo_id=args.repo_id, root=ds_root)
    n_total = ds.meta.total_episodes
    print(f"  {n_total} episode(s), {ds.meta.total_frames} total frames @ {ds.fps} Hz")

    if args.episodes:
        ep_indices = [i for i in args.episodes if 0 <= i < n_total]
    elif args.all:
        ep_indices = list(range(n_total))
    else:
        ep_indices = list(range(max(0, n_total - args.last), n_total))

    if not ep_indices:
        sys.stderr.write("ERROR: no episodes match selection\n")
        sys.exit(1)

    print(f"  rendering {len(ep_indices)} episode(s): {ep_indices}")
    body = ""
    for i, idx in enumerate(ep_indices, 1):
        print(f"    [{i}/{len(ep_indices)}] episode {idx}")
        body += _html_for_episode(ds, idx, Path(args.root))

    title = f"{args.repo_id} — episodes {ep_indices[0]}..{ep_indices[-1]} ({len(ep_indices)} of {n_total})"
    Path(args.out).write_text(_html_skeleton(title, body))
    out_abs = Path(args.out).resolve()
    print(f"\nwrote {out_abs}")

    if args.open:
        try:
            subprocess.Popen(["xdg-open", str(out_abs)])
        except FileNotFoundError:
            print("(xdg-open not available — open the file manually)")


if __name__ == "__main__":
    main()
