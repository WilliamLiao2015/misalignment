import os

from pathlib import Path

import matplotlib.pyplot as plt

from trajdata.data_structures import AgentBatch
from trajdata.visualization.interactive_animation import InteractiveAnimation, animate_agent_batch_interactive
from trajdata.visualization.interactive_vis import plot_agent_batch_interactive
from trajdata.visualization.vis import plot_agent_batch

def visualize(cache_path: Path, batch: AgentBatch, batch_idx: int, interactive: bool = False) -> None:
    if not interactive:
        plot_agent_batch(batch, batch_idx=batch_idx, show=False, close=False)
        plt.savefig(os.path.join(os.path.dirname(__file__), "../data/original-scenario.png"), bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plot_agent_batch_interactive(batch, batch_idx=batch_idx, cache_path=cache_path)

        animation = InteractiveAnimation(
            animate_agent_batch_interactive,
            batch=batch,
            batch_idx=batch_idx,
            cache_path=cache_path
        )
        animation.show()
