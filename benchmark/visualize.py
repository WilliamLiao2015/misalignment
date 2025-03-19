from pathlib import Path

from trajdata.data_structures import AgentBatch
from trajdata.visualization.interactive_animation import InteractiveAnimation, animate_agent_batch_interactive
from trajdata.visualization.interactive_vis import plot_agent_batch_interactive
from trajdata.visualization.vis import plot_agent_batch

def visualize(cache_path: Path, batch: AgentBatch, batch_idx: int, interactive: bool = False) -> None:
    if not interactive:
        plot_agent_batch(batch, batch_idx=batch_idx)
    else:
        plot_agent_batch_interactive(batch, batch_idx=batch_idx, cache_path=cache_path)

        animation = InteractiveAnimation(
            animate_agent_batch_interactive,
            batch=batch,
            batch_idx=batch_idx,
            cache_path=cache_path
        )
        animation.show()
