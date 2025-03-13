import argparse
import os
import sys

folder = os.path.dirname(__file__)
sys.path.append(os.path.join(folder, "../ProSim"))

import torch
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader
from torch import tensor, nan
from typing import List

from prosim.config.default import Config, get_config
from prosim.core.registry import registry
from prosim.dataset.data_utils import rotate
from prosim.demo.vis import plot_full_map, plot_batch_prompts, plot_model_output, plot_demo_fig, extract_lane_vecs

# copied from https://stackoverflow.com/a/45669280/16082247
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def modify_batch_condition(batch, type_control_nidx_dict, type_input_dict):
    all_agent_names = batch.extras['prompt']['motion_pred']['agent_ids'][0]
    controlled_agent_names = []

    if type(batch.extras['condition']) is not dict:
        batch.extras['condition'] = batch.extras['condition'].all_cond

    for ptype in type_control_nidx_dict.keys():
        control_indices = type_control_nidx_dict[ptype]
        for idx in control_indices:
            aname = all_agent_names[idx]
            if aname not in controlled_agent_names:
                controlled_agent_names.append(aname)

    if 'v_action_tag' in batch.extras['condition'].keys():
        del batch.extras['condition']['v_action_tag']

    for ptype in batch.extras['condition'].keys():
        cond_data = batch.extras['condition'][ptype]

        if ptype == 'llm_text_OneText':
            batch.extras['condition'][ptype]['mask'][0] = False
            batch.extras['condition'][ptype]['prompt_mask'][0, :] = False
            continue
        elif ptype not in type_control_nidx_dict:
            batch.extras['condition'][ptype]['mask'][0, :] = False
            batch.extras['condition'][ptype]['prompt_mask'][0, :] = False
            continue

        prompt_inputs = cond_data['input'][0]
        prompt_masks = cond_data['mask'][0, :]
        prompt_idxes = cond_data['prompt_idx'][0, :, 0]
        prompt_pmasks = cond_data['prompt_mask'][0, :]

        if ptype not in type_input_dict:
            type_input_dict[ptype] = {}

        for cidx, pidx in enumerate(prompt_idxes):
            if pidx not in type_control_nidx_dict[ptype]:
                prompt_masks[cidx] = False
                prompt_pmasks[pidx] = False
            else:
                prompt_masks[cidx] = True
                prompt_pmasks[pidx] = True
                if pidx.item() in type_input_dict[ptype]:
                    prompt_inputs[cidx] = type_input_dict[ptype][pidx.item()]
                else:
                    type_input_dict[ptype][pidx.item()] = prompt_inputs[pidx]

        batch.extras['condition'][ptype]['input'][0] = prompt_inputs
        batch.extras['condition'][ptype]['mask'][0] = prompt_masks
        batch.extras['condition'][ptype]['prompt_mask'][0] = prompt_pmasks


    for ptype in type_control_nidx_dict:
        if ptype in batch.extras['condition'].keys():
            continue
        prompt_inputs = []
        prompt_masks = []
        prompt_idx = []
        for pidx in type_control_nidx_dict[ptype]:
            prompt_inputs.append(type_input_dict[ptype][pidx])
            prompt_idx.append(pidx)
            prompt_masks.append(True)

        prompt_inputs = torch.stack(prompt_inputs)[None, :]
        # prompt_idx = torch.tensor(prompt_idx)[None, None, :]
        prompt_idx = torch.tensor(prompt_idx)[None, :]
        prompt_masks = torch.tensor(prompt_masks)[None, :]
        prompt_pmasks = torch.tensor(prompt_masks)

        batch.extras['condition'][ptype] = {}
        batch.extras['condition'][ptype]['input'] = prompt_inputs
        batch.extras['condition'][ptype]['mask'] = prompt_masks
        batch.extras['condition'][ptype]['prompt_mask'] = prompt_pmasks
        batch.extras['condition'][ptype]['prompt_idx'] = prompt_idx

    return batch, type_input_dict, controlled_agent_names

def obtain_agent_name_to_color(batch, controlled_agent_names):
    agent_name_to_color = {}

    prompt_type_to_color = {'v_action_tag': np.array([187, 152, 71]) / 255, 'goal': np.array([63, 138, 226]) / 255, 'drag_point': np.array([96, 179, 186]) / 255, 'text': (0.9677975592919913, 0.44127456009157356, 0.5358103155058701)}

    agent_name_to_color = {}

    for aname in controlled_agent_names:
        agent_name_to_color[aname] = prompt_type_to_color['text']

    return agent_name_to_color

def text_control(batch, text_input, text_idxs):
    text_control_agent_names = []

    for idx in text_idxs:
        text_aname = batch.extras['prompt']['motion_pred']['agent_ids'][0][idx]
        if text_aname not in text_control_agent_names:
            text_control_agent_names.append(text_aname)

    batch.extras['condition']['llm_text_OneText']['input'] = [text_input]
    batch.extras['condition']['llm_text_OneText']['mask'][0] = True
    batch.extras['condition']['llm_text_OneText']['prompt_mask'][0, :] = False
    for idx in text_idxs:
        batch.extras['condition']['llm_text_OneText']['prompt_mask'][0, idx] = True

    controlled_agent_names = (text_control_agent_names)

    return batch, controlled_agent_names

def get_trajectories(output):
    vis_indices = []
    for idx, name in enumerate(output['pair_names']):
        batch_id, _, T = name.split('-')
        if batch_id == '0' and int(T) == 0:
            vis_indices.append(idx)

    trajectories = []

    for vidx in vis_indices:
        name = output['pair_names'][vidx]
        rollout_id = '-'.join(name.split('-')[:2])
        roll_traj = output['rollout_trajs'][rollout_id]['traj'].cpu().detach().numpy()
        init_pos = output['rollout_trajs'][rollout_id]['init_pos'].cpu().detach().numpy()
        init_heading = output['rollout_trajs'][rollout_id]['init_heading'].cpu().detach().numpy()

        roll_traj = rotate(roll_traj[..., 0], roll_traj[..., 1], init_heading) + init_pos
        
        trajectories.append(roll_traj)

    return np.asarray(trajectories)

if __name__ == "__main__":
    with HiddenPrints():
        parser = argparse.ArgumentParser(description="Generate a scenario from the given text.")
        parser.add_argument("--text", type=str, help="The text to generate the scenario from.", default="Let <A11> stop moving; make <A12> stop moving.")
        parser.add_argument("--text_indices", type=int, nargs='+', help="The indices of the text to use.", default=[11, 12])
        parser.add_argument("--example_index", type=int, help="The index of the example to use.", default=5)
        parser.add_argument("--save_image", action="store_true", help="Save the generated image.")
        args = parser.parse_args()

        config = get_config(os.path.join(folder, '../ProSim/prosim_demo/cfg/waymo_demo.yaml'), cluster='local')
        dataset = registry.get_dataset("prosim_imitation")(config, 'train')
        all_data_index = dataset._data_index

        sample_data_index = [all_data_index[args.example_index]]
        dataset._data_index = sample_data_index
        dataset._data_len = len(sample_data_index)

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.get_collate_fn(), num_workers=0)

        for batch in dataloader: break

        vec_map = batch.vector_maps[0]
        center_in_world_xyzh = batch.centered_agent_state.as_format('x,y,z,h').cpu().numpy()[0]
        vis_vecs = extract_lane_vecs(vec_map, center_in_world_xyzh, 150)

        config = get_config('configs/prosim.yaml', cluster='local')
        full_ckpt = 'ProSim/prosim_demo/ckpt/prosim_demo_model.ckpt'
        LLAMA_PATH = os.path.join(folder, '../models/Meta-Llama-3-8B-Instruct') # Replace with your local path to Meta-Llama-3-8B-Instruct huggingface model directory

        config.MODEL.CONDITION_TRANSFORMER.CONDITION_ENCODER.TEXT.LLM.MODEL_PATH['LLAMA3_8B_INSTRUCT'] = LLAMA_PATH

        model_cls = registry.get_model("prosim_policy_relpe_T_step_temporal_close_loop")
        model = model_cls.load_from_checkpoint(full_ckpt, config=config, strict=False, map_location='cpu')

        batch, controlled_agent_names = text_control(batch, text_input=args.text, text_idxs=args.text_indices)
        batch = modify_batch_condition(batch, {}, {})[0]

        agent_name_to_color = obtain_agent_name_to_color(batch, controlled_agent_names)
        output = model.forward(batch, 'val')['motion_pred']
        plot_demo_fig(batch, controlled_names=controlled_agent_names, vis_vecs=vis_vecs, output=output, show_all_name=False, agent_name_to_color=agent_name_to_color)

        if args.save_image:
            plt.savefig("data/prosim-scenario.png")
            plt.close()

    print(get_trajectories(output).tolist())
