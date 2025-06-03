import torch
from .torch_utils import quat_apply

def draw_axes(gym, viewer, env, pos, rot):
    if isinstance(pos, torch.Tensor):
        device = pos.device
    else:
        device = 'cpu'
        pos = torch.tensor(pos, device=device)
        rot = torch.tensor(rot, device=device)
    posx = (pos + quat_apply(rot.float(), torch.tensor([1, 0, 0], device=device) * 0.2)).cpu().numpy()
    posy = (pos + quat_apply(rot.float(), torch.tensor([0, 1, 0], device=device) * 0.2)).cpu().numpy()
    posz = (pos + quat_apply(rot.float(), torch.tensor([0, 0, 1], device=device) * 0.2)).cpu().numpy()
    pos = pos.cpu().numpy()
    gym.add_lines(viewer, env, 1, [pos[0], pos[1], pos[2], posx[0], posx[1], posx[2]], [1.0, 0, 0])
    gym.add_lines(viewer, env, 1, [pos[0], pos[1], pos[2], posy[0], posy[1], posy[2]], [0, 1.0, 0])
    gym.add_lines(viewer, env, 1, [pos[0], pos[1], pos[2], posz[0], posz[1], posz[2]], [0, 0, 1.0])
    
def clear_axes(task):
    task.gym.clear_lines(task.viewer)