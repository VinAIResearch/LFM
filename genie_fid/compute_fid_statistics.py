import os
import argparse
import torch
import numpy as np
import pickle

from stylegan3.dataset import ImageFolderDataset
from dnnlib.util import open_url

def get_activations(dl, model, batch_size, device, max_samples):
    pred_arr = []
    total_processed = 0

    print('Starting to sample.')
    for batch in dl:
        # ignore labels
        if isinstance(batch, list):
            batch = batch[0]

        batch = batch.to(device)
        if batch.shape[1] == 1:  # if image is gray scale
            batch = batch.repeat(1, 3, 1, 1)
        elif len(batch.shape) == 3:  # if image is gray scale
            batch = batch.unsqueeze(1).repeat(1, 3, 1, 1)


        with torch.no_grad():
            pred = model(batch, return_features=True).unsqueeze(-1).unsqueeze(-1)

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr.append(pred)
        total_processed += pred.shape[0]
        if max_samples is not None and total_processed > max_samples:
            print('Max of %d samples reached.' % max_samples)
            break

    pred_arr = np.concatenate(pred_arr, axis=0)
    if max_samples is not None:
        pred_arr = pred_arr[:max_samples]

    return pred_arr


def main(args):
    if not os.path.exists(args.fid_dir):
        os.makedirs(args.fid_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = ImageFolderDataset(args.path)
    queue = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, pin_memory=True, num_workers=1)

    with open_url('https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl') as f:
        model = pickle.load(f).to(device)
        model.eval()

    act = get_activations(queue, model, batch_size=args.batch_size, device=device, max_samples=args.max_samples)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    file_path = os.path.join(args.fid_dir, args.file)
    np.savez(file_path, mu=mu, sigma=sigma)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--fid_dir', type=str, default='assets/stats/')
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    main(args)
