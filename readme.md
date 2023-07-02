# Latent Flow Matching

## Installation

Python 3.8 and Pytorch 1.13.1 are used in this implementation.
Please install required libraries:

```
pip install -r requirements.txt
```

## Training

All training scripts are wrapped in [run.sh](run.sh). Simply comment/uncomment the relevant commands and run `bash run.sh`.

## Testing

Please modify some arguments in [run_test.sh](run_test.sh) for corresponding experiments and then run `bash run_test.sh`.
These arguments are specifies as follows:

```bash
MODEL_TYPE=DiT-L/2
EPOCH_ID=475
DATASET=celeba_256
EXP=celeb_f8_dit
METHOD=dopri5
STEPS=0
USE_ORIGIN_ADM=False
```

Detailed arguments and checkpoints are provided below:

<table>
  <tr>
    <th>Exp</th>
    <th>Args</th>
    <th>FID</th>
    <th>Checkpoints</th>
  </tr>

  <tr>
    <td> celeb_f8_dit </td>
    <td><a href="test_args/celeb256_dit.txt"> test_args/celeb256_dit.txt</a></td>
    <td>5.26</td>
    <td><a href="https://drive.google.com/drive/folders/1tbd1t0Yt3ix1v_OCGWJ7xyeubhCi99ql?usp=share_link">model_475.pth</a></td>
  </tr>

  <tr>
    <td> ffhq_f8_dit </td>
    <td><a href="test_args/ffhq_dit.txt"> test_args/ffhq_dit.txt</a></td>
    <td>4.55</td>
    <td><a href="https://drive.google.com/drive/folders/1jn6xHlaQ72hKk9RtJKo5lvr7SvYMCobU?usp=share_link">model_475.pth</a></td>
  </tr>

  <tr>
    <td> bed_f8_dit </td>
    <td><a href="test_args/bed_dit.txt"> test_args/bed_dit.txt</a></td>
    <td>4.92</td>
    <td><a href="https://drive.google.com/drive/folders/1o1uDrTAPIENHRh56CdVdGiEHGNqKcaC8?usp=share_link">model_550.pth</a></td>
  </tr>

  <tr>
    <td> church_f8_dit </td>
    <td><a href="test_args/church_dit.txt"> test_args/church_dit.txt</a></td>
    <td>5.54</td>
    <td><a href="https://drive.google.com/drive/folders/15ONlqM2eNbA91j7BikWPQG_6RH80NUwz?usp=share_link">model_575.pth</a></td>
  </tr>

  <tr>
    <td> celeb512_f8_adm </td>
    <td><a href="test_args/celeb256_adm.txt"> test_args/celeb512_adm.txt</a></td>
    <td>6.35</td>
    <td><a href="https://drive.google.com/drive/folders/1lWE9hCqzZ2Q1mS2BmTsA3nYWB_T25wqV?usp=share_link">model_575.pth</a></td>
  </tr>

  <tr>
    <td> celeba_f8_adm </td>
    <td><a href="test_args/celeb256_adm.txt"> test_args/celeb256_adm.txt</a></td>
    <td>5.82</td>
    <td>---</td>
  </tr>

  <tr>
    <td> ffhq_f8_adm </td>
    <td><a href="test_args/ffhq_adm.txt"> test_args/ffhq_adm.txt</a></td>
    <td>5.82</td>
    <td>---</td>
  </tr>

  <tr>
    <td> bed_f8_adm </td>
    <td><a href="test_args/bed_adm.txt"> test_args/bed_adm.txt</a></td>
    <td>7.05</td>
    <td>---</td>
  </tr>

  <tr>
    <td> church_f8_adm </td>
    <td><a href="test_args/church_adm.txt"> test_args/church_adm.txt</a></td>
    <td>7.7</td>
    <td>---</td>

  </tr>

</table>

> All attached links are made by an anonymous account.

Please put downloaded pre-trained models in `saved_info/latent_flow/<DATASET>/<EXP>` directory where `<DATASET>` is defined as in [run.sh](run.sh).

To evaluate FID scores, please download pre-computed stats from [here](https://drive.google.com/drive/folders/1BXCqPUD36HSdrOHj2Gu_vFKA3M3hJspI?usp=share_link) and put it to `pytorch_fid`.
