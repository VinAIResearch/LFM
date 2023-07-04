##### Table of contents

1. [Installation](#Installation)
2. [Dataset preparation](#Dataset-preparation)
3. [Training](#Training)
4. [Testing](#Testing)
5. [Acknowledgments](#Acknowledgments)
6. [Contacts](#Contacts)

# Official PyTorch implementation of "Flow Matching in Latent Space"

<div align="center">
  <a href="https://quandao10.github.io/" target="_blank">Quan&nbsp;Dao</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://hao-pt.github.io/" target="_blank">Hao&nbsp;Phung</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://tbng.github.io/" target="_blank">Binh&nbsp;Nguyen</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://sites.google.com/site/anhttranusc/" target="_blank">Anh&nbsp;Tran</a>
  <br> <br>
  <a href="https://www.vinai.io/">VinAI Research</a>
  <!-- <br> <br>
  <a href="https://arxiv.org/abs/2211.16152">[Paper]</a> &emsp;&emsp;
  <a href="https://drive.google.com/file/d/1LSEYfdhS4Zjtx1VRrctmVt6xjEjgmpVA/view?usp=sharing">[Poster]</a> &emsp;&emsp;
  <a href="https://drive.google.com/file/d/11JE-RFtYJWx6XdXH8zZxgzRAvGJ6-IV2/view?usp=sharing">[Slides]</a> &emsp;&emsp;
  <a href="https://youtu.be/KaIMMamhKsU">[Video]</a> -->
</div>
<br>
<div align="center">
    <img width="1000" alt="teaser" src="assets/single_wavelet.png"/>
</div>

> Abstract: Flow matching is a recent framework to train generative models that exhibits impressive empirical performance while being relatively easier to train compared with diffusion-based models.
> Despite its advantageous properties, prior methods still face the challenges of expensive computing and a large number of function evaluations of off-the-shelf solvers in the pixel space. Furthermore, although latent-based generative methods have shown great success in recent years, this particular model type remains underexplored in this area. In this work, we propose to apply flow matching in the latent spaces of pretrained autoencoders, which offers improved computational efficiency and scalability for high-resolution image synthesis. This enables flow-matching training on constrained computational resources while maintaining their quality and flexibility. Through extensive experiments, our approach demonstrates its effectiveness in both quantitative and qualitative results on various datasets, such as CelebA-HQ, FFHQ, LSUN Church \& Bedroom, and ImageNet. We also provide a theoretical control of the Wasserstein-2 distance between the reconstructed latent flow distribution and true data distribution, showing it is upper-bounded by the latent flow matching objective.

Details of the model architecture and experimental results can be found in [our following paper](https://arxiv.org/abs/2211.16152):

```bibtex
@article{dao2023ldm,
    author    = {Dao, Quan and Phung, Hao and Tran, Anh},
    title     = {Flow Matching in Latent Space},
    journal   = {arXiv preprint},
    volume    = {arXiv:<id>},
    year      = {2023}
}
```

**Please CITE** our paper whenever this repository is used to help produce published results or incorporated into other software.

# Latent Flow Matching

## Installation

Python 3.8 and Pytorch 1.13.1 are used in this implementation.
Please install required libraries:

```
pip install -r requirements.txt
```

## Dataset preparation

### Image generation

For CelebA HQ 256, FFHQ 256 and LSUN, please check [NVAE's instructions](https://github.com/NVlabs/NVAE#set-up-file-paths-and-data) out.

For higher resolution datasets (CelebA HQ 512 & 1024), please refer to [WaveDiff's documents](https://github.com/VinAIResearch/WaveDiff.git).

For ImageNet dataset, please download it directly from [the official website](https://www.image-net.org/download.php).

### Downstream tasks

ToDo

## Training

### Image generation

All training scripts are wrapped in [run.sh](run.sh). Simply comment/uncomment the relevant commands and run `bash run.sh`.

### Downstream tasks

For downstream tasks as image inpaiting and semantic synthesis, we use the below commands.

**Image inpaiting**

```
python train_flow_latent_inpainting.py --exp inpainting_kl --dataset celeba_256 \
  --batch_size 64 --lr 5e-5 --scale_factor 0.18215 --num_epoch 500 --image_size 256 \
  --num_in_channels 9 --num_out_channels 4 --ch_mult 1 2 3 4 --attn_resolution 16 8 \
  --num_process_per_node 2 --save_content
```

**Semantic Synthesis**

```
python train_flow_latent_semantic_syn.py --exp semantic_kl --dataset celeba_256  \
--batch_size 64 --lr 5e-5 --scale_factor 0.18215 --num_epoch 175 --image_size 256 \
--num_in_channels 8 --num_out_channels 4 --ch_mult 1 2 3 4 --attn_resolution 16 8 \
--num_process_per_node 2 --save_content
```

## Testing

### Image generation

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

### Downstream tasks

For downstream tasks, we firstly run `test_flow_latent_semantic_syn.py` and `test_flow_latent_inpainting.py` to generate the synthesis data based on given conditions. After that, we can evaluate the metric using below commands.

**Image Inpainting**

```
python pytorch_fid/fid_score.py <path_to_generated_data> <path_to_gt_data>
```

**Semantic Synthesis**

```
python pytorch_fid/cal_inpainting.py <path_to_generated_data> <path_to_gt_data>
```

## Acknowledgments

Our codes are accumulated from different sources: [EDM](https://github.com/NVlabs/edm), [DiT](https://github.com/facebookresearch/DiT.git), [ADM](https://github.com/openai/guided-diffusion), [CD](https://github.com/openai/consistency_models.git), [Flow Matching in 100 LOC by François Rozet](https://gist.github.com/fd6a820e052157f8ac6e2aa39e16c1aa.git) and [WaveDiff](https://github.com/VinAIResearch/WaveDiff). We greatly appreciate these publicly available resources for research and development.

## Contacts

If you have any problems, please open an issue in this repository or ping an email to [v.quandm7@vinai.io](mailto:v.quandm7@vinai.io)/[tienhaophung@gmail.com](mailto:tienhaophung@gmail.com).
