## Notes
The fpconv code would be upgraded  later.

---
## Dependencies
- Ubuntu: 18.04 or higher
- PyTorch: 1.9.0 
- CUDA: 11.1 
- Hardware: 4GPUs (NVIDIA A40) 
- To create conda environment, command as follows:

  ```
  sh env_setup.sh pt
  ```

## Dataset preparation
- Download S3DIS [dataset](https://drive.google.com/uc?export=download&id=1KUxWagmEWnvMhEb4FRwq2Mj0aa3U3xUf) and symlink the paths to them as follows:

     ```
     mkdir -p dataset
     ln -s /path_to_s3dis_dataset dataset/s3dis
     ```

## Usage
- Semantic segmantation on S3DIS Area 5
  - Train

    - Specify the gpu used in config and then do training:

      ```
      sh tool/train.sh s3dis fptransformer_repro
      ```

  - Test

    - Afer training, you can test the checkpoint as follows:

      ```
      CUDA_VISIBLE_DEVICES=0 sh tool/test.sh s3dis fptransformer_repro
      ```
  ---

```

## Acknowledgement
The code is based on [Point Transformer](https://arxiv.org/abs/2012.09164).

