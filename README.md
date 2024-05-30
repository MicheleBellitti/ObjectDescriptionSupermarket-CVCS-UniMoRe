# Object Detection and Scene Description in a Supermarket

This course project for [Computer Vision and Cognitive System](https://offertaformativa.unimore.it/corso/insegnamento?cds_cod=20-262&aa_ord_id=2021&pds_cod=20-262-6&aa_off_id=2022&lang=eng&ad_cod=IIM-65&aa_corso=1&fac_id=10005&coorte=2022&anno_corrente=2022&durata=2) taught at [DIEF, UniMoRe](https://inginf.unimore.it/laurea-magistrale-ing-inf/).
- Course Project Report: [https://www.overleaf.com/read/mkmstpgmgphq#9e596f](https://www.overleaf.com/read/mkmstpgmgphq#9e596f)
- Model Weights: [https://drive.google.com/drive/folders/1QY-fAs8u-BdupzJeAMYeGz4cKXeUWnnR?usp=drive_link](https://drive.google.com/drive/folders/1QY-fAs8u-BdupzJeAMYeGz4cKXeUWnnR?usp=drive_link)

## Datasets
- For the Object Detection task, we use the [SKU110K dataset](https://github.com/eg4000/SKU110K_CVPR19?tab=readme-ov-file#dataset).
- For the Product Classification and Embeddings for the Product Retrieval task, we use the [GroceryStoreDataset](https://github.com/marcusklasson/GroceryStoreDataset).

## Training and Experimentations

### For training the Faster RCNN model for Object detection:
```bash
sbatch frcnn.slurm
```
### For training the DenseNet 121 model for Product Classification and Embeddings for the Product Retrieval:
```bash
sbatch clf.slurm
```

## Implementation and Inference

### Object Detection and Scene Description
- For the implementation of the complete pipeline:
    - Classical Scene Image Preprocessing (Histogram Equalization)
    - Inference of both models: Faster RCNN  _and DenseNet 121 (commented out)_
    - Shelf numbering: K Means with Silhouette Analysis
    - _Dominant colour recognition (commented out)_
    - Zero-Shot Product Detection using CLIP (Contrastive Language-Image Pre-training) model
    - Spatial Description through geometrical templating
    - Concise Scene Description using ChatGPT 3.5 Turbo through OpenAI API
        - Setup [OpenAI API account](https://platform.openai.com/docs/quickstart)

```bash
export OPENAI_API_KEY=entergeneratedAPIKey

sbatch inference.slurm
```
![pipeline](https://github.com/MicheleBellitti/ObjectDescriptionSupermarket-CVCS-UniMoRe/assets/41593068/3a1fb28d-157c-4d4b-821c-5a86a01682ae)

### Retrieval Mechanism
```bash
sbatch retrival.slurm
```
![retrival](https://github.com/MicheleBellitti/ObjectDescriptionSupermarket-CVCS-UniMoRe/assets/41593068/1a62bd58-f896-4e26-a233-a18886096a36)

_(Additional modifications can be made by editing the Python scripts mentioned in the corresponding slurm files.)_
