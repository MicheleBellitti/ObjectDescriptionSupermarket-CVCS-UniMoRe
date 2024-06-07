# Object Detection and Scene Description in a Supermarket

This is a course project for the postgraduate level course of [Computer Vision and Cognitive System](https://unimore.coursecatalogue.cineca.it/insegnamenti/2022/26380/2021/10003/10300?coorte=2022&schemaid=20024) taught at [DIEF, UniMoRe](https://inginf.unimore.it/laurea-magistrale-ing-inf/).

- Project Presentation: [https://docs.google.com/presentation/d/1oa5Y8bHkKgFodULnyd5vInzbxH9hIJZVeqW5iYaaxxA/edit?usp=sharing](https://docs.google.com/presentation/d/1oa5Y8bHkKgFodULnyd5vInzbxH9hIJZVeqW5iYaaxxA/edit?usp=sharing)
- Project Report: [https://drive.google.com/file/d/1YyxH2Q-KSBpQzCXXfHmBY9ajPjs3Un1R/view](https://drive.google.com/file/d/1YyxH2Q-KSBpQzCXXfHmBY9ajPjs3Un1R/view)
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

Retrieval was initially experimented using _Google Colab_: [https://colab.research.google.com/drive/1HXn3XRod3_6CHOes7aB0bJltz-IJagRP?usp=sharing](https://colab.research.google.com/drive/1HXn3XRod3_6CHOes7aB0bJltz-IJagRP?usp=sharing)

```bash
sbatch retrival.slurm
```
![retrival](https://github.com/MicheleBellitti/ObjectDescriptionSupermarket-CVCS-UniMoRe/assets/41593068/1a62bd58-f896-4e26-a233-a18886096a36)

_(Additional modifications can be made by editing the Python scripts mentioned in the corresponding slurm files.)_
