# Label-free Concept Bottleneck Models

## Creating Concept Sets (Optional):
1. Create initial concept set using GPT-3 - `GPT_initial_concepts.ipynb`, do this for all 3 prompt types (can be skipped if using the concept sets we have provided).

2. Process and filter the conceptset by running `GPT_conceptset_processor.ipynb` (Alternatively get ConceptNet concepts running ConceptNet_conceptset.ipynb)

## Train LF-CBM
3. Train a concept bottleneck model by running 
`python train_cbm.py --concept_set data/concept_sets/cifar10_filtered.txt`

 By default this trains a model on CIFAR10. To reproduce our results on other datasets use the following commands:

CIFAR100: 
`python train_cbm.py --dataset cifar100 --concept_set data/concept_sets/cifar100_filtered.txt`

CUB200: 
`python train_cbm.py --dataset cub --backbone resnet18_cub --concept_set data/concept_sets/cub_filtered.txt --feature_layer features.final_pool --clip_cutoff 0.26 --n_iters 5000 --lam 0.0002`

Places365:
`python train_cbm.py --dataset cub --backbone resnet50 --concept_set data/concept_sets/places365_filtered.txt --clip_cutoff 0.28 --n_iters 80 --lam 0.0003`

ImageNet:
`python train_cbm.py --dataset imagenet --backbone resnet50 --concept_set data/concept_sets/imagenet_filtered.txt --clip_cutoff 0.28 --n_iters 80 --lam 0.0001`

## Evaluate trained models

4. Evaluate the trained models by running `evaluate_cbm.ipynb`. This measures model accuracy, creates barplots explaining individual decisions and prints final layer weights which are the basis for creating weight visualizations.
