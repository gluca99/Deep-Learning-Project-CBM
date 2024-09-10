## CLIP-Dissect

Automatic neuron label generation levaraging multimodal CLIP-model.

## Quickstart:

This will dissect 5 layers of ResNet-50 using Broden as the probing dataset. Results will be saved in 'results/{target_name}_all.csv'.

```
bash dlbroden.sh
python describe_neurons.py --d_probe broden
```

## Useful downloads

Download Broden dataset(images only) for probing: 'bash dlbroden.sh'
Download ResNet-18 pretrained on Places-365: 'bash dlzoo_example.sh'

We do not provide download instructions for ImageNet data, to evaluate using your own copy of ImageNet validation set you must set 
the correct path in `DATASET_ROOTS["imagenet_val"]` variable in `utils.py`. 

## Sources:

- CLIP: https://github.com/openai/CLIP
- Text datasets(10k and 20k): https://github.com/first20hours/google-10000-english
- Text dataset(3k): https://www.ef.edu/english-resources/english-vocabulary/top-3000-words/
- Broden download script based on: https://github.com/CSAILVision/NetDissect-Lite

## Recreating experiments

To recreate the results of our paper, run the notebooks in the experiments directory.

The qualitative figures 1, 7, and 8 can be recreated from scratch by running `experiments/fig_1_7_8.ipynb`. Which figure is plotted can be controlled by changing the `target_model`, `target_layer` and `neurons_to_display` variables.

`experiments/table1_fig3.ipynb` recreates the cosine similarities of table1. Different rows for our model can be evaluated by changing the `concept_set` and `d_probe_variables`. In addition it recreates the example in Figure3 and the calculations needed for it.

Table 2 can be recreated in 'experiments\table2.ipynb'.

Table 3 can be recreated by running `experiments/table3.ipynb`. Different elements of the table can be run by varying the `d_probe` and `similarity_fn` parameters. Also reports some additional metrics not discussed in paper.

Runtime of table 4 can be recreated by running `python describe_neurons.py`. The time was measured with cached CLIP model activations but without cached target model activations and can vary based on what you have cached (the code automatically caches everything into 'saved_activations' to avoid rerunning). This was chosen as CLIP activations can be reused when analyzing different networks. This saves descriptions for all neurons in target layers into 'results/{target_name}_all.csv'.

Figure 4 results will be reproduced by running qualitative.ipynb with default parameters. To explore hidden layer explanations change `target_layer='layer4'` for example, and use a different `d_probe` for different results, i.e. `d_probe=imagenet_broden`.

Figure 5 (compositional explanations) results recreated by running `experiments/fig5.ipynb`.

Figure 6 and other analysis regarding the connection between weight and concept similarity can be reproduced in `experiments/fig6.ipynb`.
