# Geographic entity alignment based a semi-supervised dual strategy iteration
This is the implementation for the paper "Geographic entity alignment based a semi-supervised dual strategy iteration".

## Data
> Please download the datasets [here](https://drive.google.com/file/d/1uJ2omzIs0NCtJsGQsyFCBHCXUhoK1mkO/view?usp=sharing) and extract them into 'data/' directory.
  * ent_ids_1: ids for entities in source KG;
  * ent_ids_2: ids for entities in target KG;
  * ref_ent_ids: entity links encoded by ids;
  * triples_1: relation triples encoded by ids in source KG;
  * triples_2: relation triples encoded by ids in target KG;
  * lang_vectorList: the word vector embeddings of all entities in source KG and target KG.

## Codes
This repository contains the following codes:
  * 'paris': Contains the code for calculating similarity.
  * 'classifier.py': Contains the classifier implementation code that classifies relationships within the entity embedding.
  * 'conf.py': Contains the code for experimental parameter configuration.
  * 'data.py': Contains the code for data preprocessing, involving reading data files, renumbering entities, and extracting related relationships and neighboring entities.
  * 'label_generate.py': Contains the code for identifying candidate seed entity pairs, including the implementation of dual-strategy iteration and optimal selection strategy.
  * 'loss.py': Contains the code for the implementation of the loss function.
  * 'model.py': Contains the code for the implementation of relation and entity embeddings, including two versions: one with the classifier and one without the classifier.
  * 'train.py': Contains the code for training the DSGEA model.
  * 'utils.py': Contains the code for other functionalities, including relation inversion, obtaining training batches, calculating hit rates, and computing intersections.


## Running

1、The comparative experiments of DSGEA on public dataset DBP15K and Self-constructed geographic dataset GeoLinkSet (corresponding to Table 4 and 5 in the paper).
Run the model on GeoLinkSet, use:
```
python train.py --data data/DBP15K --lang ga_en
```
Run the model on DBP15K(zh_en, ja_en and fr_en), use(take zh_en as an example):
```
python train.py --data data/DBP15K --lang zh_en --em_iteration_num 40
```
2、Ablation study (corresponding to Table 6 in the paper).
Run the DSGEA-1, use:
```
python train.py --data data/DBP15K --lang ga_en --classify false
```
Run the DSGEA-2, use:
```
python train.py --data data/DBP15K --lang ga_en ----DAA false
```
Run the DSGEA-3, use:
```
python train.py --data data/DBP15K --lang ga_en
```
3、Varying the training dataset (corresponding to Figure 4 in the paper).
Taking the language of zh_en with a training ratio of 0.2 as an example, use:
```
python train.py --data data/DBP15K --lang zh_en --rate 0.2  --em_iteration_num 40
```

## Requirements
```
apex==0.1
pytorch>=3.8
torch_geometric
tqdm
numpy
scikit-learn
```
The installation command for Apex is:
```
cd apex
python setup.py install --user
```

