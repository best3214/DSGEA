# GSGEA
Source code and datasets for paper: [***Geographic entity alignment based a semi-supervised dual strategy iteration***]

## Datasets
> Please download the datasets [here](https://drive.google.com/file/d/1uJ2omzIs0NCtJsGQsyFCBHCXUhoK1mkO/view?usp=sharing) and extract them into 'data/' directory.
  * ent_ids_1: ids for entities in source KG;
  * ent_ids_2: ids for entities in target KG;
  * ref_ent_ids: entity links encoded by ids;
  * triples_1: relation triples encoded by ids in source KG;
  * triples_2: relation triples encoded by ids in target KG;
  * lang_vectorList: the word vector embeddings of all entities in source KG and target KG.

## Environment

```
apex==0.1
pytorch>=3.8
torch_geometric
tqdm
numpy
scikit-learn 
```

## Running

Run the model on GA_EN(GeoLinkSet), use:
```
python train.py --data data/DBP15K --lang ga_en --intersect 1 --repeat1 6 --repeat2 4
```
Run the model on DBP15K(zh_en, ja_en and fr_en), use(take zh_en as an example):
```
python train.py --data data/DBP15K --lang zh_en 
```


