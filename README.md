# SA-GNAS

## About

This repository is the implementation of paper *SA-GNAS: Seed Architecure Expansion for Efficient Large-scale Graph Neural Architecture Search*

## Requirements

```
./env_install.sh
```

## Architecture Search

For Coauthor CS and Physics datasets:

```
python author_search.py  
```

For Ogbn datasets (Arxiv, Porducts, Papers100M):
```
python arch_search.py  
```

## Architecture Evaluation

```
python model_author.py --data [Coauthor_Physics/Coauthor_CS] --arch cs_arch # Physics & CS
python model_arxiv.py --arch arxiv_arch # Ogbn-arxiv
python model_papers.py --arch papers_arch # Ogbn-papers
python model_products.py --arch products_arch # Ogbn-products
```

