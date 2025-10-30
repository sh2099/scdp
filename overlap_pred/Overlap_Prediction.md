### Overlap Integral Prediction using GNNs

#### Generating Data

- Molecular data comes from the scdp data
- To generate the data in the correct form

Example Usage:

```
python overlap_pred/gen_ovlp_data.py --full_dataset --use_vnodes --use_custom_gtos --add_diversity
```

A full list of the args can be found in the file but the key ones are as follows:
- ```--full_dataset``` runs over all molecules
- ```--use_vnodes``` generates the data using virtual nodes
- ```--use_custom_gtos``` constructs GTOs in a uniform way across all nodes and vnodes, using up to L=5
- ```--add_diversity``` log randomly generates the initial exponent value for each molecule, to create a continuous exponent range across the dataset

**Note:** The scdp data is accessed from the scratch folder of either plippman or mklockow depending on whether vnodes are used or not. The longevity of this data is not guaranteed. Please check with Peter or Manuel if you are having issues accessing the scdp data.

The Generated data can be inspected using: djkljaf



#### Data Analysis

A number of scripts exist to produce plots of the data to investigate the structure and impact of normalisation. The key scripts are listed below:
- ```analyse_overlaps.py``` this script creates plots of the unnormalised data, showing the overlap values against the exponent values, including a set of interactive plots. The code should be run as follows:
```
python scripts/analyse_exponent_overlap.py   --data-dir /export/data/hmichael/scdp/data/full_log_gen_new   --sample-size 20 --outdir plots/overlap_analysis --interactive-l 0  
```


#### The ML model

The model used is based directly from the GNN used in the scdp code.\\
Two major changes have been instigated:
- An additional input is added: exponent value - this value is embedded in an equivariant way.
- The model is trained to predict the overlap integrals instead of coefficients to reconstruct the charge density.




