### Overlap Integral Prediction using GNNs

#### File Structure


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
- ```analyse_overlaps.py``` this script creates plots of the data, showing the overlap values against the exponent values, including a set of interactive plots. The code should be run as follows:
  ```
  python scripts/analyse_exponent_overlap.py   --data-dir /export/data/hmichael/scdp/data/full_log_gen_new   --sample-size 20 --outdir plots/overlap_analysis --interactive-l 0  
  ```
  The key args to be set are:
  - ```--sample-size``` this determines how many molecules to use for the plots - the molecules are selected randomly from the full dataset, and points are plotted in a random order. - Note that when generating the interactive plots the recommended number of molecules to use is <100. Using more molecules can result in html files which take too long to render.
  - ```--curve-norm-dir``` and ```--curve-norm-plots``` These are used to enact the normalisation procedure on the data for plotting. The path to the saved curves can be input to the first arg, and the second triggers the use of those plots to normalise the data.
 
- ```apply_overlap_normalisation.py``` this script enacts the normalisation of the overlap integrals using binned regions. The mean and standard deviations are computed for each bin, and then smooth curves are fitted over each of these. There are separate curves per L value for equivariance. These curves are then used as part of the transforms for the data during training.
  ```
  python scripts/apply_overlap_normalisation --blah blah
  ```
  The key args to be set are:
  - ```--sample-size```
  - ```--bins```




#### The ML model

The model used is based directly from the GNN used in the scdp code.\\
Two major changes have been instigated:
- An additional input is added: exponent value - this value is embedded in an equivariant way.
- The model is trained to predict the overlap integrals instead of coefficients to reconstruct the charge density.




