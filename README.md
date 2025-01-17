# Cool-Title
This repo is an extension of an existing model [Original Repo link](https://github.com/ChenxiLiu-HNU/ST-LLM).

**Modifications** 
* Added Dask DDP
* Added an early-version of index-batching
* Added the ability to train with METR-LA
* Added the ability to train with PEMS-BAY
* Added a preprocessing script to get h5 data in the needed format

If the index-batching or DDP are useful, please consider TODO

**Notes**
Below is a collection of helpful information I figured out while working with the code base. Hopefully this is useful for anyone who wants to modify the original model further. 

* To recreate their data from the original h5 files, you need to select either drop or pickup data from the h5 file keys.
* ST-LLM requires the preprocessed data to have time-of-day and day-of-week data added. In the case of the provided datasets, they defined the first item as occurring at time 0, and then increased in 30 minutes increments, using those time-stamps as the basis for time-of-day and day-of-week.
* ST-LLM assumes day data and week day are in the [...,1] and [...,2] dimensions, respectively,  of the input data  
* To use a new dataset, you need to preprocess the data in a slight different way than prior works. I provided a script called `generate_preprocessed_npz.py` which will preprocess traffic data. 
* To generate the time-embedding  in `model_ST_LLM.py`, they manually set the `time` variable based on what dataset you passed in (and it is only coded to support the taxi and bike dataset described in the paper). To calculate what you should set it to: (24 * 60) / (data_time_interval) 
* ST-LLM manually sets the number of graph nodes in the data 

**TODO**
* Allow users to pass in `data_time_interval` and `num_nodes` as command line parameters 


