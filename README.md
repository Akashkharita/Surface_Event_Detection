- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)


# Surface Event Detection

This repository contains notebooks that show how to use my trained ML model to detect surface events (Avalanches/Rockfalls/Debris Flows) through continuous seismograms from multiple stations. 
The model was trained on over 200k seismic events in the Pacific northwest. [Ni et al. 2023](https://seismica.library.mcgill.ca/article/view/368/868)


## Installation

Instructions on how to install...

First, let's setup a conda environment using the following command. 

```
conda create -n surface python=3.9.5
```

Activate the environment

```

conda activate surface
```

Then we will install the required dependencies 
```
pip install -r requirements.txt
```

Then we will add the conda environment to jupyter hub 
```
conda install ipykernel
```
```
python -m ipykernel install --user --name=surface
```


Now we are all set to go! 😃

## Usage
The folder [Common_Scripts](Common_Scripts) contains following - 
- [cfg_file_reduced.json](Common_Scripts/cfg_file_reduced.json) - this file contains the dictionary of tsfel features that were found to be among the top 50 features for seismic event classification in the pacific northwest. We need to specify these feature dictionary to direct tsfel to extract only these features.
- [common_processing_functions.py](Common_Scripts/common_processing_functions.py) - This file contains the commonly used processing functions that will be used to process the waveforms before feature extractions and some visualization functions 

The [notebook](Notebooks/Testing_Surface_Event_Detection_on_verified_events.ipynb) shows an example of how to detect surface events through continuous seismograms and visualize the results with detailed documentation. 


## Contributing

Guidelines for contributing...

## License

Information about the project's license...
