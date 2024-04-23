- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)


# My Project

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
The folder [Common_Scripts](Common_Scripts)


The [notebook](Notebooks/Testing_Surface_Event_Detection_on_verified_events.ipynb) shows an example of how to detect surface events through continuous seismograms and visualize the results with detailed documentation. 


## Contributing

Guidelines for contributing...

## License

Information about the project's license...
