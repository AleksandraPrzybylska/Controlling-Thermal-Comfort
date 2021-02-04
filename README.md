# Controlling-Thermal-Comfort

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)

## General info
The data was provided by the LERTA company (https://www.lerta.energy/).
It was collected as part of a project aimed at improving methods of controlling thermal comfort in buildings.
The data refers to an office building located in Poznan and it contains:
- room temperature (for several temperature sensors located in different places) [°C],
- the degree of opening of the radiator valve [%],
- set temperature [°C],
- the size of the room,
- the number of people in room,
- the window orientation and placement.

#The aim of the project is to make two predictions:
- average temperature value of the indicated sensor for 8 hours (from 6:00 a.m. to 2:00 p.m.),
- the value of the average valve opening degree in 8 hours (from 6:00 - 14:00)


## Technologies
Project is created with:
* Python version: 3.8.6
* Pandas version: 1.1.5
	
## Setup
All the necessary libraries are in the requirements file. You can install with a command:
```
$ pip install -r requirements.txt
```

To run this project, you need to pass two arguments:
 - path to input json file,
 - path to output result csv file

```
$ python main.py input_file results_file_csv
```
