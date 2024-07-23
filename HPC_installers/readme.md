# Viking (University of York)
## Installation
* Register for a login using [this form](https://docs.google.com/forms/d/e/1FAIpQLSfXkL10ypU6EQCBB2jS5oDwTpRMo77ppl7dvdbLnXm5zrKR7Q/viewform).
* Login to viking using ssh
    * Run```ssh <york-username>@ssh.york.ac.uk```
    * After any verification steps, enter ```viking``` when prompted to select a machine
* ```git clone``` this repository into an appropriate directory
*  After cloning navigate into the new directory and run ```source HPC_installers/Viking/viking_setup.sh```
## Copying your data into viking
### Setup
The helper functions for fitting the data will only work when the data is provided as **a text (.txt) file with three columns, in the order time-current-potential and no column titles or footers**. 
As different potentiostat manufacturers save their data in different formats, you will need to do this processing step yourself.
If ```time```, ```current``` and ```potential``` are three one-dimensional python lists of the same length, then you can save data using the following commands
```
save_array=numpy.column_stack((time, current, potential))
numpy.savetxt(viking_data.txt, save_array)
```
### Using scp/pscp to send data to viking
To copy this processed data into viking, you can use ```scp```. When logged into viking run:
```
scp scp.york.ac.uk:/shared/storage/home/userfs/a/abc123/path/to/file/filename .

```
where ```abc123``` is replaced by your username and ```a``` the first letter of your username. For for a directory run

```
scp -r scp.york.ac.uk:/shared/storage/home/userfs/a/abc123/path/to/directory/ .

```
These commands (after the required verification steps) will copy the appropriate file(s) into the viking directory which you run the command from. 

## Getting your results back from viking
Pooled results from the inference run are stored under ```Results/PooledResults_<today's-date>/ ```. This means that you will have to copy these results after every job submission to prevent them from being overwritten!
To copy back to your personal filestore, run
```
scp -r /path/to/directory scp.york.ac.uk:/shared/storage/home/userfs/a/abc123/
```

  
