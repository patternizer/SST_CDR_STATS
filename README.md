![ESA-SSI-CCI](http://cci.esa.int/sites/default/files/esa_cci_sst_logo_0.GIF)

# ESA SST-CCI CDR2.1 statistics

Development code for calculation of statistics of Level-2 SST CDR data and uncertainties.

To get started with SST_CDR_STATS, the paths to the Level-2 daily orbit files from AVHRR and ATSR need setting as well as the path to the output directory. Running join_sst.sh will generate the daily netCDF summaries. Runtime failures can be fixed by running fix_nc.sh. 

After installation, command-line utilities include:

    plot_sst.py

which has the option to include a plot of the landsea_mask and ocean fraction:

    plot_landsea_mask.py

## Contact information

* Michael Taylor (michael.taylor@reading.ac.uk)

