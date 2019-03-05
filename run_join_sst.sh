#!/bin/sh

# PROGRAM: run_join_sst.sh
#
# Copyright (C) 2019 Michael Taylor, University of Reading                                 
# This code was developed for the EC project "Fidelity and Uncertainty in                 
# Climate Data Records from Earth Observations (FIDUCEO).                                 
# Grant Agreement: 638822                                                                 
#                                                                                          
# The code is distributed under terms and conditions of the MIT license: 
# https://opensource.org/licenses/MIT.

echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.AVHRR07.sh
echo './join_sst_AVHRR07.sh' >> run.AVHRR07.sh
echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.AVHRR09.sh
echo './join_sst_AVHRR09.sh' >> run.AVHRR09.sh
echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.AVHRR11.sh
echo './join_sst_AVHRR11.sh' >> run.AVHRR11.sh
echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.AVHRR12.sh
echo './join_sst_AVHRR12.sh' >> run.AVHRR12.sh
echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.AVHRR14.sh
echo './join_sst_AVHRR14.sh' >> run.AVHRR14.sh
echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.AVHRR15.sh
echo './join_sst_AVHRR15.sh' >> run.AVHRR15.sh
echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.AVHRR16.sh
echo './join_sst_AVHRR16.sh' >> run.AVHRR16.sh
echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.AVHRR17.sh
echo './join_sst_AVHRR17.sh' >> run.AVHRR17.sh
echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.AVHRR18.sh
echo './join_sst_AVHRR18.sh' >> run.AVHRR18.sh
echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.AVHRR19.sh
echo './join_sst_AVHRR19.sh' >> run.AVHRR19.sh
echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.AVHRRMTA.sh
echo './join_sst_AVHRRMTA.sh' >> run.AVHRRMTA.sh
echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.AATSR.sh
echo './join_sst_AATSR.sh' >> run.AATSR.sh
echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.ATSR1.sh
echo './join_sst_ATSR1.sh' >> run.ATSR1.sh
echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.ATSR2.sh
echo './join_sst_ATSR2.sh' >> run.ATSR2.sh

bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.AVHRR07.log < run.AVHRR07.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.AVHRR09.log < run.AVHRR09.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.AVHRR11.log < run.AVHRR11.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.AVHRR12.log < run.AVHRR12.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.AVHRR14.log < run.AVHRR14.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.AVHRR15.log < run.AVHRR15.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.AVHRR16.log < run.AVHRR16.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.AVHRR17.log < run.AVHRR17.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.AVHRR18.log < run.AVHRR18.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.AVHRR19.log < run.AVHRR19.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.AVHRRMTA.log < run.AVHRRMTA.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.AATSR.log < run.AATSR.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.ATSR1.log < run.ATSR1.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.ATSR2.log < run.ATSR2.sh


