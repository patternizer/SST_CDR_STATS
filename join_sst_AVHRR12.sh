#!/bin/sh

FIDHOME=/gws/nopw/j04/fiduceo/Users/mtaylor/sst
INMYVENV=${FIDHOME}/inmyvenv.sh
GENERATE=${FIDHOME}/join_sst.py
MEM_REQ=60000 # in MB
MEM_MAX=60000 # in MB 

join(){

    SAT=$1     
    START=$2 
    END=$3     

    for year in $(seq $START $END)  
    do
        for month in $(seq -w 01 12)      
	do
	    for day in $(seq -w 01 31) 
	    do
                JOBNAME="${SAT}${year}${month}${day}"
		LOGDIR="$FIDHOME/${SAT}/${year}/${month}/${day}"	       
		if [ -d "$LOGDIR" ]; then
		    python join_sst.py ${SAT} ${year} ${month} ${day} 
		    echo ${JOBNAME}
#                   LOGFILE=${LOGDIR}/run.log		    		   
# 		   bsub -q short-serial -W24:00 -R "rusage[mem=$MEM_REQ]" -M $MEM_MAX -oo ${LOGFILE} -J "$JOBNAME" $INMYVENV $GENERATE "${SAT}" "${year}" "${month}" "${day}"
		fi
	    done
        done
    done    
}

join AVHRR12_G 1991 1998
wait










