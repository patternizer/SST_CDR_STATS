x#!/bin/sh

FIDHOME=/gws/nopw/j04/fiduceo/Users/mtaylor/sst
INMYVENV=${FIDHOME}/inmyvenv.sh
GENERATE=${FIDHOME}/join_sst.py
MEM_REQ=10000 # in MB
MEM_MAX=10000 # in MB 

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
                   LOGFILE=${LOGDIR}/run.log		    
 		    bsub -q short-serial -W24:00 -R "rusage[mem=$MEM_REQ]" -M $MEM_MAX -oo ${LOGFILE} -J "$JOBNAME" $INMYVENV $GENERATE "${SAT}" "${year}" "${month}" "${day}"
		fi

	    done
        done
    done
    
}

join AVHRR07_G 1981 1985
join AVHRR09_G 1985 1992
join AVHRR11_G 1988 1995
join AVHRR12_G 1991 1998
join AVHRR14_G 1995 2002
join AVHRR15_G 1998 2010
join AVHRR16_G 2000 2010
join AVHRR17_G 2002 2010
join AVHRR18_G 2005 2017
join AVHRR19_G 2009 2017
join AVHRRMTA_G 2006 2017 
join AATSR 2002 2012
join ATSR1 1991 1997
join ATSR2 1995 2008
wait










