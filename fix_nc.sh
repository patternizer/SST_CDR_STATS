#!/bin/sh

#FIDHOME=/group_workspaces/cems2/fiduceo/Users/mtaylor/sst
FIDHOME=/gws/nopw/j04/fiduceo/Users/mtaylor/sst
INMYVENV=${FIDHOME}/inmyvenv.sh
GENERATE=${FIDHOME}/count_sst.py
MEM_REQ=60000 # in MB     
MEM_MAX=60000 # in MB       

fix(){

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
#		    if grep -q "TERM_MEMLIMIT" ${LOGFILE}; then
#			echo "exceeded memory limit ${JOBNAME}"
#		    fi
		    if ! grep -q "Successfully completed" ${LOGFILE}; then
#			echo "submitting job ${JOBNAME}"
			bsub -q short-serial -W24:00 -R "rusage[mem=$MEM_REQ]" -M $MEM_MAX -oo ${LOGFILE} -J "$JOBNAME" $INMYVENV $GENERATE "${SAT}" "${year}" "${month}" "${day}"
		    fi
		fi
	    done
        done
    done
}

fix AVHRR06_G 1979 1982
fix AVHRR07_G 1981 1985
fix AVHRR08_G 1983 1985
fix AVHRR09_G 1985 1992
fix AVHRR10_G 1986 1991
fix AVHRR11_G 1988 1995
fix AVHRR12_G 1991 1998
fix AVHRR14_G 1995 2002
fix AVHRR15_G 1998 2010
fix AVHRR16_G 2000 2010
fix AVHRR17_G 2002 2010
fix AVHRR18_G 2005 2017
fix AVHRR19_G 2009 2017
fix AVHRRMTA_G 2006 2017 
fix AATSR 2002 2012
fix ATSR1 1991 1997
fix ATSR2 1995 2008
wait








