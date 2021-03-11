#!/bin/bash
# Author: wanghan
# Created Time : Tue 17 Jul 2018 07:35:49 PM CST
# File Name: init.sh
# Description:

function loop_exe()
{
    local ex_count=0
    CMDLINE=$1
    while true ; do
        #command
        sleep 1
        echo The command is \"$CMDLINE\"
        ${CMDLINE}
        if [ $? == 0 ] ; then
            echo The command execute OK!
            break;
        else
            (( ex_count = ${ex_count} + 1 ))
            echo ERROR : The command execute fialed! ex_count = ${ex_count}.
        fi
    done
}


function main()
{
    echo --- Start ---
    
    topic=10
    for ((i=1; i<=14; i++))
    do  
	    alpha=0.1
	    for ((j=1; j<=15; j++))
	    do  

	    loop_exe "./slda est ./sample-data/train-data.dat ./sample-data/train-data-feature.dat ./sample-data/train-label.dat settings.txt ${alpha}  ${topic}  seeded ./model${i}${j}"
	    chmod -R 777  ./model${i}${j}
	    alpha=`echo "0.1+$alpha"|bc`
	    done
    
	    topic=`echo "5+$topic"|bc`
    done


    for ((i=1; i<=14; i++))
    do  
	    for ((j=1; j<=15; j++))
   do  

	    loop_exe "./slda inf ./sample-data/test-data.dat  ./sample-data/test-data-feature.dat  ./sample-data/test-label.dat settings.txt ./model${i}${j}/final.model ./test_out${i}${j}"
            chmod -R 777 ./test_out${i}${j}
	    done
    done



    echo --- Done ---
}

main
