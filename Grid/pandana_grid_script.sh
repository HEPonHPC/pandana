#!/bin/bash

# This script is designed to be run by submit_pandana.py, not by hand.

exec &> >(tee -a log.txt)

echo `whoami`@`hostname`:`pwd` at `date`

echo Args: $@

copy_log()
{
    infix=""
    if [ $NJOBS -gt 1 ]; then
    	infix="${SUFFIX}."
    fi

    target="$OUTDIR/${JOBSUBJOBID}.${infix}log.txt"
    echo Copy log to $target
    ifdh cp ../log.txt $target
}

# Make sure we always attempt to copy the logs
abort()
{
    copy_log
    exit 3
}

checkFile()
{
    if [ ! -f $2"/"$1 ]
    then
        echo File $1 somehow not copied to $2
        abort
    fi
}

while [[ $1 == *pandana_grid_script.sh ]]
do
    echo Braindead scripting passed us our own script name as first argument, discarding
    shift 1
done

function usage
{
    echo
    echo pandana_grid_script.sh -- designed to be run by submit_pandana.py, not by hand
    echo Usage: pandana_grid_script.sh [-h] -t PANDANA_TAR -r MACRO -o OUTDIR -n NJOBS [-f NFILES] [-m PYEXT] [-c COPY] [USERARGS...]
    echo
    echo "-h, --help          Print help and exit"
    echo "-t, --tarball       Name of PandAna tarball"
    echo "-r, --macro         User macro to run"
    echo "-o, --outdir        Send output to this directory"
    echo "-n, --njobs         This job is part of NJOBS total"
    echo "-f, --files_per_job Number of files to process per job"
    echo "-m, --module        External python modules sent by user"
    echo "-c, --copyOut       script to copy back relevant output files"
    echo
}

while [ $# -gt 0 ]
do
    echo "args: \$1 '$1' \$2 '$2'"
    case $1 in
        -h | --help )
            usage
            exit
            ;;
        -t | --tarball )
            PANDANA_TAR=$2
            shift 2
            ;;
        -r | --macro )
            MACRO=$2
            shift 2
            ;;
        -o | --outdir )
            OUTDIR=$2
            shift 2
            ;;
        -n | --njobs )
            NJOBS=$2
            shift 2
            ;;
        -f | --files_per_job )
            NFILES=$2
            shift 2
            ;;
        -m | --module )
            PYEXT=$2
            shift 2
            ;;
        -c | --copyOut )
            COPY=$2
            shift 2
            ;;
        *)
            # Unrecognized, must be user arguments from here to the end
            ARGS=$@
            break
            ;;
    esac
done

let COUNT=PROCESS+1
let FILENO=COUNT-1
SUFFIX=${COUNT}_of_${NJOBS}

BAD=0
[ x$MACRO == x ] && echo Must set MACRO parameter && BAD=1
[ x$OUTDIR == x ] && echo Must set OUTDIR parameter && BAD=1
[ x$PANDANA_TAR == x ] && echo Must set PANDANA_TAR parameter && BAD=1
[ x$NJOBS == x ] && echo Must set NJOBS parameter && BAD=1

if [ $BAD == 1 ]
    then usage
    exit 1 # can't abort yet, because ifdh isn't set up
fi


find_ups() {

    # use our slf6 stuff for systems with 3.x kernels (i.e. MTW2)
    case `uname -r` in
	3.*) export UPS_OVERRIDE="-H Linux64bit+2.6-2.12";;
	4.*) export UPS_OVERRIDE="-H Linux64bit+2.6-2.12";;
    esac
    
    for path in ${CVMFS_DISTRO_BASE}/externals /nusoft/app/externals
    do
	if [ -r $path/setup ]; then
	    source $path/setup
	    return 0
	fi
    done
    return 1
}

if [ x$_CONDOR_SCRATCH_DIR == x ]; then
  echo "Can't find scratch directory! Exiting"
  exit 1
fi

echo -n Will run $MACRO' '
echo -n as part of $NJOBS jobs' '
[ x"$ARGS" != x ] && echo -n with user arguments $ARGS
echo


find_ups
RELEASE="development"
echo Setting up $RELEASE
export CVMFS_DISTRO_BASE=/cvmfs/nova.opensciencegrid.org/
source $CVMFS_DISTRO_BASE/novasoft/slf6/novasoft/setup/setup_nova.sh -r $RELEASE -b maxopt || abort

echo Setting up PandAna
ls -la
export PYTHONPATH=`pwd`:$PYTHONPATH
echo "PYTHONPATH is now "$PYTHONPATH
export FW_RELEASE_BASE=`pwd`
echo 

BASEDIR=`pwd`
mkdir output

cd ${CONDOR_DIR_INPUT}
echo "Checking if macro was copied"
checkFile $MACRO ${CONDOR_DIR_INPUT}
echo "Yes!"
echo

if [ x$PYEXT != x ]; then
  echo "Checking if external modules were copied"
  checkFile $PYEXT ${CONDOR_DIR_INPUT}
  echo "Yes!"
  echo

  mkdir -p $BASEDIR"/external/lib/site-packages"
  cd $BASEDIR"/external/lib/site-packages"
  echo "Setting up external modules"
  tar zvxf $PYEXT
  export PYTHONPATH=`pwd`:$PYTHONPATH
  echo "PYTHONPATH is now "$PYTHONPATH
  cd ${CONDOR_DIR_INPUT}
fi

if [ x$COPY != x ]; then
  echo "Checking if copyOut script was copied"
  checkFile $COPY ${CONDOR_DIR_INPUT}
  echo "Yes!"
  echo
  mv $COPY $BASEDIR"/output/."
fi
cd $BASEDIR"/output"

MACRO=${CONDOR_DIR_INPUT}/$MACRO

export PANDANA_STRIDE=$NJOBS
export PANDANA_OFFSET=$FILENO
if [ x$NFILES != x ]; then
  export PANDANA_LIMIT=$NFILES
fi 

exit_flag=0
CMD="python $MACRO $ARGS"
echo $CMD
time $CMD || exit_flag=$?

ls -lh

if [ x$COPY != x ]; then
  echo "Running copy back script on output files"
  source $COPY || exit_flag=$?
else
  echo "Files in output directory to be copied back to ${OUTDIR} are:"
  ls
  echo

  # Is there a nicer way to check for an empty directory?
  if [[ `ls | wc -l` == 0 ]]
  then
      echo Nothing to copy back
      abort
  fi

  for k in *
  do
      if [ $NJOBS -gt 1 ]
      then
      # Insert the suffix after the last dot
          DEST=`echo $k | sed "s/\(.*\)\.\(.*\)/\1.${SUFFIX}.\2/"`
          if [ $DEST = $k ]
          then
          # $k had no dots in it?
              DEST=${k}.$SUFFIX
          fi
      else
          DEST=$k
      fi
      CMD="ifdh cp $k $OUTDIR/$DEST"
      echo $CMD
      $CMD || exit_flag=$?
  done
fi

echo Done!

echo Job $JOBSUBJOBID exited with code $exit_flag

copy_log

exit $exit_flag

echo also done copying log. Will now exit.
