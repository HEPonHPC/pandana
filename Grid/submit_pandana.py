#!/usr/bin/env python

import argparse
import os
import sys
import re
from datetime import datetime
import subprocess

# Various utility functions
import NovaGridUtils
from NovaGridUtils import warn, fail

def test_not_dcache(l, warnOnly = False):
    loc = os.path.expandvars(l)
    for bad in ['/nova/app/', '/nova/ana/', '/nova/data/', '/grid']:
        if loc.startswith(bad):
            txt = "Location %s cannot be on %s it must be in dCache /pnfs/nova/" % (loc, bad)
            if loc.startswith('/nova/app'):
                txt = "Jobs can no longer access BlueArc directly. Test releases will be tarred up and sent to worker nodes, however input files should be moved to dCache."
            if warnOnly:
                warn(txt)
            else:
                fail(txt)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'Submit a pandana macro to run on the grid')

    parser.add_argument('-f', '--files_per_job', type=int, default=None,
                        help='Number of files to process in each worker node')

    parser.add_argument('-n', '--njobs', type=int, required=True, metavar='N', 
                        help='Number of grid processes')

    parser.add_argument('-i', '--input_file', action="append", 
                        help="Copy this input file to work area on worker node")

    parser.add_argument('-o', '--outdir', required=True, metavar='DIR', 
                        help='Directory output files will go to')

    parser.add_argument('-m', '--module', default=None,
                        help='Path to tar file containing external module libraries which the macro may depend on')
    
    parser.add_argument('macro.py',
                        help='The pandana macro to run')

    parser.add_argument('args', nargs='*',
                        help='Arguments to the macro')

    parser.add_argument('-c', '--copyOut',
                        help='User-provided script to copy back output files from macro. Default behaviour is to copy back all hdf5 files generated by the macro')
    
    job_control_args = parser.add_argument_group("Job control options", "These optional arguments help control where and how your jobs land.") 
    ###tarball control options.
    tarball_gp = job_control_args.add_mutually_exclusive_group(required=False)
    tarball_gp.add_argument('-t', '--testrel', metavar='DIR', 
                            help='Folder which contains pandana. It will be tarred up, and sent to the worker node. (Conflicts with --user_tarball)',
                            default=None)
    
    tarball_gp.add_argument("--user_tarball",
                            help="Use existing tarball of folder containing pandana in specified location rather than having jobsub make one for you (conflicts with --testrel, and is redunant with --reuse_tarball)",
                            type=str)

    ###general job control
    job_control_args.add_argument('--reuse_tarball',
                            help='Do you want to reuse a tarball that is already in resilient space? If using this option avoid trailing slash in --testrel option.  (redundant with --user_tarball)',
                            action='store_true',default=False)
    
    job_control_args.add_argument('--dedicated',
                        help='Only run on dedicated nodes on fermigrid (default is to run opportunistically)',
                        action='store_true',default=False)
    
    job_control_args.add_argument('--disk',
                        help='Local disk space requirement for worker node in MB (default is 2000MB).',
                        type=int, default=2000)

    job_control_args.add_argument('--memory',
                        help='Local memory requirement for worker node in MB (default is 1900MB).',
                        type=int, default=3000)

    job_control_args.add_argument('--lifetime',
                        help='Expected job lifetime. Valid values are an integer number of seconds. (default is 10800=3h)',
                        type=int, default="10800")
    
    debugging_args = parser.add_argument_group("Debugging options", "These optional arguments can help debug your submission.")
    
    debugging_args.add_argument('--print_jobsub',
                        help='Print jobsub command and exit',
                        action='store_true',default=False)

    debugging_args.add_argument('--test',
                        help='Run test job over 1 file',
                        action='store_true',default=False)
    
    opts = parser.parse_args()

    macro = os.path.abspath(vars(opts)['macro.py'])
    
    input_files = []
    # Some sanity checks
    NovaGridUtils.get_credentials('Analysis')
    NovaGridUtils.check_file(macro)
    test_not_dcache(opts.outdir)
    NovaGridUtils.check_dir(opts.outdir)
    NovaGridUtils.check_is_group_writable(opts.outdir)

    if not (opts.testrel or opts.user_tarball or opts.reuse_tarball):
        print ("Please provide test release option or path to tarball of folder containing pandana!")
        sys.exit(2)
    pandana_tar = None
    if opts.testrel:
        NovaGridUtils.check_dir(opts.testrel+'/pandana')
    if opts.user_tarball:
        pandana_tar = opts.user_tarball
    if opts.reuse_tarball:
        pandana_tar = os.path.basename(opts.testrel)+'.tar'
    if pandana_tar:
        NovaGridUtils.check_file(pandana_tar)

    if opts.input_file:
        for inFile in opts.input_file:
            NovaGridUtils.check_file(inFile)
        input_files += opts.input_file
    if opts.module:
      if not opts.module.endswith('.tar'):
        print ("External module file needs to end with .tar")
        sys.exit(2)
      NovaGridUtils.check_file(opts.module)
    if opts.copyOut:
      NovaGridUtils.check_file(opts.copyOut)
    
    grid_script = os.getenv('FW_RELEASE_BASE')+'/pandana/Grid/pandana_grid_script.sh'
    NovaGridUtils.check_file(grid_script)
    
    # jobsub control
    jobsub_opts = ""
    
    usage_models=["DEDICATED"]
    if not opts.dedicated:
    	usage_models.append("OPPORTUNISTIC")
    
    jobsub_opts += "    --resource-provides=usage_model=%s \\\n" % (",".join(usage_models))
    #disk
    if opts.disk:
        disk_opt="    --disk=%sMB \\\n" % (opts.disk)
        jobsub_opts += disk_opt        
    #memory
    if opts.memory:
        mem_opt="    --memory=%sMB \\\n" % (opts.memory)
        jobsub_opts += mem_opt       
    jobsub_opts += "    --mail_never \\\n"
    # Never kill me for being over time. Find a node with enough time left in
    # its glidein for what I requested, but don't penalize me for going over
    # time.
    life_opt=("    --expected-lifetime=0s \\\n    --append_condor_requirements='(((TARGET.GLIDEIN_ToDie-CurrentTime)>%s)||isUndefined(TARGET.GLIDEIN_ToDie))' \\\n" % opts.lifetime)
    jobsub_opts += life_opt
    
    #Process the debugging_args
    test=opts.test
    print_jobsub=opts.print_jobsub
    if test :
        print_jobsub=True
        opts.files_per_job = 1
        opts.njobs = 1
        out = "/pnfs/nova/scratch/users/"+os.getenv('USER')+"/"
        testfolder = "test_"+datetime.now().strftime('%Y%m%d_%H%M%S')
        opts.outdir = out + testfolder
        subprocess.call(['mkdir', '-p', opts.outdir])
        subprocess.call(['chmod', 'oug=wrx', opts.outdir])
        print(("Running test job on 1 file, output directory is %s" % opts.outdir))
    
    cmd = 'jobsub_submit '
    
    cmd += '-N '+str(opts.njobs)+' '
    cmd += "\\\n"+jobsub_opts
    cmd += '    -f dropbox://%s \\\n' % (macro)
    for f in input_files:
        cmd += '    -f dropbox://%s \\\n' % (f)
    if opts.module:
        cmd += '    -f dropbox://%s \\\n' % opts.module
    if opts.copyOut:
        cmd += '    -f dropbox://%s \\\n' % opts.copyOut

    
    # Test release options.
    tarname = None
    if pandana_tar:
      cmd += '    --tar_file_name dropbox://'+pandana_tar
      tarname = pandana_tar
    elif opts.testrel:
      cmd += '    --tar_file_name tardir://'+opts.testrel+'\\\n'
      tarname = os.path.basename(opts.testrel)+'.tar'
    
    # grid script and arguments
    cmd += '    file://%s \\\n' % (grid_script)
    cmd += '      --macro %s \\\n' % (os.path.basename(macro))
    cmd += '      --outdir %s \\\n' % (opts.outdir)
    cmd += '      --njobs %s \\\n' % (str(opts.njobs))
    cmd += '      --tarball %s \\\n' % (tarname)
    # optional arguments to grid script
    if opts.files_per_job:
      cmd += '      --files_per_job %s \\\n' % (str(opts.files_per_job))
    if opts.module:
      cmd += '      --module %s \\\n' % (os.path.basename(opts.module))
    if opts.copyOut:
      cmd += '      --copyOut %s \\\n' % (os.path.basename(opts.copyOut))
    
    # arguments to macro
    for a in opts.args: cmd += ' '+a
    
    if print_jobsub:
        print(cmd)
        sys.stdout.flush()
        sys.stderr.flush()
   
    if not opts.print_jobsub:
        os.system(cmd)
