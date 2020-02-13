import os
import subprocess
import sys
from datetime import datetime
from itertools import count
import time
import re
import hashlib

class SAMProjectSource():
  def __init__(self, projname, limit = -1):
    from ifdh import ifdh
    import samweb_client
    self.SAM = samweb_client.SAMWebClient(experiment='nova')
    self.ifdh = ifdh()
    
    self.limit = limit
    self.nfiles = limit
    self.projname = projname
    self.hasTicket = False
    self.projurl = ''
    self.processID = 0
    self.currfile = 0
    
    self.isgrid = os.getenv('_CONDOR_SCRATCH_DIR') is not None 
    
    self.setup()
    self.gen = self.getnextfile()

  def setup(self):
    self.checkproxy()
    self.establishProcess()

  def checkproxy(self):
    try:
      subprocess.check_output(["klist", "-l"])
      self.hasTicket = True
    except subprocess.CalledProcessError:
      try: 
        subprocess.check_call(["setup_fnal_security", "-k"])
        self.hasTicket = True
      except subprocess.CalledProcessError:
        self.hasTicket = False
        print((sys.exit(2), "Authentication failed. Please run setup_fnal_security -k"))
    if not os.getenv("X509_USER_PROXY"):
      uid = subprocess.check_output(["id", "-u"]).strip('\n')
      os.environ["X509_USER_PROXY"] = "/tmp/x509up_u"+uid

  def establishProcess(self):
    self.ifdh.set_debug('0')

    # find user-provided project and establish an ifdh process over it
    self.projurl = self.ifdh.findProject(self.projname, "nova")
    userstr = os.getenv('USER') or os.getenv('GRID_USER')
    self.processID = self.ifdh.establishProcess(self.projurl, "demo", "1",
                                                os.getenv('HOSTNAME'),
                                                userstr, "nova", "", self.limit)
    print(("Connecting to project %s with process %s" % (self.projurl, self.processID)))
    if self.nfiles < 0:
      self.nfiles = self.SAM.projectSummary(self.projname)['files_in_snapshot']

  
  def getnextfile(self):
    while True:
      if(self.currfile):
        # set status to consumed
        os.unlink(self.currfile)
        self.ifdh.updateFileStatus(self.projurl, self.processID, self.currfile, "consumed")
      uri = self.ifdh.getNextFile(self.projurl, self.processID)
      # end of project
      if not uri:
        self.ifdh.endProcess(self.projurl, self.processID)
        self.ifdh.cleanup()

        # If running interactively, just stop the project.
        # On the grid, there may be other processes running over this project. 
        # Which is why we're only stopping it once all the files in the project are used up
        if not self.isgrid:
          self.SAM.stopProject(self.projurl)
        else:
          summary = self.SAM.projectSummary(self.projname)
          nused = summary['file_counts']['consumed'] + summary['file_counts']['failed']
          ntot = summary['files_in_snapshot']
          if nused == ntot:
            self.SAM.stopProject(self.projurl)
        # Project stats needs some delay to show the correct status once it ends
        time.sleep(5)
        self.processStats(self.processID)
        break

      # fetch the file
      self.currfile = self.ifdh.fetchInput(uri)
      assert self.currfile

      self.ifdh.updateFileStatus(self.projurl, self.processID, self.currfile, "transferred")
      yield self.currfile

  def __call__(self):
    return next(self.gen)

  def nFiles(self):
    return self.nfiles

  def processStats(self, processID):
    stats = self.SAM.projectSummary(self.projname)
    process_stats = stats.pop('processes')
    print(process_stats)
    print(processID)
    process = [k for k in process_stats if k['process_id'] == int(processID)]
    if not process:
      print("Warning!! Something went wrong. unable to get process information.")
      print("\nProject summary :")
      print("===================")
      print((self.SAM.projectSummaryText(self.projname)))
      return
    stats.update({'process':process[0]})
    import yaml
    print("\nProcess summary :")
    print("===================")
    print((yaml.dump(stats)))
    return

# wrap sam queries around a project.
# The philosophy here is to start up a project for each SAM query and fetch each file to a temporary scratch location.
# This is better than allowing multiple users to open the same files directly from their location.
# CAFAna, in contrast, doesn't have a problem doing the above because the data is streamed via xrootd anyway, thus preventing concurrency issues.
# The pros of using projects are that we don't have to worry about taking snapshots ourselves and a project can be shared across all grid jobs for example
# The cons are that if we aren't careful the user can end up with many stale running projects and there's an experiment-wide cap
class SAMQuerySource(SAMProjectSource):
  _instanceid = count(0)
  # number of running projects allowed per user
  _MAX_INT_PROJECTS = 10    #interactive
  _MAX_GRID_PROJECTS = 100  #grid

  def __init__(self, query):
    import samweb_client
    self.SAM = samweb_client.SAMWebClient(experiment='nova')
    
    self.query = query
    # don't create separate projects if the query differs only by limit
    if 'with limit' in self.query:
      try:
        self.limit = int(re.sub(r'^.* with limit ([0-9]*)$', '\\1', self.query))
      except ValueError:
        print ("Invalid limit number in SAM query!")
        sys.exit(2)
    else:
      self.limit = -1
   
    # keep track of class instances in a given user macro
    self.instanceid = next(SAMQuerySource._instanceid)
    self.user = os.getenv('USER') or os.getenv('GRID_USER')
    self.checkDefinition = False
    self.checkSnapshot = False
    self.definition = None
    self.snapshot_id = None
    
    self.isgrid = os.getenv('_CONDOR_SCRATCH_DIR') is not None 
    
    self.setupProject()

  def isPlainDefinition(self):

    # check if query doesn't have extraneous metadata fields
    if not self.checkDefinition:
      # strip defname/def_snapshot/dataset_def_name_newest_snapshot off of query
      definition = re.sub(r'(dataset_def_name_newest_snapshot |defname: |def_snapshot | with limit [0-9]*$)', '', self.query)
      self.isdefinition = len(self.SAM.listDefinitions(defname=definition)) > 0
      self.checkDefinition = True
      if self.isdefinition:
        self.definition = definition
    return self.isdefinition
    
  def isPlainSnapshot(self):

    if not self.checkSnapshot: 
      # check if query asks for snapshot and get the snapshot id if so
      self.issnapshot = bool(re.search(r'(dataset_def_name_newest_snapshot |def_snapshot |snapshot_id)', self.query))
      if self.issnapshot:
        if self.isPlainDefinition():
          self.snapshot_id = self.SAM.snapshotInfo(defname=self.definition)['snapshot_id']
        else:
          try: 
            self.snapshot_id = int(re.sub('snapshot_id ([0-9]*) with limit [0-9]*$', '\\1', self.query))
          except ValueError:
            self.snapshot_id = None
            self.issnapshot = False
            pass

      self.checkSnapshot = True

    return self.issnapshot

  def setupProject(self):

    if not(self.isPlainDefinition() or self.isPlainSnapshot()):
      # Query is something else. Let's create our own definition. The hashing strips out bad characters 
      batchname = re.sub(r' with limit [0-9]*$', '', self.query)
      defname = "%s_pandana_defn_%s" % (self.user, hashlib.md5(batchname).hexdigest())

      if not self.SAM.listDefinitions(defname=defname):
        self.SAM.createDefinition(defname=defname, dims=batchname)
      self.definition = defname
    
    assert self.definition or self.snapshot_id
    # use snapshot unless its not there
    if self.snapshot_id:
      self.definition = None
    if self.definition:
      self.snapshot_id = None
   
    # need a unique project name but it needs to be shared across each grid job. 
    # Therefore, create one per query per Loader instance
    batchname = re.sub(r' with limit [0-9]*$', '', self.query)
    projname = "%s_pandana_proj%d_%s" % (self.user, self.instanceid, hashlib.md5(batchname).hexdigest())
    uniqueid = "time"+datetime.now().strftime('%Y%m%d_%H%M%S')
    checkprojid = "_time"
    MAX_PROJECTS = SAMQuerySource._MAX_INT_PROJECTS
   
    # CLUSTER is unique for every job. Therefore, create a new project for every new submission.
    if self.isgrid: 
      projname += "_cluster%s" % os.getenv('CLUSTER')
      checkprojid = "_cluster"
      MAX_PROJECTS = SAMQuerySource._MAX_GRID_PROJECTS

    projlist = self.SAM.listProjects(defname=self.definition, snapshot_id=self.snapshot_id, state='running')
    projexist = [projname in l for l in projlist]
    
    # SAM doesn't allow us to create projects with old names but put this check in anyway
    projserialexist = [projname+"_"+uniqueid in l for l in projlist]
    assert projserialexist.count(True) <= 1, "Multiple projects running with the same name. This shouldn't be possible"
 
    # don't allow too many stale projects
    projcheck = [projname+checkprojid in l for l in projlist]
    if projcheck.count(True) > MAX_PROJECTS:
      print((
      """
      More than %d projects are running for current query already.
      Most likely, this is because of leftover projects from faulty/interrupted runs.
      Use samweb list-projects --defname=%s --snapshot_id=%d --state=running | grep %s 
      and stop each one of them with samweb stop-project first and then re-run.
      You can also call pandana.utils.misc.StopAllUserProjects() in your script to 
      stop all your running interactive and grid projects 
      """ % (samquersource._MAX_PROJECTS, self.definition, self.snapshot_id, projname+checkprojid)
      ))
      sys.exit(2)

    # allow only one per cluster for a given Loader instance
    if self.isgrid and projexist.count(True):
      assert projexist.count(True) == 1, "Multiple projects of the same Loader instance running on the same cluster. This shouldn't be possible"
      # Some other grid job has already created our project. Connect to it
      projname = projlist[projexist.index(True)]
    else:
      # No one's created our project. We can go for it
      self.SAM.startProject(projname+"_"+uniqueid, defname=self.definition, snapshot_id=self.snapshot_id)
      projname = projname+"_"+uniqueid
 
    # Establish an IFDH process over it 
    SAMProjectSource.__init__(self, projname, self.limit)
    return
