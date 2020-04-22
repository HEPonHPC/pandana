import os
import subprocess
import sys

# Contains classes that return a generator over a file source 

# simple wrapper around list of files provided by the user
class ListSource():
  def __init__(self, filelist):
    self.files = filelist
    self.gen = self.getnextfile()
 
  def getnextfile(self):
    for f in self.files:
      assert os.path.isfile(f), "File {} doesn't exist!".format(f)
      yield f

  def __call__(self):
    return next(self.gen)

  def nFiles(self):
    return len(self.files)

# pass a glob instead
class GlobSource():
  def __init__(self, globstr, stride=1, offset=0, limit=None):
    import glob
    self.files = glob.glob(globstr)[offset::stride]
    if limit: self.files = self.files[:limit]
    self.gen = self.getnextfile()

  def getnextfile(self):
    for f in self.files:
      assert os.path.isfile(f), "File {} doesn't exist!".format(f)
      yield f

  def __call__(self):
    return next(self.gen)

  def nFiles(self):
    return len(self.files)

class SourceWrapper():
  def __init__(self, query, stride=1, offset = 0, limit = None):
    self.query = query
    self.stride = stride
    self.offset = offset
    self.limit = limit

  def sourceArgsID(self):
    args_id = "stride%d_offset%d" % (self.stride, self.offset)
    if self.limit:
      args_id = "stride%d_offset%d_limit%d" % (self.stride, self.offset, self.limit)
    return args_id
  
  def getQuery(self):
    return self.query

  # define an id based on query and stride arguments to check if multiple loaders use the same source
  def __eq__(self, other_source):
    return (self.getQuery() == other_source.getQuery() and \
            self.sourceArgsID() == other_source.sourceArgsID())
  
  def islist(self):
    return type(self.query) is list

  def isglob(self):
    return type(self.query) is str and ' ' not in self.query

  def isproj(self):
    from ifdh import ifdh
    i = ifdh()
    url = i.findProject(self.query, "nova")
    exitcode = 1
    with open(os.devnull, 'wb') as null:
      try:
        # for some reason I'm not able to use ifdh projectStatus directly with subprocess.
        exitcode = subprocess.check_call(['bash', '-c', 'ifdh projectStatus {}'.format(url)], stdout=null, stderr=null)
      except subprocess.CalledProcessError:
        exitcode = 1
    return not exitcode

  def issamquery(self):
    import samweb_client
    SAM = samweb_client.SAMWebClient(experiment='nova')

    try: 
      SAM.listFilesSummary(dimensions=self.query)
      return True
    except samweb_client.exceptions.DimensionError:
      return False
    return False

  def __call__(self):
    if self.islist():
      if os.getenv('PANDANA_STRIDE'):
        self.stride = int(os.getenv('PANDANA_STRIDE'))
      if os.getenv('PANDANA_LIMIT'):
        self.limit = int(os.getenv('PANDANA_LIMIT'))
      if os.getenv('PANDANA_OFFSET'):
        self.offset = int(os.getenv('PANDANA_OFFSET'))
      
      filelist = self.query[self.offset::self.stride]
      if self.limit: filelist = filelist[:self.limit]
      return ListSource(filelist)

    elif self.isglob():
      if os.getenv('PANDANA_STRIDE'):
        self.stride = int(os.getenv('PANDANA_STRIDE'))
      if os.getenv('PANDANA_LIMIT'):
        self.limit = int(os.getenv('PANDANA_LIMIT'))
      if os.getenv('PANDANA_OFFSET'):
        self.offset = int(os.getenv('PANDANA_OFFSET'))
      
      return GlobSource(self.query, self.stride, self.offset, self.limit)
    
    elif self.isproj():
      if self.stride > 1 or self.offset > 0:
        print ("Can't use stride and offset for SAM projects. Use a query instead to start an internal project")
        sys.exit(2)
      if os.getenv('PANDANA_LIMIT'):
        self.limit = int(os.getenv('PANDANA_LIMIT'))
      
      if not self.limit: self.limit = -1
      
      return SAMProjectSource(self.query, self.limit)
    
    elif self.issamquery():
      if os.getenv('PANDANA_LIMIT'):
        self.limit = int(os.getenv('PANDANA_LIMIT'))
      if self.stride > 1:
        self.query += ' with stride %d' % self.stride
      if self.offset > 0:
        self.query += ' with offset %d' % self.offset
      if self.limit:
        self.query += ' with limit %d' % self.limit
      
      return SAMQuerySource(self.query)
    
    else:
      print ("Invalid Loader query!")
      sys.exit(2)
