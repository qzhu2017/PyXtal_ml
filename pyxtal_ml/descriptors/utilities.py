import numpy as np
import time
import os
import sys
import tarfile
from getpass import getuser
try:
    import cPickle as pickle    # Python2
except ImportError:
    import pickle               # Python3
    
class FileDatabase:
    """Using a database file, such as shelve or sqlitedict, that can handle
    multiple processes writing to the file is hard.

    Therefore, we take the stupid approach of having each database entry be
    a separate file. This class behaves essentially like shelve, but saves each
    dictionary entry as a plain pickle file within the directory, with the
    filename corresponding to the dictionary key (which must be a string).

    Like shelve, this also keeps an internal (memory dictionary) representation
    of the variables that have been accessed.

    Also includes an archive feature, where files are instead added to a file
    called 'archive.tar.gz' to save disk space. If an entry exists in both the
    loose and archive formats, the loose is taken to be the new (correct)
    value.
    """

    def __init__(self, filename):
        """Open the filename at specified location. flag is ignored; this
        format is always capable of both reading and writing."""
        if not filename.endswith(os.extsep + 'ampdb'):
            filename += os.extsep + 'ampdb'
        self.path = filename
        self.loosepath = os.path.join(self.path, 'loose')
        self.tarpath = os.path.join(self.path, 'archive.tar.gz')
        if not os.path.exists(self.path):
            try:
                os.mkdir(self.path)
            except OSError:
                # Many simultaneous processes might be trying to make the
                # directory at the same time.
                pass
            try:
                os.mkdir(self.loosepath)
            except OSError:
                pass
        self._memdict = {}  # Items already accessed; stored in memory.

    @classmethod
    def open(Cls, filename, flag=None):
        """Open present for compatibility with shelve. flag is ignored; this
        format is always capable of both reading and writing.
        """
        return Cls(filename=filename)

    def close(self):
        """Only present for compatibility with shelve.
        """
        return

    def keys(self):
        """Return list of keys, both of in-memory and out-of-memory
        items.
        """
        keys = os.listdir(self.loosepath)
        if os.path.exists(self.tarpath):
            with tarfile.open(self.tarpath) as tf:
                keys = list(set(keys + tf.getnames()))
        return keys

    def values(self):
        """Return list of values, both of in-memory and out-of-memory
        items. This moves all out-of-memory items into memory.
        """
        keys = self.keys()
        return [self[key] for key in keys]

    def __len__(self):
        return len(self.keys())

    def __setitem__(self, key, value):
        self._memdict[key] = value
        path = os.path.join(self.loosepath, str(key))
        if os.path.exists(path):
            with open(path, 'rb') as f:
                contents = self._repeat_read(f)
                if pickle.dumps(contents) == pickle.dumps(value):
                    # Using pickle as a hash...
                    return  # Nothing to update.
        with open(path, 'wb') as f:
            pickle.dump(value, f, protocol=0)

    def _repeat_read(self, f, maxtries=5, sleep=0.2):
        """If one process is writing, the other process cannot read without
        errors until it finishes. Reads file-like object f checking for
        errors, and retries up to 'maxtries' times, sleeping 'sleep' sec
        between tries."""
        tries = 0
        while tries < maxtries:
            try:
                contents = pickle.load(f)
            except (UnicodeDecodeError, EOFError, pickle.UnpicklingError):
                time.sleep(0.2)
                tries += 1
            else:
                return contents
        raise IOError('Too many file read attempts.')

    def __getitem__(self, key):
        if key in self._memdict:
            return self._memdict[key]
        keypath = os.path.join(self.loosepath, key)
        if os.path.exists(keypath):
            with open(keypath, 'rb') as f:
                return self._repeat_read(f)
        elif os.path.exists(self.tarpath):
            with tarfile.open(self.tarpath) as tf:
                return pickle.load(tf.extractfile(key))
        else:
            raise KeyError(str(key))

    def update(self, newitems):
        for key, value in newitems.items():
            self.__setitem__(key, value)

    def archive(self):
        """Cleans up to save disk space and reduce huge number of files.

        That is, puts all files into an archive.  Compresses all files in
        <path>/loose and places them in <path>/archive.tar.gz.  If archive
        exists, appends/modifies.
        """
        loosefiles = os.listdir(self.loosepath)
        print('Contains %i loose entries.' % len(loosefiles))
        if len(loosefiles) == 0:
            print(' -> No action taken.')
            return
        if os.path.exists(self.tarpath):
            with tarfile.open(self.tarpath) as tf:
                names = [_ for _ in tf.getnames() if _ not in
                         os.listdir(self.loosepath)]
                for name in names:
                    tf.extract(member=name, path=self.loosepath)
        loosefiles = os.listdir(self.loosepath)
        print('Compressing %i entries.' % len(loosefiles))
        with tarfile.open(self.tarpath, 'w:gz') as tf:
            for file in loosefiles:
                tf.add(name=os.path.join(self.loosepath, file),
                       arcname=file)
        print('Cleaning up: removing %i files.' % len(loosefiles))
        for file in loosefiles:
            os.remove(os.path.join(self.loosepath, file))


class Logger:

    """Logger that can also deliver timing information.

    Parameters
    ----------
    file : str
        File object or path to the file to write to.  Or set to None for
        a logger that does nothing.
    """

    def __init__(self, file):
        if file is None:
            self.file = None
            return
        if isinstance(file, str):
            self.filename = file
            file = open(file, 'a')
        self.file = file
        self.tics = {}

    def tic(self, label=None):
        """Start a timer.

        Parameters
        ----------
        label : str
            Label for managing multiple timers.
        """
        if self.file is None:
            return
        if label:
            self.tics[label] = time.time()
        else:
            self._tic = time.time()

    def __call__(self, message, toc=None, tic=False):
        """Writes message to the log file.

        Parameters
        ---------
        message : str
            Message to be written.
        toc : bool or str
            If toc=True or toc=label, it will append timing information in
            minutes to the timer.
        tic : bool or str
            If tic=True or tic=label, will start the generic timer or a timer
            associated with label. Equivalent to self.tic(label).
        """
        if self.file is None:
            return
        dt = ''
        if toc:
            if toc is True:
                tic = self._tic
            else:
                tic = self.tics[toc]
            dt = (time.time() - tic) / 60.
            dt = ' %.1f min.' % dt
        if self.file.closed:
            self.file = open(self.filename, 'a')
        self.file.write(message + dt + '\n')
        self.file.flush()
        if tic:
            if tic is True:
                self.tic()
            else:
                self.tic(label=tic)
                
    
class Data:
    """
    Serves as a container (dictionary-like) for (key, value) pairs that
    also serves to calculate them.

    Works by default with python's shelve module, but something that is built
    to share the same commands as shelve will work fine; just specify this in
    dbinstance.

    Designed to hold things like neighborlists, which have a hash, value
    format.

    This will work like a dictionary in that items can be accessed with
    data[key], but other advanced dictionary functions should be accessed with
    through the .d attribute:

    >>> data = Data(...)
    >>> data.open()
    >>> keys = data.d.keys()
    >>> values = data.d.values()
    """

    def __init__(self, filename, db=FileDatabase, calculator=None):
        self.calc = calculator
        self.db = db
        self.filename = filename
        self.d = None

    def calculate_items(self, images, parallel, log=None):
        """Calculates the data value with 'calculator' for the specified
        images.

        images is a dictionary, and the same keys will be used for the current
        database.
        """
        if log is None:
            log = Logger(None)
        if self.d is not None:
            self.d.close()
            self.d = None
        log(' Data stored in file %s.' % self.filename)
        d = self.db.open(self.filename, 'r')
        calcs_needed = list(set(images.keys()).difference(d.keys()))
        dblength = len(d)
        d.close()
        log(' File exists with %i total images, %i of which are needed.' %
            (dblength, len(images) - len(calcs_needed)))
        log(' %i new calculations needed.' % len(calcs_needed))
        if len(calcs_needed) == 0:
            return
        if parallel['cores'] == 1:
            d = self.db.open(self.filename, 'c')
            for key in calcs_needed:
                d[key] = self.calc.calculate(images[key], key)
            d.close()  # Necessary to get out of write mode and unlock?
            log(' Calculated %i new images.' % len(calcs_needed))
        else:
            python = sys.executable
            workercommand = '%s -m %s' % (python, self.calc.__module__)
            sessions = setup_parallel(parallel, workercommand, log)
            server = sessions['master']
            n_pids = sessions['n_pids']

            globals = self.calc.globals
            keyed = self.calc.keyed

            keys = make_sublists(calcs_needed, n_pids)
            results = {}

            # All incoming requests will be dictionaries with three keys.
            # d['id']: process id number, assigned when process created above.
            # d['subject']: what the message is asking for / telling you
            # d['data']: optional data passed from the worker.

            active = 0  # count of processes actively calculating
            log(' Parallel calculations starting...', tic='parallel')
            active = n_pids  # currently active workers
            while True:
                message = server.recv_pyobj()
                if message['subject'] == '<purpose>':
                    server.send_pyobj(self.calc.parallel_command)
                elif message['subject'] == '<request>':
                    request = message['data']  # Variable name.
                    if request == 'images':
                        server.send_pyobj({k: images[k] for k in
                                           keys[int(message['id'])]})
                    elif request in keyed:
                        server.send_pyobj({k: keyed[request][k] for k in
                                           keys[int(message['id'])]})
                    else:
                        server.send_pyobj(globals[request])
                elif message['subject'] == '<result>':
                    result = message['data']
                    server.send_string('meaningless reply')
                    active -= 1
                    log('  Process %s returned %i results.' %
                        (message['id'], len(result)))
                    results.update(result)
                elif message['subject'] == '<info>':
                    server.send_string('meaningless reply')
                if active == 0:
                    break
            log('  %i new results.' % len(results))
            log(' ...parallel calculations finished.', toc='parallel')
            log(' Adding new results to database.')
            d = self.db.open(self.filename, 'c')
            d.update(results)
            d.close()  # Necessary to get out of write mode and unlock?

        self.d = None

    def __getitem__(self, key):
        self.open()
        return self.d[key]

    def close(self):
        """Safely close the database.
        """
        if self.d:
            self.d.close()
        self.d = None

    def open(self, mode='r'):
        """Open the database connection with mode specified.
        """
        if self.d is None:
            self.d = self.db.open(self.filename, mode)

    def __del__(self):
        self.close()

class FileDatabase:
    """Using a database file, such as shelve or sqlitedict, that can handle
    multiple processes writing to the file is hard.

    Therefore, we take the stupid approach of having each database entry be
    a separate file. This class behaves essentially like shelve, but saves each
    dictionary entry as a plain pickle file within the directory, with the
    filename corresponding to the dictionary key (which must be a string).

    Like shelve, this also keeps an internal (memory dictionary) representation
    of the variables that have been accessed.

    Also includes an archive feature, where files are instead added to a file
    called 'archive.tar.gz' to save disk space. If an entry exists in both the
    loose and archive formats, the loose is taken to be the new (correct)
    value.
    """

    def __init__(self, filename):
        """Open the filename at specified location. flag is ignored; this
        format is always capable of both reading and writing."""
        if not filename.endswith(os.extsep + 'ampdb'):
            filename += os.extsep + 'ampdb'
        self.path = filename
        self.loosepath = os.path.join(self.path, 'loose')
        self.tarpath = os.path.join(self.path, 'archive.tar.gz')
        if not os.path.exists(self.path):
            try:
                os.mkdir(self.path)
            except OSError:
                # Many simultaneous processes might be trying to make the
                # directory at the same time.
                pass
            try:
                os.mkdir(self.loosepath)
            except OSError:
                pass
        self._memdict = {}  # Items already accessed; stored in memory.

    @classmethod
    def open(Cls, filename, flag=None):
        """Open present for compatibility with shelve. flag is ignored; this
        format is always capable of both reading and writing.
        """
        return Cls(filename=filename)

    def close(self):
        """Only present for compatibility with shelve.
        """
        return

    def keys(self):
        """Return list of keys, both of in-memory and out-of-memory
        items.
        """
        keys = os.listdir(self.loosepath)
        if os.path.exists(self.tarpath):
            with tarfile.open(self.tarpath) as tf:
                keys = list(set(keys + tf.getnames()))
        return keys

    def values(self):
        """Return list of values, both of in-memory and out-of-memory
        items. This moves all out-of-memory items into memory.
        """
        keys = self.keys()
        return [self[key] for key in keys]

    def __len__(self):
        return len(self.keys())

    def __setitem__(self, key, value):
        self._memdict[key] = value
        path = os.path.join(self.loosepath, str(key))
        if os.path.exists(path):
            with open(path, 'rb') as f:
                contents = self._repeat_read(f)
                if pickle.dumps(contents) == pickle.dumps(value):
                    # Using pickle as a hash...
                    return  # Nothing to update.
        with open(path, 'wb') as f:
            pickle.dump(value, f, protocol=0)

    def _repeat_read(self, f, maxtries=5, sleep=0.2):
        """If one process is writing, the other process cannot read without
        errors until it finishes. Reads file-like object f checking for
        errors, and retries up to 'maxtries' times, sleeping 'sleep' sec
        between tries."""
        tries = 0
        while tries < maxtries:
            try:
                contents = pickle.load(f)
            except (UnicodeDecodeError, EOFError, pickle.UnpicklingError):
                time.sleep(0.2)
                tries += 1
            else:
                return contents
        raise IOError('Too many file read attempts.')

    def __getitem__(self, key):
        if key in self._memdict:
            return self._memdict[key]
        keypath = os.path.join(self.loosepath, key)
        if os.path.exists(keypath):
            with open(keypath, 'rb') as f:
                return self._repeat_read(f)
        elif os.path.exists(self.tarpath):
            with tarfile.open(self.tarpath) as tf:
                return pickle.load(tf.extractfile(key))
        else:
            raise KeyError(str(key))

    def update(self, newitems):
        for key, value in newitems.items():
            self.__setitem__(key, value)

    def archive(self):
        """Cleans up to save disk space and reduce huge number of files.

        That is, puts all files into an archive.  Compresses all files in
        <path>/loose and places them in <path>/archive.tar.gz.  If archive
        exists, appends/modifies.
        """
        loosefiles = os.listdir(self.loosepath)
        print('Contains %i loose entries.' % len(loosefiles))
        if len(loosefiles) == 0:
            print(' -> No action taken.')
            return
        if os.path.exists(self.tarpath):
            with tarfile.open(self.tarpath) as tf:
                names = [_ for _ in tf.getnames() if _ not in
                         os.listdir(self.loosepath)]
                for name in names:
                    tf.extract(member=name, path=self.loosepath)
        loosefiles = os.listdir(self.loosepath)
        print('Compressing %i entries.' % len(loosefiles))
        with tarfile.open(self.tarpath, 'w:gz') as tf:
            for file in loosefiles:
                tf.add(name=os.path.join(self.loosepath, file),
                       arcname=file)
        print('Cleaning up: removing %i files.' % len(loosefiles))
        for file in loosefiles:
            os.remove(os.path.join(self.loosepath, file))

            
def make_sublists(masterlist, n):
    """Randomly divides the masterlist into n sublists of roughly
    equal size.

    The intended use is to divide a keylist and assign
    keys to each task in parallel processing. This also destroys
    the masterlist (to save some memory).
    """
    masterlist = list(masterlist)
    np.random.shuffle(masterlist)
    N = len(masterlist)
    sublist_lengths = [
        N // n if _ >= (N % n) else N // n + 1 for _ in range(n)]
    sublists = []
    for sublist_length in sublist_lengths:
        sublists.append([masterlist.pop() for _ in range(sublist_length)])
    return sublists


def setup_parallel(parallel, workercommand, log, setup_publisher=False):
    """Starts the worker processes and the master to control them.

    This makes an SSH connection to each node (including the one the master
    process runs on), then creates the specified number of processes on each
    node through its SSH connection. Then sets up ZMQ for efficienty
    communication between the worker processes and the master process.

    Uses the parallel dictionary as defined in amp.Amp. log is an Amp logger.
    module is the name of the module to be called, which is usually
    given by self.calc.__module, etc.
    workercommand is stub of the command used to start the servers,
    typically like "python -m amp.descriptor.gaussian". Appended to
    this will be " <pid> <serversocket> &" where <pid> is the unique ID
    assigned to each process and <serversocket> is the address of the
    server, like 'node321:34292'.

    If setup_publisher is True, also sets up a publisher instead of just
    a reply socket.

    Returns
    -------
    server : (a ZMQ socket)
        The ssh connections (pxssh instances; if these objects are destroyed
        pxssh will close the sessions)

        the pid_count, which is the total number of workers started. Each
        worker can be communicated directly through its PID, an integer
        between 0 and pid_count
    """
    import zmq
    from socket import gethostname

    log(' Parallel processing.')
    serverhostname = gethostname()

    # Establish server session.
    context = zmq.Context()
    server = context.socket(zmq.REP)
    port = server.bind_to_random_port('tcp://*')
    serversocket = '%s:%s' % (serverhostname, port)
    log(' Established server at %s.' % serversocket)
    sessions = {'master': server,
                'mastersocket': serversocket}
    if setup_publisher:
        publisher = context.socket(zmq.PUB)
        port = publisher.bind_to_random_port('tcp://*')
        publishersocket = '{}:{}'.format(serverhostname, port)
        log(' Established publisher at {}.'.format(publishersocket))
        sessions['publisher'] = publisher
        sessions['publisher_socket'] = publishersocket

    workercommand += ' %s ' + serversocket

    log(' Establishing worker sessions.')
    connections = []
    pid_count = 0
    for workerhostname, nprocesses in parallel['cores'].items():
        pids = range(pid_count, pid_count + nprocesses)
        pid_count += nprocesses
        connections.append(start_workers(pids,
                                         workerhostname,
                                         workercommand,
                                         log,
                                         parallel['envcommand']))

    sessions['n_pids'] = pid_count
    sessions['connections'] = connections
    return sessions


def start_workers(process_ids, workerhostname, workercommand, log,
                  envcommand):
    """A function to start a new SSH session and establish processes on
    that session.
    """
    if workerhostname != 'localhost':
        workercommand += ' &'
        log(' Starting non-local connections.')
        pxssh = importer('pxssh')
        ssh = pxssh.pxssh()
        ssh.login(workerhostname, getuser())
        if envcommand is not None:
            log('Environment command: %s' % envcommand)
            ssh.sendline(envcommand)
            ssh.readline()
        for process_id in process_ids:
            ssh.sendline(workercommand % process_id)
            ssh.expect('<amp-connect>')
            ssh.expect('<stderr>')
            log('  Session %i (%s): %s' %
                (process_id, workerhostname, ssh.before.strip()))
        return ssh
    import pexpect
    log(' Starting local connections.')
    children = []
    for process_id in process_ids:
        child = pexpect.spawn(workercommand % process_id)
        child.expect('<amp-connect>')
        child.expect('<stderr>')
        log('  Session %i (%s): %s' %
            (process_id, workerhostname, child.before.strip()))
        children.append(child)
    return children


def importer(name):
    """Handles strange import cases, like pxssh which might show
    up in either the package pexpect or pxssh.
    """

    if name == 'pxssh':
        try:
            import pxssh
        except ImportError:
            try:
                from pexpect import pxssh
            except ImportError:
                raise ImportError('pxssh not found!')
        return pxssh
    elif name == 'NeighborList':
        try:
            from ase.neighborlist import NeighborList
        except ImportError:
            # We're on ASE 3.10 or older
            from ase.calculators.neighborlist import NeighborList
        return NeighborList