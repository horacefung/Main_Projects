# Chapter 4: High-Level Concurrency in Python
import argparse
import collections
import math
import multiprocessing
import os
import sys

# Project packages
import Image
import Qtrac

'''
Intro:

Writing & maintaining concurrent programs is harder (sometimes much harder)
than nonconcurrent ones. Furthermore, they can actually perform much worse
than nonconcurrent ones. 

The most important difference in how we implement concurrency is whether
we are accessing shared data. This can be directly (e.g. shared memory)
or indirectly (e.g. inter-process communication IPC). 

Threaded concurrency is where separate concurrent threads of execution 
operate within the same system process. These threads typically access 
shared data using serialized access to shared memory (serialization is
the process of converting an object to a stream of bytes to store or 
transmit in to memory). A locking mechanism is necessary to prevent 
conflict in the serialization.

Process-based concurrency (multiprocessing) is where separate processes
execute indepedently. Concurrent processes typically access shared data
using PIC (alhtough they can use shared memory if supported).

Concurrent waiting is another sort-of approach, rather than concurrent
execution. This is implemented through asynchronous I/O. Async is when
a process can happen outside of the main process, e.g. returning some
message to user while the database search is happening.
'''

'''
Packages:

Python has conventional threading support and higher level multiprocessing
support. In addition, the multiprocessing support uses the same abstractions
as threading which makes it easy to switch between the two (when shared
memory isn't used).

Due to GIL (Global Interpreter Lock), the default Python interpreter itself
can only execute on one processor core at a time. Hence, we may not get
the full speed up we hope for when threading. Hence, developers would often
i) use Cython instead which can reach 100x speedup since C code is not 
restricted by GIL or ii) use multiprocessing and avoid contending for the
same interpreter under GIL (for CPU-Bound processing).

For I/O bound processing (e.g. networking), concurrency can produce 
dramatic speedups. In these cases, threading vs multiprocessing generally
doesn't matter, the bottleneck is network latency. 
'''

'''
Levels of Concurrency:

Low-Level: This is for library writers and make explicit use of 
atomic operations.

Mid-Level: This has explicit locks and python provides support with
classes like `threading.Semaphore`, `threading.Lock` and 
`multiprocessing.Lock`.

High-Level: There are no explicit atomic operations or locks.
Classes like `concurrent.futures`, `concurrent.queue` and
`multiprocessing`.
'''

#------------------------------------------#
# CPU-Bound Concurrency
#------------------------------------------#
def handle_commandline():
    parser = argparse.ArgumentParser()
    # Default we can use the number of cores
    parser.add_argument("-c", "--concurrency", type=int,
            default=multiprocessing.cpu_count(),
            help="specify the concurrency (for debugging and "
                "timing) [default: %(default)d]")
    parser.add_argument("-s", "--size", default=400, type=int,
            help="make a scaled image that fits the given dimension "
                "[default: %(default)d]")
    parser.add_argument("-S", "--smooth", action="store_true",
            help="use smooth scaling (slow but good for text)")
    parser.add_argument("source",
            help="the directory containing the original .xpm images")
    parser.add_argument("target",
            help="the directory for the scaled .xpm images")
    args = parser.parse_args()
    source = os.path.abspath(args.source)
    target = os.path.abspath(args.target)
    if source == target:
        args.error("source and target must be different")
    if not os.path.exists(args.target):
        os.makedirs(target)
    return args.size, args.smooth, source, target, args.concurrency

# Using Queues and Multiprocessing

# Collections module offers containers. Containers are objects
# to store different objects and allow for ways to access and iterate
# over them. Examples include lists, tuples, dictionaries etc.

# namedtuple lets you create a container to produce tuples
# with named elements that can be refereced by name.
Result = collections.namedtuple('Result', 'copied scaled name')
Summary = collections.namedtuple('Summary', 'todo copied scaled canceled')
# sample = Result('one', 'two', 'three)  # three elements
# sample[0]
# sample.copied 

def main():
    size, smooth, source, target, concurrency = handle_commandline()
    Qtrac.report('starting...')
    # scale() is our multiprocess function. 
    summary = scale(size, smooth, source, target, concurrency)
    summarize(summary, concurrency)

# This is a queue-based concurrent program. The steps are:
# 1. create a JoinableQueue() of jobs and unjoinable one for results
# 2. create_processes() will loop through concurrency to start process
# 3. add_jobs() takes the queue and populate each image into the queue of jobs
# 4. Include KeyboardInterrupt for clean cancellation of jobs
# 5. worker() is the function thats actually doing stuff
def scale(size, smooth, source, target, concurrency): 
    canceled = False
    jobs = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()
    create_processes(size, smooth, jobs, results, concurrency) 
    todo = add_jobs(source, target, jobs)
    # jobs is a JoinableQueue()
    # results is a Queue()
    try:
        jobs.join()
    except KeyboardInterrupt: # May not work on Windows
            Qtrac.report("canceling...")
            canceled = True
    copied = scaled = 0
    while not results.empty(): # Safe because all jobs have finished
            result = results.get_nowait()
            copied += result.copied
            scaled += result.scaled
    return Summary(todo, copied, scaled, canceled)

def create_processes(size, smooth, jobs, results, concurrency):
    # Loop through number of jobs we want to spin up
    for _ in range(concurrency):
        # For each job, create a Process. Specify the worker function
        # and the inputs into the function (size, smooth, jobs, results)

        # Within the worker function, it will fetch inputs from the job Queue
        # do the work and then append to the results Queue.
        process = multiprocessing.Process(target=worker, args=(size,
                smooth, jobs, results))
        process.daemon = True
        process.start()

def worker(size, smooth, jobs, results):
    # while True means loop forever
    while True:
        try:
            # get() will fetch an item from the Queue and remove it from
            # Queue for all other workers
            sourceImage, targetImage = jobs.get()
            try:
                # Do the scale_one() operation
                result = scale_one(size, smooth, sourceImage, targetImage)
                Qtrac.report("{} {}".format("copied" if result.copied else
                        "scaled", os.path.basename(result.name)))
                # Put results into the Results queue
                results.put(result)
            except Image.Error as err:
                Qtrac.report(str(err), True)
        # finally is the final statement, it will run regardless
        # of whether try produces an error
        finally:
            # Tells the queue the processing of the task is complete
            jobs.task_done()

def add_jobs(source, target, jobs):
    for todo, name in enumerate(os.listdir(source), start=1):
        sourceImage = os.path.join(source, name)
        targetImage = os.path.join(target, name)
        jobs.put((sourceImage, targetImage))
    return todo

def scale_one(size, smooth, sourceImage, targetImage):
    oldImage = Image.from_file(sourceImage)
    if oldImage.width <= size and oldImage.height <= size:
        oldImage.save(targetImage)
        return Result(1, 0, targetImage)
    else:
        if smooth:
            scale = min(size / oldImage.width, size / oldImage.height)
            newImage = oldImage.scale(scale)
        else:
            stride = int(math.ceil(max(oldImage.width / size,
                                       oldImage.height / size)))
            newImage = oldImage.subsample(stride)
        newImage.save(targetImage)
        return Result(0, 1, targetImage)


def summarize(summary, concurrency):
    message = "copied {} scaled {} ".format(summary.copied, summary.scaled)
    difference = summary.todo - (summary.copied + summary.scaled)
    if difference:
        message += "skipped {} ".format(difference)
    message += "using {} processes".format(concurrency)
    if summary.canceled:
        message += " [canceled]"
    Qtrac.report(message)
    print()


if __name__ == '__main__':
    None


# CPU Bound Multiprocess (with Queues) Framework:

# Flow of things:
'''
1. Initiate the empty queues before starting a parallel process. The empty
queues are `jobs = mp.JoinableQueue()` and `results = mp.Queue()`. We need
a queue with locking to handle the inputs as well as collecting the outputs.

2. Initiate the n processes with `mp.Process(target=worker, args=(..., jobs, results))`
n times. This will assign a worker function and direc the process to the jobs input
queue and results output queue.

3. Actually put real inputs into the queue. The `add_jobs()` function will get
the list of image inputs and ultimated call `job.put((...))` and place inputs into
the job queue.

4. Now with the queue filled up, the processes intitiated in create_processes()
will begin doing their thing. The workers will do work on their inputs until completion,
which is indicated by jobs.task_done(). 

---> Details: `process.daemon = True` is a boolean flag to assign this as a subprocess.
That means when the parent process completes or is terminated (the overall script), all the 
subprocesses will be automatically terminated as well.  `process.start()` will begin the
process's activity. 

4. `jobs.join()` is a wait function that blocks everything (outside of workers) until
the jobs queue is truly empty.

5. After the processes are done, we can use `get(False)` or `get_nowait()` to get
the output stored in our results queue and iterate to retrieve each output.

'''