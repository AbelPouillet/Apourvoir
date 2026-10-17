"""
Thread Pool for Apourvoir SDK

NUMA-aware thread pool for optimal performance on multi-socket systems.
"""

import logging
import os
import threading
import multiprocessing
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Tuple, Union, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from queue import Queue, Empty

import psutil
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ThreadConfig:
    """Configuration for a thread."""
    thread_id: int = 0
    name: str = ""
    cpu_affinity: List[int] = field(default_factory=list)
    numa_node: int = 0
    priority: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "thread_id": self.thread_id,
            "name": self.name,
            "cpu_affinity": self.cpu_affinity,
            "numa_node": self.numa_node,
            "priority": self.priority,
        }


@dataclass
class ThreadPoolStats:
    """Statistics for thread pool."""
    num_threads: int = 0
    active_threads: int = 0
    completed_tasks: int = 0
    pending_tasks: int = 0
    avg_task_time_ms: float = 0.0
    total_task_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_threads": self.num_threads,
            "active_threads": self.active_threads,
            "completed_tasks": self.completed_tasks,
            "pending_tasks": self.pending_tasks,
            "avg_task_time_ms": self.avg_task_time_ms,
            "total_task_time_ms": self.total_task_time_ms,
        }


class NUMAAwareThreadPool:
    """
    NUMA-aware thread pool for optimal performance on multi-socket systems.
    
    Features:
    - Automatic detection of NUMA nodes
    - CPU affinity setting for threads
    - Load balancing across NUMA nodes
    - Task queue with priority support
    - Statistics tracking
    
    Example:
        >>> from apourvoir.optimizer import NUMAAwareThreadPool
        >>> 
        >>> # Create thread pool
        >>> pool = NUMAAwareThreadPool(max_workers=8, numa_aware=True)
        >>> 
        >>> # Submit tasks
        >>> futures = [pool.submit(my_func, arg) for arg in args]
        >>> 
        >>> # Get results
        >>> results = [f.result() for f in futures]
        >>> 
        >>> # Shutdown
        >>> pool.shutdown()
    """
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        numa_aware: bool = True,
        thread_name_prefix: str = "numa_thread",
        daemon: bool = True,
        priority_queue: bool = False,
    ):
        """
        Initialize the NUMA-aware thread pool.
        
        Args:
            max_workers: Maximum number of worker threads
            numa_aware: Whether to use NUMA-aware scheduling
            thread_name_prefix: Prefix for thread names
            daemon: Whether threads should be daemon threads
            priority_queue: Whether to use priority queue
        """
        # Detect hardware
        self._detect_hardware()
        
        # Configuration
        if max_workers is None:
            max_workers = self._default_workers()
        
        self.max_workers = max_workers
        self.numa_aware = numa_aware and self._has_numa()
        self.thread_name_prefix = thread_name_prefix
        self.daemon = daemon
        self.priority_queue = priority_queue
        
        # State
        self._running = False
        self._workers: List[threading.Thread] = []
        self._task_queue: Optional[Union[Queue, Any]] = None
        self._result_queue: Optional[Queue] = None
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        
        # Statistics
        self._stats = ThreadPoolStats()
        self._task_times: List[float] = []
        self._start_time = None
        
        # Thread local storage
        self._thread_local = threading.local()
        
        # Create queues
        self._create_queues()
        
        # Start workers
        self.start()
        
        logger.info(f"NUMAAwareThreadPool initialized: workers={max_workers}, "
                   f"numa_aware={self.numa_aware}, numa_nodes={self._numa_nodes}")
    
    def _detect_hardware(self) -> None:
        """Detect hardware configuration."""
        # Number of CPU cores
        self._cpu_count = multiprocessing.cpu_count()
        self._logical_cpu_count = psutil.cpu_count(logical=True)
        self._physical_cpu_count = psutil.cpu_count(logical=False)
        
        # Detect NUMA nodes
        self._numa_nodes = 1
        self._cpu_per_node = []
        
        try:
            # Try to detect NUMA nodes
            if hasattr(os, 'sched_getaffinity'):
                # Linux with NUMA
                try:
                    with open('/proc/cpuinfo', 'r') as f:
                        content = f.read()
                        physical_ids = set()
                        cores_per_node = {}
                        
                        for line in content.split('\n'):
                            if line.startswith('physical id'):
                                phys_id = int(line.split(':')[1].strip())
                                physical_ids.add(phys_id)
                            elif line.startswith('processor'):
                                proc_id = int(line.split(':')[1].strip())
                                if 'physical id' in cores_per_node:
                                    cores_per_node[phys_id].append(proc_id)
                                else:
                                    cores_per_node[phys_id] = [proc_id]
                        
                        self._numa_nodes = len(physical_ids)
                        self._cpu_per_node = [cores_per_node.get(i, []) for i in range(self._numa_nodes)]
                except:
                    pass
            
            # Alternative method
            if self._numa_nodes == 1:
                try:
                    import numactl
                    self._numa_nodes = numactl.NUMA().num_nodes()
                    self._cpu_per_node = [
                        numactl.NUMA().get_node_cpus(i)
                        for i in range(self._numa_nodes)
                    ]
                except ImportError:
                    pass
        except Exception as e:
            logger.warning(f"Failed to detect NUMA: {e}")
    
    def _has_numa(self) -> bool:
        """Check if system has NUMA support."""
        return self._numa_nodes > 1
    
    def _default_workers(self) -> int:
        """Get default number of workers."""
        if self._numa_nodes > 1:
            return self._logical_cpu_count
        else:
            return min(self._logical_cpu_count, 8)
    
    def _create_queues(self) -> None:
        """Create task and result queues."""
        if self.priority_queue:
            # Use priority queue
            self._task_queue = PriorityQueue()
        else:
            self._task_queue = Queue()
        
        self._result_queue = Queue()
    
    def start(self) -> None:
        """Start worker threads."""
        if self._running:
            return
        
        self._running = True
        self._workers = []
        
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                name=f"{self.thread_name_prefix}_{i}",
                daemon=self.daemon,
            )
            worker.start()
            self._workers.append(worker)
        
        self._start_time = time.time()
        logger.info(f"Started {self.max_workers} worker threads")
    
    def shutdown(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """
        Shutdown the thread pool.
        
        Args:
            wait: Whether to wait for all tasks to complete
            timeout: Maximum time to wait (seconds)
        """
        if not self._running:
            return
        
        self._running = False
        
        # Add sentinel values to wake up workers
        for _ in range(self.max_workers):
            self._task_queue.put(None)
        
        # Wait for workers to finish
        if wait:
            for worker in self._workers:
                worker.join(timeout=timeout)
        
        self._workers = []
        logger.info("Thread pool shutdown")
    
    def _worker_loop(self, worker_id: int) -> None:
        """Worker thread main loop."""
        # Set CPU affinity if NUMA-aware
        if self.numa_aware:
            self._set_cpu_affinity(worker_id)
        
        # Set thread name
        threading.current_thread().name = f"{self.thread_name_prefix}_{worker_id}"
        
        while self._running:
            try:
                # Get task
                task = self._task_queue.get()
                
                if task is None:
                    # Sentinel value - shutdown
                    break
                
                # Execute task
                func, args, kwargs, future = task
                
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    result = None
                    success = False
                    error = e
                end_time = time.time()
                
                # Update statistics
                with self._stats_lock:
                    self._stats.completed_tasks += 1
                    task_time = (end_time - start_time) * 1000
                    self._stats.total_task_time_ms += task_time
                    self._task_times.append(task_time)
                    if len(self._task_times) > 100:
                        self._task_times.pop(0)
                    self._stats.avg_task_time_ms = np.mean(self._task_times) if self._task_times else 0
                
                # Put result in queue
                self._result_queue.put((future, result, success, error))
                
            except Empty:
                # Queue is empty, wait briefly
                time.sleep(0.001)
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
    
    def _set_cpu_affinity(self, worker_id: int) -> None:
        """Set CPU affinity for a worker thread."""
        if not self.numa_aware or self._numa_nodes <= 1:
            return
        
        try:
            # Assign worker to a NUMA node
            node = worker_id % self._numa_nodes
            cpus = self._cpu_per_node[node]
            
            if cpus:
                # Set affinity
                os.sched_setaffinity(0, cpus)
                logger.debug(f"Worker {worker_id} assigned to NUMA node {node}, CPUs {cpus}")
        except Exception as e:
            logger.warning(f"Failed to set CPU affinity for worker {worker_id}: {e}")
    
    def submit(
        self,
        func: Callable,
        *args,
        priority: int = 0,
        **kwargs,
    ) -> Future:
        """
        Submit a task to the thread pool.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            priority: Task priority (higher = more important)
            **kwargs: Keyword arguments
        
        Returns:
            Future object for the task
        """
        if not self._running:
            raise RuntimeError("Thread pool is not running")
        
        # Create future
        future = Future()
        
        # Create task
        task = (func, args, kwargs, future)
        
        if self.priority_queue:
            # Add to priority queue
            self._task_queue.put((priority, task))
        else:
            # Add to regular queue
            self._task_queue.put(task)
        
        # Update statistics
        with self._stats_lock:
            self._stats.pending_tasks += 1
        
        return future
    
    def map(
        self,
        func: Callable,
        *iterables,
        priority: int = 0,
        **kwargs,
    ) -> List[Future]:
        """
        Map a function across iterables.
        
        Args:
            func: Function to execute
            *iterables: Iterables to map over
            priority: Task priority
            **kwargs: Keyword arguments
        
        Returns:
            List of Future objects
        """
        # Validate iterables
        if not iterables:
            raise ValueError("At least one iterable must be provided")
        
        # Get length from first iterable
        try:
            length = len(iterables[0])
        except:
            raise ValueError("Iterables must have a length")
        
        # Validate all iterables have same length
        for iterable in iterables[1:]:
            if len(iterable) != length:
                raise ValueError("All iterables must have the same length")
        
        # Submit tasks
        futures = []
        for i in range(length):
            args = tuple(iterable[i] for iterable in iterables)
            future = self.submit(func, *args, priority=priority, **kwargs)
            futures.append(future)
        
        return futures
    
    def execute_parallel(
        self,
        funcs: List[Callable],
        args_list: Optional[List[Tuple]] = None,
        kwargs_list: Optional[List[Dict]] = None,
        priority: int = 0,
    ) -> List[Future]:
        """
        Execute multiple functions in parallel.
        
        Args:
            funcs: List of functions to execute
            args_list: List of argument tuples
            kwargs_list: List of keyword argument dictionaries
            priority: Task priority
        
        Returns:
            List of Future objects
        """
        if args_list is None:
            args_list = [()] * len(funcs)
        if kwargs_list is None:
            kwargs_list = [{}] * len(funcs)
        
        if len(funcs) != len(args_list) or len(funcs) != len(kwargs_list):
            raise ValueError("funcs, args_list, and kwargs_list must have the same length")
        
        futures = []
        for func, args, kwargs in zip(funcs, args_list, kwargs_list):
            future = self.submit(func, *args, priority=priority, **kwargs)
            futures.append(future)
        
        return futures
    
    def get_results(self, futures: List[Future]) -> List[Any]:
        """
        Get results from a list of futures.
        
        Args:
            futures: List of Future objects
        
        Returns:
            List of results in the same order as futures
        """
        results = []
        for future in futures:
            result, success, error = self._get_result(future)
            if success:
                results.append(result)
            else:
                raise error
        return results
    
    def get_results_as_completed(self, futures: List[Future]) -> Iterator[Tuple[Future, Any]]:
        """
        Get results as they complete.
        
        Args:
            futures: List of Future objects
        
        Yields:
            Tuple of (future, result) as each completes
        """
        # Create a mapping from our futures to concurrent.futures.Future
        # For simplicity, we'll just poll
        remaining = set(futures)
        
        while remaining:
            for future in list(remaining):
                if future in self._completed_futures:
                    result, success, error = self._completed_futures.pop(future)
                    if success:
                        yield future, result
                    else:
                        raise error
                    remaining.remove(future)
            
            if remaining:
                time.sleep(0.001)
    
    def _get_result(self, future: Future) -> Tuple[Any, bool, Optional[Exception]]:
        """Get result from a future."""
        # This is a simplified version
        # In a real implementation, we'd need to track futures properly
        
        # For now, we'll just wait for the result
        while True:
            with self._stats_lock:
                if future in self._completed_futures:
                    return self._completed_futures.pop(future)
            time.sleep(0.001)
    
    def wait(self, futures: Optional[List[Future]] = None) -> bool:
        """
        Wait for all tasks to complete.
        
        Args:
            futures: Optional list of futures to wait for (all if None)
        
        Returns:
            True if all tasks completed successfully
        """
        if futures is None:
            # Wait for all pending tasks
            while True:
                with self._stats_lock:
                    if self._stats.pending_tasks == 0:
                        return True
                time.sleep(0.01)
        else:
            # Wait for specific futures
            for future in futures:
                result, success, error = self._get_result(future)
                if not success:
                    return False
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get thread pool statistics."""
        with self._stats_lock:
            # Update active threads
            self._stats.active_threads = self._get_active_threads()
            self._stats.pending_tasks = self._task_queue.qsize() if self._task_queue else 0
            
            return self._stats.to_dict()
    
    def _get_active_threads(self) -> int:
        """Get number of active threads."""
        # This is an approximation
        return len([t for t in self._workers if t.is_alive()])
    
    def reset_stats(self) -> None:
        """Reset all statistics."""
        with self._stats_lock:
            self._stats = ThreadPoolStats()
            self._task_times.clear()


class PriorityQueue:
    """Priority queue for tasks."""
    
    def __init__(self):
        self._queue = []
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
    
    def put(self, item: tuple) -> None:
        """Add an item to the queue."""
        with self._lock:
            # Item is (priority, task)
            priority, task = item
            self._queue.append((priority, task))
            self._queue.sort(key=lambda x: -x[0])  # Higher priority first
            self._condition.notify()
    
    def get(self) -> tuple:
        """Get an item from the queue."""
        with self._lock:
            while not self._queue:
                self._condition.wait()
            
            priority, task = self._queue.pop(0)
            return task
    
    def qsize(self) -> int:
        """Get queue size."""
        with self._lock:
            return len(self._queue)
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        with self._lock:
            return len(self._queue) == 0


class TaskScheduler:
    """
    Advanced task scheduler with load balancing.
    """
    
    def __init__(
        self,
        thread_pool: NUMAAwareThreadPool,
        batch_size: int = 1,
    ):
        """
        Initialize task scheduler.
        
        Args:
            thread_pool: Thread pool to use
            batch_size: Number of tasks to batch
        """
        self.thread_pool = thread_pool
        self.batch_size = batch_size
        self._lock = threading.Lock()
        self._task_buffer: List[tuple] = []
    
    def submit(self, func: Callable, *args, priority: int = 0, **kwargs) -> Future:
        """
        Submit a task with batching.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            priority: Task priority
            **kwargs: Keyword arguments
        
        Returns:
            Future object
        """
        with self._lock:
            self._task_buffer.append((func, args, kwargs, priority))
            
            if len(self._task_buffer) >= self.batch_size:
                return self._flush_buffer()
            else:
                # Return a future that will be set later
                future = Future()
                self._pending_futures.append(future)
                return future
    
    def _flush_buffer(self) -> Optional[Future]:
        """Flush the task buffer."""
        if not self._task_buffer:
            return None
        
        # Submit all buffered tasks
        futures = []
        for func, args, kwargs, priority in self._task_buffer:
            future = self.thread_pool.submit(func, *args, priority=priority, **kwargs)
            futures.append(future)
        
        self._task_buffer.clear()
        
        # Return first future
        return futures[0] if futures else None


if __name__ == "__main__":
    # Test the thread pool
    import time
    
    print("NUMAAwareThreadPool Test")
    print("=" * 80)
    
    # Create thread pool
    pool = NUMAAwareThreadPool(max_workers=4, numa_aware=True)
    
    print(f"Thread pool created: {pool.get_stats()}")
    print()
    
    # Define a test function
    def test_func(x):
        time.sleep(0.1)
        return x * x
    
    # Submit tasks
    futures = [pool.submit(test_func, i) for i in range(10)]
    
    # Get results
    results = pool.get_results(futures)
    
    print(f"Results: {results}")
    print(f"Final stats: {pool.get_stats()}")
    
    # Shutdown
    pool.shutdown()
