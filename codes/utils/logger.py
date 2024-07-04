__all__ = ['ExpLogger']

from datetime import datetime
import time
from pathlib import Path

import psutil
from multiprocessing import Process

def measure_memory(process: psutil.Process, log_file_memory:Path):
    """
    Measure the memory usage of the main process every second.
    """
    with open(log_file_memory, 'a') as f:
        while True:
            memory = process.memory_info().rss
            f.write(f"{time.time()},{memory}\n")
            time.sleep(0.1)
    

class ExpLogger:
    """
    [Experiment Logger]
    
    Logs the execution time and memory usage of the experiment.
    Execution time:
        'start_measure_time()' method records the start time of the experiment.
        'end_measure_time()' method records the end time of the experiment, calculates the execution time, and logs it.
    
    Memory usage:
        'start_measure_memory()' method creates a process and records the memory usage of the main process.
        'end_measure_memory()' method ends the process.
    
    """
    def __init__(self, log_dir:str, log_accuracy:bool=True, log_memory:bool=True, log_time:bool=True):
        self.log_start_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        
        self.log_dir = Path(log_dir) / self.log_start_time
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file_accuracy = self.log_dir / 'accuracy.csv' if log_accuracy else None
        self.log_file_memory = self.log_dir / 'memory_usage.csv' if log_memory else None
        self.log_file_time = self.log_dir / 'execution_time.csv' if log_time else None
        
        # for log_file in [self.log_file_accuracy, self.log_memory, self.log_time]:
        #     if log_file is not None:
        #         with open(log_file, 'w') as f:
        #             f.write("")
        
        self.start_time = {}
        self.memory_process = None
        self.main_process = psutil.Process()
    
    def start_measure_time(self, name:str="main"):
        if self.log_file_time is None:
            return
        
        self.start_time[name] = time.time()
        
    def end_measure_time(self, name:str="main"):
        if self.log_file_time is None:
            return
        
        exec_time = time.time() - self.start_time.pop(name)
        with open(self.log_file_time, 'a') as f:
            f.write(f"{name},{exec_time}\n")  
    
    def start_measure_memory(self):
        """
        Start a process to log the memory usage
        """
        if self.log_file_memory is None:
            return
        
        self.memory_process = Process(target=measure_memory, args=(self.main_process,))
        self.memory_process.start()
    
    def end_measure_memory(self):
        """
        End the thread to log the memory usage
        """
        if self.log_file_memory is None:
            return
        
        self.memory_process.terminate()
        self.memory_process.join()