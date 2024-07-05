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
            current_time = time.time()
            memory = process.memory_info().rss
            f.write(f"{current_time},{memory}\n")
            f.flush()
            time.sleep(0.5 - (time.time() - current_time - 0.01))
    

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
    def __init__(self, log_dir:str, exp_name:str, tag:str, default_overwrite=None, log_start_time=True, log_accuracy:bool=True, log_memory:bool=True, log_time:bool=True):
        log_start_time = datetime.now(tz="Asia/Seoul").strftime("%Y-%m-%d_%H:%M:%S") if log_start_time else ""
        
        self.log_dir = Path(log_dir) / exp_name / f"{log_start_time}_{tag}"
        
        self.terminate = False
        
        self.log_file_accuracy = self.log_dir / 'accuracy.csv' if log_accuracy else None
        self.log_file_memory = self.log_dir / 'memory_usage.csv' if log_memory else None
        self.log_file_time = self.log_dir / 'execution_time.csv' if log_time else None
        
        if self.log_dir.exists():
            # check file only has header
            for log_file in [self.log_file_accuracy, self.log_file_memory, self.log_file_time]:
                if log_file is not None:
                    with open(log_file, 'r') as f:
                        if len(f.readlines()) == 1:
                            default_overwrite = False
                            break
            
            if default_overwrite is None:
                # ask the user if they want to overwrite the existing log
                print(f"Log directory '{self.log_dir}' already exists.")
                print("Do you want to overwrite the existing log? (y/n)")
                answer = input()
                if answer.lower() != 'y':
                    self.terminate = True
                    return
                else:
                    import shutil
                    shutil.rmtree(self.log_dir)
            elif default_overwrite:
                import shutil
                shutil.rmtree(self.log_dir)
            else:
                self.terminate = True
                return
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        
        # for log_file in [self.log_file_accuracy, self.log_memory, self.log_time]:
        #     if log_file is not None:
        #         with open(log_file, 'w') as f:
        #             f.write("")
        
        if self.log_file_accuracy is not None:
            with open(self.log_file_accuracy, 'w') as f:
                f.write("name,accuracy\n")
        
        if self.log_file_time is not None:
            with open(self.log_file_time, 'w') as f:
                f.write("name,time\n")
        
        if self.log_file_memory is not None:
            with open(self.log_file_memory, 'w') as f:
                f.write("time,memory\n")
        
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
                
        self.memory_process = Process(target=measure_memory, args=(self.main_process, self.log_file_memory))
        self.memory_process.start()
    
    def end_measure_memory(self):
        """
        End the thread to log the memory usage
        """
        if self.log_file_memory is None:
            return
        
        self.memory_process.terminate()
        self.memory_process.join()
        
    def write_accuracy(self, name, accuracy):
        if self.log_file_accuracy is None:
            return
        
        with open(self.log_file_accuracy, 'a') as f:
            f.write(f"{name},{accuracy}\n")