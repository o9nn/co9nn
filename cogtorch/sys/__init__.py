"""
CogTorch System Utilities Module
System utilities inspired by torch.sys

Provides timing, memory management, and system information utilities.
"""

import time
import sys as python_sys
import os
import platform
from typing import Dict, Any, Optional


class Clock:
    """
    High-resolution timer for performance measurement
    """
    
    def __init__(self):
        self.start_time = None
        self.elapsed = 0.0
    
    def tic(self):
        """Start the timer"""
        self.start_time = time.perf_counter()
        return self
    
    def toc(self) -> float:
        """
        Stop the timer and return elapsed time
        
        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            raise RuntimeError("Clock not started - call tic() first")
        
        self.elapsed = time.perf_counter() - self.start_time
        return self.elapsed
    
    def reset(self):
        """Reset the timer"""
        self.start_time = None
        self.elapsed = 0.0
    
    def __repr__(self):
        return f"sys.Clock(elapsed={self.elapsed:.6f}s)"


def tic() -> float:
    """
    Global timer start
    Returns current time
    """
    global _global_start_time
    _global_start_time = time.perf_counter()
    return _global_start_time


def toc() -> float:
    """
    Global timer stop
    Returns elapsed time since last tic()
    """
    global _global_start_time
    if _global_start_time is None:
        raise RuntimeError("No global timer started - call tic() first")
    
    elapsed = time.perf_counter() - _global_start_time
    return elapsed


# Global timer state
_global_start_time = None


def sleep(seconds: float):
    """
    Sleep for specified number of seconds
    
    Args:
        seconds: Number of seconds to sleep
    """
    time.sleep(seconds)


def getpid() -> int:
    """Get process ID"""
    return os.getpid()


def os_name() -> str:
    """Get operating system name"""
    return platform.system()


def os_release() -> str:
    """Get operating system release"""
    return platform.release()


def machine() -> str:
    """Get machine architecture"""
    return platform.machine()


def hostname() -> str:
    """Get hostname"""
    return platform.node()


def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system information
    
    Returns:
        Dictionary with system information
    """
    return {
        'os': os_name(),
        'release': os_release(),
        'machine': machine(),
        'hostname': hostname(),
        'python_version': python_sys.version,
        'python_implementation': platform.python_implementation(),
        'pid': getpid(),
        'cpu_count': os.cpu_count(),
    }


class MemoryTracker:
    """
    Memory usage tracker
    """
    
    def __init__(self):
        self.allocations = {}
        self.total_allocated = 0
    
    def track(self, name: str, size: int):
        """
        Track a memory allocation
        
        Args:
            name: Name of allocation
            size: Size in bytes
        """
        self.allocations[name] = size
        self.total_allocated += size
    
    def untrack(self, name: str):
        """
        Untrack a memory allocation
        
        Args:
            name: Name of allocation to remove
        """
        if name in self.allocations:
            size = self.allocations[name]
            del self.allocations[name]
            self.total_allocated -= size
    
    def get_total(self) -> int:
        """Get total tracked memory in bytes"""
        return self.total_allocated
    
    def get_total_mb(self) -> float:
        """Get total tracked memory in megabytes"""
        return self.total_allocated / (1024 * 1024)
    
    def get_allocations(self) -> Dict[str, int]:
        """Get all tracked allocations"""
        return self.allocations.copy()
    
    def reset(self):
        """Reset tracking"""
        self.allocations.clear()
        self.total_allocated = 0
    
    def __repr__(self):
        return f"sys.MemoryTracker(total={self.get_total_mb():.2f} MB, allocations={len(self.allocations)})"


# Global memory tracker
_global_memory_tracker = MemoryTracker()


def track_memory(name: str, size: int):
    """Track memory allocation in global tracker"""
    _global_memory_tracker.track(name, size)


def untrack_memory(name: str):
    """Untrack memory allocation from global tracker"""
    _global_memory_tracker.untrack(name)


def get_memory_info() -> Dict[str, Any]:
    """Get memory tracking information"""
    return {
        'total_mb': _global_memory_tracker.get_total_mb(),
        'allocations': _global_memory_tracker.get_allocations()
    }


class Benchmark:
    """
    Benchmark utility for timing code execution
    """
    
    def __init__(self, name: str = "Benchmark"):
        """
        Args:
            name: Name of the benchmark
        """
        self.name = name
        self.timings = []
        self.clock = Clock()
    
    def __enter__(self):
        """Context manager entry"""
        self.clock.tic()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        elapsed = self.clock.toc()
        self.timings.append(elapsed)
        return False
    
    def get_average(self) -> float:
        """Get average timing"""
        if not self.timings:
            return 0.0
        return sum(self.timings) / len(self.timings)
    
    def get_min(self) -> float:
        """Get minimum timing"""
        if not self.timings:
            return 0.0
        return min(self.timings)
    
    def get_max(self) -> float:
        """Get maximum timing"""
        if not self.timings:
            return 0.0
        return max(self.timings)
    
    def get_total(self) -> float:
        """Get total time"""
        return sum(self.timings)
    
    def get_count(self) -> int:
        """Get number of timings"""
        return len(self.timings)
    
    def report(self) -> str:
        """Generate benchmark report"""
        if not self.timings:
            return f"{self.name}: No timings recorded"
        
        report = f"\n{self.name} Benchmark Report:\n"
        report += f"  Count:   {self.get_count()}\n"
        report += f"  Total:   {self.get_total():.6f}s\n"
        report += f"  Average: {self.get_average():.6f}s\n"
        report += f"  Min:     {self.get_min():.6f}s\n"
        report += f"  Max:     {self.get_max():.6f}s\n"
        return report
    
    def reset(self):
        """Reset all timings"""
        self.timings.clear()
    
    def __repr__(self):
        return f"sys.Benchmark('{self.name}', count={self.get_count()})"


def ffi_available() -> bool:
    """Check if FFI (Foreign Function Interface) is available"""
    try:
        import cffi
        return True
    except ImportError:
        return False


def get_env(var: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get environment variable
    
    Args:
        var: Variable name
        default: Default value if not found
        
    Returns:
        Environment variable value or default
    """
    return os.environ.get(var, default)


def set_env(var: str, value: str):
    """
    Set environment variable
    
    Args:
        var: Variable name
        value: Variable value
    """
    os.environ[var] = value


__all__ = [
    'Clock',
    'tic',
    'toc',
    'sleep',
    'getpid',
    'os_name',
    'os_release',
    'machine',
    'hostname',
    'get_system_info',
    'MemoryTracker',
    'track_memory',
    'untrack_memory',
    'get_memory_info',
    'Benchmark',
    'ffi_available',
    'get_env',
    'set_env'
]
