#!/usr/bin/env python3
"""Check system memory limits and constraints."""

import resource
import os
import subprocess

print("=" * 80)
print("System Memory Limits Check")
print("=" * 80)
print()

# Check ulimit
print("1. Checking ulimit (virtual memory limit)...")
try:
    result = subprocess.run(['ulimit', '-v'], shell=True, capture_output=True, text=True)
    print(f"   ulimit -v: {result.stdout.strip()}")
except:
    pass

# Check Python resource limits
print("\n2. Python resource limits:")
soft, hard = resource.getrlimit(resource.RLIMIT_AS)  # Virtual memory
print(f"   RLIMIT_AS (virtual memory): soft={soft}, hard={hard}")
if soft != -1:
    print(f"   Virtual memory limit: {soft / (1024**3):.2f} GB")
else:
    print("   No virtual memory limit")

soft, hard = resource.getrlimit(resource.RLIMIT_DATA)  # Data segment
print(f"   RLIMIT_DATA (data segment): soft={soft}, hard={hard}")
if soft != -1:
    print(f"   Data segment limit: {soft / (1024**3):.2f} GB")
else:
    print("   No data segment limit")

# Check available memory
print("\n3. System memory:")
try:
    result = subprocess.run(['free', '-h'], capture_output=True, text=True)
    print(result.stdout)
except:
    pass

# Check if there's a memory cgroup limit
print("\n4. Checking for cgroup memory limits...")
try:
    with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
        limit = int(f.read().strip())
        if limit < 2**63:  # Not unlimited
            print(f"   Cgroup memory limit: {limit / (1024**3):.2f} GB")
        else:
            print("   No cgroup memory limit")
except:
    print("   No cgroup limits found (or not in cgroup)")

print("\n" + "=" * 80)

