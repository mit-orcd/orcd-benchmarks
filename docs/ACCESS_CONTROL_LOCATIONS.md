# Reservation Access Control Implementation - Location Summary

This document provides a quick reference showing exactly where in the repository an operator is prevented from launching a job on a reservation unless they have explicit access.

## Primary Implementation Location

### Core Access Control Logic

**File:** `/lib/reservation_access_control.sh`

This is the **primary location** where access control is enforced. The key function is:

```bash
validate_reservation_access() {
    local reservation="$1"
    local current_user="${USER}"
    
    # If reservation is empty, "none", or not specified, allow the job
    if [ -z "$reservation" ] || [ "$reservation" = "none" ] || [ "$reservation" = "NONE" ]; then
        return 0
    fi
    
    # Check if scontrol command is available
    if ! command -v scontrol &> /dev/null; then
        echo "WARNING: scontrol command not found. Cannot validate reservation access." >&2
        echo "Proceeding without validation. SLURM will enforce access control." >&2
        return 0
    fi
    
    # Get reservation information
    local reservation_info
    reservation_info=$(scontrol show reservation="${reservation}" 2>&1)
    
    # Check if reservation exists
    if echo "$reservation_info" | grep -q "Reservation .* not found"; then
        echo "ERROR: Reservation '${reservation}' does not exist." >&2
        echo "Cannot submit job to non-existent reservation." >&2
        return 1
    fi
    
    # Extract the Users and Accounts fields from reservation info
    local users_field
    local accounts_field
    users_field=$(echo "$reservation_info" | grep -oP '(?<=Users=)[^ ]+' || echo "")
    accounts_field=$(echo "$reservation_info" | grep -oP '(?<=Accounts=)[^ ]+' || echo "")
    
    # Check if the reservation has unrestricted access (Users=(null) or empty)
    if [ -z "$users_field" ] || [ "$users_field" = "(null)" ]; then
        # Check accounts field
        if [ -z "$accounts_field" ] || [ "$accounts_field" = "(null)" ]; then
            # Unrestricted reservation - all users have access
            return 0
        fi
    fi
    
    # Check if current user is in the Users list
    if [ -n "$users_field" ] && [ "$users_field" != "(null)" ]; then
        # Convert comma-separated list to array and check
        if echo "$users_field" | grep -qw "$current_user"; then
            return 0
        fi
    fi
    
    # If we get here, user doesn't have explicit access
    echo "ERROR: User '${current_user}' does not have explicit access to reservation '${reservation}'." >&2
    echo "Reservation details:" >&2
    echo "  Allowed Users: ${users_field:-'(not restricted by user)'}" >&2
    echo "  Allowed Accounts: ${accounts_field:-'(not restricted by account)'}" >&2
    echo "" >&2
    echo "Please request access to this reservation or use a different reservation." >&2
    return 1
}
```

**Lines:** 14-76 in `/lib/reservation_access_control.sh`

## Integration Points - Where Validation is Called

The validation is enforced in the following job submission scripts. In each script, the validation happens **before** any `sbatch` commands are executed.

### 1. GPU Burn Tests
**File:** `gpu-burn-r8/run/run.sh`  
**Lines:** 9-18

```bash
# Source the reservation access control library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")/lib"
source "${LIB_DIR}/reservation_access_control.sh"

# Validate reservation access before submitting jobs
if ! validate_reservation_access "$reservation"; then
    echo "ERROR: Cannot submit jobs due to reservation access restrictions." >&2
    exit 1
fi
```

### 2. NCCL Tests (Single Node)
**File:** `nccl-tests/run/run.sh`  
**Lines:** 9-18

### 3. NCCL Tests (Two Nodes)
**File:** `nccl-tests/run/run-2node.sh`  
**Lines:** 9-18

### 4. NVIDIA HPC Benchmarks
**File:** `nvidia-hpc-benchmarks/run/run.sh`  
**Lines:** 9-18

### 5. MPI Point-to-Point Tests
**File:** `mpi-p2p/run/run.sh`  
**Lines:** 6-15

### 6. MPI Pi Calculation
**File:** `mpi-calc-pi/run/run.sh`  
**Lines:** 7-16

### 7. OpenMP Tests
**File:** `openmp/run/run.sh`  
**Lines:** 6-15

## How It Prevents Unauthorized Access

### Step-by-Step Flow

1. **Script Execution**: User runs a benchmark script (e.g., `./gpu-burn-r8/run/run.sh`)
2. **Parameter Parsing**: Script receives reservation name as parameter
3. **Library Loading**: Script sources `/lib/reservation_access_control.sh`
4. **Validation Call**: Script calls `validate_reservation_access "$reservation")`
5. **SLURM Query**: Function queries SLURM for reservation details via `scontrol show reservation`
6. **Access Check**: Function checks if current user is in allowed users list
7. **Decision**:
   - ✓ **Allow**: If user has access, function returns 0 (success)
   - ✗ **Deny**: If user lacks access, function returns 1 (failure) with error message
8. **Script Action**:
   - ✓ **Allow**: Script continues and submits jobs with `sbatch`
   - ✗ **Deny**: Script exits immediately with error, **NO jobs are submitted**

### Example: Access Denied

```
ERROR: User 'operator1' does not have explicit access to reservation 'restricted_res'.
Reservation details:
  Allowed Users: admin,researcher1,researcher2
  Allowed Accounts: (not restricted by account)

Please request access to this reservation or use a different reservation.
ERROR: Cannot submit jobs due to reservation access restrictions.
```

The script **exits with status 1** and **NO jobs are submitted to SLURM**.

## Testing and Verification

### Test Scripts
- **Unit Tests**: `/test/test_reservation_access.sh`
- **Demonstration**: `/test/demo_reservation_access.sh`

### Documentation
- **Full Documentation**: `/docs/RESERVATION_ACCESS_CONTROL.md`
- **Quick Reference**: `README.md` (lines 3-20)

## Summary

**The access control is enforced at:**
1. **Primary Location**: `/lib/reservation_access_control.sh` (function `validate_reservation_access`)
2. **Integration Points**: All 7 job submission scripts listed above
3. **Enforcement Point**: Before any `sbatch` command is executed
4. **Effect**: Scripts exit immediately if user lacks reservation access

This ensures operators **cannot** launch jobs on reservations unless they have **explicit access** granted in SLURM.
