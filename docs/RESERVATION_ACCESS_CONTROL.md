# Reservation Access Control

## Overview

This repository implements explicit access control validation for SLURM reservations. Before submitting any job to a reservation, the system validates that the operator (user) has explicit access to that reservation.

## Purpose

The reservation access control mechanism prevents operators from launching jobs on reservations unless they have been explicitly granted access. This ensures:

1. **Security**: Only authorized users can utilize reserved resources
2. **Resource Management**: Prevents accidental or unauthorized use of reserved compute time
3. **Clear Error Messages**: Provides immediate feedback when access is denied
4. **Early Detection**: Catches access issues before job submission, saving time

## How It Works

### Access Control Library

The core access control logic is implemented in `/lib/reservation_access_control.sh`. This library provides a `validate_reservation_access()` function that:

1. **Checks reservation validity**: Verifies the reservation exists in SLURM
2. **Retrieves access lists**: Queries SLURM for the Users and Accounts allowed on the reservation
3. **Validates current user**: Checks if the current user (`$USER`) is in the allowed users list
4. **Returns appropriate status**: Returns 0 (success) if access is allowed, 1 (failure) if denied

### Integration Points

The validation is integrated into all job submission scripts in the repository:

- `gpu-burn-r8/run/run.sh`
- `nccl-tests/run/run.sh`
- `nccl-tests/run/run-2node.sh`
- `nvidia-hpc-benchmarks/run/run.sh`
- `mpi-p2p/run/run.sh`
- `mpi-calc-pi/run/run.sh`
- `openmp/run/run.sh`

Each script sources the access control library and calls `validate_reservation_access()` before submitting jobs.

### Code Example

Here's how the validation is integrated into a typical run.sh script:

```bash
#!/bin/bash
nodes=($1)
partition=$2
reservation=$3
qos=$4

# Source the reservation access control library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")/lib"
source "${LIB_DIR}/reservation_access_control.sh"

# Validate reservation access before submitting jobs
if ! validate_reservation_access "$reservation"; then
    echo "ERROR: Cannot submit jobs due to reservation access restrictions." >&2
    exit 1
fi

# Continue with job submission...
sbatch --reservation=$reservation ...
```

## Access Control Rules

The validation follows these rules:

1. **Empty or 'none' reservations**: Always allowed (no reservation specified)
2. **Non-existent reservations**: Access denied with error message
3. **Unrestricted reservations**: Allowed if both Users and Accounts fields are null/empty
4. **User-restricted reservations**: Allowed only if current user is in the Users list
5. **Account-restricted reservations**: Checked via Users field (primary mechanism)

## Error Messages

When access is denied, the system provides clear error messages:

```
ERROR: User 'username' does not have explicit access to reservation 'reservation_name'.
Reservation details:
  Allowed Users: user1,user2,user3
  Allowed Accounts: (not restricted by account)

Please request access to this reservation or use a different reservation.
ERROR: Cannot submit jobs due to reservation access restrictions.
```

## Special Cases

### scontrol Not Available

If the `scontrol` command is not available (e.g., on systems without SLURM installed), the validation:
- Prints a warning message
- Allows the job to proceed
- Relies on SLURM's native access control at submission time

This ensures the scripts work in development/testing environments without SLURM.

### Development and Testing

The validation gracefully handles environments where:
- SLURM is not installed
- Reservations don't exist yet
- Users are testing scripts outside the production cluster

## Testing

Two test scripts are provided to verify and demonstrate the validation logic:

### 1. Unit Test Script

`/test/test_reservation_access.sh` verifies the validation logic works correctly:

```bash
./test/test_reservation_access.sh
```

The test script validates:
- Empty reservation handling
- 'none' reservation handling
- Real reservation access (if SLURM available)
- Non-existent reservation handling

### 2. Demonstration Script

`/test/demo_reservation_access.sh` demonstrates the access control mechanism with realistic scenarios:

```bash
./test/demo_reservation_access.sh
```

The demo script shows:
- Unrestricted reservation access (allows all users)
- Restricted reservation access (only allows explicitly listed users)
- Non-existent reservation handling (clear error messages)

This helps users understand how the access control works in practice.

## Benefits

1. **Fail Fast**: Detects access issues immediately, before waiting in queue
2. **Clear Feedback**: Users know exactly why their job was rejected
3. **Audit Trail**: Access control decisions are logged in script output
4. **Consistent**: Same validation logic across all benchmarks
5. **Maintainable**: Centralized in one library file

## Security Model

This implementation provides a **defense-in-depth** approach:
- **First Layer** (this code): Pre-submission validation and clear error messages
- **Second Layer** (SLURM): Native access control at the scheduler level

Even if a user bypasses the script validation, SLURM's native access control will still enforce the reservation restrictions.

## Maintenance

When adding new benchmark scripts:

1. Source the library: `source "${LIB_DIR}/reservation_access_control.sh"`
2. Call validation: `validate_reservation_access "$reservation"`
3. Check return code and exit if validation fails

The centralized library ensures all scripts benefit from updates and bug fixes automatically.
