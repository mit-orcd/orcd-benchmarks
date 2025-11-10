Benchmark programs for ORCD computing clusters. 

## Reservation Access Control

This repository implements explicit access control for SLURM reservations. Before submitting jobs to reservations, the system validates that the operator has explicit access.

For detailed information about the reservation access control mechanism, see [docs/RESERVATION_ACCESS_CONTROL.md](docs/RESERVATION_ACCESS_CONTROL.md).

### Quick Summary

- **What**: Validates user access to SLURM reservations before job submission
- **Where**: Integrated into all `run.sh` scripts across benchmark directories
- **Why**: Prevents unauthorized use of reserved compute resources
- **How**: Uses `validate_reservation_access()` function from `/lib/reservation_access_control.sh`

Key locations:
- Access control library: `/lib/reservation_access_control.sh`
- Documentation: `/docs/RESERVATION_ACCESS_CONTROL.md`
- Test script: `/test/test_reservation_access.sh`

