#!/bin/bash
#
# Test script for reservation access control validation
#
# This script demonstrates how the reservation access control works
# and can be used to test the validation function.
#

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="$(dirname "$SCRIPT_DIR")/lib"

# Source the reservation access control library
source "${LIB_DIR}/reservation_access_control.sh"

echo "=========================================="
echo "Reservation Access Control Test"
echo "=========================================="
echo ""
echo "Current user: ${USER}"
echo ""

# Test cases
test_reservation_access() {
    local reservation="$1"
    local description="$2"
    
    echo "Test: ${description}"
    echo "Reservation: ${reservation}"
    
    if validate_reservation_access "$reservation"; then
        echo "✓ PASS: User has access to reservation '${reservation}'"
    else
        echo "✗ FAIL: User does NOT have access to reservation '${reservation}'"
    fi
    echo ""
}

# Test Case 1: Empty reservation (should pass)
test_reservation_access "" "Empty reservation name"

# Test Case 2: 'none' reservation (should pass)
test_reservation_access "none" "Reservation name 'none'"

# Test Case 3: Real reservation (requires scontrol and actual reservation)
if command -v scontrol &> /dev/null; then
    # Get first available reservation
    first_reservation=$(scontrol show reservation | grep -oP '(?<=ReservationName=)[^ ]+' | head -1)
    
    if [ -n "$first_reservation" ]; then
        test_reservation_access "$first_reservation" "Real reservation: ${first_reservation}"
    else
        echo "Test: Real reservation"
        echo "Note: No reservations found in SLURM. Skipping this test."
        echo ""
    fi
else
    echo "Test: Real reservation"
    echo "Note: scontrol not available. Skipping reservation validation tests."
    echo ""
fi

# Test Case 4: Non-existent reservation (should fail if scontrol available)
if command -v scontrol &> /dev/null; then
    test_reservation_access "nonexistent_reservation_12345" "Non-existent reservation"
fi

echo "=========================================="
echo "Test Complete"
echo "=========================================="
