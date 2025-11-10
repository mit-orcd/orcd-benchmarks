#!/bin/bash
#
# Demonstration of Reservation Access Control
#
# This script demonstrates what happens when a user tries to submit
# a job to a reservation they don't have access to.
#

echo "=========================================="
echo "Reservation Access Control Demonstration"
echo "=========================================="
echo ""

# Mock scontrol command for demonstration purposes
mock_scontrol() {
    local cmd="$1"
    local reservation="$2"
    
    if [ "$cmd" = "show" ] && [[ "$reservation" =~ reservation= ]]; then
        local res_name=$(echo "$reservation" | sed 's/reservation=//')
        
        case "$res_name" in
            "orcd_testing")
                # Simulate a restricted reservation
                cat << EOF
ReservationName=orcd_testing StartTime=2024-01-01T00:00:00 EndTime=2024-12-31T23:59:59 Duration=365-00:00:00
   Nodes=node[4100-4199] NodeCnt=100 CoreCnt=4800 Features=(null) PartitionName=mit_normal_gpu Flags=
   TRES=cpu=4800,mem=9600G,node=100,billing=4800
   Users=alice,bob,charlie Accounts=(null) Licenses=(null) State=ACTIVE BurstBuffer=(null) Watts=n/a
   MaxStartDelay=(null)
EOF
                ;;
            "nonexistent_reservation_12345")
                echo "Reservation nonexistent_reservation_12345 not found" >&2
                return 1
                ;;
            *)
                cat << EOF
ReservationName=$res_name StartTime=2024-01-01T00:00:00 EndTime=2024-12-31T23:59:59 Duration=365-00:00:00
   Nodes=ALL NodeCnt=1000 CoreCnt=48000 Features=(null) PartitionName=(null) Flags=
   TRES=cpu=48000
   Users=(null) Accounts=(null) Licenses=(null) State=ACTIVE BurstBuffer=(null) Watts=n/a
   MaxStartDelay=(null)
EOF
                ;;
        esac
        return 0
    fi
}

# Export mock function
export -f mock_scontrol

echo "Scenario 1: User 'runner' tries to access unrestricted reservation"
echo "-------------------------------------------------------------------"
cat << 'EOF' | bash
source /home/runner/work/orcd-benchmarks/orcd-benchmarks/lib/reservation_access_control.sh
# Override scontrol with mock
scontrol() { mock_scontrol "$@"; }
export -f scontrol
if validate_reservation_access "public_reservation"; then
    echo "✓ SUCCESS: User has access"
else
    echo "✗ DENIED: User does not have access"
fi
EOF
echo ""

echo "Scenario 2: User 'runner' tries to access restricted reservation 'orcd_testing'"
echo "--------------------------------------------------------------------------------"
echo "(Allowed users: alice, bob, charlie)"
cat << 'EOF' | bash
source /home/runner/work/orcd-benchmarks/orcd-benchmarks/lib/reservation_access_control.sh
# Override scontrol with mock
scontrol() { mock_scontrol "$@"; }
export -f scontrol
if validate_reservation_access "orcd_testing"; then
    echo "✓ SUCCESS: User has access"
else
    echo "✗ DENIED: User does not have access"
fi
EOF
echo ""

echo "Scenario 3: User tries to access non-existent reservation"
echo "----------------------------------------------------------"
cat << 'EOF' | bash
source /home/runner/work/orcd-benchmarks/orcd-benchmarks/lib/reservation_access_control.sh
# Override scontrol with mock
scontrol() { mock_scontrol "$@"; }
export -f scontrol
if validate_reservation_access "nonexistent_reservation_12345"; then
    echo "✓ SUCCESS: User has access"
else
    echo "✗ DENIED: User does not have access"
fi
EOF
echo ""

echo "=========================================="
echo "Demonstration Complete"
echo "=========================================="
echo ""
echo "Summary:"
echo "--------"
echo "1. Unrestricted reservations: Allow all users"
echo "2. Restricted reservations: Only allow explicitly listed users"
echo "3. Non-existent reservations: Deny access with clear error message"
echo ""
echo "This prevents operators from launching jobs on reservations"
echo "unless they have explicit access."
