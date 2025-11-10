#!/bin/bash
#
# Reservation Access Control Library
# 
# This module provides functions to validate that a user has explicit access
# to a SLURM reservation before allowing job submission.
#
# Usage:
#   source /path/to/reservation_access_control.sh
#   validate_reservation_access "reservation_name"
#

# Check if a reservation exists and if the current user has access to it
# Arguments:
#   $1 - reservation name
# Returns:
#   0 if user has access (or reservation is empty/none)
#   1 if user does not have access or validation fails
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

# Export the function so it's available to scripts that source this file
export -f validate_reservation_access
