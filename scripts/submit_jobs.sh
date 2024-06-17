#!/bin/bash

# Read the number of sessions from jobs.txt
num_sessions=$(wc -l < jobs1.txt)

# Loop to create tmux sessions and run jobs
for ((i=0; i<$num_sessions; i++)); do
    # Create a new tmux session
    tmux new-session -d -s "session_$i"

    # Run the job command for the session
    job_command=$(sed -n "$((i+1))p" jobs1.txt)
    tmux send-keys -t "session_$i" "$job_command" C-m
done