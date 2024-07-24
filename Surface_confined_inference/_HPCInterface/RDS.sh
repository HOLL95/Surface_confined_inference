#!/usr/bin/env bash

output=$(flight desktop start gnome)
# Extract the full text of the Identity and Password lines
identity_line=$(echo "$output" | grep "Identity")
password_line=$(echo "$output" | grep "Password")

# Extract just the Identity value and its first 5 characters
identity=$(echo "$identity_line" | awk '{print $2}' | cut -d "-" -f1)

# Extract just the Password value
password=$(echo "$password_line" | awk '{print $2}')

# Print the results
hostip=$(echo "$output" | grep "Host IP")
extracted_ip=$(echo "$hostip" | awk '{print $3}')
extracted_port=$(echo "$output" |  awk '/Port/ && !/WebSocket/ {print $2; exit}')
sshscript="ssh -L ${extracted_port}:${extracted_ip}:${extracted_port} ${USER}@${extracted_ip}"
echo
echo "Open a new terminal on your LOCAL machine, and run"
echo $sshscript
echo
echo "Open a second new terminal on your LOCAL machine and run (depending on OS)"
echo
echo "On Linux"
echo "remmina vnc://${USER}:${password}@localhost:${extracted_port}"
echo
echo "IMPORTANT: when you are done, run"
echo  "flight desktop kill ${identity}"


