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
echo "Open a new terminal/powershell on your LOCAL machine, and run"
echo $sshscript
echo
echo "If this command just hangs indefintiely, you may need to be on the York VPN"
echo
echo
echo "On your local Linux machine, enter in a new terminal:"
echo
echo "remmina vnc://${USER}:${password}@localhost:${extracted_port}"
echo
echo
echo "On your local Windows machine, you need to download a VNC client (the unversity recommends TightVNC). Open the program and enter:"
echo
echo "localhost:${extracted_port}"
echo
echo "and give the password: ${password}"
echo
echo
echo "Mac has not been tested, but you can try (under Finder -> Go -> Connect to server)"
echo
echo "vnc://localhost:${extracted_port}"
echo
echo "and use the password ${password}"
echo
echo
echo "IMPORTANT: when you are done on a REMOTE (i.e. in Viking) machine, run:"
echo
echo  "flight desktop kill ${identity}"


