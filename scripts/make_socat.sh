#!/usr/bin/env bash
set -euo pipefail
rm -f /tmp/ttyV0 /tmp/ttyV1
socat -d -d pty,raw,echo=0,link=/tmp/ttyV0 pty,raw,echo=0,link=/tmp/ttyV1
