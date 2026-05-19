#!/usr/bin/env sh

# Start FastAPI (bind locally only)
uvicorn api:app --host 127.0.0.1 --port 8000 &

# Start Nginx (expects nginx installed and nginx.conf present)
# -g 'daemon off;' keeps Nginx in foreground so the script doesn't exit.
nginx -c "$(pwd)/nginx.conf" -g 'daemon off;'

