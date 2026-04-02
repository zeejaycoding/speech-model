release: bash build.sh
web: gunicorn -w 1 -b 0.0.0.0:$PORT --timeout 120 --access-logfile - api:app
