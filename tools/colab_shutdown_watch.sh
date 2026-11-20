#!/bin/bash
# Release the Colab VM once training is no longer running (finished or crashed), to stop using compute.
#   nohup bash tools/colab_shutdown_watch.sh <run_dir> <train_log> > /content/shutdown_watch.log 2>&1 &
RUN=${1:-/content/drive/MyDrive/BitRoss2/runs/bitross-small}
LOG=${2:-/content/train.log}
gone=0
while true; do
  if pgrep -f "bitross.train" > /dev/null || pgrep -f "after_label.sh" > /dev/null; then gone=0; else gone=$((gone+1)); fi
  [ $gone -ge 4 ] && break          # absent for 2 minutes in a row
  sleep 30
done
echo "training stopped at $(date); waiting 5 min for Drive to sync"
sleep 300
cp "$LOG" "$RUN/train_console.log" 2>/dev/null
echo "$(date): training process ended, releasing the runtime" > "$RUN/SHUTDOWN.txt"
tail -n 5 "$LOG" >> "$RUN/SHUTDOWN.txt"
sync; sleep 60
curl -s -X POST "http://${TBE_RUNTIME_ADDR}/unassign" -o /dev/null -w "unassign http %{http_code}\n"
