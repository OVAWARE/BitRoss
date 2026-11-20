#!/bin/bash
# After training: render a showcase of the final model to Drive, then release the Colab VM.
#   setsid nohup bash tools/colab_finish.sh <run_dir> <train_log> > /content/finish.log 2>&1 < /dev/null &
RUN=${1:-/content/drive/MyDrive/BitRoss2/runs/bitross-small}
LOG=${2:-/content/train.log}
cd "$(dirname "$0")/.."
gone=0
while true; do
  if pgrep -f "[b]itross[.]train" > /dev/null; then gone=0; else gone=$((gone+1)); fi
  [ $gone -ge 4 ] && break          # training absent for 2 minutes in a row
  sleep 30
done
echo "training stopped at $(date)"
if grep -q "^done\." "$LOG"; then
  echo "rendering showcase"
  timeout 1200 python -m bitross.showcase --model "$RUN/bitross.pt" --out "$RUN/showcase" 2>&1 | grep -vE "Warning|warn|Loading weights"
fi
cp "$LOG" "$RUN/train_console.log" 2>/dev/null
echo "$(date): training process ended, releasing the runtime" > "$RUN/SHUTDOWN.txt"
tail -n 5 "$LOG" >> "$RUN/SHUTDOWN.txt"
echo "waiting 4 min for Drive to sync"
sync; sleep 240
curl -s -X POST "http://${TBE_RUNTIME_ADDR}/unassign" -o /dev/null -w "unassign http %{http_code}\n"
