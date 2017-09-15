#!/bin/bash

if [ $# -eq 0 ]; then
    LOG=$(ls -1 logs/*.json | peco)
else
    LOG=$1
fi

TITLE=$(head -1 $LOG | jq -r '._info._commandline | "\(.labels) / \(.unlabels)"')
TMP=`mktemp`
tail -n +2 $LOG > $TMP
echo $TITLE $TMP
visplot --smoothing 2 -x epoch -y val_clf_loss,val_clf_acc --title "$TITLE" "$TMP"
rm $TMP
