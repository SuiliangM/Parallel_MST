BSIZE=256
BCOUNT=8
INPUT="../data/graph_1.txt"
OUTPUT="output.txt"
METHOD="tpe"
USEMEM="no"

ARGS="--input $INPUT --bsize $BSIZE --bcount $BCOUNT --output $OUTPUT --method $METHOD --usemem $USEMEM"

./sssp $ARGS
