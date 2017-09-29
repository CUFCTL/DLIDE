#!/bin/bash
# Usage training_info.sh caffe.log

if [ "$#" -lt 1 ]; then
	echo "Usage ./training_info.sh /path/to/your.log"
	exit
fi

cityscapes=2975

# calculate total memory required
grep 'GPU ' $1 | sed 's:.*] ::'
grep 'Memory' $1 | awk '{print $9}' > memory.txt
awk '{s+=$1} END {printf "Memory required: %g B\n", s}' memory.txt
rm memory.txt

# seperate model from log
# cat $1 | sed -n '/Creating layer/,/Network initialization done/p' > model.txt

# calculate average itereation
# grep 'Iteration' $1 | grep 'loss = ' | awk '{printf "%s %s\n", $1, $2}' > aux_time.txt
# IFS='.I:'
# days=$(expr $(echo $end | awk '{print $1}') - $(echo $start | awk '{print $1}'))
# hours=$(expr $(echo $end | awk '{print $2}') + 24 \* $days - $(echo $start | awk '{print $2}'))
# minutes=$(expr $(echo $end | awk '{print $3}') + 60 \* $hours - $(echo $start | awk '{print $3}'))
# IFS=''

batch_size=$(grep 'batch_size' $1 | awk '{printf "%s\n", $2}' | head -1)
start=$(grep 'Iteration' $1 | grep 'loss = ' | awk '{printf "%s %s\n", $1, $2}' | head -1)
end=$(grep 'Iteration' $1 | grep 'loss = ' | awk '{printf "%s %s\n", $1, $2}' | tail -1)
iters=$(grep 'max_iter' $1 | awk '{print $2}')
epochs=$(expr $iters \* $batch_size / $cityscapes)

IFS='.I:'
days=$(expr $(echo $end | awk '{print $1}') - $(echo $start | awk '{print $1}'))
hours=$(expr $(echo $end | awk '{print $2}') + 24 \* $days - $(echo $start | awk '{print $2}'))
minutes=$(expr $(echo $end | awk '{print $3}') + 60 \* $hours - $(echo $start | awk '{print $3}'))
IFS=''

echo 'Start time:' $start
echo 'End time:  ' $end
echo 'Train time:' $(expr $hours / 24) 'D,' $(expr $hours % 24) 'H,' $(expr $minutes % 60) 'M' 
echo 'Iters:     ' $iters
echo 'Batch size:' $batch_size
echo 'Epochs:    ' $epochs
echo 'Images/sec:' $(expr $(expr $epochs \* $cityscapes) / $(expr $minutes \* 60))
