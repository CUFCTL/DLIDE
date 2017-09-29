#!/bin/bash
# Usage parse_log.sh caffe.log

# get the dirname of the script
DIR="$( cd "$(dirname "$0")" ; pwd -P )"

if [ "$#" -lt 1 ]; then
	echo "Usage parse_log.sh /path/to/your.log"
	exit
fi

LOG=`basename $1`
LOGDIR="$( cd "$(dirname "$1")" ; pwd -P)"
BASE="$LOGDIR/$LOG"

# extract relevant information from log
sed -n '/Iteration .* Testing net/,/Iteration *. loss/p' $1 > aux.txt
grep 'Test' aux.txt > aux_testing.txt
grep 'Train\|Iteration' aux.txt > aux_training.txt
sed -i '/Test/d' aux_training.txt

# parse training log
grep 'lr' aux_training.txt | sed  's/.*Iteration \([[:digit:]]*\).*/\1/' > aux_iter.txt
grep 'lr' aux_training.txt | awk '{print $9}' > aux_lr.txt

grep 'Train net output #0:' aux_training.txt | awk '{print $11}' > aux_accuracy.txt
grep 'Train net output #1:' aux_training.txt | awk '{print $11}' > aux_loss.txt
grep 'Train net output #2:' aux_training.txt | awk '{print $11}' > aux_road.txt
grep 'Train net output #3:' aux_training.txt | awk '{print $11}' > aux_sidewalk.txt
grep 'Train net output #4:' aux_training.txt | awk '{print $11}' > aux_building.txt
grep 'Train net output #5:' aux_training.txt | awk '{print $11}' > aux_wall.txt
grep 'Train net output #6:' aux_training.txt | awk '{print $11}' > aux_fence.txt
grep 'Train net output #7:' aux_training.txt | awk '{print $11}' > aux_pole.txt
grep 'Train net output #8:' aux_training.txt | awk '{print $11}' > aux_traffic_light.txt
grep 'Train net output #9:' aux_training.txt | awk '{print $11}' > aux_traffic_sign.txt
grep 'Train net output #10:' aux_training.txt | awk '{print $11}' > aux_vegetation.txt
grep 'Train net output #11:' aux_training.txt | awk '{print $11}' > aux_terrain.txt
grep 'Train net output #12:' aux_training.txt | awk '{print $11}' > aux_sky.txt
grep 'Train net output #13:' aux_training.txt | awk '{print $11}' > aux_person.txt
grep 'Train net output #14:' aux_training.txt | awk '{print $11}' > aux_rider.txt
grep 'Train net output #15:' aux_training.txt | awk '{print $11}' > aux_car.txt
grep 'Train net output #16:' aux_training.txt | awk '{print $11}' > aux_truck.txt
grep 'Train net output #17:' aux_training.txt | awk '{print $11}' > aux_bus.txt
grep 'Train net output #18:' aux_training.txt | awk '{print $11}' > aux_train.txt
grep 'Train net output #19:' aux_training.txt | awk '{print $11}' > aux_motorcycle.txt
grep 'Train net output #20:' aux_training.txt | awk '{print $11}' > aux_bicycle.txt

# format train information
echo '#Iters LearningRate Accuracy Loss Road Sidewalk Building Wall Fence Pole TrafficLight TrafficSign Vegetation Terrain Sky Person Rider Car Truck Bus Train Motorcycle Bicycle' > $BASE.train
paste aux_iter.txt aux_lr.txt aux_accuracy.txt aux_loss.txt aux_road.txt aux_sidewalk.txt aux_building.txt aux_wall.txt aux_fence.txt aux_pole.txt aux_traffic_light.txt aux_traffic_sign.txt aux_vegetation.txt aux_terrain.txt aux_sky.txt aux_person.txt aux_rider.txt aux_car.txt aux_truck.txt aux_bus.txt aux_train.txt aux_motorcycle.txt aux_bicycle.txt | column -t >> $BASE.train
rm aux_lr.txt aux_training.txt aux_accuracy.txt aux_loss.txt aux_road.txt aux_sidewalk.txt aux_building.txt aux_wall.txt aux_fence.txt aux_pole.txt aux_traffic_light.txt aux_traffic_sign.txt aux_vegetation.txt aux_terrain.txt aux_sky.txt aux_person.txt aux_rider.txt aux_car.txt aux_truck.txt aux_bus.txt aux_train.txt aux_motorcycle.txt aux_bicycle.txt

# parse testing log
grep 'Iteration ' aux_testing.txt | sed  's/.*Iteration \([[:digit:]]*\).*/\1/' > aux_iter.txt
grep 'Test net output #0:' aux_testing.txt | awk '{print $11}' > aux_accuracy.txt
grep 'Test net output #1:' aux_testing.txt | awk '{print $11}' > aux_loss.txt
grep 'Test net output #2:' aux_testing.txt | awk '{print $11}' > aux_road.txt
grep 'Test net output #3:' aux_testing.txt | awk '{print $11}' > aux_sidewalk.txt
grep 'Test net output #4:' aux_testing.txt | awk '{print $11}' > aux_building.txt
grep 'Test net output #5:' aux_testing.txt | awk '{print $11}' > aux_wall.txt
grep 'Test net output #6:' aux_testing.txt | awk '{print $11}' > aux_fence.txt
grep 'Test net output #7:' aux_testing.txt | awk '{print $11}' > aux_pole.txt
grep 'Test net output #8:' aux_testing.txt | awk '{print $11}' > aux_traffic_light.txt
grep 'Test net output #9:' aux_testing.txt | awk '{print $11}' > aux_traffic_sign.txt
grep 'Test net output #10:' aux_testing.txt | awk '{print $11}' > aux_vegetation.txt
grep 'Test net output #11:' aux_testing.txt | awk '{print $11}' > aux_terrain.txt
grep 'Test net output #12:' aux_testing.txt | awk '{print $11}' > aux_sky.txt
grep 'Test net output #13:' aux_testing.txt | awk '{print $11}' > aux_person.txt
grep 'Test net output #14:' aux_testing.txt | awk '{print $11}' > aux_rider.txt
grep 'Test net output #15:' aux_testing.txt | awk '{print $11}' > aux_car.txt
grep 'Test net output #16:' aux_testing.txt | awk '{print $11}' > aux_truck.txt
grep 'Test net output #17:' aux_testing.txt | awk '{print $11}' > aux_bus.txt
grep 'Test net output #18:' aux_testing.txt | awk '{print $11}' > aux_train.txt
grep 'Test net output #19:' aux_testing.txt | awk '{print $11}' > aux_motorcycle.txt
grep 'Test net output #20:' aux_testing.txt | awk '{print $11}' > aux_bicycle.txt

# format test information
echo '#Iters Accuracy Loss Road Sidewalk Building Wall Fence Pole TrafficLight TrafficSign Vegetation Terrain Sky Person Rider Car Truck Bus Train Motorcycle Bicycle' > $BASE.test
paste aux_iter.txt aux_accuracy.txt aux_loss.txt aux_road.txt aux_sidewalk.txt aux_building.txt aux_wall.txt aux_fence.txt aux_pole.txt aux_traffic_light.txt aux_traffic_sign.txt aux_vegetation.txt aux_terrain.txt aux_sky.txt aux_person.txt aux_rider.txt aux_car.txt aux_truck.txt aux_bus.txt aux_train.txt aux_motorcycle.txt aux_bicycle.txt | column -t >> $BASE.test
rm aux.txt aux_testing.txt aux_iter.txt aux_accuracy.txt aux_loss.txt aux_road.txt aux_sidewalk.txt aux_building.txt aux_wall.txt aux_fence.txt aux_pole.txt aux_traffic_light.txt aux_traffic_sign.txt aux_vegetation.txt aux_terrain.txt aux_sky.txt aux_person.txt aux_rider.txt aux_car.txt aux_truck.txt aux_bus.txt aux_train.txt aux_motorcycle.txt aux_bicycle.txt