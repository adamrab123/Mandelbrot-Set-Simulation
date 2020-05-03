#!/bin/sh

pwd

# remove log files
rm log_*
rm -rf images
mkdir images
rm -rf output

# --------- STRONG SCALING 1 ---------

echo "Starting Strong Scaling 1"

num_ranks_total=(1 2 4 8 16 32 64)
block_size=(16 32 64 128 256 512 1024)

for i in ${!num_ranks_total[@]};
do
	num_rank=${num_ranks_total[$i]}
  	block=${block_size[$i]}
  	
  	num_gpus=1

  	num_nodes=1
	ranks_per_node=$num_rank

	if [ $num_ranks_total -eq 64 ]
	then
	    ranks_per_node=32
	    num_nodes=2
	fi

	
	# creating the directory name
	step_size=0.001
	iterations=100
	file=${step_size}_${iterations}_${num_rank}_${block}
	time_dir="output/strong1/$file/"

	# echo $time_dir

	mkdir -p "output"
	mkdir -p "output/strong1"
	mkdir -p $time_dir
	

	# echo $ranks_per_node

  	sbatch --nodes=$num_nodes --ntasks-per-node=$ranks_per_node --gres=gpu:$num_gpus "sim/slurmTest.sh" \
  		--block-size=$block_size --step-size=$step_size --iterations=$iterations --time-dir=$time_dir --output-file="images/$file.bmp"
done

# --------- STRONG SCALING 2 ---------

echo "Starting Strong Scaling 1"

num_ranks_total=(1 2 4 8 16 32 64)
block_size=(16 32 64 128 256 512 1024)

for i in ${!num_ranks_total[@]};
do
	num_rank=${num_ranks_total[$i]}
  	block=${block_size[$i]}
  	
  	num_gpus=1

  	num_nodes=1
	ranks_per_node=$num_rank

	if [ $num_ranks_total -eq 64 ]
	then
	    ranks_per_node=32
	    num_nodes=2
	fi

	
	# creating the directory name
	step_size=0.0001
	iterations=100
	file=${step_size}_${iterations}_${num_rank}_${block}
	time_dir="output/strong2/$file/"

	# echo $time_dir

	mkdir -p "output"
	mkdir -p "output/strong2"
	mkdir -p $time_dir
	

	# echo $ranks_per_node

  	sbatch --nodes=$num_nodes --ntasks-per-node=$ranks_per_node --gres=gpu:$num_gpus slurmTest.sh \
  		--block-size=$block_size --step-size=$step_size --iterations=$iterations --time-dir=$time_dir --output-file="images/$file.bmp"
done

# --------- WEAK SCALING 2 ---------

echo "Starting Weak Scaling 1"

num_ranks_total=(1 2 4 8 16 32 64)
block_size=(16 32 64 128 256 512 1024)
step_size=(0.032 0.016 0.008 0.004 0.002 0.001 0.0005)

for i in ${!num_ranks_total[@]};
do
	num_rank=${num_ranks_total[$i]}
  	block=${block_size[$i]}
  	step_size=${step_size[$i]}
  	
  	num_gpus=1

  	num_nodes=1
	ranks_per_node=$num_rank

	if [ $num_ranks_total -eq 64 ]
	then
	    ranks_per_node=32
	    num_nodes=2
	fi

	
	# creating the directory name
	
	iterations=100
	file=${step_size}_${iterations}_${num_rank}_${block}
	time_dir="output/weak/$file/"

	# echo $time_dir

	mkdir -p "output"
	mkdir -p "output/weak"
	mkdir -p $time_dir
	

	# echo $ranks_per_node

  	sbatch --nodes=$num_nodes --ntasks-per-node=$ranks_per_node --gres=gpu:$num_gpus slurmTest.sh \
  		--block-size=$block_size --step-size=$step_size --iterations=$iterations --time-dir=$time_dir --output-file="images/$file.bmp"
done