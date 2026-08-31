#!/bin/bash

session="train"

tmux new-session -d -s $session

window=0
tmux rename-window -t $session:$window 'p11'
tmux send-keys -t $session:$window 'cd ~/Projects/SmokeMon/git-repo/thermo-smoking/Pipeline/2_Detection' C-m
tmux send-keys -t $session:$window 'CUDA_VISIBLE_DEVICES=0 python3.8 train.py --config ./configs/mobilenet_v2_3d/p11.json' C-m

#window=1
#tmux new-window -t $session:$window -n 'p7'
#tmux send-keys -t $session:$window 'cd ~/Projects/SmokeMon/git-repo/thermo-smoking/Pipeline/2_Detection' C-m
#tmux send-keys -t $session:$window 'CUDA_VISIBLE_DEVICES=1 python3.8 train.py --config ./configs/mobilenet_v2_3d/p7.json' C-m
#
#window=2
#tmux new-window -t $session:$window -n 'p8'
#tmux send-keys -t $session:$window 'cd ~/Projects/SmokeMon/git-repo/thermo-smoking/Pipeline/2_Detection' C-m
#tmux send-keys -t $session:$window 'CUDA_VISIBLE_DEVICES=2 python3.8 train.py --config ./configs/mobilenet_v2_3d/p8.json' C-m
#
#window=3
#tmux new-window -t $session:$window -n 'p9'
#tmux send-keys -t $session:$window 'cd ~/Projects/SmokeMon/git-repo/thermo-smoking/Pipeline/2_Detection' C-m
#tmux send-keys -t $session:$window 'CUDA_VISIBLE_DEVICES=3 python3.8 train.py --config ./configs/mobilenet_v2_3d/p9.json' C-m
#
#window=4
#tmux new-window -t $session:$window -n 'p10'
#tmux send-keys -t $session:$window 'cd ~/Projects/SmokeMon/git-repo/thermo-smoking/Pipeline/2_Detection' C-m
#tmux send-keys -t $session:$window 'CUDA_VISIBLE_DEVICES=2 python3.8 train.py --config ./configs/mobilenet_v2_3d/p10.json' C-m
#
#window=5
#tmux new-window -t $session:$window -n 'p11'
#tmux send-keys -t $session:$window 'cd ~/Projects/SmokeMon/git-repo/thermo-smoking/Pipeline/2_Detection' C-m
#tmux send-keys -t $session:$window 'CUDA_VISIBLE_DEVICES=3 python3.8 train.py --config ./configs/mobilenet_v2_3d/p11.json' C-m
#
#window=6
#tmux new-window -t $session:$window -n 'p5'
#tmux send-keys -t $session:$window 'cd ~/Projects/SmokeMon/git-repo/thermo-smoking/Pipeline/2_Detection' C-m
#tmux send-keys -t $session:$window 'CUDA_VISIBLE_DEVICES=0 python3.8 train.py --config ./configs/mobilenet_v2_3d/p5.json' C-m
#
#window=7
#tmux new-window -t $session:$window -n 'p1'
#tmux send-keys -t $session:$window 'cd ~/Projects/SmokeMon/git-repo/thermo-smoking/Pipeline/2_Detection' C-m
#tmux send-keys -t $session:$window 'CUDA_VISIBLE_DEVICES=1 python3.8 train.py --config ./configs/mobilenet_v2_3d/p1.json' C-m


tmux attach-session -t $session






#tmux new-session -d -s $session
#
#window=0
#tmux rename-window -t $session:$window 'p6'
#tmux send-keys -t $session:$window 'cd ~/Projects/SmokeMon/git-repo/thermo-smoking/Pipeline/2_Detection' C-m
#tmux send-keys -t $session:$window 'CUDA_VISIBLE_DEVICES=0 python3.8 train.py --config ./configs/mnist_temporal/p6.json' C-m
#
#window=1
#tmux new-window -t $session:$window -n 'p7'
#tmux send-keys -t $session:$window 'cd ~/Projects/SmokeMon/git-repo/thermo-smoking/Pipeline/2_Detection' C-m
#tmux send-keys -t $session:$window 'CUDA_VISIBLE_DEVICES=1 python3.8 train.py --config ./configs/mnist_temporal/p7.json' C-m
#
#window=2
#tmux new-window -t $session:$window -n 'p8'
#tmux send-keys -t $session:$window 'cd ~/Projects/SmokeMon/git-repo/thermo-smoking/Pipeline/2_Detection' C-m
#tmux send-keys -t $session:$window 'CUDA_VISIBLE_DEVICES=2 python3.8 train.py --config ./configs/mnist_temporal/p8.json' C-m
#
#window=3
#tmux new-window -t $session:$window -n 'p9'
#tmux send-keys -t $session:$window 'cd ~/Projects/SmokeMon/git-repo/thermo-smoking/Pipeline/2_Detection' C-m
#tmux send-keys -t $session:$window 'CUDA_VISIBLE_DEVICES=3 python3.8 train.py --config ./configs/mnist_temporal/p9.json' C-m
#
#window=4
#tmux new-window -t $session:$window -n 'p10'
#tmux send-keys -t $session:$window 'cd ~/Projects/SmokeMon/git-repo/thermo-smoking/Pipeline/2_Detection' C-m
#tmux send-keys -t $session:$window 'CUDA_VISIBLE_DEVICES=2 python3.8 train.py --config ./configs/mnist_temporal/p10.json' C-m
#
#window=5
#tmux new-window -t $session:$window -n 'p11'
#tmux send-keys -t $session:$window 'cd ~/Projects/SmokeMon/git-repo/thermo-smoking/Pipeline/2_Detection' C-m
#tmux send-keys -t $session:$window 'CUDA_VISIBLE_DEVICES=3 python3.8 train.py --config ./configs/mnist_temporal/p11.json' C-m
#
#
#tmux attach-session -t $session
#
#
#


