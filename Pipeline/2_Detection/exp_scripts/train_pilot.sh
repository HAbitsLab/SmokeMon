#!/bin/bash

session="train1"

tmux new-session -d -s $session

window=0
tmux rename-window -t $session:$window 'p0'
tmux send-keys -t $session:$window 'cd ~/Projects/SmokeMon/git-repo/thermo-smoking/Pipeline/2_Detection' C-m
tmux send-keys -t $session:$window 'CUDA_VISIBLE_DEVICES=0 python3.8 train.py --config ./configs/mobilenet_v2_3d/pilot/P0.json' C-m

window=1
tmux new-window -t $session:$window -n 'p1'
tmux send-keys -t $session:$window 'cd ~/Projects/SmokeMon/git-repo/thermo-smoking/Pipeline/2_Detection' C-m
tmux send-keys -t $session:$window 'CUDA_VISIBLE_DEVICES=1 python3.8 train.py --config ./configs/mobilenet_v2_3d/pilot/P1.json' C-m

window=2
tmux new-window -t $session:$window -n 'p2'
tmux send-keys -t $session:$window 'cd ~/Projects/SmokeMon/git-repo/thermo-smoking/Pipeline/2_Detection' C-m
tmux send-keys -t $session:$window 'CUDA_VISIBLE_DEVICES=2 python3.8 train.py --config ./configs/mobilenet_v2_3d/pilot/P2.json' C-m

window=3
tmux new-window -t $session:$window -n 'p3'
tmux send-keys -t $session:$window 'cd ~/Projects/SmokeMon/git-repo/thermo-smoking/Pipeline/2_Detection' C-m
tmux send-keys -t $session:$window 'CUDA_VISIBLE_DEVICES=3 python3.8 train.py --config ./configs/mobilenet_v2_3d/pilot/P3.json' C-m

window=4
tmux new-window -t $session:$window -n 'p4'
tmux send-keys -t $session:$window 'cd ~/Projects/SmokeMon/git-repo/thermo-smoking/Pipeline/2_Detection' C-m
tmux send-keys -t $session:$window 'CUDA_VISIBLE_DEVICES=0 python3.8 train.py --config ./configs/mobilenet_v2_3d/pilot/P4.json' C-m

window=5
tmux new-window -t $session:$window -n 'p5'
tmux send-keys -t $session:$window 'cd ~/Projects/SmokeMon/git-repo/thermo-smoking/Pipeline/2_Detection' C-m
tmux send-keys -t $session:$window 'CUDA_VISIBLE_DEVICES=1 python3.8 train.py --config ./configs/mobilenet_v2_3d/pilot/P5.json' C-m

window=6
tmux new-window -t $session:$window -n 'p6'
tmux send-keys -t $session:$window 'cd ~/Projects/SmokeMon/git-repo/thermo-smoking/Pipeline/2_Detection' C-m
tmux send-keys -t $session:$window 'CUDA_VISIBLE_DEVICES=2 python3.8 train.py --config ./configs/mobilenet_v2_3d/pilot/P6.json' C-m

window=7
tmux new-window -t $session:$window -n 'p7'
tmux send-keys -t $session:$window 'cd ~/Projects/SmokeMon/git-repo/thermo-smoking/Pipeline/2_Detection' C-m
tmux send-keys -t $session:$window 'CUDA_VISIBLE_DEVICES=3 python3.8 train.py --config ./configs/mobilenet_v2_3d/pilot/P7.json' C-m


tmux attach-session -t $session



