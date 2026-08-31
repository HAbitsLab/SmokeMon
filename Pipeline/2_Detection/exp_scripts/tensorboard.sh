#!/bin/bash

session="tensorboard"


tmux new-session -d -s $session

window=0
tmux rename-window -t $session:$window 'experiments'
tmux send-keys -t $session:$window 'cd ~/Projects/SmokeMon/git-repo/thermo-smoking/Pipeline/2_Detection/saved/log' C-m
tmux send-keys -t $session:$window 'tensorboard --logdir ./' C-m
tmux attach-session -t $session





