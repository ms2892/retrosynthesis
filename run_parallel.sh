sess1='sess1'
sess2='sess2'
sess3='sess3'
sess4='sess4'
sess5='sess5'
sess6='sess6'

tmux new-session -d -s "$sess1"
tmux send-keys -t "$sess1" "export CUDA_VISIBLE_DEVICES=''" Enter
tmux send-keys -t "$sess1" "export OMP_NUM_THREADS='3'" Enter
tmux send-keys -t "$sess1" "python compare_heuristic_value_functions_rand.py --smiles_file guacamol_250_chunk1.smiles" Enter

tmux new-session -d -s "$sess2"
tmux send-keys -t "$sess2" "export CUDA_VISIBLE_DEVICES=''" Enter
tmux send-keys -t "$sess2" "export OMP_NUM_THREADS='3'" Enter
tmux send-keys -t "$sess2" "python compare_heuristic_value_functions_rand.py --smiles_file guacamol_250_chunk2.smiles" Enter

tmux new-session -d -s "$sess3"
tmux send-keys -t "$sess3" "export CUDA_VISIBLE_DEVICES=''" Enter
tmux send-keys -t "$sess3" "export OMP_NUM_THREADS='3'" Enter
tmux send-keys -t "$sess3" "python compare_heuristic_value_functions_rand.py --smiles_file guacamol_250_chunk3.smiles" Enter

tmux new-session -d -s "$sess4"
tmux send-keys -t "$sess4" "export CUDA_VISIBLE_DEVICES=''" Enter
tmux send-keys -t "$sess4" "export OMP_NUM_THREADS='3'" Enter
tmux send-keys -t "$sess4" "python compare_heuristic_value_functions_rand_second.py --smiles_file guacamol_250_chunk1.smiles" Enter

tmux new-session -d -s "$sess5"
tmux send-keys -t "$sess5" "export CUDA_VISIBLE_DEVICES=''" Enter
tmux send-keys -t "$sess5" "export OMP_NUM_THREADS='3'" Enter
tmux send-keys -t "$sess5" "python compare_heuristic_value_functions_rand_second.py --smiles_file guacamol_250_chunk2.smiles" Enter

tmux new-session -d -s "$sess6"
tmux send-keys -t "$sess6" "export CUDA_VISIBLE_DEVICES=''" Enter
tmux send-keys -t "$sess6" "export OMP_NUM_THREADS='3'" Enter
tmux send-keys -t "$sess6" "python compare_heuristic_value_functions_rand_second.py --smiles_file guacamol_250_chunk3.smiles" Enter