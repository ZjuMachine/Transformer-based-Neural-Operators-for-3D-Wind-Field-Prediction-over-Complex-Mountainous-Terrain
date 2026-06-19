# train model
bash scripts/Patch_solver.sh

# infer model

bash scripts/Evaluation_valnew.sh

# infer model on the zero shot task

bash scripts/zero_patchsolver.sh
