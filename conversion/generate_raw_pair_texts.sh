for worker in {0..9}; do
    OMP_NUM_THREADS=1 python conversion/generate_comb_dist_data.py $1 --n_workers 10 --worker $worker --mapping data/geneid2uniprot.json  --species rat,mouse,rabbit,hamster &
done



