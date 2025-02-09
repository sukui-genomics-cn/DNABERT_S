# Scripts for our experiments
export PATH_TO_RESULT_DICT=/home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/DNABERT_S/outputs/big_batch # (e.g., ./results)
export PATH_TO_DATA_DICT=/home/share/huadjyin/home/s_sukui/02_data/07_genomics_data/Seq_class/Homo_sapiens # (e.g., /root/data/DNABERT_s_data)
export CUDA_VISIBLE_DEVICES=1

# 1. Main Experiment:

# Curriculum contrastive learning (DNABERT-S): Weighted SimCLR + Manifold i-Mix
python main_t2t.py \
    --resdir $PATH_TO_RESULT_DICT \
    --datapath $PATH_TO_DATA_DICT \
    --train_dataname inter_intron_cds_sample_chunks_1998 \
    --val_dataname inter_intron_cds_sample_chunks_1998 \
    --seed 1 \
    --logging_step 400 \
    --logging_num 1000 \
    --max_length 1000 \
    --train_batch_size 12 \
    --val_batch_size 12 \
    --lr 3e-06 \
    --lr_scale 100 \
    --epochs 10 \
    --feat_dim 128 \
    --temperature 1 \
    --con_method same_species \
    --mix \
    --mix_alpha 1.0 \
    --mix_layer_num -1 \
    --curriculum
