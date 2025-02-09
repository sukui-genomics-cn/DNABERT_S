# Scripts for our experiments
export PATH_TO_RESULT_DICT=/home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/DNABERT_S/outputs/ # (e.g., ./results)
export PATH_TO_DATA_DICT=/home/share/huadjyin/home/s_sukui/02_data/07_genomics_data/Seq_class/Homo_sapiens # (e.g., /root/data/DNABERT_s_data)
export CUDA_VISIBLE_DEVICES=1

# 1. Main Experiment:

# Curriculum contrastive learning (DNABERT-S): Weighted SimCLR + Manifold i-Mix
# /home/share/huadjyin/home/s_sukui/02_data/01_model/DNABERT-2-117M
# /home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/DNABERT_S/outputs/epoch3.inter_intron_cds_sample_chunks_1998.lr3e-06.lrscale100.bs8.maxlength1000.tmp0.05.seed1.con_methodsame_species.mixTrue.mix_layer_num-1.curriculumTrue/1200
python eval_clustering_classification_v1.py \
    --test_model_dir /home/share/huadjyin/home/s_sukui/02_data/01_model/DNABERT-2-117M \
    --data_dir $PATH_TO_DATA_DICT \
    --model_list "test"